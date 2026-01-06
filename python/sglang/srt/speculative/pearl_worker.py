import copy
import dataclasses
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch
from transformers import AutoTokenizer
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.pearl_draft_worker import PearlDraftWorker
from sglang.srt.speculative.pearl_info import PearlVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool
from sglang.srt.utils import next_power_of_2
from sglang.srt.utils.common import get_bool_env_var

logger = logging.getLogger(__name__)
_PEARL_DEBUG = get_bool_env_var("SGLANG_PEARL_DEBUG")


class PearlWorker:
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_num_steps = server_args.speculative_num_steps
        if server_args.speculative_num_steps != -1:
            if (
                server_args.speculative_num_draft_tokens is not None
                and server_args.speculative_num_draft_tokens != server_args.speculative_num_steps
            ):
                logger.warning(
                    "PEARL ignores speculative_num_draft_tokens; using speculative_num_steps=%s.",
                    server_args.speculative_num_steps,
                )
            server_args.speculative_num_draft_tokens = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self._auto_steps_enabled = server_args.speculative_num_steps == -1
        self._adaptive_steps: Optional[int] = None
        self._accept_rate_ema: Optional[float] = None
        if target_worker.model_runner.device == "cuda":
            self.device = torch.device(f"cuda:{target_worker.model_runner.gpu_id}")
        else:
            self.device = target_worker.model_runner.device
        self.target_vocab_size = target_worker.model_runner.model_config.vocab_size
        self.target_gpu_id = target_worker.model_runner.gpu_id

        # Share allocator and request pool with the target worker to align slots.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        self.draft_worker = PearlDraftWorker(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            moe_ep_rank=moe_ep_rank,
            nccl_port=nccl_port,
        )
        self._cuda_graph_draft_tokens = None
        if (
            server_args.speculative_num_steps == -1
            and not server_args.disable_cuda_graph
        ):
            graph_runner = getattr(self.target_worker.model_runner, "graph_runner", None)
            if graph_runner is not None:
                self._cuda_graph_draft_tokens = graph_runner.num_tokens_per_bs
            else:
                self._cuda_graph_draft_tokens = server_args.speculative_num_draft_tokens
        self._auto_steps_buckets = None
        if self.speculative_num_steps == -1:
            self._auto_steps_buckets = self._auto_tune_steps()
            self._apply_auto_steps(batch_size=1)
        logger.info(
            "PEARL init target_model=%s target_gpu=%s draft_model=%s draft_gpu=%s",
            server_args.model_path,
            self.target_gpu_id,
            server_args.speculative_draft_model_path,
            server_args.speculative_draft_gpu_id,
        )
        self._draft_executor = ThreadPoolExecutor(max_workers=1)
        self._token_id_map_cpu = self._build_token_id_map()
        self._token_id_map_device: Optional[torch.Tensor] = None

        self.model_runner = self.target_worker.model_runner
        self.model_config = self.target_worker.model_config

        if torch.cuda.is_available():
            torch.cuda.set_device(self.target_gpu_id)

    def _build_sampling_info(self, reqs):
        class _DummyBatch:
            __slots__ = ("reqs", "device", "__weakref__")

            def __init__(self, reqs, device):
                self.reqs = reqs
                self.device = device

        dummy = _DummyBatch(reqs=reqs, device=self.device)
        return SamplingBatchInfo.from_schedule_batch(dummy, self.target_vocab_size)

    def _profile_model_speed(
        self, model_runner, batch_size: int, profile_steps: int, skip_steps: int
    ) -> float:
        if batch_size <= 0:
            return 0.0
        if model_runner.device == "cuda":
            torch.cuda.set_device(model_runner.gpu_id)
        device_module = torch.get_device_module(model_runner.device)
        timings = []
        with torch.inference_mode():
            for _ in range(profile_steps):
                device_module.synchronize()
                start_time = time.time()
                model_runner._dummy_run(batch_size)
                device_module.synchronize()
                timings.append(time.time() - start_time)
        timings = timings[skip_steps:] if len(timings) > skip_steps else timings
        if not timings:
            return 0.0
        avg = sum(timings) / len(timings)
        return batch_size / max(avg, 1e-6)

    def _auto_tune_steps(self) -> list[tuple[int, int]]:
        default_buckets = [1, 2, 4, 8, 16, 32]
        buckets = default_buckets
        profile_steps = 12
        skip_steps = 2
        auto_steps = (self.server_args.speculative_auto_steps or "").strip()
        max_steps = None
        steps_mult = None
        if auto_steps:
            for chunk in auto_steps.split(","):
                chunk = chunk.strip()
                if not chunk:
                    continue
                if "=" not in chunk:
                    raise ValueError(
                        "PEARL auto steps must use key=value pairs, got %r" % chunk
                    )
                key, value = chunk.split("=", 1)
                key = key.strip().lower()
                value = value.strip()
                if key == "max":
                    max_steps = int(value)
                elif key == "mult":
                    steps_mult = float(value)
                else:
                    raise ValueError(
                        "PEARL auto steps unknown key %r (use max, mult)" % key
                    )
        if max_steps is None:
            max_steps = 8
        if steps_mult is None:
            steps_mult = 4.0
        max_steps = max(1, max_steps)
        if steps_mult <= 0:
            raise ValueError("PEARL auto steps multiplier must be positive.")
        if profile_steps <= 0:
            raise ValueError("PEARL auto steps profile_steps must be positive.")
        if skip_steps < 0:
            raise ValueError("PEARL auto steps skip_steps must be non-negative.")
        logger.info(
            "PEARL auto steps profiling: buckets=%s profile_steps=%s skip_steps=%s "
            "max_steps=%s mult=%s",
            buckets,
            profile_steps,
            skip_steps,
            max_steps,
            steps_mult,
        )

        results = []
        for bucket in buckets:
            max_req = min(
                self.draft_worker.model_runner.req_to_token_pool.size,
                self.target_worker.model_runner.req_to_token_pool.size,
            )
            profile_bs = min(bucket, max_req)
            if profile_bs <= 0:
                continue
            if profile_bs != bucket:
                logger.info(
                    "PEARL auto steps: bucket=%s capped to %s for profiling",
                    bucket,
                    profile_bs,
                )
            draft_speed = self._profile_model_speed(
                self.draft_worker.model_runner, profile_bs, profile_steps, skip_steps
            )
            target_speed = self._profile_model_speed(
                self.target_worker.model_runner, profile_bs, profile_steps, skip_steps
            )
            ratio = draft_speed / max(target_speed, 1e-6)
            ratio *= steps_mult
            steps = int(round(ratio))
            steps = max(1, min(max_steps, steps))
            logger.info(
                "PEARL auto steps bucket=%s draft_speed=%.2f tok/s target_speed=%.2f tok/s steps=%s",
                bucket,
                draft_speed,
                target_speed,
                steps,
            )
            results.append((bucket, steps))
        if not results:
            logger.warning("PEARL auto steps: no valid buckets; falling back to 1.")
            results.append((1, 1))
        return results

    def _apply_auto_steps(self, batch_size: int) -> None:
        if not self._auto_steps_buckets:
            return
        bucket_steps = None
        for bucket, steps in self._auto_steps_buckets:
            if batch_size <= bucket:
                bucket_steps = steps
                break
        if bucket_steps is None:
            bucket_steps = self._auto_steps_buckets[-1][1]
        if self._adaptive_steps is not None:
            bucket_steps = min(bucket_steps, self._adaptive_steps)
        if bucket_steps == self.speculative_num_steps:
            return
        self.speculative_num_steps = bucket_steps
        self.speculative_num_draft_tokens = bucket_steps
        self.server_args.speculative_num_steps = bucket_steps
        self.server_args.speculative_num_draft_tokens = bucket_steps
        self.draft_worker.set_speculative_num_draft_tokens(bucket_steps)
        if _PEARL_DEBUG:
            logger.info(
                "PEARL auto steps applied: batch_size=%s steps=%s",
                batch_size,
                bucket_steps,
            )

    def _update_adaptive_steps(self, extra_accept_rate: float) -> None:
        if not self._auto_steps_enabled or self.speculative_num_steps <= 1:
            return
        if self._accept_rate_ema is None:
            self._accept_rate_ema = extra_accept_rate
        else:
            self._accept_rate_ema = (
                0.8 * self._accept_rate_ema + 0.2 * extra_accept_rate
            )

        ema = self._accept_rate_ema
        if ema < 0.5:
            scale = 0.4
        elif ema < 0.7:
            scale = 0.6
        elif ema < 0.85:
            scale = 0.8
        else:
            scale = 1.0
        new_steps = max(1, int(round(self.speculative_num_steps * scale)))
        if self._adaptive_steps != new_steps:
            self._adaptive_steps = new_steps
            if _PEARL_DEBUG:
                logger.info(
                    "PEARL adaptive steps: extra_accept_rate=%.3f ema=%.3f steps=%s",
                    extra_accept_rate,
                    ema,
                    new_steps,
                )

    def _should_disable_cuda_graph(self) -> bool:
        return False

    def _get_graph_token_num(self) -> int:
        if self.server_args.disable_cuda_graph:
            return self.speculative_num_draft_tokens
        graph_runner = getattr(self.target_worker.model_runner, "graph_runner", None)
        if graph_runner is not None:
            return int(graph_runner.num_tokens_per_bs)
        if self._cuda_graph_draft_tokens is not None:
            return self._cuda_graph_draft_tokens
        return self.speculative_num_draft_tokens

    def _pad_verify_tokens(
        self, tokens: torch.Tensor | list[torch.Tensor], draft_token_num: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(tokens, list):
            lengths = [min(int(t.numel()), draft_token_num) for t in tokens]
            padded = []
            for t in tokens:
                t = t.to(self.device)
                if t.numel() > draft_token_num:
                    t = t[:draft_token_num]
                if t.numel() < draft_token_num:
                    pad = torch.zeros(
                        (draft_token_num - t.numel(),),
                        dtype=t.dtype,
                        device=self.device,
                    )
                    t = torch.cat([t, pad], dim=0)
                padded.append(t)
            out = torch.stack(padded, dim=0)
            valid_lens = torch.tensor(lengths, dtype=torch.int32, device=self.device)
            return out, valid_lens

        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        cur = tokens.shape[1]
        valid_lens = torch.full(
            (tokens.shape[0],), cur, dtype=torch.int32, device=self.device
        )
        if cur == draft_token_num:
            return tokens, valid_lens
        if cur > draft_token_num:
            return tokens[:, :draft_token_num], valid_lens.clamp(max=draft_token_num)
        pad = torch.zeros(
            (tokens.shape[0], draft_token_num - cur),
            dtype=tokens.dtype,
            device=tokens.device,
        )
        return torch.cat([tokens, pad], dim=1), valid_lens

    def clear_cache_pool(self):
        # Allocator is shared with target worker.
        pass

    def _build_token_id_map(self) -> Optional[torch.Tensor]:
        draft_path = self.server_args.speculative_draft_model_path
        target_path = self.server_args.model_path
        if not draft_path or not target_path or draft_path == target_path:
            return None
        try:
            draft_tok = AutoTokenizer.from_pretrained(
                draft_path, local_files_only=True, use_fast=True
            )
            target_tok = AutoTokenizer.from_pretrained(
                target_path, local_files_only=True, use_fast=True
            )
        except Exception as exc:
            logger.warning("PEARL tokenizer load failed: %s", exc)
            return None

        draft_vocab = draft_tok.get_vocab()
        target_vocab = target_tok.get_vocab()
        if len(draft_vocab) == len(target_vocab) and draft_vocab == target_vocab:
            return None

        missing_in_target = sum(1 for token in draft_vocab if token not in target_vocab)
        missing_in_draft = sum(1 for token in target_vocab if token not in draft_vocab)
        logger.error(
            "PEARL requires identical draft/target tokenizers. "
            "draft_vocab=%d target_vocab=%d missing_in_target=%d missing_in_draft=%d",
            len(draft_vocab),
            len(target_vocab),
            missing_in_target,
            missing_in_draft,
        )
        raise RuntimeError(
            "PEARL tokenizer mismatch: use a draft model with the same tokenizer as the target."
        )

    def _allocate_draft_slots(
        self, batch: ScheduleBatch, draft_token_num: Optional[int] = None
    ):
        draft_token_num = draft_token_num or self.speculative_num_draft_tokens
        if self.page_size == 1:
            out_cache_loc = alloc_token_slots(
                batch.tree_cache,
                batch.batch_size() * draft_token_num,
            )
            end_offset = batch.seq_lens + draft_token_num
        else:
            prefix_lens = batch.seq_lens
            prefix_lens_cpu = batch.seq_lens_cpu
            end_offset = prefix_lens + draft_token_num
            end_offset_cpu = prefix_lens_cpu + draft_token_num
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                prefix_lens,
            )
            out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                prefix_lens,
                prefix_lens_cpu,
                end_offset,
                end_offset_cpu,
                last_loc,
                batch.batch_size() * draft_token_num,
            )

        bs = batch.batch_size()
        assign_req_to_token_pool[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            next_power_of_2(bs),
        )
        return out_cache_loc

    def _build_positions(
        self, batch: ScheduleBatch, draft_token_num: Optional[int] = None
    ) -> torch.Tensor:
        draft_token_num = draft_token_num or self.speculative_num_draft_tokens
        base = batch.seq_lens.to(self.device, non_blocking=True).unsqueeze(1)
        offsets = torch.arange(
            draft_token_num, device=self.device
        ).unsqueeze(0)
        positions = (base + offsets).reshape(-1)
        return positions

    def _build_custom_mask(
        self,
        batch: ScheduleBatch,
        draft_token_num: Optional[int] = None,
        valid_draft_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        draft_token_num = draft_token_num or self.speculative_num_draft_tokens
        masks = []
        base_draft_mask = torch.tril(
            torch.ones(
                (draft_token_num, draft_token_num),
                dtype=torch.bool,
                device=self.device,
            )
        )
        if valid_draft_lens is None:
            valid_lens = [draft_token_num] * batch.batch_size()
        elif isinstance(valid_draft_lens, torch.Tensor):
            valid_lens = [int(x) for x in valid_draft_lens.tolist()]
        else:
            valid_lens = [int(x) for x in valid_draft_lens]

        for idx, seq_len in enumerate(batch.seq_lens_cpu.tolist()):
            prefix_len = max(int(seq_len), 0)
            prefix_mask = torch.ones(
                (draft_token_num, prefix_len),
                dtype=torch.bool,
                device=self.device,
            )
            valid_len = valid_lens[idx] if idx < len(valid_lens) else draft_token_num
            if valid_len < draft_token_num:
                if valid_len <= 0:
                    prefix_mask.zero_()
                    draft_mask = torch.zeros(
                        (draft_token_num, draft_token_num),
                        dtype=torch.bool,
                        device=self.device,
                    )
                else:
                    prefix_mask[valid_len:, :] = False
                    draft_mask = base_draft_mask.clone()
                    draft_mask[valid_len:, :] = False
                    draft_mask[:, valid_len:] = False
            else:
                draft_mask = base_draft_mask
            masks.append(torch.cat([prefix_mask, draft_mask], dim=1).reshape(-1))
        if not masks:
            return torch.empty((0,), dtype=torch.bool, device=self.device)
        return torch.cat(masks, dim=0)

    def _compute_prefix_logits(self, batch: ScheduleBatch):
        max_context_len = self.req_to_token_pool.req_to_token.shape[1]
        max_reqs = self.req_to_token_pool.req_to_token.shape[0]
        seq_lens_cpu = batch.seq_lens_cpu.tolist()
        req_pool_indices_cpu = batch.req_pool_indices.cpu().tolist()
        pre_indices = []
        for i, req in enumerate(batch.reqs):
            if not getattr(req, "pre_verify", True):
                continue
            if getattr(req, "pearl_prev_logits", None) is not None:
                # We already have the prefix logits from a prior target decode.
                continue
            seq_len = int(seq_lens_cpu[i])
            req_pool_idx = int(req_pool_indices_cpu[i])
            if seq_len <= 0 or seq_len >= max_context_len:
                continue
            if req_pool_idx < 0 or req_pool_idx >= max_reqs:
                continue
            pre_indices.append(i)
        if not pre_indices:
            return None, None

        input_ids = []
        seq_lens = []
        seq_lens_cpu = []
        req_pool_indices = []
        for idx in pre_indices:
            req = batch.reqs[idx]
            last_token = (
                req.output_ids[-1]
                if req.output_ids
                else req.origin_input_ids[-1]
            )
            input_ids.append(last_token)
            seq_len = int(batch.seq_lens_cpu[idx])
            seq_lens.append(seq_len)
            seq_lens_cpu.append(seq_len)
            req_pool_indices.append(int(req_pool_indices_cpu[idx]))

        out_cache_loc = self.token_to_kv_pool_allocator.alloc(len(pre_indices))
        if out_cache_loc is None:
            raise RuntimeError("PEARL prefix logits KV allocation failed.")

        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device)
        seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=self.device)
        seq_lens_cpu = torch.tensor(seq_lens_cpu, dtype=torch.int64)
        req_pool_indices = torch.tensor(
            req_pool_indices, dtype=torch.int64, device=self.device
        )
        out_cache_loc = out_cache_loc.to(self.device)

        # Snapshot existing token mapping to avoid corrupting speculative slots.
        orig_mapping = self.req_to_token_pool.req_to_token[
            req_pool_indices, seq_lens
        ].clone()

        model_worker_batch = ModelWorkerBatch(
            forward_mode=ForwardMode.DECODE,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=int(seq_lens.sum().item()),
            return_logprob=False,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            global_num_tokens=None,
            global_num_tokens_for_logprob=None,
            is_extend_in_batch=False,
            can_run_dp_cuda_graph=False,
            tbo_split_seq_index=None,
            global_forward_mode=None,
            extend_num_tokens=None,
            extend_seq_lens=None,
            extend_prefix_lens=None,
            extend_logprob_start_lens=None,
            extend_input_logprob_token_ids=None,
            multimodal_inputs=[batch.reqs[i].multimodal_inputs for i in pre_indices],
            encoder_cached=None,
            encoder_lens=None,
            encoder_lens_cpu=None,
            encoder_out_cache_loc=None,
            lora_ids=[batch.reqs[i].lora_id for i in pre_indices],
            sampling_info=batch.sampling_info,
            input_embeds=None,
            token_type_ids=None,
            spec_algorithm=None,
            spec_info=None,
            hicache_consumer_index=-1,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            disable_cuda_graph=True,
            is_prefill_only=False,
            dimensions=None,
            dllm_block_offsets=None,
            dllm_config=None,
            reqs=[batch.reqs[i] for i in pre_indices],
            has_grammar=False,
            mamba_track_indices=None,
            mamba_track_mask=None,
            mamba_track_seqlens=None,
        )

        forward_batch = ForwardBatch.init_new(model_worker_batch, self.target_worker.model_runner)
        forward_batch.positions = (seq_lens - 1).to(self.device)
        out = self.target_worker.model_runner.forward(forward_batch)
        logits_output = out.logits_output
        prefix_logits = logits_output.next_token_logits

        # Restore mapping and free temporary KV slots.
        self.req_to_token_pool.write(
            (req_pool_indices, seq_lens), orig_mapping.to(torch.int32)
        )
        self.token_to_kv_pool_allocator.free(out_cache_loc)
        return pre_indices, prefix_logits

    def _slice_out_cache_loc(
        self,
        out_cache_loc: torch.Tensor,
        indices: list[int],
        draft_token_num: Optional[int] = None,
    ):
        if out_cache_loc is None or not indices:
            return None
        stride = draft_token_num or self.speculative_num_draft_tokens
        segments = []
        for idx in indices:
            start = idx * stride
            end = start + stride
            segments.append(out_cache_loc[start:end])
        if not segments:
            return out_cache_loc[:0]
        return torch.cat(segments, dim=0)

    def _record_out_cache_segments(
        self,
        out_cache_segments: list[Optional[torch.Tensor]],
        indices: list[int],
        out_cache_loc: Optional[torch.Tensor],
        accept_length_list: list[int],
    ) -> None:
        if out_cache_loc is None or not indices:
            return
        cursor = 0
        for local_idx, batch_idx in enumerate(indices):
            keep = int(accept_length_list[local_idx]) + 1
            if keep <= 0:
                out_cache_segments[batch_idx] = out_cache_loc[:0]
                continue
            out_cache_segments[batch_idx] = out_cache_loc[cursor : cursor + keep]
            cursor += keep

    def _build_last_token_ids(self, batch: ScheduleBatch) -> torch.Tensor:
        token_ids = []
        for req in batch.reqs:
            if req.output_ids:
                token_ids.append(req.output_ids[-1])
            else:
                token_ids.append(req.origin_input_ids[-1])
        return torch.tensor(token_ids, dtype=torch.int64, device=self.device)

    def _verify_subset(
        self,
        batch: ScheduleBatch,
        indices: list[int],
        verify_tokens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        prefix_logits: Optional[torch.Tensor] = None,
        prefix_logits_mask: Optional[torch.Tensor] = None,
        next_window_tokens: Optional[torch.Tensor] = None,
        disable_cuda_graph: bool = False,
        draft_token_num: Optional[int] = None,
        valid_draft_lens: Optional[torch.Tensor] = None,
    ):
        if not indices:
            return None

        sub_reqs = [batch.reqs[i] for i in indices]
        sampling_info = self._build_sampling_info(sub_reqs)
        sub_batch = dataclasses.replace(batch)
        sub_batch.reqs = sub_reqs
        sub_batch.disable_cuda_graph = disable_cuda_graph
        sub_batch.req_pool_indices = batch.req_pool_indices[indices]
        sub_batch.seq_lens = batch.seq_lens[indices]
        sub_batch.seq_lens_cpu = batch.seq_lens_cpu[indices]
        sub_batch.seq_lens_sum = int(sub_batch.seq_lens.sum().item())
        if batch.orig_seq_lens is not None:
            sub_batch.orig_seq_lens = batch.orig_seq_lens[indices]
        sub_batch.multimodal_inputs = [req.multimodal_inputs for req in sub_reqs]
        sub_batch.return_logprob = batch.return_logprob
        if sub_batch.return_logprob:
            sub_batch.top_logprobs_nums = [batch.top_logprobs_nums[i] for i in indices]
            sub_batch.token_ids_logprobs = [
                batch.token_ids_logprobs[i] for i in indices
            ]
        else:
            sub_batch.top_logprobs_nums = None
            sub_batch.token_ids_logprobs = None
        sub_batch.has_stream = any(req.stream for req in sub_reqs)
        sub_batch.has_grammar = False
        sub_batch.sampling_info = sampling_info
        sub_batch.forward_mode = ForwardMode.TARGET_VERIFY
        sub_batch.spec_algorithm = SpeculativeAlgorithm.PEARL
        sub_batch.input_ids = verify_tokens.reshape(-1)
        sub_batch.out_cache_loc = out_cache_loc

        token_num = draft_token_num or self.speculative_num_draft_tokens
        positions = self._build_positions(sub_batch, token_num)
        custom_mask = self._build_custom_mask(
            sub_batch, token_num, valid_draft_lens=valid_draft_lens
        )
        spec_info = PearlVerifyInput(
            sub_batch.input_ids,
            custom_mask,
            positions,
            token_num,
            self.target_vocab_size,
            next_window_tokens=next_window_tokens,
            prefix_logits=prefix_logits,
            prefix_logits_mask=prefix_logits_mask,
            valid_draft_lens=valid_draft_lens,
        )
        sub_batch.spec_info = spec_info

        model_worker_batch = sub_batch.get_model_worker_batch()
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )

        logits_output = batch_result.logits_output
        (
            logits_output,
            _verified_id,
            _num_accepted_tokens,
            accept_length_list,
        ) = spec_info.verify(sub_batch, logits_output, self.page_size)
        return {
            "sub_batch": sub_batch,
            "spec_info": spec_info,
            "logits_output": logits_output,
            "accept_length_list": accept_length_list,
        }

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        self._apply_auto_steps(batch.batch_size())
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return self.target_worker.forward_batch_generation(
                batch.get_model_worker_batch()
            )

        if any(req.grammar is not None for req in batch.reqs):
            logger.warning("PEARL does not support grammar; falling back to target.")
            return self.target_worker.forward_batch_generation(
                batch.get_model_worker_batch()
            )

        if torch.cuda.is_available():
            torch.cuda.set_device(self.target_gpu_id)

        batch.spec_info = None
        graph_token_num = self._get_graph_token_num()
        out_cache_loc = self._allocate_draft_slots(batch, graph_token_num)
        draft_start = time.perf_counter()
        draft_elapsed = None
        draft_future = self._draft_executor.submit(self.draft_worker.run_draft, batch)

        post_indices = []
        pre_indices = []
        for i, req in enumerate(batch.reqs):
            prev_window = getattr(req, "pearl_prev_window", None)
            if not getattr(req, "pre_verify", True) and prev_window is not None:
                post_indices.append(i)
            else:
                pre_indices.append(i)

        pre_logits = None
        prefix_logits = None
        prefix_logits_mask = None
        if pre_indices:
            pre_logits_indices, pre_logits = self._compute_prefix_logits(batch)
            if pre_logits_indices:
                prefix_logits = torch.zeros(
                    (batch.batch_size(), self.target_vocab_size),
                    dtype=pre_logits.dtype,
                    device=pre_logits.device,
                )
                prefix_logits_mask = torch.zeros(
                    (batch.batch_size(),), dtype=torch.bool, device=pre_logits.device
                )
                for row, idx in enumerate(pre_logits_indices):
                    prefix_logits[idx] = pre_logits[row]
                    prefix_logits_mask[idx] = True

        batch.spec_algorithm = SpeculativeAlgorithm.PEARL
        batch.forward_mode = ForwardMode.TARGET_VERIFY

        allow_split = (
            post_indices
            and pre_indices
            and not batch.return_logprob
            and not any(req.return_hidden_states for req in batch.reqs)
        )

        accept_length_list_full = [0 for _ in range(batch.batch_size())]
        out_cache_segments: list[Optional[torch.Tensor]] = [
            None for _ in range(batch.batch_size())
        ]
        logits_output = None
        can_run_cuda_graph = False

        if allow_split:
            disable_cuda_graph = self._should_disable_cuda_graph()
            verify_tokens = []
            for idx in post_indices:
                prev_window = batch.reqs[idx].pearl_prev_window
                if prev_window is not None and prev_window.numel() > self.speculative_num_steps:
                    prev_window = prev_window[: self.speculative_num_steps]
                verify_tokens.append(prev_window.to(self.device))
                if _PEARL_DEBUG:
                    logger.info(
                        "PEARL verify window req=%s mode=post prev_window_len=%d",
                        batch.reqs[idx].rid,
                        prev_window.numel(),
                    )
            verify_tokens, valid_lens = self._pad_verify_tokens(
                verify_tokens, graph_token_num
            )
            post_out_cache_loc = self._slice_out_cache_loc(
                out_cache_loc, post_indices, graph_token_num
            )
            post_result = self._verify_subset(
                batch,
                post_indices,
                verify_tokens,
                post_out_cache_loc,
                prefix_logits=(
                    prefix_logits[post_indices] if prefix_logits is not None else None
                ),
                prefix_logits_mask=(
                    prefix_logits_mask[post_indices]
                    if prefix_logits_mask is not None
                    else None
                ),
                disable_cuda_graph=disable_cuda_graph,
                draft_token_num=graph_token_num,
                valid_draft_lens=valid_lens,
            )
            if post_result is not None:
                logits_output = post_result["logits_output"]
                accept_length_list = post_result["accept_length_list"]
                for local_idx, batch_idx in enumerate(post_indices):
                    accept_length_list_full[batch_idx] = accept_length_list[local_idx]
                self._record_out_cache_segments(
                    out_cache_segments,
                    post_indices,
                    post_result["sub_batch"].out_cache_loc,
                    accept_length_list,
                )

            draft_tokens = draft_future.result()
            draft_elapsed = time.perf_counter() - draft_start
            if self.target_vocab_size:
                draft_tokens = torch.where(
                    draft_tokens < self.target_vocab_size,
                    draft_tokens,
                    torch.zeros_like(draft_tokens),
                )
            if draft_tokens.device != self.device:
                draft_tokens = draft_tokens.to(self.device, non_blocking=True)

            pre_verify_tokens = draft_tokens[pre_indices]
            if _PEARL_DEBUG:
                for local_idx, idx in enumerate(pre_indices):
                    logger.info(
                        "PEARL verify window req=%s mode=pre current_window_len=%d",
                        batch.reqs[idx].rid,
                        pre_verify_tokens[local_idx].numel(),
                    )
            pre_verify_tokens, valid_lens = self._pad_verify_tokens(
                pre_verify_tokens, graph_token_num
            )
            pre_out_cache_loc = self._slice_out_cache_loc(
                out_cache_loc, pre_indices, graph_token_num
            )
            pre_result = self._verify_subset(
                batch,
                pre_indices,
                pre_verify_tokens,
                pre_out_cache_loc,
                prefix_logits=(
                    prefix_logits[pre_indices] if prefix_logits is not None else None
                ),
                prefix_logits_mask=(
                    prefix_logits_mask[pre_indices]
                    if prefix_logits_mask is not None
                    else None
                ),
                next_window_tokens=draft_tokens[pre_indices],
                disable_cuda_graph=disable_cuda_graph,
                draft_token_num=graph_token_num,
                valid_draft_lens=valid_lens,
            )
            if pre_result is not None:
                logits_output = pre_result["logits_output"]
                accept_length_list = pre_result["accept_length_list"]
                for local_idx, batch_idx in enumerate(pre_indices):
                    accept_length_list_full[batch_idx] = accept_length_list[local_idx]
                self._record_out_cache_segments(
                    out_cache_segments,
                    pre_indices,
                    pre_result["sub_batch"].out_cache_loc,
                    accept_length_list,
                )

            for idx in post_indices:
                req = batch.reqs[idx]
                if not getattr(req, "pre_verify", True) and not req.finished():
                    req.pearl_prev_window = draft_tokens[idx]
            can_run_cuda_graph = False
        else:
            post_verify_only = not pre_indices
            verify_tokens = []
            draft_tokens = None
            if post_verify_only:
                for req in batch.reqs:
                    prev_window = req.pearl_prev_window
                    if prev_window is not None and prev_window.numel() > self.speculative_num_steps:
                        prev_window = prev_window[: self.speculative_num_steps]
                    verify_tokens.append(prev_window.to(self.device))
                    if _PEARL_DEBUG:
                        logger.info(
                            "PEARL verify window req=%s mode=post prev_window_len=%d",
                            req.rid,
                            prev_window.numel(),
                        )
                verify_tokens, valid_lens = self._pad_verify_tokens(
                    verify_tokens, graph_token_num
                )
            else:
                draft_tokens = draft_future.result()
                draft_elapsed = time.perf_counter() - draft_start
                if self.target_vocab_size:
                    draft_tokens = torch.where(
                        draft_tokens < self.target_vocab_size,
                        draft_tokens,
                        torch.zeros_like(draft_tokens),
                    )
                if draft_tokens.device != self.device:
                    draft_tokens = draft_tokens.to(self.device, non_blocking=True)

                for i, req in enumerate(batch.reqs):
                    prev_window = getattr(req, "pearl_prev_window", None)
                    if not getattr(req, "pre_verify", True) and prev_window is not None:
                        if prev_window.numel() > self.speculative_num_steps:
                            prev_window = prev_window[: self.speculative_num_steps]
                        verify_tokens.append(prev_window.to(self.device))
                        if _PEARL_DEBUG:
                            logger.info(
                                "PEARL verify window req=%s mode=post prev_window_len=%d",
                                req.rid,
                                prev_window.numel(),
                            )
                    else:
                        verify_tokens.append(draft_tokens[i])
                        if _PEARL_DEBUG:
                            logger.info(
                                "PEARL verify window req=%s mode=%s current_window_len=%d",
                                req.rid,
                                "pre" if getattr(req, "pre_verify", True) else "post",
                                draft_tokens[i].numel(),
                            )
                verify_tokens, valid_lens = self._pad_verify_tokens(
                    verify_tokens, graph_token_num
                )

            batch.out_cache_loc = out_cache_loc
            batch.input_ids = verify_tokens.reshape(-1)
            batch.disable_cuda_graph = self._should_disable_cuda_graph()
            spec_info = PearlVerifyInput(
                batch.input_ids,
                self._build_custom_mask(
                    batch, graph_token_num, valid_draft_lens=valid_lens
                ),
                self._build_positions(batch, graph_token_num),
                graph_token_num,
                self.target_vocab_size,
                next_window_tokens=draft_tokens if not post_verify_only else None,
                prefix_logits=prefix_logits,
                prefix_logits_mask=prefix_logits_mask,
                valid_draft_lens=valid_lens,
            )
            batch.spec_info = spec_info

            model_worker_batch = batch.get_model_worker_batch()
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, is_verify=True
            )

            logits_output, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.can_run_cuda_graph,
            )
            (
                logits_output,
                _verified_id,
                _num_accepted_tokens,
                accept_length_list,
            ) = spec_info.verify(batch, logits_output, self.page_size)
            accept_length_list_full = accept_length_list
            self._record_out_cache_segments(
                out_cache_segments,
                list(range(batch.batch_size())),
                batch.out_cache_loc,
                accept_length_list,
            )

            if post_verify_only:
                draft_tokens = draft_future.result()
                draft_elapsed = time.perf_counter() - draft_start
                if self.target_vocab_size:
                    draft_tokens = torch.where(
                        draft_tokens < self.target_vocab_size,
                        draft_tokens,
                        torch.zeros_like(draft_tokens),
                    )
                if draft_tokens.device != self.device:
                    draft_tokens = draft_tokens.to(self.device, non_blocking=True)
                for i, req in enumerate(batch.reqs):
                    if not getattr(req, "pre_verify", True) and not req.finished():
                        req.pearl_prev_window = draft_tokens[i]

        new_seq_lens_cpu = torch.tensor(
            [
                len(req.origin_input_ids) + len(req.output_ids)
                for req in batch.reqs
            ],
            dtype=torch.int64,
        )
        batch.seq_lens_cpu = new_seq_lens_cpu
        batch.seq_lens = new_seq_lens_cpu.to(self.device, non_blocking=True)

        if all(segment is not None for segment in out_cache_segments):
            batch.out_cache_loc = torch.cat(out_cache_segments, dim=0)

        spec_info_full = PearlVerifyInput(
            torch.zeros(
                (batch.batch_size() * graph_token_num,),
                dtype=torch.int64,
                device=self.device,
            ),
            self._build_custom_mask(batch, graph_token_num),
            self._build_positions(batch, graph_token_num),
            graph_token_num,
            self.target_vocab_size,
        )
        spec_info_full.accept_length = torch.tensor(
            accept_length_list_full, dtype=torch.int32, device=self.device
        )
        batch.spec_info = spec_info_full
        if accept_length_list_full:
            steps = max(self.speculative_num_steps, 1)
            extra_accept_max = max(steps - 1, 1)
            extra_accept_rate = sum(accept_length_list_full) / (
                len(accept_length_list_full) * extra_accept_max
            )
            accept_count_sum = sum(
                int(getattr(req, "pearl_accept_count", 0)) for req in batch.reqs
            )
            total_accept_rate = accept_count_sum / (len(accept_length_list_full) * steps)
            self._update_adaptive_steps(extra_accept_rate)
            avg_accept = sum(accept_length_list_full) / len(accept_length_list_full)
            avg_accept_count = accept_count_sum / len(accept_length_list_full)
            logger.info(
                "PEARL batch stats: bs=%d pre=%d post=%d steps=%d accept_rate=%.3f "
                "extra_accept_rate=%.3f avg_accept=%.2f avg_accept_count=%.2f draft_ms=%.2f",
                batch.batch_size(),
                len(pre_indices),
                len(post_indices),
                steps,
                total_accept_rate,
                extra_accept_rate,
                avg_accept,
                avg_accept_count,
                (draft_elapsed or 0.0) * 1000.0,
            )

        self._recompute_revised_kv(batch, spec_info_full)
        self.draft_worker.update_after_verify(batch.reqs)

        batch.forward_mode = ForwardMode.DECODE
        for req in batch.reqs:
            if req.finished():
                self.draft_worker.release_req(req)

        next_token_ids = self._build_last_token_ids(batch)
        num_accepted_tokens = int(sum(accept_length_list_full)) + batch.batch_size()

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            num_accepted_tokens=num_accepted_tokens,
            accept_length_per_req_cpu=accept_length_list_full,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def _recompute_revised_kv(self, batch: ScheduleBatch, spec_info: PearlVerifyInput):
        if spec_info.accept_length is None:
            return

        accept_lengths = spec_info.accept_length.cpu().tolist()
        revised_entries = []
        prefix = 0
        total_slots = (
            int(batch.out_cache_loc.numel()) if batch.out_cache_loc is not None else 0
        )
        for i, req in enumerate(batch.reqs):
            revised = getattr(req, "pearl_revised_token", None)
            keep = accept_lengths[i] + 1
            if revised is not None:
                offset, token_id = revised
                if offset < keep:
                    cache_index = prefix + offset
                    if cache_index >= total_slots:
                        req.pearl_revised_token = None
                        prefix += keep
                        continue
                    base_len = int(batch.seq_lens[i].item()) - keep
                    position = base_len + offset
                    revised_entries.append((i, cache_index, token_id, position))
            prefix += keep
            req.pearl_revised_token = None

        if not revised_entries:
            return

        indices = [entry[0] for entry in revised_entries]
        cache_indices = [entry[1] for entry in revised_entries]
        token_ids = [entry[2] for entry in revised_entries]
        positions = [entry[3] for entry in revised_entries]

        input_ids = torch.tensor(token_ids, dtype=torch.int64, device=self.device)
        seq_lens = torch.tensor(positions, dtype=torch.int64, device=self.device)
        seq_lens_cpu = torch.tensor(positions, dtype=torch.int64)
        req_pool_indices = batch.req_pool_indices[indices].to(
            self.device, non_blocking=True
        )
        out_cache_loc = batch.out_cache_loc[cache_indices].to(
            self.device, non_blocking=True
        )

        sampling_info = batch.sampling_info
        if len(indices) != len(sampling_info):
            sampling_info = copy.deepcopy(sampling_info)
            keep_indices_device = torch.tensor(indices, device=self.device)
            sampling_info.filter_batch(indices, keep_indices_device)

        model_worker_batch = ModelWorkerBatch(
            forward_mode=ForwardMode.DECODE,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=int(seq_lens.sum().item()),
            return_logprob=False,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            global_num_tokens=None,
            global_num_tokens_for_logprob=None,
            is_extend_in_batch=False,
            can_run_dp_cuda_graph=False,
            tbo_split_seq_index=None,
            global_forward_mode=None,
            extend_num_tokens=None,
            extend_seq_lens=None,
            extend_prefix_lens=None,
            extend_logprob_start_lens=None,
            extend_input_logprob_token_ids=None,
            multimodal_inputs=[batch.reqs[i].multimodal_inputs for i in indices],
            encoder_cached=None,
            encoder_lens=None,
            encoder_lens_cpu=None,
            encoder_out_cache_loc=None,
            lora_ids=[batch.reqs[i].lora_id for i in indices],
            sampling_info=sampling_info,
            input_embeds=None,
            token_type_ids=None,
            spec_algorithm=None,
            spec_info=None,
            hicache_consumer_index=-1,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            disable_cuda_graph=True,
            is_prefill_only=False,
            dimensions=None,
            dllm_block_offsets=None,
            dllm_config=None,
            reqs=[batch.reqs[i] for i in indices],
            has_grammar=False,
            mamba_track_indices=None,
            mamba_track_mask=None,
            mamba_track_seqlens=None,
        )

        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output = batch_result.logits_output
        if logits_output is not None and logits_output.next_token_logits is not None:
            for idx, entry in enumerate(revised_entries):
                req = batch.reqs[entry[0]]
                req.pearl_prev_logits = logits_output.next_token_logits[idx].detach()
