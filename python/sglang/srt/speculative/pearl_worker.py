import copy
import logging
import os
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

    def _parse_int_list_env(self, name: str, default: list[int]) -> list[int]:
        raw = os.environ.get(name)
        if not raw:
            return default
        values = []
        for chunk in raw.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                value = int(chunk)
            except ValueError:
                logger.warning("PEARL auto steps: invalid %s entry %r", name, chunk)
                continue
            if value <= 0:
                logger.warning("PEARL auto steps: non-positive %s entry %r", name, chunk)
                continue
            values.append(value)
        return values or default

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
        buckets = self._parse_int_list_env(
            "SGLANG_PEARL_AUTO_STEPS_BS", default_buckets
        )
        buckets = sorted(set(buckets))
        profile_steps = int(os.environ.get("SGLANG_PEARL_AUTO_STEPS_PROFILE_STEPS", "12"))
        skip_steps = int(os.environ.get("SGLANG_PEARL_AUTO_STEPS_SKIP_STEPS", "2"))
        max_steps = int(os.environ.get("SGLANG_PEARL_AUTO_STEPS_MAX", "16"))
        max_steps = max(1, max_steps)
        if profile_steps <= 0:
            raise ValueError("PEARL auto steps profile_steps must be positive.")
        if skip_steps < 0:
            raise ValueError("PEARL auto steps skip_steps must be non-negative.")
        logger.info(
            "PEARL auto steps profiling: buckets=%s profile_steps=%s skip_steps=%s max_steps=%s",
            buckets,
            profile_steps,
            skip_steps,
            max_steps,
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

    def _allocate_draft_slots(self, batch: ScheduleBatch):
        if self.page_size == 1:
            out_cache_loc = alloc_token_slots(
                batch.tree_cache,
                batch.batch_size() * self.speculative_num_draft_tokens,
            )
            end_offset = batch.seq_lens + self.speculative_num_draft_tokens
        else:
            prefix_lens = batch.seq_lens
            prefix_lens_cpu = batch.seq_lens_cpu
            end_offset = prefix_lens + self.speculative_num_draft_tokens
            end_offset_cpu = prefix_lens_cpu + self.speculative_num_draft_tokens
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
                batch.batch_size() * self.speculative_num_draft_tokens,
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

    def _build_positions(self, batch: ScheduleBatch) -> torch.Tensor:
        base = batch.seq_lens.to(self.device, non_blocking=True).unsqueeze(1)
        offsets = torch.arange(
            self.speculative_num_draft_tokens, device=self.device
        ).unsqueeze(0)
        positions = (base + offsets).reshape(-1)
        return positions

    def _build_custom_mask(self, batch: ScheduleBatch) -> torch.Tensor:
        masks = []
        draft_mask = torch.tril(
            torch.ones(
                (self.speculative_num_draft_tokens, self.speculative_num_draft_tokens),
                dtype=torch.bool,
                device=self.device,
            )
        )
        for seq_len in batch.seq_lens_cpu.tolist():
            prefix_len = max(int(seq_len), 0)
            prefix_mask = torch.ones(
                (self.speculative_num_draft_tokens, prefix_len),
                dtype=torch.bool,
                device=self.device,
            )
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
        out_cache_loc = self._allocate_draft_slots(batch)
        draft_future = self._draft_executor.submit(self.draft_worker.run_draft, batch)

        post_verify_only = True
        for req in batch.reqs:
            prev_window = getattr(req, "pearl_prev_window", None)
            if getattr(req, "pre_verify", True) or prev_window is None:
                post_verify_only = False
                break

        pre_indices = []
        pre_logits = None
        prefix_logits = None
        prefix_logits_mask = None

        if torch.cuda.is_available():
            torch.cuda.set_device(self.target_gpu_id)
        positions = self._build_positions(batch)
        custom_mask = self._build_custom_mask(batch)

        batch.spec_algorithm = SpeculativeAlgorithm.PEARL
        batch.forward_mode = ForwardMode.TARGET_VERIFY

        if post_verify_only:
            verify_tokens = []
            for i, req in enumerate(batch.reqs):
                prev_window = req.pearl_prev_window
                verify_tokens.append(prev_window.to(self.device))
                if _PEARL_DEBUG:
                    logger.info(
                        "PEARL verify window req=%s mode=post prev_window_len=%d",
                        req.rid,
                        prev_window.numel(),
                    )
            verify_tokens = torch.stack(verify_tokens, dim=0)
            draft_tokens = None
        else:
            pre_indices, pre_logits = self._compute_prefix_logits(batch)
            draft_tokens = draft_future.result()
            if self.target_vocab_size:
                draft_tokens = torch.where(
                    draft_tokens < self.target_vocab_size,
                    draft_tokens,
                    torch.zeros_like(draft_tokens),
                )
            if draft_tokens.device != self.device:
                draft_tokens = draft_tokens.to(self.device, non_blocking=True)

            verify_tokens = []
            for i, req in enumerate(batch.reqs):
                prev_window = getattr(req, "pearl_prev_window", None)
                if not getattr(req, "pre_verify", True) and prev_window is not None:
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
            verify_tokens = torch.stack(verify_tokens, dim=0)

        batch.out_cache_loc = out_cache_loc
        batch.input_ids = verify_tokens.reshape(-1)
        if pre_indices:
            prefix_logits = torch.zeros(
                (batch.batch_size(), self.target_vocab_size),
                dtype=pre_logits.dtype,
                device=pre_logits.device,
            )
            prefix_logits_mask = torch.zeros(
                (batch.batch_size(),), dtype=torch.bool, device=pre_logits.device
            )
            for row, idx in enumerate(pre_indices):
                prefix_logits[idx] = pre_logits[row]
                prefix_logits_mask[idx] = True

        spec_info = PearlVerifyInput(
            batch.input_ids,
            custom_mask,
            positions,
            self.speculative_num_draft_tokens,
            self.target_vocab_size,
            next_window_tokens=draft_tokens if not post_verify_only else None,
            prefix_logits=prefix_logits,
            prefix_logits_mask=prefix_logits_mask,
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
            verified_id,
            num_accepted_tokens,
            accept_length_list,
        ) = spec_info.verify(batch, logits_output, self.page_size)

        if post_verify_only:
            draft_tokens = draft_future.result()
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

        self._recompute_revised_kv(batch, spec_info)
        self.draft_worker.update_after_verify(batch.reqs)

        batch.forward_mode = ForwardMode.DECODE
        for req in batch.reqs:
            if req.finished():
                self.draft_worker.release_req(req)

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=verified_id,
            num_accepted_tokens=num_accepted_tokens,
            accept_length_per_req_cpu=accept_length_list,
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
