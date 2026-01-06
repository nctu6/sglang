import copy
import logging
import os
import threading
from typing import Dict, Optional

import torch

from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import get_bool_env_var
from sglang.srt.mem_cache.common import write_cache_indices
from sglang.srt.utils import empty_context

logger = logging.getLogger(__name__)
_PEARL_DRAFT_SYNC = get_bool_env_var("SGLANG_PEARL_DRAFT_SYNC")


class PearlDraftWorker:
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
    ):
        self.server_args = server_args
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        draft_gpu_id = (
            server_args.speculative_draft_gpu_id
            if server_args.speculative_draft_gpu_id is not None
            else gpu_id
        )
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible:
            visible_ids = [int(x) for x in cuda_visible.split(",") if x.strip() != ""]
            if draft_gpu_id in visible_ids:
                draft_gpu_id = visible_ids.index(draft_gpu_id)
            else:
                logger.warning(
                    "PEARL draft GPU id %s not in CUDA_VISIBLE_DEVICES=%s; using it as local index.",
                    draft_gpu_id,
                    cuda_visible,
                )

        draft_backend_backup = server_args.speculative_draft_attention_backend
        disable_cuda_graph_backup = server_args.disable_cuda_graph
        sampling_backend_backup = server_args.sampling_backend
        if draft_backend_backup is None:
            # Prefer triton backend for draft worker stability.
            server_args.speculative_draft_attention_backend = "triton"
        server_args.disable_cuda_graph = True
        # Use pytorch sampling backend for draft to avoid custom kernel crashes.
        server_args.sampling_backend = "pytorch"

        with (
            empty_context(),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self.draft_worker = TpModelWorker(
                server_args=server_args,
                gpu_id=draft_gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # FIXME
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
            )

        server_args.speculative_draft_attention_backend = draft_backend_backup
        server_args.disable_cuda_graph = disable_cuda_graph_backup
        server_args.sampling_backend = sampling_backend_backup

        self.model_runner = self.draft_worker.model_runner
        self.model_config = self.draft_worker.model_config
        self.req_to_token_pool = self.model_runner.req_to_token_pool
        self.token_to_kv_pool_allocator = self.model_runner.token_to_kv_pool_allocator
        if self.model_runner.device == "cuda":
            self.device = torch.device(f"cuda:{self.model_runner.gpu_id}")
        else:
            self.device = self.model_runner.device

        if hasattr(self.model_runner, "attn_backend"):
            # Force paged prefill path for draft cache sync to avoid ragged failures.
            self.model_runner.attn_backend.enable_deterministic = True
        if self.model_runner.page_size != 1:
            raise ValueError("PEARL draft worker currently requires page_size == 1.")

        self._cache_lens: Dict[str, int] = {}
        self._cache_indices: Dict[str, torch.Tensor] = {}
        self._req_pool_map: Dict[int, int] = {}
        self._draft_buffers: Dict[str, list[int]] = {}
        self._lock = threading.Lock()
        self._vectorize_draft = server_args.pearl_vectorize_draft

    def clear_cache_pool(self):
        # Allocator is shared with target worker.
        pass

    def set_speculative_num_draft_tokens(self, num_tokens: int) -> None:
        if num_tokens <= 0:
            raise ValueError("PEARL speculative_num_draft_tokens must be positive.")
        if num_tokens == self.speculative_num_draft_tokens:
            return
        self.speculative_num_draft_tokens = num_tokens
        self.server_args.speculative_num_draft_tokens = num_tokens
        for rid, buffer in self._draft_buffers.items():
            if len(buffer) > num_tokens:
                self._draft_buffers[rid] = buffer[:num_tokens]

    def release_req(self, req):
        with self._lock:
            rid = req.rid
            target_req_pool_idx = req.req_pool_idx
            if target_req_pool_idx in self._req_pool_map:
                draft_idx = self._req_pool_map.pop(target_req_pool_idx)
                self.req_to_token_pool.free(draft_idx)

            indices = self._cache_indices.pop(rid, None)
            if indices is not None and indices.numel() > 0:
                self.token_to_kv_pool_allocator.free(indices)
            self._cache_lens.pop(rid, None)
            self._draft_buffers.pop(rid, None)

    def _build_sampling_info(self, reqs):
        class _DummyBatch:
            __slots__ = ("reqs", "device", "__weakref__")

            def __init__(self, reqs, device):
                self.reqs = reqs
                self.device = device

        dummy_batch = _DummyBatch(reqs=reqs, device=self.device)
        return SamplingBatchInfo.from_schedule_batch(
            dummy_batch, self.model_config.vocab_size
        )

    def _build_greedy_sampling_info(self, batch_size: int) -> SamplingBatchInfo:
        device = self.device
        temps = torch.ones((batch_size, 1), device=device)
        top_ps = torch.ones((batch_size,), device=device)
        top_ks = torch.ones((batch_size,), dtype=torch.int32, device=device)
        min_ps = torch.zeros((batch_size,), device=device)
        return SamplingBatchInfo(
            temperatures=temps,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            is_all_greedy=True,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            vocab_size=self.model_config.vocab_size,
            grammars=None,
            vocab_mask=None,
            apply_mask_func=None,
            penalizer_orchestrator=None,
            acc_linear_penalties=None,
            has_custom_logit_processor=False,
            custom_params=None,
            custom_logit_processor=None,
            sampling_seed=None,
            device=str(device),
            logit_bias=None,
        )

    def _sample_from_logits(
        self, logits: torch.Tensor, sampling_info: SamplingBatchInfo
    ):
        bs, vocab = logits.shape
        next_tokens = torch.empty((bs,), dtype=torch.int64, device=logits.device)
        next_token_probs = torch.empty((bs,), dtype=torch.float32, device=logits.device)
        temperatures = sampling_info.temperatures.view(-1).to(logits.device)
        top_ps = sampling_info.top_ps.to(logits.device)
        top_ks = sampling_info.top_ks.to(logits.device)
        min_ps = sampling_info.min_ps.to(logits.device)

        for i in range(bs):
            temp = temperatures[i].clamp(min=1e-5)
            probs = torch.softmax(logits[i].float() / temp, dim=-1)

            top_k = int(top_ks[i].item())
            if 0 < top_k < vocab:
                vals, idx = torch.topk(probs, top_k)
                masked = torch.zeros_like(probs)
                masked.scatter_(0, idx, vals)
                probs = masked

            top_p = float(top_ps[i].item())
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cdf = torch.cumsum(sorted_probs, dim=-1)
                mask = cdf <= top_p
                if mask.numel() > 0:
                    mask[0] = True
                sorted_probs = sorted_probs * mask
                norm = sorted_probs.sum()
                if norm > 0:
                    sorted_probs = sorted_probs / norm
                probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)

            min_p = float(min_ps[i].item())
            if min_p > 0.0:
                probs = torch.where(probs >= min_p, probs, torch.zeros_like(probs))
                norm = probs.sum()
                if norm > 0:
                    probs = probs / norm

            sampled = torch.multinomial(probs, 1)
            next_tokens[i] = sampled
            next_token_probs[i] = probs[sampled]

        return next_tokens, next_token_probs

    def _sync_cache(self, batch: ScheduleBatch):
        if torch.cuda.is_available():
            torch.cuda.set_device(self.model_runner.gpu_id)
        req_pool_indices = []
        missing_tokens = []
        base_lens = []

        for idx, req in enumerate(batch.reqs):
            target_req_pool_idx = int(batch.req_pool_indices[idx].item())
            if target_req_pool_idx not in self._req_pool_map:
                draft_idx = self.req_to_token_pool.alloc(1)
                if draft_idx is None:
                    raise RuntimeError("Draft req_to_token_pool allocation failed.")
                self._req_pool_map[target_req_pool_idx] = draft_idx[0]
            req_pool_indices.append(self._req_pool_map[target_req_pool_idx])

            rid = req.rid
            target_ids = req.origin_input_ids + req.output_ids
            target_len = len(target_ids)
            buffer_len = len(self._draft_buffers.get(rid, []))
            desired_len = target_len + buffer_len

            cached_len = self._cache_lens.get(rid, 0)
            cached_indices = self._cache_indices.get(
                rid, torch.empty((0,), dtype=torch.int64, device=self.device)
            )

            if cached_len > desired_len:
                to_free = cached_indices[desired_len:]
                if to_free.numel() > 0:
                    self.token_to_kv_pool_allocator.free(to_free)
                cached_indices = cached_indices[:desired_len]
                cached_len = desired_len

            if cached_len < desired_len and buffer_len > 0:
                # Cache is behind the buffer; drop buffer to keep consistency.
                self._draft_buffers[rid] = []
                buffer_len = 0
                desired_len = target_len

            if cached_len < target_len:
                missing_ids = target_ids[cached_len:]
            else:
                missing_ids = []

            self._cache_lens[rid] = cached_len
            self._cache_indices[rid] = cached_indices
            missing_tokens.append(missing_ids)
            base_lens.append(cached_len)

        max_missing = max((len(tokens) for tokens in missing_tokens), default=0)
        if max_missing == 0:
            return

        if getattr(self.token_to_kv_pool_allocator, "page_size", 1) != 1:
            # Fallback to per-token sync for paged cache.
            req_pool_indices_tensor = torch.tensor(
                req_pool_indices, dtype=torch.int64, device=self.device
            )
            for step in range(max_missing):
                active = [
                    i for i, tokens in enumerate(missing_tokens) if len(tokens) > step
                ]
                if not active:
                    break

                active_req_pool = req_pool_indices_tensor[active]
                input_ids = torch.tensor(
                    [missing_tokens[i][step] for i in active],
                    dtype=torch.int64,
                    device=self.device,
                )
                step_seq_lens = torch.tensor(
                    [base_lens[i] + step for i in active],
                    dtype=torch.int64,
                    device=self.device,
                )
                step_seq_lens_cpu = torch.tensor(
                    [base_lens[i] + step for i in active],
                    dtype=torch.int64,
                )
                out_cache_loc = self.token_to_kv_pool_allocator.alloc(len(active))
                if out_cache_loc is None:
                    raise RuntimeError("Draft KV cache allocation failed.")
                self.req_to_token_pool.write(
                    (active_req_pool, step_seq_lens), out_cache_loc.to(torch.int32)
                )
                for idx, req_idx in enumerate(active):
                    rid = batch.reqs[req_idx].rid
                    cached_indices = self._cache_indices.get(
                        rid, torch.empty((0,), dtype=torch.int64, device=self.device)
                    )
                    self._cache_indices[rid] = torch.cat(
                        [cached_indices, out_cache_loc[idx : idx + 1]], dim=0
                    )
                    self._cache_lens[rid] = base_lens[req_idx] + step + 1

                sampling_info = self._build_sampling_info(
                    [batch.reqs[i] for i in active]
                )
                model_worker_batch = ModelWorkerBatch(
                    forward_mode=ForwardMode.DECODE,
                    input_ids=input_ids,
                    req_pool_indices=active_req_pool,
                    seq_lens=step_seq_lens,
                    out_cache_loc=out_cache_loc,
                    seq_lens_cpu=step_seq_lens_cpu,
                    seq_lens_sum=int(step_seq_lens.sum().item()),
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
                    multimodal_inputs=[batch.reqs[i].multimodal_inputs for i in active],
                    encoder_cached=None,
                    encoder_lens=None,
                    encoder_lens_cpu=None,
                    encoder_out_cache_loc=None,
                    lora_ids=[batch.reqs[i].lora_id for i in active],
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
                    reqs=[batch.reqs[i] for i in active],
                    has_grammar=False,
                    mamba_track_indices=None,
                    mamba_track_mask=None,
                    mamba_track_seqlens=None,
                )

                self.draft_worker.forward_batch_generation(
                    model_worker_batch, is_verify=True
                )
            return

        req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
        req_pool_indices_device = req_pool_indices_cpu.to(
            self.device, non_blocking=True
        )
        prefix_lens_cpu = torch.tensor(base_lens, dtype=torch.int64)
        extend_lens_cpu = torch.tensor(
            [len(tokens) for tokens in missing_tokens], dtype=torch.int64
        )
        seq_lens_cpu = prefix_lens_cpu + extend_lens_cpu
        total_missing = int(extend_lens_cpu.sum().item())
        if total_missing <= 0:
            return

        out_cache_loc = self.token_to_kv_pool_allocator.alloc(total_missing)
        if out_cache_loc is None:
            raise RuntimeError("Draft KV cache allocation failed.")

        prefix_tensors = [
            self._cache_indices.get(
                batch.reqs[i].rid,
                torch.empty((0,), dtype=torch.int64, device=self.device),
            )
            for i in range(len(batch.reqs))
        ]

        prefix_lens_device = prefix_lens_cpu.to(self.device, non_blocking=True)
        extend_lens_device = extend_lens_cpu.to(self.device, non_blocking=True)
        seq_lens_device = seq_lens_cpu.to(self.device, non_blocking=True)

        write_cache_indices(
            out_cache_loc,
            req_pool_indices_device,
            req_pool_indices_cpu,
            prefix_lens_device,
            prefix_lens_cpu,
            seq_lens_device,
            seq_lens_cpu,
            extend_lens_device,
            extend_lens_cpu,
            prefix_tensors,
            self.req_to_token_pool,
        )

        input_ids = torch.tensor(
            [token for tokens in missing_tokens for token in tokens],
            dtype=torch.int64,
            device=self.device,
        )

        sampling_info = self._build_sampling_info(batch.reqs)
        model_worker_batch = ModelWorkerBatch(
            forward_mode=ForwardMode.EXTEND,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices_device,
            seq_lens=seq_lens_device,
            out_cache_loc=out_cache_loc,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=int(seq_lens_cpu.sum().item()),
            return_logprob=False,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            global_num_tokens=None,
            global_num_tokens_for_logprob=None,
            is_extend_in_batch=False,
            can_run_dp_cuda_graph=False,
            tbo_split_seq_index=None,
            global_forward_mode=None,
            extend_num_tokens=total_missing,
            extend_seq_lens=extend_lens_cpu.tolist(),
            extend_prefix_lens=prefix_lens_cpu.tolist(),
            extend_logprob_start_lens=None,
            extend_input_logprob_token_ids=None,
            multimodal_inputs=[req.multimodal_inputs for req in batch.reqs],
            encoder_cached=None,
            encoder_lens=None,
            encoder_lens_cpu=None,
            encoder_out_cache_loc=None,
            lora_ids=[req.lora_id for req in batch.reqs],
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
            reqs=list(batch.reqs),
            has_grammar=False,
            mamba_track_indices=None,
            mamba_track_mask=None,
            mamba_track_seqlens=None,
        )

        self.draft_worker.forward_batch_generation(model_worker_batch, is_verify=True)

        cursor = 0
        for i, req in enumerate(batch.reqs):
            extend_len = int(extend_lens_cpu[i].item())
            if extend_len <= 0:
                continue
            rid = req.rid
            new_indices = out_cache_loc[cursor : cursor + extend_len]
            cached_indices = self._cache_indices.get(
                rid, torch.empty((0,), dtype=torch.int64, device=self.device)
            )
            self._cache_indices[rid] = torch.cat(
                [cached_indices, new_indices], dim=0
            )
            self._cache_lens[rid] = int(seq_lens_cpu[i].item())
            cursor += extend_len


    def update_after_verify(self, reqs):
        with self._lock:
            for req in reqs:
                rid = req.rid
                buffer = self._draft_buffers.get(rid, [])
                accept_count = getattr(req, "pearl_accept_count", 0)
                reject_pos = getattr(req, "pearl_reject_pos", None)
                used_revised = getattr(req, "pearl_used_revised", False)

                if used_revised or reject_pos is not None:
                    buffer = []
                elif accept_count > 0:
                    buffer = buffer[accept_count:]

                self._draft_buffers[rid] = buffer

                target_len = len(req.origin_input_ids) + len(req.output_ids)
                desired_len = target_len + len(buffer)
                cached_len = self._cache_lens.get(rid, 0)
                cached_indices = self._cache_indices.get(
                    rid, torch.empty((0,), dtype=torch.int64, device=self.device)
                )
                if cached_len > desired_len:
                    to_free = cached_indices[desired_len:]
                    if to_free.numel() > 0:
                        self.token_to_kv_pool_allocator.free(to_free)
                    cached_indices = cached_indices[:desired_len]
                    cached_len = desired_len

                self._cache_indices[rid] = cached_indices
                self._cache_lens[rid] = cached_len

    def _run_draft_vectorized(
        self,
        batch: ScheduleBatch,
        req_pool_indices: torch.Tensor,
        max_context_len: int,
    ) -> torch.Tensor:
        bs = batch.batch_size()
        max_draft_tokens = self.speculative_num_draft_tokens
        buffers = [list(self._draft_buffers.get(req.rid, [])) for req in batch.reqs]
        draft_tokens_tensor = torch.full(
            (bs, max_draft_tokens),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        filled = torch.zeros((bs,), dtype=torch.int64, device=self.device)
        cache_lens = torch.empty((bs,), dtype=torch.int64, device=self.device)
        current_tokens = torch.empty((bs,), dtype=torch.int64, device=self.device)

        for i, req in enumerate(batch.reqs):
            buffer = buffers[i]
            buf_len = min(len(buffer), max_draft_tokens)
            if buf_len:
                draft_tokens_tensor[i, :buf_len] = torch.tensor(
                    buffer[:buf_len], dtype=torch.int64, device=self.device
                )
                filled[i] = buf_len
                current_tokens[i] = buffer[buf_len - 1]
            else:
                current_tokens[i] = (
                    req.output_ids[-1] if req.output_ids else req.origin_input_ids[-1]
                )
            cache_lens[i] = self._cache_lens.get(req.rid, 0)

        sampling_info = self._build_greedy_sampling_info(bs)
        new_cache_chunks = [[] for _ in range(bs)]

        for _ in range(max_draft_tokens):
            active_mask = filled < max_draft_tokens
            if not torch.any(active_mask):
                break
            active_idx = torch.nonzero(active_mask, as_tuple=False).flatten()
            active_req_pool = req_pool_indices.index_select(0, active_idx)
            input_ids = current_tokens.index_select(0, active_idx)
            step_seq_lens = cache_lens.index_select(0, active_idx)

            if int(step_seq_lens.max().item()) >= max_context_len:
                raise RuntimeError(
                    "Draft seq_lens exceed context: max=%d limit=%d"
                    % (int(step_seq_lens.max().item()), max_context_len)
                )

            step_seq_lens_cpu = step_seq_lens.cpu()
            step_seq_lens_sum = int(step_seq_lens.sum().item())
            step_out_cache_loc = self.token_to_kv_pool_allocator.alloc(
                int(active_idx.numel())
            )
            if step_out_cache_loc is None:
                raise RuntimeError("Draft KV cache allocation failed.")
            self.req_to_token_pool.write(
                (active_req_pool, step_seq_lens),
                step_out_cache_loc.to(torch.int32),
            )

            active_idx_cpu = active_idx.tolist()
            model_worker_batch = ModelWorkerBatch(
                forward_mode=ForwardMode.DECODE,
                input_ids=input_ids,
                req_pool_indices=active_req_pool,
                seq_lens=step_seq_lens,
                out_cache_loc=step_out_cache_loc,
                seq_lens_cpu=step_seq_lens_cpu,
                seq_lens_sum=step_seq_lens_sum,
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
                multimodal_inputs=[
                    batch.reqs[i].multimodal_inputs for i in active_idx_cpu
                ],
                encoder_cached=None,
                encoder_lens=None,
                encoder_lens_cpu=None,
                encoder_out_cache_loc=None,
                lora_ids=[batch.reqs[i].lora_id for i in active_idx_cpu],
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
                reqs=[batch.reqs[i] for i in active_idx_cpu],
                has_grammar=False,
                mamba_track_indices=None,
                mamba_track_mask=None,
                mamba_track_seqlens=None,
            )

            batch_result = self.draft_worker.forward_batch_generation(
                model_worker_batch, is_verify=True
            )
            if _PEARL_DRAFT_SYNC and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            logits = batch_result.logits_output.next_token_logits
            next_tokens = torch.argmax(logits, dim=-1)

            draft_tokens_tensor[active_idx, filled[active_idx]] = next_tokens
            filled[active_idx] += 1
            current_tokens[active_idx] = next_tokens
            cache_lens[active_idx] += 1

            for local_i, req_idx in enumerate(active_idx_cpu):
                new_cache_chunks[req_idx].append(
                    step_out_cache_loc[local_i : local_i + 1]
                )

        for i, req in enumerate(batch.reqs):
            rid = req.rid
            buffer = draft_tokens_tensor[i, : filled[i]].tolist()
            self._draft_buffers[rid] = buffer
            cached_indices = self._cache_indices.get(
                rid, torch.empty((0,), dtype=torch.int64, device=self.device)
            )
            if new_cache_chunks[i]:
                cached_indices = torch.cat([cached_indices, *new_cache_chunks[i]], dim=0)
            self._cache_indices[rid] = cached_indices
            self._cache_lens[rid] = int(cache_lens[i].item())

        return draft_tokens_tensor

    def _run_draft_legacy(
        self,
        batch: ScheduleBatch,
        req_pool_indices: torch.Tensor,
        max_context_len: int,
    ) -> torch.Tensor:
        bs = batch.batch_size()
        buffers = {
            req.rid: list(self._draft_buffers.get(req.rid, [])) for req in batch.reqs
        }
        current_tokens = {}
        for req in batch.reqs:
            buffer = buffers[req.rid]
            if buffer:
                current_tokens[req.rid] = buffer[-1]
            else:
                current_tokens[req.rid] = (
                    req.output_ids[-1]
                    if req.output_ids
                    else req.origin_input_ids[-1]
                )

        sampling_info = self._build_sampling_info(batch.reqs)
        for _ in range(self.speculative_num_draft_tokens):
            active = [
                i
                for i, req in enumerate(batch.reqs)
                if len(buffers[req.rid]) < self.speculative_num_draft_tokens
            ]
            if not active:
                break

            active_req_pool = req_pool_indices[active]
            input_ids = torch.tensor(
                [current_tokens[batch.reqs[i].rid] for i in active],
                dtype=torch.int64,
                device=self.device,
            )
            sampling_info_active = sampling_info
            if len(active) != bs:
                sampling_info_active = copy.deepcopy(sampling_info)
                keep_indices_device = torch.tensor(active, device=self.device)
                sampling_info_active.filter_batch(active, keep_indices_device)
            step_seq_lens = torch.tensor(
                [self._cache_lens[batch.reqs[i].rid] for i in active],
                dtype=torch.int64,
                device=self.device,
            )
            if int(step_seq_lens.max().item()) >= max_context_len:
                raise RuntimeError(
                    "Draft seq_lens exceed context: max=%d limit=%d"
                    % (int(step_seq_lens.max().item()), max_context_len)
                )
            step_seq_lens_cpu = step_seq_lens.cpu()
            step_seq_lens_sum = int(step_seq_lens.sum().item())

            step_out_cache_loc = self.token_to_kv_pool_allocator.alloc(len(active))
            if step_out_cache_loc is None:
                raise RuntimeError("Draft KV cache allocation failed.")
            self.req_to_token_pool.write(
                (active_req_pool, step_seq_lens),
                step_out_cache_loc.to(torch.int32),
            )

            model_worker_batch = ModelWorkerBatch(
                forward_mode=ForwardMode.DECODE,
                input_ids=input_ids,
                req_pool_indices=active_req_pool,
                seq_lens=step_seq_lens,
                out_cache_loc=step_out_cache_loc,
                seq_lens_cpu=step_seq_lens_cpu,
                seq_lens_sum=step_seq_lens_sum,
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
                multimodal_inputs=[batch.reqs[i].multimodal_inputs for i in active],
                encoder_cached=None,
                encoder_lens=None,
                encoder_lens_cpu=None,
                encoder_out_cache_loc=None,
                lora_ids=[batch.reqs[i].lora_id for i in active],
                sampling_info=sampling_info_active,
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
                reqs=[batch.reqs[i] for i in active],
                has_grammar=False,
                mamba_track_indices=None,
                mamba_track_mask=None,
                mamba_track_seqlens=None,
            )

            batch_result = self.draft_worker.forward_batch_generation(
                model_worker_batch, is_verify=True
            )
            if _PEARL_DRAFT_SYNC and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            logits_output = batch_result.logits_output
            logits = logits_output.next_token_logits
            if sampling_info_active.has_custom_logit_processor:
                apply_custom_logit_processor(
                    logits, sampling_info_active, num_tokens_in_batch=1
                )
            if sampling_info_active.penalizer_orchestrator is not None:
                sampling_info_active.apply_logits_bias(logits)
            temperatures = (
                sampling_info_active.temperatures.view(-1)
                .to(logits.device)
                .clamp(min=1e-5)
            )
            logits = logits / temperatures.unsqueeze(1)
            next_tokens = torch.argmax(logits, dim=-1)

            for idx, req_idx in enumerate(active):
                rid = batch.reqs[req_idx].rid
                token = int(next_tokens[idx].item())
                buffers[rid].append(token)
                current_tokens[rid] = token
                cached_indices = self._cache_indices.get(
                    rid, torch.empty((0,), dtype=torch.int64, device=self.device)
                )
                self._cache_indices[rid] = torch.cat(
                    [cached_indices, step_out_cache_loc[idx : idx + 1]], dim=0
                )
                self._cache_lens[rid] = self._cache_lens.get(rid, 0) + 1

        draft_tokens = []
        for req in batch.reqs:
            buffer = buffers[req.rid]
            self._draft_buffers[req.rid] = buffer
            draft_tokens.append(buffer[: self.speculative_num_draft_tokens])

        return torch.tensor(draft_tokens, dtype=torch.int64, device=self.device)

    def run_draft(self, batch: ScheduleBatch):
        with self._lock:
            if torch.cuda.is_available():
                torch.cuda.set_device(self.model_runner.gpu_id)
            self._sync_cache(batch)
            if _PEARL_DRAFT_SYNC and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)

            req_pool_indices = torch.tensor(
                [self._req_pool_map[int(idx.item())] for idx in batch.req_pool_indices],
                dtype=torch.int64,
                device=self.device,
            )
            max_reqs = self.req_to_token_pool.req_to_token.shape[0]
            max_context_len = self.req_to_token_pool.req_to_token.shape[1]
            if (
                req_pool_indices.numel()
                and int(req_pool_indices.max().item()) >= max_reqs
            ):
                raise RuntimeError(
                    "Draft req_pool_indices out of range: max=%d size=%d"
                    % (int(req_pool_indices.max().item()), max_reqs)
                )

            if self._vectorize_draft:
                draft_tokens = self._run_draft_vectorized(
                    batch, req_pool_indices, max_context_len
                )
            else:
                draft_tokens = self._run_draft_legacy(
                    batch, req_pool_indices, max_context_len
                )

            if _PEARL_DRAFT_SYNC and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            return draft_tokens
