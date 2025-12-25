import logging
from typing import Optional

import torch

from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.pearl_info import PearlVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool
from sglang.srt.utils import empty_context, next_power_of_2

logger = logging.getLogger(__name__)


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
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.device = target_worker.model_runner.device

        # Share allocator and request pool with the target worker to align slots.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        with (
            empty_context(),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self.draft_worker = TpModelWorker(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # FIXME
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            )

    def clear_cache_pool(self):
        # Allocator is shared with target worker.
        pass

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

    def _run_draft(self, batch: ScheduleBatch, out_cache_loc: torch.Tensor):
        bs = batch.batch_size()
        draft_tokens = []

        base_seq_lens = batch.seq_lens.clone()
        base_seq_lens_cpu = batch.seq_lens_cpu.clone()

        if self.page_size != 1:
            raise ValueError("PEARL currently requires page_size == 1")

        draft_cache_loc = out_cache_loc.view(bs, self.speculative_num_draft_tokens)
        current_tokens = [
            (req.output_ids[-1] if req.output_ids else req.origin_input_ids[-1])
            for req in batch.reqs
        ]

        for step in range(self.speculative_num_draft_tokens):
            batch.forward_mode = ForwardMode.DECODE
            batch.input_ids = torch.tensor(
                current_tokens, dtype=torch.int64, device=self.device
            )
            batch.seq_lens = base_seq_lens + step
            batch.seq_lens_cpu = base_seq_lens_cpu + step
            batch.out_cache_loc = draft_cache_loc[:, step]

            model_worker_batch = batch.get_model_worker_batch()
            batch_result = self.draft_worker.forward_batch_generation(model_worker_batch)
            logits_output = batch_result.logits_output
            next_tokens = torch.argmax(logits_output.next_token_logits, dim=-1)

            draft_tokens.append(next_tokens)
            current_tokens = next_tokens.tolist()

        batch.seq_lens = base_seq_lens
        batch.seq_lens_cpu = base_seq_lens_cpu
        return torch.stack(draft_tokens, dim=1)

    def _build_positions(self, batch: ScheduleBatch) -> torch.Tensor:
        base = batch.seq_lens.unsqueeze(1)
        offsets = torch.arange(
            self.speculative_num_draft_tokens, device=self.device
        ).unsqueeze(0)
        positions = (base + offsets).reshape(-1)
        return positions

    def _build_custom_mask(self, batch: ScheduleBatch) -> torch.Tensor:
        mask = torch.tril(
            torch.ones(
                (self.speculative_num_draft_tokens, self.speculative_num_draft_tokens),
                dtype=torch.bool,
                device=self.device,
            )
        )
        return mask.repeat(batch.batch_size(), 1, 1).reshape(-1)

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return self.target_worker.forward_batch_generation(
                batch.get_model_worker_batch()
            )

        if any(req.grammar is not None for req in batch.reqs):
            logger.warning("PEARL does not support grammar; falling back to target.")
            return self.target_worker.forward_batch_generation(
                batch.get_model_worker_batch()
            )

        batch.spec_info = None
        out_cache_loc = self._allocate_draft_slots(batch)
        draft_tokens = self._run_draft(batch, out_cache_loc)

        positions = self._build_positions(batch)
        custom_mask = self._build_custom_mask(batch)

        batch.spec_algorithm = SpeculativeAlgorithm.PEARL
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.out_cache_loc = out_cache_loc
        batch.input_ids = draft_tokens.reshape(-1)

        spec_info = PearlVerifyInput(
            batch.input_ids,
            custom_mask,
            positions,
            self.speculative_num_draft_tokens,
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

        batch.forward_mode = ForwardMode.DECODE

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=verified_id,
            num_accepted_tokens=num_accepted_tokens,
            accept_length_per_req_cpu=accept_length_list,
            can_run_cuda_graph=can_run_cuda_graph,
        )
