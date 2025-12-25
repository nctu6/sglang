from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import (
    assign_req_to_token_pool,
    get_src_tgt_cache_loc,
    get_target_cache_loc,
)
from sglang.srt.utils import next_power_of_2

logger = logging.getLogger(__name__)


@dataclass
class PearlVerifyInput(SpecInput):
    def __init__(
        self,
        draft_token: torch.Tensor,
        custom_mask: torch.Tensor,
        positions: torch.Tensor,
        draft_token_num: int,
    ):
        super().__init__(SpecInputType.PEARL_VERIFY)
        self.draft_token = draft_token
        self.custom_mask = custom_mask
        self.positions = positions
        self.draft_token_num = draft_token_num
        self.device = draft_token.device
        self.accepted_indices: Optional[torch.Tensor] = None
        self.accept_length: Optional[torch.Tensor] = None
        self.verified_id: Optional[torch.Tensor] = None

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.draft_token_num, self.draft_token_num

    def prepare_for_verify(self, batch: ScheduleBatch, page_size: int):
        if batch.forward_mode.is_idle():
            return

        batch.input_ids = self.draft_token

        if page_size == 1:
            batch.out_cache_loc = alloc_token_slots(
                batch.tree_cache,
                len(batch.input_ids),
            )
            end_offset = batch.seq_lens + self.draft_token_num
        else:
            prefix_lens = batch.seq_lens
            prefix_lens_cpu = batch.seq_lens_cpu
            end_offset = prefix_lens + self.draft_token_num
            end_offset_cpu = prefix_lens_cpu + self.draft_token_num
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                prefix_lens,
            )
            batch.out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                prefix_lens,
                prefix_lens_cpu,
                end_offset,
                end_offset_cpu,
                last_loc,
                len(batch.input_ids),
            )

        bs = batch.batch_size()
        assign_req_to_token_pool[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            next_power_of_2(bs),
        )

    def verify(
        self,
        batch: ScheduleBatch,
        logits_output,
        page_size: int,
    ):
        bs = batch.batch_size()
        sampling_info: SamplingBatchInfo = batch.sampling_info

        # Apply custom logit processors (if any).
        if sampling_info.has_custom_logit_processor:
            apply_custom_logit_processor(
                logits_output.next_token_logits,
                sampling_info,
                num_tokens_in_batch=self.draft_token_num,
            )

        # Apply penalty in a relaxed way for speculative decoding.
        if sampling_info.penalizer_orchestrator.is_required:
            linear_penalty = torch.zeros(
                (bs, logits_output.next_token_logits.shape[1]),
                dtype=torch.float32,
                device=self.device,
            )
            sampling_info.apply_logits_bias(linear_penalty)
            logits_output.next_token_logits.add_(
                torch.repeat_interleave(linear_penalty, self.draft_token_num, dim=0)
            )

        draft_tokens = self.draft_token.view(bs, self.draft_token_num)
        temperatures = sampling_info.temperatures.repeat_interleave(
            self.draft_token_num
        ).clamp(min=1e-5)
        scaled_logits = logits_output.next_token_logits / temperatures
        probs = torch.softmax(scaled_logits, dim=-1)
        target_prob = probs.gather(1, self.draft_token.unsqueeze(1)).squeeze(1)
        target_prob = target_prob.view(bs, self.draft_token_num)

        accepted_indices = []
        verified_tokens = []
        accept_length_list = []
        has_finished = False

        coins = torch.rand_like(target_prob)

        for i, req in enumerate(batch.reqs):
            accept_count = 0
            reject_pos = None

            for j in range(self.draft_token_num):
                if coins[i, j] <= target_prob[i, j]:
                    accept_count += 1
                else:
                    reject_pos = j
                    break

            if reject_pos is None:
                tokens_to_append = draft_tokens[i, :accept_count].tolist()
            else:
                probs_row = probs.view(bs, self.draft_token_num, -1)[i, reject_pos]
                draft_token = draft_tokens[i, reject_pos]
                probs_row = probs_row.clone()
                probs_row[draft_token] = 0
                norm = probs_row.sum()
                if norm > 0:
                    probs_row = probs_row / norm
                    revised_token = torch.multinomial(probs_row, 1).item()
                else:
                    revised_token = draft_token.item()
                tokens_to_append = draft_tokens[i, :reject_pos].tolist() + [
                    revised_token
                ]
                accept_count = reject_pos + 1

            for offset, token_id in enumerate(tokens_to_append):
                req.output_ids.append(token_id)
                req.check_finished()
                accepted_indices.append(i * self.draft_token_num + offset)
                verified_tokens.append(token_id)
                if req.finished():
                    has_finished = True
                    accept_count = offset + 1
                    break

            req.spec_verify_ct += 1
            req.spec_accepted_tokens += max(accept_count - 1, 0)
            accept_length_list.append(max(accept_count - 1, 0))

        if has_finished:
            pass

        if accepted_indices:
            accepted_indices_tensor = torch.tensor(
                accepted_indices, dtype=torch.int64, device=self.device
            )
            logits_output.next_token_logits = logits_output.next_token_logits[
                accepted_indices_tensor
            ]
            if logits_output.hidden_states is not None:
                logits_output.hidden_states = logits_output.hidden_states[
                    accepted_indices_tensor
                ]
            self.accepted_indices = accepted_indices_tensor
            self.verified_id = torch.tensor(
                verified_tokens, dtype=torch.int64, device=self.device
            )
        else:
            self.accepted_indices = torch.empty(
                (0,), dtype=torch.int64, device=self.device
            )
            self.verified_id = torch.empty((0,), dtype=torch.int64, device=self.device)

        self.accept_length = torch.tensor(
            accept_length_list, dtype=torch.int32, device=self.device
        )

        if batch.return_logprob:
            add_output_logprobs_for_spec_v1(batch, self, logits_output)

        accept_length_cpu = self.accept_length.cpu()
        num_accepted_tokens = int(accept_length_cpu.sum().item()) + bs

        self._free_cache(batch, page_size, accept_length_cpu)

        batch.seq_lens.add_(self.accept_length + 1)
        batch.seq_lens_cpu.add_(accept_length_cpu + 1)

        return logits_output, self.verified_id, num_accepted_tokens, accept_length_list

    def _free_cache(
        self, batch: ScheduleBatch, page_size: int, accept_length_cpu: torch.Tensor
    ):
        bs = batch.batch_size()

        if page_size == 1:
            accept_mask = torch.zeros(
                (bs * self.draft_token_num,),
                dtype=torch.bool,
                device=self.device,
            )
            start = 0
            for i, accept_len in enumerate(accept_length_cpu.tolist()):
                keep = accept_len + 1
                accept_mask[start : start + keep] = True
                start += self.draft_token_num
            batch.token_to_kv_pool_allocator.free(batch.out_cache_loc[~accept_mask])
            batch.out_cache_loc = batch.out_cache_loc[accept_mask]
        else:
            accept_index = []
            for i, accept_len in enumerate(accept_length_cpu.tolist()):
                keep = accept_len + 1
                accept_index.extend(
                    range(i * self.draft_token_num, i * self.draft_token_num + keep)
                )
            accept_index = torch.tensor(
                accept_index, dtype=torch.int64, device=self.device
            )

            src_cache_loc, tgt_cache_loc, to_free_num_slots = get_src_tgt_cache_loc(
                batch.seq_lens,
                batch.out_cache_loc,
                accept_index,
                self.accept_length,
                self.draft_token_num,
                page_size,
            )

            to_free_slots = torch.empty(
                to_free_num_slots.sum(), dtype=torch.int64, device=self.device
            )
            get_target_cache_loc[(bs,)](
                tgt_cache_loc,
                to_free_slots,
                self.accept_length,
                to_free_num_slots,
                batch.out_cache_loc,
                self.draft_token_num,
                next_power_of_2(self.draft_token_num),
                next_power_of_2(bs),
            )
            batch.token_to_kv_pool_allocator.free(to_free_slots)
            batch.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
                tgt_cache_loc, src_cache_loc
            )
            batch.out_cache_loc = tgt_cache_loc

        accept_length_list = accept_length_cpu.tolist()
        for i, req in enumerate(batch.reqs):
            req.kv_committed_len += accept_length_list[i] + 1
            req.kv_allocated_len = req.kv_committed_len

        assign_req_to_token_pool[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + self.accept_length + 1,
            batch.out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            next_power_of_2(bs),
        )
