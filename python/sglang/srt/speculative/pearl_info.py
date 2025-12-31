from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch
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
from sglang.srt.utils.common import get_bool_env_var

logger = logging.getLogger(__name__)
_PEARL_DEBUG = get_bool_env_var("SGLANG_PEARL_DEBUG")


@dataclass
class PearlVerifyInput(SpecInput):
    def __init__(
        self,
        draft_token: torch.Tensor,
        custom_mask: torch.Tensor,
        positions: torch.Tensor,
        draft_token_num: int,
        vocab_size: int,
        next_window_tokens: Optional[torch.Tensor] = None,
        prefix_logits: Optional[torch.Tensor] = None,
        prefix_logits_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__(SpecInputType.PEARL_VERIFY)
        self.draft_token = draft_token
        self.custom_mask = custom_mask
        self.positions = positions
        self.draft_token_num = draft_token_num
        self.vocab_size = vocab_size
        self.device = draft_token.device
        self.next_window_tokens = next_window_tokens
        self.prefix_logits = prefix_logits
        self.prefix_logits_mask = prefix_logits_mask
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

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        bs = len(req_pool_indices)

        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=self.device)
        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        self.qo_indptr = (
            torch.arange(0, bs + 1, dtype=torch.int32, device=self.device)
            * self.draft_token_num
        )

        kv_indices = torch.empty(
            cum_kv_seq_len[-1], dtype=torch.int32, device=self.device
        )

        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        return kv_indices, cum_kv_seq_len, self.qo_indptr, self.custom_mask

    def verify(
        self,
        batch: ScheduleBatch,
        logits_output,
        page_size: int,
    ):
        bs = batch.batch_size()
        sampling_info: SamplingBatchInfo = batch.sampling_info
        raw_logits = logits_output.next_token_logits.view(bs, self.draft_token_num, -1)
        aligned_logits = torch.empty_like(raw_logits)
        aligned_logits[:, 1:] = raw_logits[:, :-1]
        prev_logits_list = []
        for i, req in enumerate(batch.reqs):
            prev_logits = getattr(req, "pearl_prev_logits", None)
            if (
                self.prefix_logits is not None
                and self.prefix_logits_mask is not None
                and self.prefix_logits_mask[i]
            ):
                prev_logits_list.append(self.prefix_logits[i])
            elif prev_logits is None or prev_logits.shape[-1] != raw_logits.shape[-1]:
                prev_logits_list.append(raw_logits[i, 0])
            else:
                prev_logits_list.append(prev_logits.to(raw_logits.device))
        aligned_logits[:, 0] = torch.stack(prev_logits_list, dim=0)

        aligned_logits = aligned_logits.reshape(bs * self.draft_token_num, -1)

        # Apply custom logit processors (if any) on the aligned logits.
        if sampling_info.has_custom_logit_processor:
            apply_custom_logit_processor(
                aligned_logits,
                sampling_info,
                num_tokens_in_batch=self.draft_token_num,
            )

        # Apply penalty/logit bias in a relaxed way for speculative decoding.
        if (
            sampling_info.penalizer_orchestrator.is_required
            or sampling_info.logit_bias is not None
        ):
            linear_penalty = torch.zeros(
                (bs, aligned_logits.shape[1]),
                dtype=torch.float32,
                device=self.device,
            )
            sampling_info.apply_logits_bias(linear_penalty)
            aligned_logits.add_(
                torch.repeat_interleave(linear_penalty, self.draft_token_num, dim=0)
            )

        # Keep logits_output in sync for logprob reporting.
        logits_output.next_token_logits = aligned_logits
        aligned_logits = aligned_logits.view(bs, self.draft_token_num, -1)

        if self.vocab_size:
            self.draft_token = torch.clamp(
                self.draft_token, min=0, max=self.vocab_size - 1
            )
        draft_tokens = self.draft_token.view(bs, self.draft_token_num)
        temperatures = (
            sampling_info.temperatures.repeat_interleave(self.draft_token_num)
            .clamp(min=1e-5)
            .unsqueeze(1)
        )
        scaled_logits = aligned_logits.reshape(bs * self.draft_token_num, -1) / temperatures
        probs = torch.softmax(scaled_logits, dim=-1)
        target_prob = probs.gather(1, self.draft_token.unsqueeze(1)).squeeze(1)
        target_prob = target_prob.view(bs, self.draft_token_num)
        probs_view = probs.view(bs, self.draft_token_num, -1)

        accepted_indices = []
        verified_tokens = []
        accept_length_list = []
        has_finished = False

        coins = torch.rand_like(target_prob)

        for i, req in enumerate(batch.reqs):
            accept_count = 0
            reject_pos = None
            pre_verify = getattr(req, "pre_verify", True)
            pre_verify_before = pre_verify
            next_window = None
            if self.next_window_tokens is not None:
                next_window = self.next_window_tokens[i].tolist()
            tokens_for_kv = None
            used_revised_token = False
            revised_offset = None
            req.pearl_revised_token = None

            if pre_verify:
                if coins[i, 0] <= target_prob[i, 0]:
                    tokens_to_append = [draft_tokens[i, 0].item()]
                    tokens_for_kv = tokens_to_append
                    accept_count = 1
                    req.pre_verify = False
                else:
                    draft_token_id = draft_tokens[i, 0].item()
                    probs_row = probs_view[i, 0].clone()
                    probs_row[draft_token_id] = 0.0
                    norm = probs_row.sum()
                    if norm > 0:
                        probs_row = probs_row / norm
                    probs_row = probs_row.unsqueeze(0)
                    position = int(batch.seq_lens[i].item())
                    positions = torch.tensor([position], device=probs_row.device)
                    sampling_seed = (
                        sampling_info.sampling_seed[i : i + 1]
                        if sampling_info.sampling_seed is not None
                        else None
                    )
                    revised_token = top_k_top_p_min_p_sampling_from_probs_torch(
                        probs_row,
                        sampling_info.top_ks[i : i + 1],
                        sampling_info.top_ps[i : i + 1],
                        sampling_info.min_ps[i : i + 1],
                        sampling_info.need_min_p_sampling,
                        sampling_seed,
                        positions,
                    )[0].item()
                    tokens_to_append = [revised_token]
                    tokens_for_kv = [revised_token]
                    accept_count = 1
                    req.pre_verify = True
                    used_revised_token = True
                    revised_offset = 0
            else:
                for j in range(self.draft_token_num):
                    if coins[i, j] <= target_prob[i, j]:
                        accept_count += 1
                    else:
                        reject_pos = j
                        break

                if reject_pos is None:
                    tokens_to_append = draft_tokens[i, :accept_count].tolist()
                    tokens_for_kv = draft_tokens[i, :accept_count].tolist()
                    req.pre_verify = False
                else:
                    draft_token_id = draft_tokens[i, reject_pos].item()
                    probs_row = probs_view[i, reject_pos].clone()
                    probs_row[draft_token_id] = 0.0
                    norm = probs_row.sum()
                    if norm > 0:
                        probs_row = probs_row / norm
                    probs_row = probs_row.unsqueeze(0)
                    position = int(batch.seq_lens[i].item()) + reject_pos
                    positions = torch.tensor([position], device=probs_row.device)
                    sampling_seed = (
                        sampling_info.sampling_seed[i : i + 1]
                        if sampling_info.sampling_seed is not None
                        else None
                    )
                    revised_token = top_k_top_p_min_p_sampling_from_probs_torch(
                        probs_row,
                        sampling_info.top_ks[i : i + 1],
                        sampling_info.top_ps[i : i + 1],
                        sampling_info.min_ps[i : i + 1],
                        sampling_info.need_min_p_sampling,
                        sampling_seed,
                        positions,
                    )[0].item()
                    tokens_to_append = draft_tokens[i, :reject_pos].tolist() + [
                        revised_token
                    ]
                    tokens_for_kv = draft_tokens[i, :reject_pos].tolist() + [
                        revised_token
                    ]
                    accept_count = reject_pos + 1
                    req.pre_verify = True
                    used_revised_token = True
                    revised_offset = reject_pos

            remaining = req.sampling_params.max_new_tokens - len(req.output_ids)
            if remaining <= 0:
                req.check_finished()
                has_finished = True
                accept_count = 0
                accept_length_list.append(0)
                continue

            if remaining < len(tokens_to_append):
                tokens_to_append = tokens_to_append[:remaining]
            if tokens_for_kv is None:
                tokens_for_kv = tokens_to_append
            verified_count = min(len(tokens_for_kv), remaining)
            tokens_for_kv = tokens_for_kv[:verified_count]

            if used_revised_token and revised_offset is not None:
                if revised_offset < len(tokens_to_append):
                    req.pearl_revised_token = (
                        revised_offset,
                        tokens_to_append[revised_offset],
                    )

            decoded_tokens = None
            if _PEARL_DEBUG and tokens_to_append and getattr(req, "tokenizer", None) is not None:
                try:
                    decoded_tokens = req.tokenizer.decode(
                        tokens_to_append,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                except Exception:
                    decoded_tokens = None

            for offset, token_id in enumerate(tokens_to_append):
                req.output_ids.append(token_id)
                req.check_finished()
                if offset < verified_count:
                    accepted_indices.append(i * self.draft_token_num + offset)
                    verified_tokens.append(tokens_for_kv[offset])
                if req.finished():
                    has_finished = True
                    verified_count = min(verified_count, offset + 1)
                    break

            if not req.pre_verify and next_window is not None:
                req.pearl_prev_window = torch.tensor(
                    next_window, dtype=torch.int64, device=self.device
                )
            else:
                req.pearl_prev_window = None

            if verified_count > 0 and not used_revised_token:
                req.pearl_prev_logits = raw_logits[i, verified_count - 1].detach()
            elif used_revised_token:
                req.pearl_prev_logits = None

            req.spec_verify_ct += 1
            req.spec_accepted_tokens += max(verified_count - 1, 0)
            accept_length_list.append(max(verified_count - 1, 0))
            if _PEARL_DEBUG:
                logger.info(
                    "PEARL verify req=%s pre_verify_before=%s pre_verify_after=%s "
                    "accept_count=%d verified_count=%d reject_pos=%s remaining=%d "
                    "append_len=%d kv_len=%d finished=%s target_prob0=%.6f appended=%s decoded=%s",
                    req.rid,
                    pre_verify_before,
                    req.pre_verify,
                    accept_count,
                    verified_count,
                    reject_pos,
                    remaining,
                    len(tokens_to_append),
                    len(tokens_for_kv) if tokens_for_kv is not None else 0,
                    req.finished(),
                    float(target_prob[i, 0].item()) if target_prob.numel() else 0.0,
                    tokens_to_append,
                    repr(decoded_tokens) if decoded_tokens is not None else None,
                )

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

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        if self.accept_length is None:
            return

        if has_been_filtered:
            # Batch already filtered during verify; only keep the leading slice.
            keep_len = int(new_indices.numel())
            self.accept_length = self.accept_length[:keep_len]
        else:
            self.accept_length = self.accept_length[new_indices.to(self.accept_length.device)]

    def merge_batch(self, spec_info: "PearlVerifyInput"):
        if spec_info is None:
            return
        if self.accept_length is None:
            self.accept_length = spec_info.accept_length
        elif spec_info.accept_length is not None:
            self.accept_length = torch.cat(
                [self.accept_length, spec_info.accept_length], dim=0
            )

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
