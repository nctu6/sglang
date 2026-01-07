from __future__ import annotations

import logging
from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch

from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.layers.sampler import multinomial_with_seed
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
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
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
        draft_probs: Optional[torch.Tensor] = None,
        next_window_tokens: Optional[torch.Tensor] = None,
        next_window_probs: Optional[torch.Tensor] = None,
        prefix_logits: Optional[torch.Tensor] = None,
        prefix_logits_mask: Optional[torch.Tensor] = None,
        valid_draft_lens: Optional[torch.Tensor] = None,
    ):
        super().__init__(SpecInputType.PEARL_VERIFY)
        self.draft_token = draft_token
        self.custom_mask = custom_mask
        self.positions = positions
        self.draft_token_num = draft_token_num
        self.vocab_size = vocab_size
        self.device = draft_token.device
        if draft_probs is not None and draft_probs.device != self.device:
            draft_probs = draft_probs.to(self.device, non_blocking=True)
        self.draft_probs = draft_probs
        self.next_window_tokens = next_window_tokens
        if next_window_probs is not None and next_window_probs.device != self.device:
            next_window_probs = next_window_probs.to(self.device, non_blocking=True)
        self.next_window_probs = next_window_probs
        self.prefix_logits = prefix_logits
        self.prefix_logits_mask = prefix_logits_mask
        self.valid_draft_lens = (
            valid_draft_lens.to("cpu") if valid_draft_lens is not None else None
        )
        self.capture_hidden_mode = CaptureHiddenMode.NULL
        self.num_tokens_per_batch = draft_token_num
        self.num_tokens_for_logprob_per_batch = draft_token_num
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
            prev_window = getattr(req, "pearl_prev_window", None)
            prev_window_logits = getattr(req, "pearl_prev_window_logits", None)
            if (
                self.prefix_logits is not None
                and self.prefix_logits_mask is not None
                and self.prefix_logits_mask[i]
            ):
                prev_logits_list.append(self.prefix_logits[i])
            elif prev_window is not None and prev_window_logits is not None:
                prev_logits_list.append(prev_window_logits.to(raw_logits.device))
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
        probs_view = probs.view(bs, self.draft_token_num, -1)
        draft_probs_view = None
        if self.draft_probs is not None:
            draft_probs_view = self.draft_probs
            if draft_probs_view.dim() == 1:
                draft_probs_view = draft_probs_view.view(bs, -1)
            if draft_probs_view.shape[0] != bs:
                draft_probs_view = None
            else:
                if draft_probs_view.shape[1] > self.draft_token_num:
                    draft_probs_view = draft_probs_view[:, : self.draft_token_num]
                elif draft_probs_view.shape[1] < self.draft_token_num:
                    pad = torch.zeros(
                        (bs, self.draft_token_num - draft_probs_view.shape[1]),
                        dtype=draft_probs_view.dtype,
                        device=draft_probs_view.device,
                    )
                    draft_probs_view = torch.cat([draft_probs_view, pad], dim=1)
                draft_probs_view = draft_probs_view.float()

        accepted_indices = []
        verified_tokens = []
        accept_length_list = []
        has_finished = False
        pre_count = 0
        pre_accept = 0
        pre_target_prob_sum = 0.0
        pre_draft_prob_sum = 0.0
        pre_accept_prob_sum = 0.0
        pre_filtered_zero = 0

        coins = torch.rand(
            (bs, self.draft_token_num), dtype=probs.dtype, device=self.device
        )
        valid_lens = (
            self.valid_draft_lens.tolist()
            if self.valid_draft_lens is not None
            else [self.draft_token_num] * bs
        )

        for i, req in enumerate(batch.reqs):
            accept_count = 0
            reject_pos = None
            pre_verify = getattr(req, "pre_verify", True)
            pre_verify_before = pre_verify
            next_window = None
            valid_len = valid_lens[i] if i < len(valid_lens) else self.draft_token_num
            if self.next_window_tokens is not None:
                next_window = self.next_window_tokens[i][:valid_len].tolist()
            tokens_for_kv = None
            used_revised_token = False
            revised_offset = None
            consumed_full_window = False
            req.pearl_revised_token = None
            target_prob0 = 0.0
            draft_prob0 = None
            accept_prob0 = None

            def _apply_sampling_filters(
                probs_row: torch.Tensor,
                top_k: int,
                top_p: float,
                min_p: float,
            ) -> torch.Tensor:
                filtered = probs_row
                vocab_size = filtered.numel()
                if top_k > 0 and top_k < vocab_size:
                    topk_vals, topk_idx = torch.topk(filtered, top_k)
                    filtered = torch.zeros_like(filtered)
                    filtered.scatter_(0, topk_idx, topk_vals)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(filtered, descending=True)
                    cdf = torch.cumsum(sorted_probs, dim=-1)
                    mask = cdf <= top_p
                    if mask.numel() > 0:
                        mask[0] = True
                    sorted_probs = sorted_probs * mask
                    filtered = torch.zeros_like(filtered).scatter_(
                        0, sorted_idx, sorted_probs
                    )
                if min_p > 0.0:
                    filtered = torch.where(
                        filtered >= min_p, filtered, torch.zeros_like(filtered)
                    )
                norm = filtered.sum()
                if norm > 0:
                    filtered = filtered / norm
                return filtered

            def _sample_from_filtered_probs(
                probs_row: torch.Tensor,
                position: int,
            ) -> int:
                if float(probs_row.sum().item()) <= 0.0:
                    probs_row = torch.full_like(
                        probs_row, 1.0 / max(probs_row.numel(), 1)
                    )
                probs_row = probs_row.unsqueeze(0)
                if sampling_info.sampling_seed is not None:
                    sampled = multinomial_with_seed(
                        probs_row,
                        sampling_info.sampling_seed[i : i + 1],
                        torch.tensor([position], device=probs_row.device),
                    )
                    return int(sampled.item())
                sampled = torch.multinomial(probs_row, num_samples=1)
                return int(sampled.item())

            def _accept_prob(target_prob: float, draft_prob: Optional[float]) -> float:
                if draft_prob is None or not math.isfinite(draft_prob) or draft_prob <= 0.0:
                    return target_prob
                ratio = target_prob / max(draft_prob, 1e-8)
                return 1.0 if ratio >= 1.0 else ratio

            if valid_len <= 0:
                tokens_to_append = []
                tokens_for_kv = []
                accept_count = 0
                req.pre_verify = True
            elif pre_verify:
                pre_count += 1
                draft_token_id = draft_tokens[i, 0].item()
                target_prob0 = float(probs_view[i, 0][draft_token_id].item())
                if draft_probs_view is not None:
                    draft_prob0 = float(draft_probs_view[i, 0].item())
                accept_prob0 = _accept_prob(target_prob0, draft_prob0)
                pre_target_prob_sum += target_prob0
                if draft_prob0 is not None:
                    pre_draft_prob_sum += draft_prob0
                if accept_prob0 is not None:
                    pre_accept_prob_sum += accept_prob0
                if (
                    float(sampling_info.top_ks[i].item()) > 0
                    or float(sampling_info.top_ps[i].item()) < 1.0
                    or float(sampling_info.min_ps[i].item()) > 0.0
                ):
                    filtered_row = _apply_sampling_filters(
                        probs_view[i, 0].clone(),
                        int(sampling_info.top_ks[i].item()),
                        float(sampling_info.top_ps[i].item()),
                        float(sampling_info.min_ps[i].item()),
                    )
                    if float(filtered_row[draft_token_id].item()) <= 0.0:
                        pre_filtered_zero += 1
                if coins[i, 0] <= accept_prob0:
                    accept_count = 1
                    reject_pos = None
                    tokens_to_append = [draft_token_id]
                    tokens_for_kv = tokens_to_append
                    req.pre_verify = False
                    consumed_full_window = valid_len <= 1
                else:
                    probs_row = probs_view[i, 0].clone()
                    probs_row = _apply_sampling_filters(
                        probs_row,
                        int(sampling_info.top_ks[i].item()),
                        float(sampling_info.top_ps[i].item()),
                        float(sampling_info.min_ps[i].item()),
                    )
                    probs_row[draft_token_id] = 0.0
                    norm = probs_row.sum()
                    if norm > 0:
                        probs_row = probs_row / norm
                    position = int(batch.seq_lens[i].item())
                    revised_token = _sample_from_filtered_probs(
                        probs_row, position
                    )
                    tokens_to_append = [revised_token]
                    tokens_for_kv = [revised_token]
                    accept_count = 1
                    reject_pos = 0
                    req.pre_verify = True
                    used_revised_token = True
                    revised_offset = 0
                if not req.pre_verify:
                    pre_accept += 1
            else:
                for j in range(valid_len):
                    draft_token_id = draft_tokens[i, j].item()
                    draft_prob = None
                    if draft_probs_view is not None:
                        draft_prob = float(draft_probs_view[i, j].item())
                    accept_prob = _accept_prob(
                        float(probs_view[i, j][draft_token_id].item()), draft_prob
                    )
                    if coins[i, j] <= accept_prob:
                        accept_count += 1
                    else:
                        reject_pos = j
                        break

                if reject_pos is None:
                    tokens_to_append = draft_tokens[i, :accept_count].tolist()
                    tokens_for_kv = draft_tokens[i, :accept_count].tolist()
                    req.pre_verify = False
                else:
                    probs_row = probs_view[i, reject_pos].clone()
                    probs_row = _apply_sampling_filters(
                        probs_row,
                        int(sampling_info.top_ks[i].item()),
                        float(sampling_info.top_ps[i].item()),
                        float(sampling_info.min_ps[i].item()),
                    )
                    draft_token_id = draft_tokens[i, reject_pos].item()
                    probs_row[draft_token_id] = 0.0
                    norm = probs_row.sum()
                    if norm > 0:
                        probs_row = probs_row / norm
                    position = int(batch.seq_lens[i].item()) + reject_pos
                    revised_token = _sample_from_filtered_probs(
                        probs_row, position
                    )
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

            req.pearl_accept_count = accept_count
            req.pearl_reject_pos = reject_pos
            req.pearl_used_revised = used_revised_token

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

            if not req.pre_verify and next_window is not None and not consumed_full_window:
                next_probs = None
                if self.next_window_probs is not None:
                    window_probs = self.next_window_probs[i]
                    if isinstance(window_probs, torch.Tensor):
                        window_probs = window_probs[:valid_len].tolist()
                    else:
                        window_probs = list(window_probs)[:valid_len]
                    next_probs = window_probs
                next_tokens = (
                    next_window[:valid_len].tolist()
                    if isinstance(next_window, torch.Tensor)
                    else list(next_window)[:valid_len]
                )
                if pre_verify_before:
                    if len(next_tokens) > 1:
                        req.pearl_prev_window = torch.tensor(
                            next_tokens[1:], dtype=torch.int64, device=self.device
                        )
                        req.pearl_prev_window_logits = aligned_logits[i, 1].detach()
                        if next_probs is not None and len(next_probs) > 1:
                            req.pearl_prev_window_probs = torch.tensor(
                                next_probs[1:],
                                dtype=torch.float32,
                                device=self.device,
                            )
                        else:
                            req.pearl_prev_window_probs = None
                    else:
                        req.pearl_prev_window = None
                        req.pearl_prev_window_logits = None
                        req.pearl_prev_window_probs = None
                else:
                    if next_tokens:
                        req.pearl_prev_window = torch.tensor(
                            next_tokens, dtype=torch.int64, device=self.device
                        )
                        if verified_count > 0:
                            req.pearl_prev_window_logits = raw_logits[
                                i, verified_count - 1
                            ].detach()
                        else:
                            req.pearl_prev_window_logits = None
                        if next_probs is not None and len(next_probs) == len(next_tokens):
                            req.pearl_prev_window_probs = torch.tensor(
                                next_probs, dtype=torch.float32, device=self.device
                            )
                        else:
                            req.pearl_prev_window_probs = None
                    else:
                        req.pearl_prev_window = None
                        req.pearl_prev_window_logits = None
                        req.pearl_prev_window_probs = None
            else:
                req.pearl_prev_window = None
                req.pearl_prev_window_logits = None
                req.pearl_prev_window_probs = None

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
                    target_prob0,
                    tokens_to_append,
                    repr(decoded_tokens) if decoded_tokens is not None else None,
                )

        self.pre_verify_count = pre_count
        self.pre_verify_accept = pre_accept
        self.pre_target_prob_sum = pre_target_prob_sum
        self.pre_draft_prob_sum = pre_draft_prob_sum
        self.pre_accept_prob_sum = pre_accept_prob_sum
        self.pre_filtered_zero = pre_filtered_zero

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
