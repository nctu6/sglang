from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import triton

from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import alloc_token_slots
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool

logger = logging.getLogger(__name__)


class DFlashDraftInput(SpecInput):
    def __init__(
        self,
        positions: torch.Tensor,
        target_hidden: torch.Tensor,
        target_lens: torch.Tensor,
    ):
        super().__init__(SpecInputType.DFLASH_DRAFT)
        self.positions = positions
        self.target_hidden = target_hidden
        self.target_lens = target_lens

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return 1, 1


class DFlashVerifyInput(SpecInput):
    def __init__(
        self,
        draft_token: torch.Tensor,
        positions: torch.Tensor,
        block_size: int,
        custom_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__(SpecInputType.DFLASH_VERIFY)
        self.draft_token = draft_token
        self.positions = positions
        self.draft_token_num = block_size
        self.custom_mask = custom_mask
        self.capture_hidden_mode = CaptureHiddenMode.FULL
        self.accept_length: Optional[torch.Tensor] = None
        self.accept_length_cpu = None
        self.accepted_indices: Optional[torch.Tensor] = None
        self.verified_id: Optional[torch.Tensor] = None

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.draft_token_num, self.draft_token_num

    def _build_custom_mask(self, batch: ScheduleBatch) -> None:
        device = self.draft_token.device
        masks = []
        for seq_len in batch.seq_lens_cpu.tolist():
            prefix_mask = torch.ones(
                (self.draft_token_num, seq_len),
                dtype=torch.bool,
                device=device,
            )
            draft_mask = torch.tril(
                torch.ones(
                    (self.draft_token_num, self.draft_token_num),
                    dtype=torch.bool,
                    device=device,
                )
            )
            masks.append(torch.cat((prefix_mask, draft_mask), dim=1).reshape(-1))

        self.custom_mask = (
            torch.cat(masks, dim=0)
            if masks
            else torch.empty((0,), dtype=torch.bool, device=device)
        )

    def prepare_for_verify(self, batch: ScheduleBatch, page_size: int):
        if batch.forward_mode.is_idle():
            return

        if page_size != 1:
            raise ValueError("DFlash only supports page_size=1.")

        batch.input_ids = self.draft_token
        batch.out_cache_loc = alloc_token_slots(
            batch.tree_cache,
            len(batch.input_ids),
        )

        bs = batch.batch_size()
        end_offset = batch.seq_lens + self.draft_token_num
        assign_req_to_token_pool[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            triton.next_power_of_2(bs),
        )

        if self.custom_mask is None:
            self._build_custom_mask(batch)

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        bs = len(req_pool_indices)
        device = req_pool_indices.device

        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        self.qo_indptr = (
            torch.arange(0, bs + 1, dtype=torch.int32, device=device)
            * self.draft_token_num
        )

        kv_indices = torch.empty(
            cum_kv_seq_len[-1], dtype=torch.int32, device=device
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

        if paged_kernel_lens_sum is None:
            paged_kernel_lens_sum = paged_kernel_lens.sum().item() - (
                self.draft_token_num * bs
            )
        if self.custom_mask is not None:
            mask_numel = paged_kernel_lens_sum * self.draft_token_num + (
                self.draft_token_num**2
            ) * bs
            if self.custom_mask.numel() < mask_numel:
                self.custom_mask = torch.cat(
                    [
                        self.custom_mask,
                        torch.full(
                            (mask_numel - self.custom_mask.numel(),),
                            True,
                            dtype=torch.bool,
                            device=device,
                        ),
                    ],
                    dim=0,
                )

        return kv_indices, cum_kv_seq_len, self.qo_indptr, self.custom_mask

    def verify(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        page_size: int,
        vocab_mask: Optional[torch.Tensor] = None,
    ):
        if batch.forward_mode.is_idle():
            return logits_output, torch.empty((0,), device=batch.device), 0

        if page_size != 1:
            raise ValueError("DFlash only supports page_size=1.")

        sampling_info = batch.sampling_info
        if not sampling_info.is_all_greedy:
            logger.warning(
                "DFlash only supports greedy sampling. Falling back to argmax."
            )

        if sampling_info.has_custom_logit_processor:
            apply_custom_logit_processor(
                logits_output.next_token_logits,
                sampling_info,
                num_tokens_in_batch=self.draft_token_num,
            )

        if sampling_info.penalizer_orchestrator.is_required:
            linear_penalty = torch.zeros(
                (batch.batch_size(), logits_output.next_token_logits.shape[1]),
                dtype=torch.float32,
                device=logits_output.next_token_logits.device,
            )
            sampling_info.apply_logits_bias(linear_penalty)
            logits_output.next_token_logits.add_(
                torch.repeat_interleave(
                    linear_penalty, self.draft_token_num, dim=0
                )
            )

        if vocab_mask is not None:
            logger.warning("DFlash grammar masking is not supported; ignoring mask.")

        bs = batch.batch_size()
        draft_tokens = self.draft_token.reshape(bs, self.draft_token_num)
        target_predict = torch.argmax(
            logits_output.next_token_logits, dim=-1
        ).reshape(bs, self.draft_token_num)

        matches = draft_tokens[:, 1:] == target_predict[:, :-1]
        accept_length = matches.cumprod(dim=1).sum(dim=1).to(torch.int32)

        accepted_indices = []
        verified_tokens = []
        adjusted_accept_lengths = []
        accept_length_cpu = accept_length.tolist()

        for i, req in enumerate(batch.reqs):
            acc = accept_length_cpu[i]
            output_tokens = draft_tokens[i, 1 : acc + 1].tolist()
            output_tokens.append(int(target_predict[i, acc].item()))

            appended = []
            for token in output_tokens:
                req.output_ids.append(token)
                appended.append(token)
                req.check_finished()
                if req.grammar is not None:
                    try:
                        req.grammar.accept_token(token)
                    except ValueError as e:
                        logger.info(
                            f"{i=}, {req=}\n"
                            f"{accept_length=}\n"
                            f"{draft_tokens=}\n"
                            f"{target_predict=}\n"
                        )
                        raise e
                if req.finished():
                    break

            num_output = len(appended)
            verified_tokens.extend(appended)
            accepted_indices.extend(
                range(i * self.draft_token_num, i * self.draft_token_num + num_output)
            )

            accept_len = max(num_output - 1, 0)
            adjusted_accept_lengths.append(accept_len)

            req.spec_verify_ct += 1
            req.spec_accepted_tokens += accept_len

        device = logits_output.next_token_logits.device
        self.accept_length = torch.tensor(
            adjusted_accept_lengths, dtype=torch.int32, device=device
        )
        self.accept_length_cpu = adjusted_accept_lengths
        self.accepted_indices = (
            torch.tensor(accepted_indices, dtype=torch.int64, device=device)
            if accepted_indices
            else torch.empty((0,), dtype=torch.int64, device=device)
        )
        self.verified_id = (
            torch.tensor(verified_tokens, dtype=torch.int64, device=device)
            if verified_tokens
            else torch.empty((0,), dtype=torch.int64, device=device)
        )

        if self.accepted_indices.numel() > 0:
            logits_output.next_token_logits = logits_output.next_token_logits[
                self.accepted_indices
            ]
            if logits_output.hidden_states is not None:
                logits_output.hidden_states = logits_output.hidden_states[
                    self.accepted_indices
                ]
        else:
            logits_output.next_token_logits = logits_output.next_token_logits[:0]
            if logits_output.hidden_states is not None:
                logits_output.hidden_states = logits_output.hidden_states[:0]

        evict_mask = torch.full_like(self.draft_token, True, dtype=torch.bool)
        if self.accepted_indices.numel() > 0:
            evict_mask[self.accepted_indices] = False
        batch.token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
        batch.out_cache_loc = batch.out_cache_loc[~evict_mask]

        accept_length_cpu = self.accept_length_cpu
        for req, accepted in zip(batch.reqs, accept_length_cpu):
            req.kv_committed_len += accepted + 1
            req.kv_allocated_len = req.kv_committed_len

        if bs > 0:
            assign_req_to_token_pool[(bs,)](
                batch.req_pool_indices,
                batch.req_to_token_pool.req_to_token,
                batch.seq_lens,
                batch.seq_lens + self.accept_length + 1,
                batch.out_cache_loc,
                batch.req_to_token_pool.req_to_token.shape[1],
                triton.next_power_of_2(bs),
            )

        batch.seq_lens.add_(self.accept_length + 1)
        batch.seq_lens_cpu.add_(
            torch.tensor(
                accept_length_cpu,
                dtype=batch.seq_lens_cpu.dtype,
                device=batch.seq_lens_cpu.device,
            )
            + 1
        )

        num_accepted_tokens = sum(accept_length_cpu)
        return logits_output, self.verified_id, num_accepted_tokens
