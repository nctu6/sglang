import inspect
import logging
from typing import Dict, Optional

import torch
from transformers import AutoModel

from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


class DFlashWorker:
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
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size
        self.block_size = server_args.speculative_num_draft_tokens
        self.draft_token_num = self.block_size - 1
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        if server_args.speculative_draft_model_path is None:
            raise ValueError("DFlash requires --speculative-draft-model-path.")

        if self.model_runner.tp_size > 1:
            logger.warning(
                "DFlash is untested with tensor parallelism; results may be incorrect."
            )
        if not server_args.disable_cuda_graph:
            logger.warning(
                "DFlash may require CUDA graph recapture after enabling aux hidden "
                "states. Consider running with --disable-cuda-graph if you see issues."
            )

        self.draft_model = AutoModel.from_pretrained(
            server_args.speculative_draft_model_path,
            revision=server_args.speculative_draft_model_revision,
            torch_dtype=self.model_runner.model_config.dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.draft_model.eval()

        try:
            draft_sig = inspect.signature(self.draft_model.forward)
            draft_params = draft_sig.parameters
            self.use_dflash_interface = (
                "noise_embedding" in draft_params and "target_hidden" in draft_params
            )
        except (TypeError, ValueError):
            self.use_dflash_interface = False

        if not self.use_dflash_interface:
            logger.warning(
                "DFlash: draft model does not support noise_embedding/target_hidden; "
                "falling back to input_ids-only draft generation."
            )

        def normalize_token_id(value, label: str) -> Optional[int]:
            if value is None:
                return None
            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    return None
                if len(value) > 1:
                    logger.warning(
                        "DFlash: %s has multiple ids; using the first one.", label
                    )
                value = value[0]
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    return None
                if value.numel() > 1:
                    logger.warning(
                        "DFlash: %s has multiple ids; using the first one.", label
                    )
                value = value.flatten()[0].item()
            if isinstance(value, bool):
                return int(value)
            return int(value)

        self.mask_token_id = server_args.speculative_dflash_mask_token_id
        if self.mask_token_id is None:
            self.mask_token_id = getattr(self.draft_model.config, "mask_token_id", None)
            if self.mask_token_id is not None:
                logger.info("DFlash: using draft model mask_token_id.")
        if self.mask_token_id is None:
            self.mask_token_id = target_worker.tokenizer.mask_token_id
            if self.mask_token_id is not None:
                logger.info("DFlash: using target tokenizer mask_token_id.")
        if self.mask_token_id is None:
            self.mask_token_id = getattr(self.draft_model.config, "pad_token_id", None)
            if self.mask_token_id is not None:
                logger.warning(
                    "DFlash: no mask token; using draft model pad_token_id as mask."
                )
        if self.mask_token_id is None:
            self.mask_token_id = target_worker.tokenizer.pad_token_id
            if self.mask_token_id is not None:
                logger.warning(
                    "DFlash: no mask token; using target tokenizer pad_token_id as mask."
                )
        if self.mask_token_id is None:
            self.mask_token_id = getattr(self.draft_model.config, "eos_token_id", None)
            if self.mask_token_id is not None:
                logger.warning(
                    "DFlash: no mask/pad token; using draft model eos_token_id as mask."
                )
        if self.mask_token_id is None:
            self.mask_token_id = target_worker.tokenizer.eos_token_id
            if self.mask_token_id is not None:
                logger.warning(
                    "DFlash: no mask/pad token; using target tokenizer eos_token_id as mask."
                )
        self.mask_token_id = normalize_token_id(self.mask_token_id, "mask_token_id")
        if self.mask_token_id is None:
            raise ValueError(
                "DFlash requires a mask token id. Set --speculative-dflash-mask-token-id."
            )

        if hasattr(self.draft_model.config, "block_size"):
            if self.draft_model.config.block_size != self.block_size:
                logger.warning(
                    "Overriding draft model block_size to match speculative_num_draft_tokens."
                )
                self.draft_model.config.block_size = self.block_size

        if self.use_dflash_interface:
            num_target_layers = self.model_runner.model_config.num_hidden_layers
            num_draft_layers = self.draft_model.config.num_hidden_layers
            self.target_layer_ids = build_target_layer_ids(
                num_target_layers, num_draft_layers
            )
            if hasattr(self.model_runner.model, "set_eagle3_layers_to_capture"):
                self.model_runner.model.set_eagle3_layers_to_capture(
                    self.target_layer_ids
                )
            else:
                raise ValueError(
                    "Target model does not support capturing auxiliary hidden states."
                )
            rotary_emb = getattr(self.draft_model, "rotary_emb", None)
            if rotary_emb is not None and hasattr(rotary_emb, "forward"):
                orig_forward = rotary_emb.forward

                def patched_forward(hidden_states, position_ids=None, *args, **kwargs):
                    if position_ids is not None and hidden_states is not None:
                        pos_len = position_ids.shape[-1]
                        q_len = hidden_states.shape[-2]
                        if pos_len > q_len:
                            dummy = hidden_states.new_empty(
                                hidden_states.shape[0],
                                pos_len,
                                hidden_states.shape[-1],
                            )
                            return orig_forward(dummy, position_ids, *args, **kwargs)
                    return orig_forward(hidden_states, position_ids, *args, **kwargs)

                rotary_emb.forward = patched_forward
        else:
            self.target_layer_ids = []

        self.embed_tokens = self.model_runner.model.get_input_embeddings()
        self.lm_head = self.model_runner.model.lm_head

        self.req_state: Dict[str, torch.Tensor] = {}

    def clear_cache_pool(self):
        self.req_state.clear()

    def _update_target_hidden_from_extend(
        self,
        batch: ScheduleBatch,
        extend_seq_lens: list[int],
        hidden_states: Optional[torch.Tensor],
    ):
        if not self.use_dflash_interface:
            return
        if hidden_states is None:
            return
        hidden_states = self._normalize_target_hidden(hidden_states, "extend")
        offset = 0
        for req, extend_len in zip(batch.reqs, extend_seq_lens):
            new_hidden = hidden_states[offset : offset + extend_len]
            offset += extend_len
            cached_tokens = getattr(req, "cached_tokens", 0)
            if cached_tokens > 0:
                existing = self.req_state.get(req.rid)
                if existing is None:
                    existing = new_hidden.new_zeros(
                        (cached_tokens, new_hidden.shape[-1])
                    )
                elif existing.shape[0] < cached_tokens:
                    pad_len = cached_tokens - existing.shape[0]
                    pad = existing.new_zeros((pad_len, existing.shape[-1]))
                    existing = torch.cat((existing, pad), dim=0)
                elif existing.shape[0] > cached_tokens:
                    existing = existing[:cached_tokens]
                self.req_state[req.rid] = existing
            if req.rid in self.req_state:
                self.req_state[req.rid] = torch.cat(
                    (self.req_state[req.rid], new_hidden), dim=0
                )
            else:
                self.req_state[req.rid] = new_hidden

    def _update_target_hidden_from_verify(
        self,
        batch: ScheduleBatch,
        accept_length: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ):
        if not self.use_dflash_interface:
            return
        if hidden_states is None:
            return
        hidden_states = self._normalize_target_hidden(hidden_states, "verify")
        lengths = (accept_length + 1).tolist()
        offset = 0
        for req, length in zip(batch.reqs, lengths):
            if length <= 0:
                continue
            new_hidden = hidden_states[offset : offset + length]
            offset += length
            if req.rid in self.req_state:
                self.req_state[req.rid] = torch.cat(
                    (self.req_state[req.rid], new_hidden), dim=0
                )
            else:
                self.req_state[req.rid] = new_hidden

    def _cleanup_req_state(self, batch: ScheduleBatch):
        active_ids = {
            req.rid
            for req in batch.reqs
            if not req.finished() and not req.is_retracted
        }
        for rid in list(self.req_state.keys()):
            if rid not in active_ids:
                del self.req_state[rid]

    def _normalize_target_hidden(
        self, hidden_states: torch.Tensor, label: str
    ) -> torch.Tensor:
        expected_dim = None
        if self.use_dflash_interface and self.target_layer_ids:
            expected_dim = (
                self.model_runner.model_config.hidden_size * len(self.target_layer_ids)
            )

        if expected_dim is None or hidden_states.shape[-1] == expected_dim:
            return hidden_states

        cur_dim = hidden_states.shape[-1]
        if expected_dim % cur_dim == 0:
            repeat_factor = expected_dim // cur_dim
            logger.warning(
                "DFlash: %s hidden dim %d -> %d by repeat.",
                label,
                cur_dim,
                expected_dim,
            )
            return hidden_states.repeat(1, repeat_factor)

        if cur_dim < expected_dim:
            logger.warning(
                "DFlash: %s hidden dim %d -> %d by padding.",
                label,
                cur_dim,
                expected_dim,
            )
            pad = hidden_states.new_zeros(
                hidden_states.shape[0], expected_dim - cur_dim
            )
            return torch.cat([hidden_states, pad], dim=-1)

        logger.warning(
            "DFlash: %s hidden dim %d -> %d by truncation.",
            label,
            cur_dim,
            expected_dim,
        )
        return hidden_states[:, :expected_dim]

    def _compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        lm_head = self.lm_head
        if hasattr(lm_head, "weight"):
            weight = lm_head.weight
            if self.model_runner.server_args.enable_fp32_lm_head:
                logits = torch.matmul(
                    hidden_states.to(torch.float32), weight.to(torch.float32).T
                )
            else:
                logits = torch.matmul(hidden_states.to(weight.dtype), weight.T)
            if getattr(lm_head, "bias", None) is not None:
                logits = logits + lm_head.bias
            return logits
        if hasattr(lm_head, "quant_method"):
            return lm_head.quant_method.apply(lm_head, hidden_states, None)
        return lm_head(hidden_states)

    def _generate_draft_tokens(self, batch: ScheduleBatch):
        bs = batch.batch_size()
        device = batch.seq_lens.device
        draft_tokens = torch.empty(
            (bs, self.block_size), dtype=torch.long, device=device
        )
        positions = torch.empty(
            (bs, self.block_size), dtype=torch.long, device=device
        )

        with torch.inference_mode():
            for i, req in enumerate(batch.reqs):
                if req.output_ids:
                    prefix_token = req.output_ids[-1]
                else:
                    prefix_token = req.origin_input_ids[-1]

                seq_len = int(batch.seq_lens_cpu[i])
                pos_ids = torch.arange(
                    seq_len, seq_len + self.block_size, device=device
                ).unsqueeze(0)
                positions[i] = pos_ids.squeeze(0)

                block_tokens = torch.full(
                    (self.block_size,),
                    self.mask_token_id,
                    device=device,
                    dtype=torch.long,
                )
                block_tokens[0] = prefix_token

                if self.use_dflash_interface:
                    noise_embedding = self.embed_tokens(block_tokens.unsqueeze(0))
                    target_hidden = self.req_state.get(req.rid)
                    if target_hidden is None:
                        target_hidden = noise_embedding.new_zeros(
                            (seq_len, expected_hidden_dim)
                        )
                        self.req_state[req.rid] = target_hidden
                    if target_hidden.shape[0] != seq_len:
                        if target_hidden.shape[0] < seq_len:
                            pad_len = seq_len - target_hidden.shape[0]
                            pad = target_hidden.new_zeros(
                                (pad_len, target_hidden.shape[-1])
                            )
                            target_hidden = torch.cat((target_hidden, pad), dim=0)
                        else:
                            target_hidden = target_hidden[:seq_len]
                        self.req_state[req.rid] = target_hidden
                    expected_hidden_dim = (
                        self.model_runner.model_config.hidden_size
                        * len(self.target_layer_ids)
                    )
                    if target_hidden.shape[-1] != expected_hidden_dim:
                        raise RuntimeError(
                            "DFlash target hidden states dimension mismatch."
                        )
                    full_pos_ids = torch.arange(
                        seq_len + self.block_size, device=device
                    ).unsqueeze(0)
                    draft_output = self.draft_model(
                        position_ids=full_pos_ids,
                        noise_embedding=noise_embedding,
                        target_hidden=target_hidden.unsqueeze(0),
                        use_cache=False,
                    )
                else:
                    draft_output = self.draft_model(
                        input_ids=block_tokens.unsqueeze(0),
                        position_ids=pos_ids,
                        use_cache=False,
                    )

                if isinstance(draft_output, torch.Tensor):
                    hidden = draft_output
                elif hasattr(draft_output, "last_hidden_state"):
                    hidden = draft_output.last_hidden_state
                else:
                    hidden = draft_output[0]
                draft_logits = self._compute_draft_logits(hidden[:, 1:, :])
                draft_sample = torch.argmax(draft_logits, dim=-1)

                draft_tokens[i, 0] = prefix_token
                draft_tokens[i, 1:] = draft_sample.squeeze(0)

        return draft_tokens.reshape(-1), positions.reshape(-1)

    def forward_batch_generation(self, batch: ScheduleBatch):
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            model_worker_batch = batch.get_model_worker_batch()
            model_worker_batch.capture_hidden_mode = (
                CaptureHiddenMode.FULL
                if self.use_dflash_interface
                else CaptureHiddenMode.NULL
            )
            batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
            logits_output, next_token_ids = (
                batch_result.logits_output,
                batch_result.next_token_ids,
            )
            self._update_target_hidden_from_extend(
                batch, model_worker_batch.extend_seq_lens, logits_output.hidden_states
            )
            self._cleanup_req_state(batch)
            return batch_result

        draft_tokens, positions = self._generate_draft_tokens(batch)
        verify_input = DFlashVerifyInput(
            draft_token=draft_tokens,
            positions=positions,
            block_size=self.block_size,
        )
        batch.spec_algorithm = SpeculativeAlgorithm.DFLASH
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = verify_input
        verify_input.prepare_for_verify(batch, self.page_size)

        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = (
            CaptureHiddenMode.FULL
            if self.use_dflash_interface
            else CaptureHiddenMode.NULL
        )
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        logits_output, verified_id, num_accepted_tokens = verify_input.verify(
            batch, logits_output, self.page_size
        )
        if batch.return_logprob:
            add_output_logprobs_for_spec_v1(batch, verify_input, logits_output)

        self._update_target_hidden_from_verify(
            batch, verify_input.accept_length, logits_output.hidden_states
        )
        self._cleanup_req_state(batch)
        batch.forward_mode = ForwardMode.DECODE

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=verified_id,
            num_accepted_tokens=num_accepted_tokens,
            can_run_cuda_graph=can_run_cuda_graph,
            accept_lens=verify_input.accept_length,
        )
