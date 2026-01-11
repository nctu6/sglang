import copy
import inspect
import logging
from typing import Dict, Optional

import torch
from transformers import AutoModel, AutoTokenizer
from transformers.cache_utils import DynamicCache

from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import alloc_token_slots
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import DFlashDraftInput, DFlashVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.common import empty_context

logger = logging.getLogger(__name__)

KNOWN_MASK_TOKENS = ("<|MASK|>", "<|mask|>", "<mask>", "[MASK]", "<MASK>")


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
        self.embed_tokens = self.model_runner.model.get_input_embeddings()
        self.lm_head = self.model_runner.model.lm_head

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

        self.draft_tokenizer = None
        self.draft_worker: Optional[TpModelWorker] = None
        self.draft_runner = None
        self.draft_req_to_token_pool: Optional[ReqToTokenPool] = None
        self.draft_token_to_kv_pool_allocator = None
        self.draft_req_pool_indices: Dict[str, int] = {}
        self.use_sglang_draft = False

        if self._init_sglang_draft_worker(
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            moe_ep_rank=moe_ep_rank,
            nccl_port=nccl_port,
        ):
            self.use_sglang_draft = True
            self.use_dflash_interface = True
            self.draft_model = self.draft_runner.model
            self.draft_model.eval()
        else:
            draft_attn_impl = self._select_draft_attn_impl()
            draft_model_kwargs = dict(
                revision=server_args.speculative_draft_model_revision,
                torch_dtype=self.model_runner.model_config.dtype,
                trust_remote_code=True,
            )
            if draft_attn_impl is not None:
                draft_model_kwargs["attn_implementation"] = draft_attn_impl
            try:
                self.draft_model = AutoModel.from_pretrained(
                    server_args.speculative_draft_model_path,
                    **draft_model_kwargs,
                ).to(self.device)
            except Exception as exc:
                if "attn_implementation" in draft_model_kwargs:
                    logger.warning(
                        "DFlash: draft model load failed with attn_implementation=%s: %s. Retrying without.",
                        draft_attn_impl,
                        exc,
                    )
                    draft_model_kwargs.pop("attn_implementation", None)
                    self.draft_model = AutoModel.from_pretrained(
                        server_args.speculative_draft_model_path,
                        **draft_model_kwargs,
                    ).to(self.device)
                else:
                    raise
            self.draft_model.eval()
            used_attn_impl = getattr(
                self.draft_model.config, "_attn_implementation", None
            )
            if used_attn_impl:
                logger.info("DFlash: draft attn_implementation=%s.", used_attn_impl)

            try:
                self.draft_tokenizer = AutoTokenizer.from_pretrained(
                    server_args.speculative_draft_model_path,
                    revision=server_args.speculative_draft_model_revision,
                    trust_remote_code=True,
                )
            except Exception as exc:
                logger.debug("DFlash: failed to load draft tokenizer: %s", exc)
                try:
                    self.draft_tokenizer = AutoTokenizer.from_pretrained(
                        server_args.speculative_draft_model_path,
                        revision=server_args.speculative_draft_model_revision,
                        trust_remote_code=True,
                        use_fast=False,
                    )
                except Exception as exc2:
                    logger.debug(
                        "DFlash: failed to load draft tokenizer (slow): %s", exc2
                    )

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
        self._aux_capture_initialized = False
        self._cuda_graph_recaptured = False
        self._hidden_dim_mismatch_seen = False
        self.mask_token_id_out_of_vocab = False
        self.mask_token_fallback_id: Optional[int] = None
        self.mask_embedding: Optional[torch.Tensor] = None

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

        def token_id_to_token(tokenizer, token_id) -> Optional[str]:
            if tokenizer is None or token_id is None:
                return None
            try:
                token = tokenizer.convert_ids_to_tokens(int(token_id))
            except Exception:
                return None
            if isinstance(token, (list, tuple)):
                return token[0] if token else None
            return token

        def is_known_mask_token(token: Optional[str]) -> bool:
            return token in KNOWN_MASK_TOKENS

        def is_known_mask_token_id(tokenizer, token_id) -> bool:
            return is_known_mask_token(token_id_to_token(tokenizer, token_id))

        def infer_mask_token_id(tokenizer, vocab_size: Optional[int] = None):
            if tokenizer is None:
                return None, None
            special_map = getattr(tokenizer, "special_tokens_map", None) or {}
            candidates = []
            mask_token = special_map.get("mask_token")
            if mask_token and is_known_mask_token(mask_token):
                candidates.append(mask_token)
            additional = special_map.get("additional_special_tokens") or []
            for token in additional:
                if is_known_mask_token(token):
                    candidates.append(token)
            candidates.extend(KNOWN_MASK_TOKENS)
            for token in candidates:
                token_id = tokenizer.convert_tokens_to_ids(token)
                unk_id = getattr(tokenizer, "unk_token_id", None)
                if token_id is not None and (unk_id is None or token_id != unk_id):
                    if vocab_size is None or token_id < vocab_size:
                        return token_id, token
            if hasattr(tokenizer, "get_vocab"):
                vocab = tokenizer.get_vocab()
                for token in KNOWN_MASK_TOKENS:
                    token_id = vocab.get(token)
                    if token_id is not None:
                        if vocab_size is None or token_id < vocab_size:
                            return token_id, token
            return None, None

        self.mask_token_id = server_args.speculative_dflash_mask_token_id
        target_vocab_size = getattr(self.model_runner.model_config, "vocab_size", None)
        if self.mask_token_id is None:
            self.mask_token_id = getattr(self.draft_model.config, "mask_token_id", None)
            if self.mask_token_id is not None:
                if (
                    self.draft_tokenizer is not None
                    and not is_known_mask_token_id(
                        self.draft_tokenizer, self.mask_token_id
                    )
                ):
                    logger.warning(
                        "DFlash: ignoring draft model mask_token_id %s; not a known mask token.",
                        self.mask_token_id,
                    )
                    self.mask_token_id = None
                else:
                    logger.info("DFlash: using draft model mask_token_id.")
        if self.mask_token_id is None:
            if self.draft_tokenizer is not None:
                self.mask_token_id = normalize_token_id(
                    self.draft_tokenizer.mask_token_id, "mask_token_id"
                )
                if self.mask_token_id is not None and is_known_mask_token_id(
                    self.draft_tokenizer, self.mask_token_id
                ):
                    logger.info("DFlash: using draft tokenizer mask_token_id.")
                else:
                    self.mask_token_id = None
        if self.mask_token_id is None:
            inferred_id, inferred_token = infer_mask_token_id(
                self.draft_tokenizer, target_vocab_size
            )
            if inferred_id is not None:
                self.mask_token_id = inferred_id
                logger.info(
                    "DFlash: inferred mask token from draft tokenizer: %s.",
                    inferred_token,
                )
        if self.mask_token_id is None:
            self.mask_token_id = normalize_token_id(
                target_worker.tokenizer.mask_token_id, "mask_token_id"
            )
            if self.mask_token_id is not None and is_known_mask_token_id(
                target_worker.tokenizer, self.mask_token_id
            ):
                logger.info("DFlash: using target tokenizer mask_token_id.")
            else:
                self.mask_token_id = None
        if self.mask_token_id is None:
            inferred_id, inferred_token = infer_mask_token_id(
                target_worker.tokenizer, target_vocab_size
            )
            if inferred_id is not None:
                self.mask_token_id = inferred_id
                logger.info(
                    "DFlash: inferred mask token from target tokenizer: %s.",
                    inferred_token,
                )
        if self.mask_token_id is None:
            self._maybe_add_mask_token(target_worker)
            target_vocab_size = getattr(
                self.model_runner.model_config, "vocab_size", target_vocab_size
            )
        if self.mask_token_id is None:
            self.mask_token_id = getattr(self.draft_model.config, "pad_token_id", None)
            if self.mask_token_id is not None:
                logger.warning(
                    "DFlash: no mask token; using draft model pad_token_id as mask."
                )
        if self.mask_token_id is None:
            if self.draft_tokenizer is not None:
                self.mask_token_id = self.draft_tokenizer.pad_token_id
                if self.mask_token_id is not None:
                    logger.warning(
                        "DFlash: no mask token; using draft tokenizer pad_token_id as mask."
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
            if self.draft_tokenizer is not None:
                self.mask_token_id = self.draft_tokenizer.eos_token_id
                if self.mask_token_id is not None:
                    logger.warning(
                        "DFlash: no mask/pad token; using draft tokenizer eos_token_id as mask."
                    )
        if self.mask_token_id is None:
            self.mask_token_id = target_worker.tokenizer.eos_token_id
            if self.mask_token_id is not None:
                logger.warning(
                    "DFlash: no mask/pad token; using target tokenizer eos_token_id as mask."
                )
        self.mask_token_id = normalize_token_id(self.mask_token_id, "mask_token_id")
        if (
            self.mask_token_id is not None
            and target_vocab_size is not None
            and self.mask_token_id >= target_vocab_size
        ):
            self.mask_token_id_out_of_vocab = True
            self.mask_token_fallback_id = (
                target_worker.tokenizer.eos_token_id
                or target_worker.tokenizer.pad_token_id
                or 0
            )
            self.mask_embedding = self._compute_mask_embedding()
            logger.warning(
                "DFlash: mask_token_id %d >= target vocab size %d; using fallback embedding.",
                self.mask_token_id,
                target_vocab_size,
            )
        if self.mask_token_id is None:
            self.mask_token_id = target_worker.tokenizer.pad_token_id
            if self.mask_token_id is not None:
                logger.warning(
                    "DFlash: no valid mask token; using target tokenizer pad_token_id as mask."
                )
        if self.mask_token_id is None:
            self.mask_token_id = target_worker.tokenizer.eos_token_id
            if self.mask_token_id is not None:
                logger.warning(
                    "DFlash: no valid mask/pad token; using target tokenizer eos_token_id as mask."
                )
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
            self._ensure_target_aux_capture()
            self._recapture_cuda_graph_if_needed()
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

        self.req_state: Dict[str, torch.Tensor] = {}
        self.draft_cache: Dict[str, DynamicCache] = {}

    def clear_cache_pool(self):
        self.req_state.clear()
        self.draft_cache.clear()
        if self.draft_req_to_token_pool is not None and self.draft_req_pool_indices:
            self.draft_req_to_token_pool.free(
                list(self.draft_req_pool_indices.values())
            )
        self.draft_req_pool_indices.clear()

    def _compute_mask_embedding(self) -> Optional[torch.Tensor]:
        embed = self.embed_tokens
        if not hasattr(embed, "weight"):
            logger.warning(
                "DFlash: cannot access embed_tokens weight; using zero mask embedding."
            )
            return torch.zeros(
                embed.embedding_dim,
                device=self.device,
                dtype=self.model_runner.model_config.dtype,
            )
        weight = embed.weight
        if hasattr(embed, "org_vocab_size"):
            base_vocab = embed.org_vocab_size
            if base_vocab > 0:
                return weight[:base_vocab].mean(dim=0)
        return weight.mean(dim=0)

    def _build_noise_embedding(self, block_tokens: torch.Tensor) -> torch.Tensor:
        if not self.mask_token_id_out_of_vocab:
            return self.embed_tokens(block_tokens.unsqueeze(0))
        safe_tokens = block_tokens.clone()
        mask_positions = safe_tokens == self.mask_token_id
        if mask_positions.any():
            safe_tokens[mask_positions] = int(self.mask_token_fallback_id)
        embeddings = self.embed_tokens(safe_tokens.unsqueeze(0))
        if mask_positions.any():
            embeddings = embeddings.clone()
            embeddings[0, mask_positions] = self.mask_embedding
        return embeddings

    def _build_dflash_input_ids(
        self,
        mem_len: int,
        seq_len: int,
        block_tokens: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        input_ids = torch.zeros(mem_len, dtype=torch.long, device=device)
        input_ids[seq_len:] = block_tokens
        return input_ids

    def _select_draft_attn_impl(self) -> Optional[str]:
        backend = self.server_args.speculative_draft_attention_backend

        def _auto_backend() -> str:
            try:
                from transformers.utils.import_utils import is_flash_attn_2_available

                if is_flash_attn_2_available():
                    return "flash_attention_2"
            except Exception:
                pass
            return "sdpa"

        if backend is None:
            return None

        backend = backend.lower()
        if backend in {"auto"}:
            return _auto_backend()
        if backend in {"flash_attention_2", "flashattn2", "fa2"}:
            return "flash_attention_2"
        if backend in {"sdpa", "torch_sdpa"}:
            return "sdpa"
        if backend in {"eager", "torch"}:
            return "eager"
        if backend in {"fa3", "flashinfer", "triton"}:
            logger.warning(
                "DFlash draft attention backend '%s' is not supported by HF; using default HF attention implementation.",
                backend,
            )
            return None
        raise ValueError(
            "DFlash draft attention backend must be one of "
            "auto, flash_attention_2, sdpa, eager, fa2/flashattn2, "
            "or one of fa3/flashinfer/triton (mapped to HF default)."
        )

    def _init_sglang_draft_worker(
        self,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
    ) -> bool:
        try:
            draft_args = copy.deepcopy(self.server_args)
            draft_args.disable_cuda_graph = True
            draft_args.skip_tokenizer_init = True
            draft_args.speculative_draft_model_path = (
                self.server_args.speculative_draft_model_path
            )
            draft_args.speculative_draft_model_revision = (
                self.server_args.speculative_draft_model_revision
            )

            draft_req_to_token_pool = self.target_worker.get_memory_pool()[0]
            if isinstance(draft_req_to_token_pool, ReqToTokenPool):
                draft_req_to_token_pool = ReqToTokenPool(
                    size=draft_req_to_token_pool.size,
                    max_context_len=draft_req_to_token_pool.max_context_len,
                    device=draft_req_to_token_pool.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                )
            else:
                logger.warning(
                    "DFlash: draft req_to_token_pool is not a plain ReqToTokenPool; using shared pool."
                )

            token_to_kv_pool_allocator = self.target_worker.get_memory_pool()[1]

            with (
                empty_context(),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                draft_worker = TpModelWorker(
                    server_args=draft_args,
                    gpu_id=gpu_id,
                    tp_rank=tp_rank,
                    pp_rank=0,
                    dp_rank=dp_rank,
                    moe_ep_rank=moe_ep_rank,
                    nccl_port=nccl_port,
                    is_draft_worker=True,
                    req_to_token_pool=draft_req_to_token_pool,
                    token_to_kv_pool_allocator=token_to_kv_pool_allocator,
                )

            self.draft_worker = draft_worker
            self.draft_runner = draft_worker.model_runner
            self.draft_req_to_token_pool = draft_req_to_token_pool
            self.draft_token_to_kv_pool_allocator = token_to_kv_pool_allocator

            if self.draft_runner.model.__class__.__name__ != "DFlashDraftModel":
                logger.warning(
                    "DFlash: draft model architecture %s is not DFlashDraftModel; falling back to HF draft.",
                    self.draft_runner.model.__class__.__name__,
                )
                self.draft_worker = None
                self.draft_runner = None
                self.draft_req_to_token_pool = None
                self.draft_token_to_kv_pool_allocator = None
                return False

            embed, head = self.target_worker.model_runner.model.get_embed_and_head()
            if hasattr(self.draft_runner.model, "set_embed_and_head"):
                self.draft_runner.model.set_embed_and_head(embed, head)
            elif hasattr(self.draft_runner.model, "set_embed"):
                self.draft_runner.model.set_embed(embed)

            if self.server_args.speculative_draft_attention_backend:
                logger.info(
                    "DFlash: draft attention backend=%s.",
                    self.server_args.speculative_draft_attention_backend,
                )
            return True
        except Exception as exc:
            logger.warning(
                "DFlash: failed to initialize SGLang draft runner: %s", exc
            )
            return False

    def _maybe_add_mask_token(self, target_worker: TpModelWorker) -> None:
        tokenizer = target_worker.tokenizer
        if tokenizer is None:
            return
        if tokenizer.mask_token_id is not None:
            token = None
            try:
                token = tokenizer.convert_ids_to_tokens(tokenizer.mask_token_id)
            except Exception:
                token = None
            if isinstance(token, (list, tuple)):
                token = token[0] if token else None
            if token in KNOWN_MASK_TOKENS:
                self.mask_token_id = tokenizer.mask_token_id
                logger.info("DFlash: using target tokenizer mask_token_id.")
                return
            logger.warning(
                "DFlash: target tokenizer mask token %s is not a known mask token; overriding with <|MASK|>.",
                token,
            )
        try:
            added = tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        except Exception as exc:
            logger.warning("DFlash: failed to add mask token to tokenizer: %s", exc)
            return
        if tokenizer.mask_token_id is not None:
            self.mask_token_id = tokenizer.mask_token_id
            if added > 0:
                logger.info(
                    "DFlash: added mask token to target tokenizer: <|MASK|>."
                )
            else:
                logger.info("DFlash: set mask token on target tokenizer: <|MASK|>.")

    def _recapture_cuda_graph_if_needed(self) -> None:
        if self._cuda_graph_recaptured:
            return
        if self.server_args.disable_cuda_graph:
            return
        if self.model_runner.device == "cpu":
            return
        if getattr(self.model_runner, "graph_runner", None) is None:
            return
        logger.info(
            "DFlash: recapturing cuda graph after enabling aux hidden states."
        )
        self.model_runner.init_device_graphs()
        self._cuda_graph_recaptured = True

    def _ensure_target_aux_capture(self) -> None:
        if not self.use_dflash_interface:
            return
        model = self.model_runner.model
        if not hasattr(model, "set_eagle3_layers_to_capture"):
            raise ValueError(
                "Target model does not support capturing auxiliary hidden states."
            )
        expected_layers = [val + 1 for val in self.target_layer_ids]
        if not getattr(model, "capture_aux_hidden_states", False):
            model.set_eagle3_layers_to_capture(self.target_layer_ids)
        if hasattr(model, "capture_aux_hidden_states"):
            model.capture_aux_hidden_states = True
        if (
            hasattr(model, "model")
            and hasattr(model.model, "capture_aux_hidden_states_post_layer")
        ):
            model.model.capture_aux_hidden_states_post_layer = False
        if (
            hasattr(model, "model")
            and hasattr(model.model, "layers_to_capture")
            and model.model.layers_to_capture != expected_layers
        ):
            model.model.layers_to_capture = expected_layers
        if not self._aux_capture_initialized:
            self._aux_capture_initialized = True
            logger.info(
                "DFlash: target aux hidden capture enabled. layers_to_capture=%s",
                getattr(
                    getattr(model, "model", None),
                    "layers_to_capture",
                    None,
                ),
            )

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
        for rid in list(self.draft_cache.keys()):
            if rid not in active_ids:
                del self.draft_cache[rid]
        for rid in list(self.draft_req_pool_indices.keys()):
            if rid not in active_ids:
                idx = self.draft_req_pool_indices.pop(rid)
                if self.draft_req_to_token_pool is not None:
                    self.draft_req_to_token_pool.free(idx)

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
        if (
            not self.server_args.disable_cuda_graph
            and getattr(self.model_runner, "graph_runner", None) is not None
            and not self._hidden_dim_mismatch_seen
        ):
            logger.warning(
                "DFlash: %s hidden dim %d != %d; will recapture cuda graph for aux hidden states.",
                label,
                cur_dim,
                expected_dim,
            )
            self._hidden_dim_mismatch_seen = True
            self._cuda_graph_recaptured = False
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
        if self.use_sglang_draft:
            return self._generate_draft_tokens_sglang(batch)

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

                seq_lens_cpu = (
                    batch.seq_lens_cpu
                    if batch.seq_lens_cpu is not None
                    else batch.seq_lens.cpu()
                )
                seq_len = int(seq_lens_cpu[i])
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
                    expected_hidden_dim = (
                        self.model_runner.model_config.hidden_size
                        * len(self.target_layer_ids)
                    )
                    noise_embedding = self._build_noise_embedding(block_tokens)
                    draft_cache = self.draft_cache.get(req.rid)
                    if draft_cache is None:
                        draft_cache = DynamicCache()
                        self.draft_cache[req.rid] = draft_cache
                    cache_len = (
                        draft_cache.get_seq_length()
                        if hasattr(draft_cache, "get_seq_length")
                        else 0
                    )
                    if cache_len > seq_len and hasattr(draft_cache, "crop"):
                        draft_cache.crop(seq_len)
                        cache_len = seq_len
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
                    if target_hidden.shape[-1] != expected_hidden_dim:
                        raise RuntimeError(
                            "DFlash target hidden states dimension mismatch."
                        )
                    if cache_len < seq_len:
                        target_hidden = target_hidden[cache_len:seq_len]
                    else:
                        target_hidden = target_hidden.new_zeros((0, expected_hidden_dim))
                    full_pos_ids = torch.arange(
                        cache_len, seq_len + self.block_size, device=device
                    ).unsqueeze(0)
                    draft_output = self.draft_model(
                        position_ids=full_pos_ids,
                        noise_embedding=noise_embedding,
                        target_hidden=target_hidden.unsqueeze(0),
                        past_key_values=draft_cache,
                        use_cache=True,
                        is_causal=False,
                    )
                    if hasattr(draft_cache, "crop"):
                        draft_cache.crop(seq_len)
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

    def _generate_draft_tokens_sglang(self, batch: ScheduleBatch):
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
                )
                positions[i] = pos_ids

                block_tokens = torch.full(
                    (self.block_size,),
                    self.mask_token_id,
                    device=device,
                    dtype=torch.long,
                )
                block_tokens[0] = prefix_token

                expected_hidden_dim = (
                    self.model_runner.model_config.hidden_size
                    * len(self.target_layer_ids)
                )
                target_hidden = self.req_state.get(req.rid)
                if target_hidden is None:
                    target_hidden = torch.zeros(
                        (seq_len, expected_hidden_dim),
                        device=device,
                        dtype=self.model_runner.model_config.dtype,
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
                if target_hidden.shape[-1] != expected_hidden_dim:
                    raise RuntimeError("DFlash target hidden states dimension mismatch.")

                mem_len = seq_len + self.block_size
                mem_positions = torch.arange(mem_len, device=device, dtype=torch.long)
                noise_embedding = self._build_noise_embedding(block_tokens).squeeze(0)
                target_placeholder = noise_embedding.new_zeros(
                    (seq_len, noise_embedding.shape[-1])
                )
                input_embeds = torch.cat((target_placeholder, noise_embedding), dim=0)
                if batch.tree_cache is not None:
                    out_cache_loc = alloc_token_slots(batch.tree_cache, mem_len)
                else:
                    out_cache_loc = self.draft_token_to_kv_pool_allocator.alloc(mem_len)
                    if out_cache_loc is None:
                        raise RuntimeError("DFlash draft KV cache allocation failed.")

                draft_pool_idx = self.draft_req_pool_indices.get(req.rid)
                if draft_pool_idx is None:
                    pool = self.draft_req_to_token_pool
                    if pool is None:
                        raise RuntimeError("DFlash draft req_to_token_pool is not set.")
                    alloc = pool.alloc(1)
                    if alloc is None:
                        raise RuntimeError(
                            "DFlash draft req_to_token_pool is full."
                        )
                    draft_pool_idx = alloc[0]
                    self.draft_req_pool_indices[req.rid] = draft_pool_idx

                token_pool = self.draft_req_to_token_pool.req_to_token
                token_pool[draft_pool_idx, :mem_len] = out_cache_loc.to(
                    token_pool.dtype
                )

                seq_lens = torch.tensor(
                    [mem_len], dtype=batch.seq_lens.dtype, device=device
                )
                seq_lens_cpu_dtype = (
                    batch.seq_lens_cpu.dtype
                    if batch.seq_lens_cpu is not None
                    else torch.int32
                )
                seq_lens_cpu = torch.tensor([mem_len], dtype=seq_lens_cpu_dtype)
                req_pool_indices = torch.tensor(
                    [draft_pool_idx], dtype=batch.req_pool_indices.dtype, device=device
                )
                forward_batch = ForwardBatch(
                    forward_mode=ForwardMode.EXTEND,
                    batch_size=1,
                    input_ids=self._build_dflash_input_ids(
                        mem_len, seq_len, block_tokens, device
                    ),
                    req_pool_indices=req_pool_indices,
                    seq_lens=seq_lens,
                    out_cache_loc=out_cache_loc,
                    seq_lens_sum=mem_len,
                )
                forward_batch.seq_lens_cpu = seq_lens_cpu
                forward_batch.positions = mem_positions
                forward_batch.extend_num_tokens = mem_len
                forward_batch.extend_seq_lens = seq_lens
                forward_batch.extend_prefix_lens = torch.zeros_like(seq_lens)
                forward_batch.extend_seq_lens_cpu = [mem_len]
                forward_batch.extend_prefix_lens_cpu = [0]
                forward_batch.req_to_token_pool = self.draft_runner.req_to_token_pool
                forward_batch.token_to_kv_pool = self.draft_runner.token_to_kv_pool
                forward_batch.attn_backend = self.draft_runner.attn_backend
                forward_batch.spec_algorithm = SpeculativeAlgorithm.DFLASH
                forward_batch.spec_info = DFlashDraftInput(
                    positions=mem_positions,
                    target_hidden=target_hidden,
                    target_lens=torch.tensor([seq_len], device=device),
                )
                forward_batch.input_embeds = input_embeds

                self.draft_runner.attn_backend.init_forward_metadata(forward_batch)
                hidden = self.draft_runner.model.forward(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                    input_embeds=forward_batch.input_embeds,
                )

                if hasattr(self.draft_token_to_kv_pool_allocator, "free"):
                    self.draft_token_to_kv_pool_allocator.free(out_cache_loc)

                draft_logits = self._compute_draft_logits(hidden[1:, :])
                draft_sample = torch.argmax(draft_logits, dim=-1)

                draft_tokens[i, 0] = prefix_token
                draft_tokens[i, 1:] = draft_sample

        return draft_tokens.reshape(-1), positions.reshape(-1)

    def forward_batch_generation(self, batch: ScheduleBatch):
        if self.use_dflash_interface:
            self._ensure_target_aux_capture()
            self._recapture_cuda_graph_if_needed()
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
