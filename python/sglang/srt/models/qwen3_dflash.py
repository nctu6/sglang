import logging
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.qwen2 import Qwen2MLP as Qwen3MLP
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix

Qwen3Config = None

logger = logging.getLogger(__name__)


class Qwen3DFlashAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        head_dim: Optional[int] = None,
        max_position_embeddings: int = 32768,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps: float = None,
        attention_bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.tp_rank = get_tensor_model_parallel_rank()

        norm_kwargs = (
            dict(
                weight_dtype=torch.float32,
                cast_x_before_out_mul=True,
            )
            if get_global_server_args().rl_on_policy_target is not None
            else {}
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            attn_type=AttentionType.ENCODER_ONLY,
            prefix=add_prefix("attn", prefix),
        )

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_by_head = q.reshape(-1, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        return q_by_head.view(q.shape), k_by_head.view(k.shape)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        forward_batch,
    ) -> torch.Tensor:
        if get_global_server_args().rl_on_policy_target is not None:
            hidden_states = hidden_states.bfloat16()
            target_hidden = target_hidden.bfloat16()

        memory_states = torch.cat((target_hidden, hidden_states), dim=0)
        qkv, _ = self.qkv_proj(memory_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v, forward_batch, save_kv_cache=True)
        output, _ = self.o_proj(attn_output)
        return output[target_hidden.shape[0] :]


class Qwen3DFlashDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        head_dim = getattr(config, "head_dim", None)
        self.self_attn = Qwen3DFlashAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        forward_batch,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            forward_batch=forward_batch,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class DFlashDraftModel(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DFlashDecoderLayer(
                    config,
                    layer_id=i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.fc = nn.Linear(
            config.hidden_size * len(self._build_target_layer_ids()),
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = getattr(config, "block_size", None)

        if config.tie_word_embeddings:
            self.lm_head = self.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

    def _build_target_layer_ids(self):
        num_target_layers = getattr(self.config, "num_target_layers", None)
        num_draft_layers = self.config.num_hidden_layers
        if num_target_layers is None:
            return [num_draft_layers // 2]
        if num_draft_layers == 1:
            return [num_target_layers // 2]
        start = 1
        end = num_target_layers - 3
        span = end - start
        return [
            int(round(start + (i * span) / (num_draft_layers - 1)))
            for i in range(num_draft_layers)
        ]

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_embed(self, embed):
        del self.embed_tokens.weight
        self.embed_tokens.weight = embed
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def set_embed_and_head(self, embed, head):
        self.set_embed(embed)
        if hasattr(self, "lm_head"):
            del self.lm_head.weight
            self.lm_head.weight = head
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if name.startswith("model."):
                name = name[len("model.") :]
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self, "layers")
                and (layer_id < 0 or layer_id >= len(self.layers))
            ):
                continue
            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning("Parameter %s not found in params_dict", name)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        input_embeds: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        spec_info = forward_batch.spec_info
        if spec_info is None or spec_info.target_hidden is None:
            raise ValueError("DFlashDraftModel requires target_hidden in spec_info.")

        hidden_states = (
            input_embeds
            if input_embeds is not None
            else self.embed_tokens(input_ids)
        )
        if forward_batch.batch_size != 1:
            raise ValueError("DFlashDraftModel currently expects batch_size=1.")
        target_len = int(spec_info.target_lens[0])
        target_hidden = spec_info.target_hidden[:target_len]
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        noise_hidden = hidden_states[target_len:]

        for layer in self.layers:
            noise_hidden = layer(
                positions=positions,
                hidden_states=noise_hidden,
                target_hidden=target_hidden,
                forward_batch=forward_batch,
            )

        noise_hidden = self.norm(noise_hidden)
        return noise_hidden


EntryClass = [DFlashDraftModel]
