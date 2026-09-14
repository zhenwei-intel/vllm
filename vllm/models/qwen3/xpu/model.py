# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3 model compatible with HuggingFace weights."""

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from transformers import Qwen3Config

import vllm._custom_ops as ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.distributed.communication_op import (
    tensor_model_parallel_reduce_scatter,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.encoder_only_attention import (
    Attention,
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.fusion.quant_activation import (
    fused_rms_norm_quant,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import (
    LocalArgmaxMixin,
    SupportsEagle,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.qwen2 import Qwen2MLP as Qwen3MLP
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backend import AttentionType

logger = init_logger(__name__)


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: dict[str, Any] | None = None,
        per_layer_sliding_window: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        attn_cls = (
            EncoderOnlyAttention
            if attn_type == AttentionType.ENCODER_ONLY
            else Attention
        )
        self.attn = attn_cls(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=per_layer_sliding_window,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
            if dual_chunk_attention_config
            else {},
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        self.use_fused_qk_norm_rope = (
            current_platform.is_xpu()
            and self.head_dim in (64, 128, 256, 512)
            and hasattr(self.rotary_emb, "cos_sin_cache")
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        if self.use_fused_qk_norm_rope:
            # Normalizes Q/K per head and applies RoPE in place (V untouched).
            ops.fused_qk_norm_rope(
                qkv,
                self.num_heads,
                self.num_kv_heads,
                self.num_kv_heads,
                self.head_dim,
                self.q_norm.variance_epsilon,
                self.q_norm.weight,
                self.k_norm.weight,
                self.rotary_emb.cos_sin_cache,
                self.rotary_emb.is_neox_style,
                positions,
            )
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q_by_head = q.view(
                *q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim
            )
            q_by_head = self.q_norm(q_by_head)
            q = q_by_head.view(q.shape)
            k_by_head = k.view(
                *k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim
            )
            k_by_head = self.k_norm(k_by_head)
            k = k_by_head.view(k.shape)
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        per_layer_sliding_window: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )

        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
            per_layer_sliding_window=per_layer_sliding_window,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config
        self.eager_sp = parallel_config.use_eager_sequence_parallel
        self.fuse_gemm_comms = (
            parallel_config.enable_eager_sp_fuse_gemm_comms and self.eager_sp
        )
        self._sp_threshold = parallel_config.eager_sp_threshold
        if self.eager_sp:
            self.self_attn.o_proj.reduce_results = False
            self.mlp.down_proj.reduce_results = False

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.eager_sp:
            return self._forward_eager_sp(positions, hidden_states, residual)
        return self._forward_standard(positions, hidden_states, residual)

    def _forward_standard(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = fused_rms_norm_quant(
            self.input_layernorm,
            getattr(self.self_attn, "qkv_proj", None),
            hidden_states,
            residual,
        )
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = fused_rms_norm_quant(
            self.post_attention_layernorm,
            getattr(self.mlp, "gate_up_proj", None),
            hidden_states,
            residual,
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def _forward_eager_sp(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states = sequence_parallel_chunk(hidden_states)
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        use_fused = (
            self.fuse_gemm_comms
            and hidden_states.shape[0] >= self._sp_threshold
        )

        if use_fused:
            hidden_states = self._attn_fused(positions, hidden_states)
        else:
            hidden_states = tensor_model_parallel_all_gather(
                hidden_states, dim=0
            )
            hidden_states = self.self_attn(
                positions=positions, hidden_states=hidden_states
            )
            hidden_states = tensor_model_parallel_reduce_scatter(
                hidden_states, dim=0
            )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )

        if use_fused:
            hidden_states = self._mlp_fused(hidden_states)
        else:
            hidden_states = tensor_model_parallel_all_gather(
                hidden_states, dim=0
            )
            hidden_states = self.mlp(hidden_states)
            hidden_states = tensor_model_parallel_reduce_scatter(
                hidden_states, dim=0
            )

        return hidden_states, residual

    def _attn_fused(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        from vllm.distributed import get_tp_group

        group_name = get_tp_group().device_group.group_name
        attn = self.self_attn

        qkv = attn.qkv_proj.fused_ag_forward(hidden_states, group_name)

        q, k, v = qkv.split(
            [attn.q_size, attn.kv_size, attn.kv_size], dim=-1
        )
        q = attn.q_norm(
            q.view(
                *q.shape[:-1], q.shape[-1] // attn.head_dim, attn.head_dim
            )
        ).view(q.shape)
        k = attn.k_norm(
            k.view(
                *k.shape[:-1], k.shape[-1] // attn.head_dim, attn.head_dim
            )
        ).view(k.shape)
        q, k = attn.rotary_emb(positions, q, k)
        attn_output = attn.attn(q, k, v)

        hidden_states = attn.o_proj.fused_rs_forward(attn_output, group_name)
        return hidden_states

    def _mlp_fused(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from vllm.distributed import get_tp_group

        group_name = get_tp_group().device_group.group_name
        mlp = self.mlp

        gate_up = mlp.gate_up_proj.fused_ag_forward(
            hidden_states, group_name
        )
        out = mlp.act_fn(gate_up)

        hidden_states = mlp.down_proj.fused_rs_forward(out, group_name)
        return hidden_states


ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen3DecoderLayer,
}


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3Model(Qwen2Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config, prefix=prefix, decoder_layer_type=Qwen3DecoderLayer
        )
        parallel_config = vllm_config.parallel_config
        self.eager_sp = parallel_config.use_eager_sequence_parallel

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        result = super().forward(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        if not self.eager_sp:
            return result
        if isinstance(result, IntermediateTensors):
            return result
        if isinstance(result, tuple):
            hidden_states, aux = result
            hidden_states = tensor_model_parallel_all_gather(
                hidden_states, dim=0
            )
            return hidden_states, aux
        return tensor_model_parallel_all_gather(result, dim=0)


class Qwen3ForCausalLM(
    LocalArgmaxMixin, nn.Module, SupportsLoRA, SupportsPP, SupportsEagle, SupportsEagle3
):
    hf_to_vllm_mapper = Qwen3Model.hf_to_vllm_mapper
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config

        self.vllm_config = vllm_config
        self.quant_config = quant_config
        self.model = Qwen3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
