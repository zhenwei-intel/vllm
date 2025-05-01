# SPDX-License-Identifier: Apache-2.0
"""
KV cache helper for store.
"""
import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.logger import init_logger
from typing import Any, Dict, List, Optional, Tuple, Type

logger = init_logger(__name__)


class model_aware_kv_ops_helper:

    def __init__(self, config: VllmConfig):
        self.is_deepseek_mla = config.model_config.is_deepseek_mla
        self.use_mla_opt = not envs.VLLM_MLA_DISABLE
        self.tp_size = config.parallel_config.tensor_parallel_size

    def get_model_args(self, model_executable: torch.nn.Module):

        model_config = model_executable.model.config
        self.model_executable = model_executable
        num_heads = int(model_config.num_key_value_heads / self.tp_size)
        hidden_size = model_config.hidden_size
        num_attention_heads = model_config.num_attention_heads

        # Deepseek's MLA (Multi-head Latent Attention) uses two different
        # kv_cache shapes based on whether VLLM_MLA_DISABLE is set to 0.
        # When VLLM_MLA_DISABLE=0 (default), forward absorb is applied,
        # resulting in a kv_cache shape of [num_blks, blk_size, 1,
        # kv_lora_rank + qk_rope_head_dim].
        # When VLLM_MLA_DISABLE=1, standard FA is used instead, leading
        # to a kv_cache shape of [2, num_blks, blk_size,
        # num_key_value_heads / tp, qk_nope_head_dim + qk_rope_head_dim].
        # For more details, see vllm/attention/backends/mla/common.py.
        if self.is_deepseek_mla and self.use_mla_opt:
            head_size = model_config.kv_lora_rank + \
                model_config.qk_rope_head_dim
            num_heads = 1
        elif self.is_deepseek_mla and not self.use_mla_opt:
            head_size = model_config.qk_nope_head_dim + \
                model_config.qk_rope_head_dim
        else:
            head_size = getattr(model_config, "head_dim",
                                int(hidden_size // num_attention_heads))

        return num_heads, head_size

    def get_kv_from_cache(self, kv_cache, num_heads, head_size):
        if self.is_deepseek_mla and self.use_mla_opt:
            key_cache = kv_cache.reshape(-1, num_heads, head_size)
            value_cache = kv_cache.reshape(-1, num_heads, head_size)
        else:
            key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
            value_cache = kv_cache[1].reshape(-1, num_heads, head_size)
        return key_cache, value_cache

    def put_kv_to_cache(self, model_executable: torch.nn.Module, keys, values,
                        layer, kv_cache, slot_mapping, start_pos, end_pos):

        model_config = model_executable.model.config

        if self.is_deepseek_mla and self.use_mla_opt:
            layer.self_attn.attn = layer.self_attn.mla_attn
            k_c_normed_k_pe = keys.squeeze(1)
            k_c_normed = k_c_normed_k_pe[:, :model_config.kv_lora_rank]
            k_pe = k_c_normed_k_pe[:, model_config.kv_lora_rank:]
            ops.concat_and_cache_mla(
                k_c_normed.to(kv_cache.device),
                k_pe.to(kv_cache.device),
                kv_cache,
                slot_mapping[start_pos:end_pos],
                layer.self_attn.attn.kv_cache_dtype,
                layer.self_attn.attn._k_scale,
            )
        else:
            if kv_cache.devicce == torch.device("xpu"):
                num_heads, head_size = self.kv_helper.get_model_args(model_executable)
                # # 最重要的地方：
                # key_cache, value_cache = kv_cache[0], kv_cache[1]  # keycache torch.Size([3922, 65536])
                # # remote_k torch.Size([5, 32, 128])
                # key_cache = kv_cache.reshape(-1, num_heads, head_size)
                # key = keys.unsqueeze(0).view(-1, self.block_size, num_heads, head_size)
                # value = values.unsqueeze(0).view(-1, self.block_size, num_heads, head_size)
                key_cache, value_cache = self.split_kv_cache(
                    kv_cache, num_heads, head_size)
                from vllm._ipex_ops import ipex_ops
                ipex_ops.reshape_and_cache(
                    keys.to(key_cache.device),
                    values.to(value_cache.device),
                    key_cache,
                    value_cache,
                    slot_mapping[start_pos:end_pos],
                    layer.self_attn.attn.kv_cache_dtype,
                    layer.self_attn.attn._k_scale,
                    layer.self_attn.attn._v_scale,
                    )
                return
            key_cache, value_cache = kv_cache[0], kv_cache[1]
            ops.reshape_and_cache_flash(
                keys.to(key_cache.device),
                values.to(value_cache.device),
                key_cache,
                value_cache,
                slot_mapping[start_pos:end_pos],
                layer.self_attn.attn.kv_cache_dtype,
                layer.self_attn.attn._k_scale,
                layer.self_attn.attn._v_scale,
            )

    def split_kv_cache(
        self,
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = 1
        num_blocks = kv_cache.shape[1]

        key_cache = kv_cache[0]
        key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x,
                                   -1, x)
        value_cache = kv_cache[1]
        value_cache = value_cache.view(num_blocks, num_kv_heads, head_size, -1)
        return key_cache, value_cache