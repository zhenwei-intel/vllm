# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU Attention backend using DeepKLOX."""

from dataclasses import dataclass
from typing import ClassVar

import torch

import vllm._custom_ops as ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.import_utils import has_deepklox
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash

if has_deepklox():
    from deepklox import flash_attn_varlen_func as deepklox_varlen_func
else:
    deepklox_varlen_func = None  # type: ignore[assignment]
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    get_kv_cache_layout,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


@dataclass
class XpuAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool

    # Split counts (decode seqs ordered before prefill seqs)
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    # For cascade attention
    use_cascade: bool
    common_prefix_len: int

    # FP8 Q descale set by an upstream fused kernel before attention forward.
    # None when no FP8 Q pre-quantization is used.
    xpu_dynamic_q_scale: torch.Tensor | None = None


class XpuAttentionMetadataBuilder(AttentionMetadataBuilder[XpuAttentionMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> XpuAttentionMetadata:
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        assert self.reorder_batch_threshold is not None
        (
            num_decodes,
            num_prefills,
            num_decode_tokens,
            num_prefill_tokens,
        ) = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.reorder_batch_threshold,
        )

        use_cascade = common_prefix_len > 0

        return XpuAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            causal=causal,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            xpu_dynamic_q_scale=None,
        )

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


class XpuAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64, 128]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return max(default_block_size, 64)

    @staticmethod
    def get_name() -> str:
        return "XPU_ATTN"

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return False

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
        )

    @staticmethod
    def get_impl_cls() -> type["XpuAttentionImpl"]:
        return XpuAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["XpuAttentionMetadataBuilder"]:
        return XpuAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            return (2, 0, 1, 3, 4, 5)
        elif cache_layout == "NHD":
            return (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            return (2, 4, 0, 1, 3, 5)
        elif cache_layout == "HND":
            return (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size % 8 == 0 and head_size <= 256

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        return kv_cache_dtype in ["auto", "float16", "bfloat16", "fp8", "fp8_e4m3"]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return False


class XpuAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.attn_type = attn_type

        self._reshape_and_cache_flash = reshape_and_cache_flash

        if deepklox_varlen_func is None:
            raise ImportError(
                "DeepKLOX is required for XPU_ATTN backend. Please install DeepKLOX."
            )
        self._flash_attn_varlen_func = deepklox_varlen_func
        logger.info_once("DeepKLOX loaded successfully.")

    def maybe_quant_query(
        self,
        query: torch.Tensor,
        layer: torch.nn.Module,
        attn_metadata: XpuAttentionMetadata,
    ) -> tuple[torch.Tensor, "torch.Tensor | None"]:
        """Optionally quantize query to FP8; return (query, raw_q_scale).

        Three cases:
        1. Q already FP8 (pre-quantized by an upstream fused kernel):
           ``attn_metadata.xpu_dynamic_q_scale`` holds the corresponding
           descale — scalar for per-tensor or ``[T, num_q_heads]`` for
           per-token-per-head; both shapes are accepted by DeepKLOX.
           The caller is responsible for setting this field before invoking
           attention forward.
        2. Q is BF16/FP16 with a calibrated per-tensor scale
           (``layer._q_scale_float != 1.0``): static per-tensor FP8 quant;
           ``raw_q_scale`` is the scalar ``layer._q_scale``.
        3. Q is BF16/FP16 with default scale (``1.0``): no quantization;
           DeepKLOX handles BF16-Q + FP8-KV natively; ``raw_q_scale`` is
           ``None``.
        """
        fp8_dtype = current_platform.fp8_dtype()
        if query.dtype == fp8_dtype:
            # Case 1: Q was pre-quantized by an upstream fused kernel (e.g. a
            # fused_rope_norm_store_kv_fp8 op).  Read back descale
            # the caller stored in attn_metadata.xpu_dynamic_q_scale.
            return query, attn_metadata.xpu_dynamic_q_scale

        q_scale_float: float = getattr(layer, "_q_scale_float", 1.0)
        if q_scale_float != 1.0:
            # Case 2: Q is BF16/FP16; quantize per-tensor using the calibrated
            # scale from the model checkpoint.
            num_tokens, num_heads, head_size = query.shape
            q_fp8, _ = ops.scaled_fp8_quant(
                query.view(num_tokens, num_heads * head_size),
                scale=layer._q_scale,
            )
            return q_fp8.view(num_tokens, num_heads, head_size), layer._q_scale

        # Case 3: scale is 1.0 (not calibrated) – skip FP8 Q quantization.
        return query, None

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: XpuAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass using flash_attn_varlen_func for all sequences.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for XpuAttentionImpl"
            )

        if attn_metadata is None:
            return output.fill_(0)

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention (no KV cache)
        if self.attn_type in (
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER,
        ):
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
            )

        key_cache, value_cache = kv_cache.unbind(0)

        query = query[:num_actual_tokens]
        output = output[:num_actual_tokens]

        # Handle FP8 KV cache: view caches as FP8 and (maybe) quantize Q.
        raw_q_scale = None
        k_descale = None
        v_descale = None
        if is_quantized_kv_cache(self.kv_cache_dtype):
            key_cache = key_cache.view(current_platform.fp8_dtype())
            value_cache = value_cache.view(current_platform.fp8_dtype())
            k_descale = layer._k_scale
            v_descale = layer._v_scale
            query, raw_q_scale = self.maybe_quant_query(query, layer, attn_metadata)

        num_seqs = attn_metadata.num_decodes + attn_metadata.num_prefills
        seq_lens_all = attn_metadata.seq_lens[:num_seqs]
        seqused_k = (
            seq_lens_all.to(torch.int32)
            if seq_lens_all.dtype != torch.int32
            else seq_lens_all
        )

        # Build kernel-specific parameters:
        #   decode (max_seqlen_q==1): seqused_k (per-seq lengths), cu_seqlens_k=None
        #   prefill/mixed (max_seqlen_q>1): cu_seqlens_k (cumulative), seqused_k=None
        if attn_metadata.max_query_len == 1:
            q = query if query.is_contiguous() else query.contiguous()
            cu_seqlens_k = None
            causal = False
            # Decode: kernel accepts q_descale as scalar (per-tensor) or
            # [total_tokens, num_heads_q] (per-token-per-head).
            q_descale = raw_q_scale
        else:
            q = query
            cu_seqlens_k = torch.zeros(
                num_seqs + 1, dtype=torch.int32, device=seqused_k.device
            )
            cu_seqlens_k[1:] = seqused_k.cumsum(0)
            seqused_k = None
            causal = attn_metadata.causal
            # Prefill/mixed: scalar q_descale is passed directly (per-tensor).
            # Per-token-per-head [T, H] must be reshaped to
            # [batch, num_heads_q, padded_seqlen] where
            # padded_seqlen = ceil(max_seqlen_q / 128) * 128.
            if raw_q_scale is None or raw_q_scale.numel() == 1:
                q_descale = raw_q_scale
            else:
                padded = (attn_metadata.max_query_len + 127) // 128 * 128
                total_tokens = raw_q_scale.shape[0]
                q_scale_3d = torch.zeros(
                    num_seqs,
                    self.num_heads,
                    padded,
                    dtype=torch.float32,
                    device=raw_q_scale.device,
                )
                # Avoid host sync from int() indexing GPU tensors in a loop.
                # Use searchsorted to compute batch_idx and local_pos on GPU.
                cu_q = attn_metadata.query_start_loc[: num_seqs + 1]
                token_range = torch.arange(
                    total_tokens, dtype=cu_q.dtype, device=cu_q.device
                )
                # cu_q[1:] = [end_seq0, end_seq1, ...], right=True maps
                # token t → batch i where cu_q[i] <= t < cu_q[i+1].
                batch_idx = torch.searchsorted(
                    cu_q[1:].contiguous(), token_range, right=True
                )
                local_pos = token_range - cu_q[batch_idx]
                # Scatter: q_scale_3d[b, :, pos] = raw_q_scale[t, :]
                # permute to [num_seqs, padded, num_heads] for easy indexing.
                q_scale_3d.permute(0, 2, 1)[batch_idx, local_pos] = raw_q_scale
                q_descale = q_scale_3d

        self._flash_attn_varlen_func(
            q=q,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=attn_metadata.query_start_loc,
            cu_seqlens_k=cu_seqlens_k,
            seqused_k=seqused_k,
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=causal,
            out=output,
            block_table=attn_metadata.block_table,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            window_size=self.sliding_window,
            softcap=self.logits_soft_cap,
        )
        return output

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER,
        ):
            return

        key_cache, value_cache = kv_cache.unbind(0)
        self._reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: XpuAttentionMetadata,
    ) -> torch.Tensor:
        """Encoder attention without KV cache (bidirectional)."""
        self._flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=attn_metadata.query_start_loc,
            cu_seqlens_k=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_query_len,
            softmax_scale=self.scale,
            causal=False,
            out=output,
            window_size=self.sliding_window,
            softcap=self.logits_soft_cap,
        )
        return output
