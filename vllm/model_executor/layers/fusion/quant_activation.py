# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
A QuantizedActivation is a pre-quantized activation produced by a fused kernel
and consumed directly by a linear layer, letting the layer skip its own input
quantization. A linear advertises the key its kernel can consume via
expose_input_quant_key; the kernel validates and reads the activation via
as_quantized_activation.
"""

from dataclasses import dataclass

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic64Sym,
    kFp8Dynamic128Sym,
    kFp8StaticTensorSym,
    kMxfp8Dynamic,
)


@dataclass
class QuantizedActivation:
    """A quantized activation paired with its scale and original metadata.

    The quant_key describes how data and scale are to be interpreted (dtype,
    scale granularity, value packing). Details the key does not capture, such
    as blockscale layout or activation padding, must follow the consumer
    kernel's convention.

    TODO(mgoin): Encode layout and padding requirements in the contract so
    producers can match consumer kernels without relying on convention.
    """

    data: torch.Tensor
    scale: torch.Tensor
    orig_dtype: torch.dtype
    orig_shape: torch.Size
    quant_key: QuantKey


def expose_input_quant_key(layer: torch.nn.Module, kernel) -> None:
    """Advertise the kernel's pre-quantized input key on the layer, if any.

    This is the bridge from a kernel's input_quant_key() to the
    layer.input_quant_key attribute that fusion call sites read. The attribute
    is left unset when the kernel quantizes its own input, so non-supporting
    backends never receive a QuantizedActivation.

    TODO(mgoin): Producers also need the consumer's quantization scales (e.g.
    static input scale, global scale). Expose those here as well so producers
    do not reach into kernel-specific layer attributes.
    """
    key = kernel.input_quant_key()
    if key is not None:
        layer.input_quant_key = key


def as_quantized_activation(
    x: "torch.Tensor | QuantizedActivation", expected_key: QuantKey | None
) -> "QuantizedActivation | None":
    """Validate and narrow a pre-quantized activation for a consumer kernel.

    Returns the QuantizedActivation when x is one whose key matches the
    kernel's declared expected_key, and None when x is a plain tensor (the
    caller quantizes in-kernel). Raises on a key mismatch so a wrongly routed
    activation fails loudly instead of being silently re-quantized.
    """
    if not isinstance(x, QuantizedActivation):
        return None
    assert x.quant_key == expected_key, (
        f"QuantizedActivation key {x.quant_key} != consumer kernel "
        f"input_quant_key {expected_key}"
    )
    return x


def fused_silu_and_mul_quant(
    down_proj: torch.nn.Module, gate_up: torch.Tensor
) -> "QuantizedActivation | None":
    key = getattr(down_proj, "input_quant_key", None)
    hidden_size = gate_up.shape[-1] // 2

    if key == kFp8StaticTensorSym:
        input_scale = getattr(down_proj, "input_scale", None)
        if input_scale is None:
            return None
        out = torch.empty(
            (*gate_up.shape[:-1], hidden_size),
            dtype=key.dtype,
            device=gate_up.device,
        )
        torch.ops._C.silu_and_mul_quant(out, gate_up, input_scale)
        scale = input_scale
    elif key in (kFp8Dynamic128Sym, kFp8Dynamic64Sym, kMxfp8Dynamic):
        from vllm import _custom_ops as ops

        group_size = key.scale.group_shape.col
        out, scale = ops.silu_and_mul_per_block_quant(
            gate_up, group_size, key.dtype, scale_ue8m0=key == kMxfp8Dynamic
        )
    else:
        return None

    return QuantizedActivation(
        data=out,
        scale=scale,
        orig_dtype=gate_up.dtype,
        orig_shape=out.shape,
        quant_key=key,
    )


def fused_rms_norm_quant(
    rms_norm: torch.nn.Module,
    next_linear: "torch.nn.Module | None",
    hidden: torch.Tensor,
    residual: "torch.Tensor | None",
) -> "tuple[QuantizedActivation | torch.Tensor, torch.Tensor]":
    key = (
        getattr(next_linear, "input_quant_key", None)
        if next_linear is not None
        else None
    )
    can_fuse = (
        key is not None
        and getattr(rms_norm, "variance_size_override", None) is None
        and getattr(rms_norm, "has_weight", True)
    )
    if can_fuse:
        eps = rms_norm.variance_epsilon
        weight = rms_norm.weight.data
        out = None
        if key == kFp8StaticTensorSym:
            input_scale = getattr(next_linear, "input_scale", None)
            if input_scale is not None:
                out = torch.empty(hidden.shape, dtype=key.dtype, device=hidden.device)
                if residual is None:
                    torch.ops._C.rms_norm_static_fp8_quant(
                        out, hidden, weight, input_scale, eps
                    )
                    residual_out = hidden
                else:
                    torch.ops._C.fused_add_rms_norm_static_fp8_quant(
                        out, hidden, residual, weight, input_scale, eps
                    )
                    residual_out = residual
                scale = input_scale
        elif key in (kFp8Dynamic128Sym, kFp8Dynamic64Sym, kMxfp8Dynamic):
            from vllm import _custom_ops as ops

            group_size = key.scale.group_shape.col
            out, scale = ops.rms_norm_per_block_quant(
                hidden,
                weight,
                eps,
                key.dtype,
                [1, group_size],
                residual=residual,
                scale_ue8m0=key == kMxfp8Dynamic,
            )
            residual_out = hidden if residual is None else residual
        if out is not None:
            qa = QuantizedActivation(
                data=out,
                scale=scale,
                orig_dtype=hidden.dtype,
                orig_shape=out.shape,
                quant_key=key,
            )
            return qa, residual_out

    if residual is None:
        return rms_norm(hidden), hidden
    return rms_norm(hidden, residual)
