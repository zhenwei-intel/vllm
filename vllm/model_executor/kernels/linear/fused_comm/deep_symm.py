# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .base import FusedCommLinearKernel

if TYPE_CHECKING:
    from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
        FP8ScaledMMLinearKernel,
    )


class DeepSymmBF16FusedCommKernel(FusedCommLinearKernel):
    """Unquantized fused comm+GEMM using deep_symm async_tp on XPU."""

    @classmethod
    def is_supported(cls) -> tuple[bool, str | None]:
        from vllm.platforms import current_platform

        if not current_platform.is_xpu():
            return False, "deep_symm requires XPU"
        try:
            from deep_symm import async_tp  # noqa: F401

            return True, None
        except ImportError:
            return False, "deep_symm not installed"

    def fused_ag_gemm(
        self,
        layer: torch.nn.Module,
        x_shard: torch.Tensor,
        bias: torch.Tensor | None,
        group_name: str,
    ) -> torch.Tensor:
        from deep_symm import async_tp

        W = layer.weight.t()
        output = async_tp.fused_all_gather_matmul(
            x_shard, W, gather_dim=0, group_name=group_name
        )
        if bias is not None:
            output = output + bias
        return output

    def fused_gemm_rs(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
        group_name: str,
    ) -> torch.Tensor:
        from deep_symm import async_tp

        W = layer.weight.t()
        output = async_tp.fused_matmul_reduce_scatter(x, W, "sum", 0, group_name)
        if bias is not None:
            output = output + bias
        return output


class DeepSymmFP8FusedCommKernel(FusedCommLinearKernel):
    """FP8 W8A8 fused comm+GEMM using deep_symm async_tp on XPU."""

    def __init__(self, fp8_kernel: FP8ScaledMMLinearKernel) -> None:
        self._fp8_kernel = fp8_kernel

    @classmethod
    def is_supported(cls) -> tuple[bool, str | None]:
        from vllm.platforms import current_platform

        if not current_platform.is_xpu():
            return False, "deep_symm requires XPU"
        try:
            from deep_symm import async_tp  # noqa: F401

            return True, None
        except ImportError:
            return False, "deep_symm not installed"

    def fused_ag_gemm(
        self,
        layer: torch.nn.Module,
        x_shard: torch.Tensor,
        bias: torch.Tensor | None,
        group_name: str,
    ) -> torch.Tensor:
        from deep_symm import async_tp

        w = layer.weight
        w_scale = layer.weight_scale
        x_scale = getattr(layer, "input_scale", None)
        x_scale_ub = getattr(layer, "input_scale_upper_bound", None)

        x_2d = x_shard.view(-1, x_shard.shape[-1])
        x_q, a_scale = self._fp8_kernel.quant_fp8(x_2d, x_scale, x_scale_ub)

        return async_tp.fused_all_gather_scaled_matmul(
            x_q,
            w,
            a_scale,
            w_scale,
            gather_dim=0,
            group_name=group_name,
            bias=bias,
            result_scale=None,
            out_dtype=torch.bfloat16,
            use_fast_accum=False,
        )

    def fused_gemm_rs(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
        group_name: str,
    ) -> torch.Tensor:
        from deep_symm import async_tp

        w = layer.weight
        w_scale = layer.weight_scale
        x_scale = getattr(layer, "input_scale", None)
        x_scale_ub = getattr(layer, "input_scale_upper_bound", None)

        x_2d = x.view(-1, x.shape[-1])
        x_q, a_scale = self._fp8_kernel.quant_fp8(x_2d, x_scale, x_scale_ub)

        output_shape = [x_2d.shape[0], w.shape[1]]
        return async_tp.fused_scaled_matmul_reduce_scatter(
            x_q,
            w,
            a_scale,
            w_scale,
            "sum",
            0,
            0,
            group_name,
            output_shape,
            bias,
            None,
            torch.bfloat16,
            False,
        )
