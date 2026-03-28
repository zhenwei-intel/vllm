# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for GGUF dequantization on Intel XPU.

These tests validate that the XPU dequantization path (Python gguf-library
fallback + optional SYCL kernel) produces numerically accurate results for
the Q8_0 and Q4_K quantization types.

The tests are designed to run on any device that supports PyTorch (CPU,
CUDA, XPU).  On non-XPU platforms the XPU code-path is exercised by
directly calling ``_ggml_dequantize_xpu`` against a CPU tensor – this
ensures correctness is verified in CI even without real XPU hardware.

On a machine with Intel XPU hardware, run with:
    pytest tests/kernels/quantization/test_gguf_xpu.py -v
"""

from pathlib import Path

import pytest
import torch
from gguf import GGMLQuantizationType, GGUFReader, ReaderTensor
from gguf import dequantize as gguf_dequantize
from huggingface_hub import snapshot_download

from vllm._custom_ops import _ggml_dequantize_xpu

GGUF_SAMPLE = snapshot_download("Isotr0py/test-gguf-sample")

QUANT_TYPES_XPU = [
    GGMLQuantizationType.Q8_0,
    GGMLQuantizationType.Q4_K,
    # Standard quants that also exercise the Python-library fallback
    GGMLQuantizationType.Q4_0,
    GGMLQuantizationType.Q5_0,
    GGMLQuantizationType.Q5_K,
    GGMLQuantizationType.Q6_K,
]

HIDDEN_SIZES = [256, 1024]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def _get_sample_tensors(
    hidden_size: int, quant_type: GGMLQuantizationType
) -> list[ReaderTensor]:
    filename = f"Quant_{quant_type.name}_{hidden_size}.gguf"
    return GGUFReader(Path(GGUF_SAMPLE) / filename).tensors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _xpu_device() -> torch.device:
    """Return the first available XPU device, or CPU as fallback."""
    if (
        hasattr(torch, "xpu")
        and torch.xpu.is_available()
        and torch.xpu.device_count() > 0
    ):
        return torch.device("xpu:0")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Test: dequantize accuracy on XPU / CPU
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_type", QUANT_TYPES_XPU)
@torch.inference_mode()
def test_ggml_dequantize_xpu(
    hidden_size: int,
    dtype: torch.dtype,
    quant_type: GGMLQuantizationType,
) -> None:
    """Verify that ``_ggml_dequantize_xpu`` matches the reference Python
    implementation from the ``gguf`` library for every supported quant type.
    """
    device = _xpu_device()
    tensors = _get_sample_tensors(hidden_size, quant_type)
    for tensor in tensors:
        shape_str = tensor.name.split("_")[-1]
        m, n = (int(s) for s in shape_str.split("x"))

        # Reference: Python gguf library (float32)
        ref = (
            torch.from_numpy(gguf_dequantize(tensor.data, quant_type))
            .view(m, n)
            .to(dtype)
        )

        # XPU path (works on any device via Python fallback)
        qweight = torch.from_numpy(tensor.data).to(device)
        out = _ggml_dequantize_xpu(qweight, int(quant_type), m, n, dtype)

        # Tolerances are generous because we are comparing fp16/bf16 to fp32
        torch.testing.assert_close(
            out.cpu(),
            ref,
            atol=1e-2,
            rtol=4e-2,
            msg=f"Mismatch for {quant_type.name} shape=({m},{n}) dtype={dtype}",
        )


# ---------------------------------------------------------------------------
# Test: linear layer matmul with quantized weights on XPU
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_tokens", [1, 7, 64])
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "quant_type",
    [
        GGMLQuantizationType.Q8_0,
        GGMLQuantizationType.Q4_K,
    ],
)
@torch.inference_mode()
def test_gguf_linear_xpu(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    quant_type: GGMLQuantizationType,
) -> None:
    """Verify that the full dequantize + matmul path used by
    ``GGUFLinearMethod.apply`` on XPU matches the reference output obtained
    by dequantizing on CPU and performing the matmul there.
    """
    device = _xpu_device()
    tensors = _get_sample_tensors(hidden_size, quant_type)
    for tensor in tensors:
        shape_str = tensor.name.split("_")[-1]
        out_size, in_size = (int(s) for s in shape_str.split("x"))

        # Quantized weight on device
        qweight = torch.from_numpy(tensor.data).to(device)

        # Random input
        torch.manual_seed(42)
        x = torch.rand(num_tokens, in_size, dtype=dtype, device=device)

        # XPU path: dequantize on device + matmul
        weight_xpu = _ggml_dequantize_xpu(
            qweight, int(quant_type), out_size, in_size, dtype
        )
        out_xpu = x @ weight_xpu.T

        # Reference: dequantize on CPU + matmul
        weight_ref = (
            torch.from_numpy(gguf_dequantize(tensor.data, quant_type))
            .view(out_size, in_size)
            .to(dtype)
        )
        out_ref = x.cpu() @ weight_ref.T

        torch.testing.assert_close(
            out_xpu.cpu(),
            out_ref,
            atol=1.0,
            rtol=1e-1,
            msg=(
                f"Linear mismatch for {quant_type.name} "
                f"tokens={num_tokens} hidden={in_size} dtype={dtype}"
            ),
        )
