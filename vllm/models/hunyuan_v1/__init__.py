# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HunYuanV1 model — vendor-specific entry point.

Dispatches to ``xpu/`` on XPU platforms, otherwise falls back to the
upstream model in vllm.model_executor.models.
"""

from vllm import envs
from vllm.platforms import current_platform

if current_platform.is_xpu() and envs.VLLM_XPU_USE_CUSTOM_MODEL:
    from .xpu.model import HunYuanMoEV1ForCausalLM  # type: ignore[assignment]
else:
    from vllm.model_executor.models.hunyuan_v1 import (  # type: ignore[assignment]
        HunYuanMoEV1ForCausalLM,
    )

__all__ = [
    "HunYuanMoEV1ForCausalLM",
]
