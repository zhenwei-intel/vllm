# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Optional, cast, Union

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.kv_cache_interface import (AttentionSpec, FullAttentionSpec,
                                        KVCacheConfig, KVCacheSpec,
                                        SlidingWindowSpec)
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.v1.utils import bind_kv_cache

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


class XPUModelRunner(GPUModelRunner):
    """A model runner for XPU devices."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(vllm_config, device)
        # FIXME: To be verified.
        self.cascade_attn_enabled = False

    def _init_device_properties(self) -> None:
        self.num_sms = None

    def _sync_device(self) -> None:
        torch.xpu.synchronize()

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        self.kv_cache_config = kv_cache_config
        self.may_reinitialize_input_batch(kv_cache_config)
        self.initialize_attn_backend(kv_cache_config)
        kv_caches = self.initialize_kv_cache_tensors(kv_cache_config)

        if self.speculative_config and self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)
            # validate all draft model layers belong to the same kv cache
            # group
            self.drafter.validate_same_kv_cache_group(kv_cache_config)

        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(kv_caches)
            get_kv_transfer_group().set_host_xfer_buffer_ops(d2h_copy_blocks, h2d_copy_blocks)

def _make_src_and_dst_indices(
    src_block_ids: list[int],
    dst_block_ids: list[int],
    src_device: Union[torch.device, str],
    dst_device: Union[torch.device, str],
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    src_indices = torch.tensor(src_block_ids,
                               device=src_device,
                               dtype=torch.int64)
    dst_indices = torch.tensor(dst_block_ids,
                               device=dst_device,
                               dtype=torch.int64)
    return src_indices, dst_indices

def _insert_blocks_to_xpu(
    src_cache: torch.Tensor,
    xpu_cache: torch.Tensor,
    xpu_block_indices: torch.Tensor,
) -> None:
    # No buffer donor op for XPU, just assign
    xpu_cache[:, xpu_block_indices] = src_cache

def _swap_out_xpu_blocks(
    xpu_cache: torch.Tensor,
    cpu_cache: torch.Tensor,
    xpu_block_indices: torch.Tensor,
    cpu_block_indices: torch.Tensor,
) -> None:
    """ xpu blocks to cpu blocks"""
    _xpu_cache = xpu_cache[:, xpu_block_indices]
    cpu_cache[:, cpu_block_indices] = _xpu_cache.cpu()

def h2d_copy_blocks(
    cpu_kv_caches: dict[torch.Tensor],
    xpu_kv_caches: dict[torch.Tensor],
    cpu_block_ids: list[int],
    xpu_block_ids: list[int],
    xpu_device: str,
) -> None:
    """Copy kv blocks from host xfer buffer to device."""
    if not cpu_block_ids or not xpu_block_ids or len(cpu_block_ids) != len(xpu_block_ids):
        return
    host_indices, device_indices = _make_src_and_dst_indices(
        src_block_ids=cpu_block_ids,
        dst_block_ids=xpu_block_ids,
        src_device="cpu",
        dst_device=xpu_device)
    for layer_name in cpu_kv_caches:
        host_tensor = cpu_kv_caches[layer_name]
        device_tensor = xpu_kv_caches[layer_name]
        sliced_device_tensor = host_tensor[:, host_indices].to(xpu_device)
        _insert_blocks_to_xpu(sliced_device_tensor, device_tensor, device_indices)

def d2h_copy_blocks(
    cpu_kv_caches: dict[torch.Tensor],
    xpu_kv_caches: dict[torch.Tensor],
    cpu_block_ids: list[int],
    xpu_block_ids: list[int],
    xpu_device: str,
) -> None:
    """Copy kv blocks from device to host xfer buffer."""
    if not cpu_block_ids or not xpu_block_ids or len(cpu_block_ids) != len(xpu_block_ids):
        return
    device_indices, host_indices = _make_src_and_dst_indices(
        src_block_ids=xpu_block_ids,
        dst_block_ids=cpu_block_ids,
        src_device=xpu_device,
        dst_device="cpu")
    for layer_name in cpu_kv_caches:
        host_tensor = cpu_kv_caches[layer_name]
        device_tensor = xpu_kv_caches[layer_name]
        _swap_out_xpu_blocks(
            xpu_cache=device_tensor,
            cpu_cache=host_tensor,
            xpu_block_indices=device_indices,
            cpu_block_indices=host_indices)
