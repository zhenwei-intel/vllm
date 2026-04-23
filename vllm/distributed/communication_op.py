# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.distributed

import os

from .parallel_state import get_tp_group


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    if os.environ.get("VLLM_SKIP_TP_COMM", "0") == "1":
        # 假设 world_size=4，模拟 all_gather 行为
        world_size = torch.distributed.get_world_size(get_tp_group())
        input_size = input_.size()
        output_size = (input_size[0] * world_size,) + input_size[1:]
        output_tensor = input_.repeat(world_size, *[1] * (input_.dim() - 1))
        output_tensor = output_tensor.reshape((world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim]
            + (world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )
        return output_tensor
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_reduce_scatter(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""
    return get_tp_group().reduce_scatter(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> torch.Tensor | None:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: dict[Any, torch.Tensor | Any] | None = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
