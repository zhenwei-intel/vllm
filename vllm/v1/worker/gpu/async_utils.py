# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import os
import time

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.outputs import AsyncModelRunnerOutput, LogprobsTensors, ModelRunnerOutput
from vllm.v1.worker.gpu.sample.output import SamplerOutput

logger = init_logger(__name__)

_DEBUG_ASYNC_OUTPUT = os.getenv("VLLM_DEBUG_ASYNC_OUTPUT", "0") == "1"


def _log_async_output(
    event: str, req_ids: list[str], **fields: float | int | str | bool
) -> None:
    if not _DEBUG_ASYNC_OUTPUT:
        return
    details = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.warning(
        "ASYNC_OUTPUT_DBG event=%s ts=%.6f req_ids=%s %s",
        event,
        time.perf_counter(),
        req_ids[:4],
        details,
    )


def _tensor_copy_fields(x: torch.Tensor) -> dict[str, str | int]:
    return {
        "shape": str(tuple(x.shape)),
        "numel": x.numel(),
        "bytes": x.numel() * x.element_size(),
        "dtype": str(x.dtype),
    }


def _log_copy_enqueue_start(
    req_ids: list[str], label: str, x: torch.Tensor | None = None, **fields: object
) -> float:
    if x is not None:
        fields = {**_tensor_copy_fields(x), **fields}
    _log_async_output(
        f"copy_enqueue_start:{label}",
        req_ids,
        **{k: str(v) if not isinstance(v, (int, float, bool)) else v for k, v in fields.items()},
    )
    return time.perf_counter()


def _log_copy_enqueue_done(req_ids: list[str], label: str, start: float,
                           **fields: object) -> None:
    serialized_fields = {
        k: str(v) if not isinstance(v, (int, float, bool)) else v
        for k, v in fields.items()
    }
    _log_async_output(
        f"copy_enqueue_done:{label}",
        req_ids,
        enqueue_ms=round((time.perf_counter() - start) * 1000, 3),
        **serialized_fields,
    )


class AsyncOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampler_output: SamplerOutput,
        num_sampled_tokens: torch.Tensor,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
        copy_event: torch.cuda.Event,
        ttft_request_arrival_times: dict[str, float] | None = None,
    ):
        # NOTE(woosuk): We must retain references to the GPU tensors,
        # as the copy operations are performed on a different CUDA stream than
        # the one where the tensors were created.
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        self.copy_event = copy_event
        self.ttft_request_arrival_times = ttft_request_arrival_times or {}

        _log_async_output("init_enter", self.model_runner_output.req_ids)

        with stream(copy_stream, main_stream):
            copy_stream.wait_stream(main_stream)

            copy_start = _log_copy_enqueue_start(
                self.model_runner_output.req_ids,
                "sampled_token_ids",
                sampler_output.sampled_token_ids,
            )
            self.sampled_token_ids = async_copy_to_np(sampler_output.sampled_token_ids)
            _log_copy_enqueue_done(
                self.model_runner_output.req_ids,
                "sampled_token_ids",
                copy_start,
            )

            self.logprobs_tensors: LogprobsTensors | None = None
            if sampler_output.logprobs_tensors is not None:
                copy_start = _log_copy_enqueue_start(
                    self.model_runner_output.req_ids,
                    "logprobs_tensors",
                )
                self.logprobs_tensors = (
                    sampler_output.logprobs_tensors.to_cpu_nonblocking()
                )
                _log_copy_enqueue_done(
                    self.model_runner_output.req_ids,
                    "logprobs_tensors",
                    copy_start,
                )

            self.num_nans: np.ndarray | None = None
            if sampler_output.num_nans is not None:
                copy_start = _log_copy_enqueue_start(
                    self.model_runner_output.req_ids,
                    "num_nans",
                    sampler_output.num_nans,
                )
                self.num_nans = async_copy_to_np(sampler_output.num_nans)
                _log_copy_enqueue_done(
                    self.model_runner_output.req_ids,
                    "num_nans",
                    copy_start,
                )

            copy_start = _log_copy_enqueue_start(
                self.model_runner_output.req_ids,
                "num_sampled_tokens",
                num_sampled_tokens,
            )
            self.num_sampled_tokens_np = async_copy_to_np(num_sampled_tokens)
            _log_copy_enqueue_done(
                self.model_runner_output.req_ids,
                "num_sampled_tokens",
                copy_start,
            )

            prompt_logprobs_count = len(self.model_runner_output.prompt_logprobs_dict)
            prompt_logprobs_non_null = sum(
                value is not None
                for value in self.model_runner_output.prompt_logprobs_dict.values()
            )
            copy_start = _log_copy_enqueue_start(
                self.model_runner_output.req_ids,
                "prompt_logprobs_dict",
                entries=prompt_logprobs_count,
                non_null_entries=prompt_logprobs_non_null,
            )
            self.prompt_logprobs_dict = {
                k: v.to_cpu_nonblocking() if v is not None else None
                for k, v in self.model_runner_output.prompt_logprobs_dict.items()
            }
            _log_copy_enqueue_done(
                self.model_runner_output.req_ids,
                "prompt_logprobs_dict",
                copy_start,
                entries=prompt_logprobs_count,
                non_null_entries=prompt_logprobs_non_null,
            )
            self.copy_event.record(copy_stream)
            _log_async_output("copy_recorded", self.model_runner_output.req_ids)

    def get_output(self) -> ModelRunnerOutput:
        sync_start = time.perf_counter()
        _log_async_output("get_output_enter", self.model_runner_output.req_ids)
        self.copy_event.synchronize()
        output_ready_time = time.time()
        _log_async_output(
            "get_output_after_sync",
            self.model_runner_output.req_ids,
            wait_ms=round((time.perf_counter() - sync_start) * 1000, 3),
        )
        for req_id, arrival_time in self.ttft_request_arrival_times.items():
            _log_async_output(
                "prefill_ttft_ready",
                [req_id],
                prefill_ttft_ms=round((output_ready_time - arrival_time) * 1000, 3),
            )

        # NOTE(woosuk): The following code is to ensure compatibility with
        # the existing model runner.
        # Going forward, we should keep the data structures as NumPy arrays
        # rather than Python lists.
        sampled_token_ids: list[list[int]] = self.sampled_token_ids.tolist()
        num_sampled_tokens: list[int] = self.num_sampled_tokens_np.tolist()
        for token_ids, num_tokens in zip(sampled_token_ids, num_sampled_tokens):
            del token_ids[num_tokens:]
        self.model_runner_output.sampled_token_ids = sampled_token_ids

        if self.num_nans is not None:
            self.model_runner_output.num_nans_in_logits = dict(
                zip(self.model_runner_output.req_ids, self.num_nans.tolist())
            )

        if self.logprobs_tensors is not None:
            self.model_runner_output.logprobs = self.logprobs_tensors.tolists()
        self.model_runner_output.prompt_logprobs_dict = self.prompt_logprobs_dict
        _log_async_output("get_output_return", self.model_runner_output.req_ids)
        return self.model_runner_output


class AsyncPoolingOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        pooler_output: torch.Tensor,
        is_valid: torch.Tensor | None,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
        copy_event: torch.cuda.Event,
    ):
        self.model_runner_output = model_runner_output
        self.pooler_output = pooler_output
        self.is_valid = is_valid
        self.copy_event = copy_event

        with stream(copy_stream, main_stream):
            copy_stream.wait_stream(main_stream)
            self.pooler_output_cpu = self.pooler_output.to("cpu", non_blocking=True)
            if self.is_valid is not None:
                self.is_valid_cpu = self.is_valid.to("cpu", non_blocking=True)
            else:
                self.is_valid_cpu = None
            self.copy_event.record(copy_stream)

    def get_output(self) -> ModelRunnerOutput:
        pooler_output = list(self.pooler_output_cpu.unbind(dim=0))
        self.copy_event.synchronize()
        if self.is_valid_cpu is not None:
            is_valid_cpu = self.is_valid_cpu.tolist()
            for i, is_valid in enumerate(is_valid_cpu):
                if not is_valid:
                    pooler_output[i] = None
        self.model_runner_output.pooler_output = pooler_output
        return self.model_runner_output


def async_copy_to_np(x: torch.Tensor) -> np.ndarray:
    return x.to("cpu", non_blocking=True).numpy()


@contextlib.contextmanager
def stream(to_stream: torch.cuda.Stream, from_stream: torch.cuda.Stream):
    """Lightweight version of torch.cuda.stream() context manager which
    avoids current_stream and device lookups.
    """
    try:
        torch.cuda.set_stream(to_stream)
        yield
    finally:
        torch.cuda.set_stream(from_stream)
