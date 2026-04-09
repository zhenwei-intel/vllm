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


def _log_async_output(event: str,
                      req_ids: list[str],
                      **fields: float | int | str | bool) -> None:
    if not _DEBUG_ASYNC_OUTPUT:
        return
    details = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.warning("ASYNC_OUTPUT_DBG event=%s ts=%.6f req_ids=%s %s",
                   event, time.perf_counter(), req_ids[:4], details)


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
        req_id_to_last_output_ready_time: dict[str, float] | None = None,
    ):
        # NOTE(woosuk): We must retain references to the GPU tensors,
        # as the copy operations are performed on a different CUDA stream than
        # the one where the tensors were created.
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        self.copy_event = copy_event
        self.ttft_request_arrival_times = ttft_request_arrival_times or {}
        self.req_id_to_last_output_ready_time = (
            req_id_to_last_output_ready_time
            if req_id_to_last_output_ready_time is not None else {})

        _log_async_output("init_enter", self.model_runner_output.req_ids)

        with stream(copy_stream, main_stream):
            copy_stream.wait_stream(main_stream)

            self.sampled_token_ids = async_copy_to_np(sampler_output.sampled_token_ids)
            self.logprobs_tensors: LogprobsTensors | None = None
            if sampler_output.logprobs_tensors is not None:
                self.logprobs_tensors = (
                    sampler_output.logprobs_tensors.to_cpu_nonblocking()
                )
            self.num_nans: np.ndarray | None = None
            if sampler_output.num_nans is not None:
                self.num_nans = async_copy_to_np(sampler_output.num_nans)
            self.num_sampled_tokens_np = async_copy_to_np(num_sampled_tokens)
            self.prompt_logprobs_dict = {
                k: v.to_cpu_nonblocking() if v is not None else None
                for k, v in self.model_runner_output.prompt_logprobs_dict.items()
            }
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
        for req_id, sampled_count in zip(self.model_runner_output.req_ids,
                                         num_sampled_tokens):
            prev_output_ready_time = self.req_id_to_last_output_ready_time.get(req_id)
            if prev_output_ready_time is not None:
                _log_async_output(
                    "decode_tpot_ready",
                    [req_id],
                    decode_tpot_ms=round(
                        (output_ready_time - prev_output_ready_time) * 1000, 3),
                    num_tokens=sampled_count,
                )
            self.req_id_to_last_output_ready_time[req_id] = output_ready_time
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
