#!/usr/bin/env python3
"""
Usage examples:

1) Default eager mode (no TP/EP), stop server after client exits:
    python3 tools/run_vllm_e2e_server_mode.py

2) Keep server running after client exits:
    python3 tools/run_vllm_e2e_server_mode.py keep

3) TP mode (server appends -tp=<N>):
    python3 tools/run_vllm_e2e_server_mode.py --parallel-mode tp2
    python3 tools/run_vllm_e2e_server_mode.py --parallel-mode tp4

4) EP mode (server appends --enable-expert-parallel and -tp=<N>):
    python3 tools/run_vllm_e2e_server_mode.py --parallel-mode ep2
    python3 tools/run_vllm_e2e_server_mode.py --parallel-mode ep4
    python3 tools/run_vllm_e2e_server_mode.py --parallel-mode ep8

5) Graph mode:
    python3 tools/run_vllm_e2e_server_mode.py --mode graph --parallel-mode tp4

Notes:
- Client command is unchanged across TP/EP modes.
- Use --no-dtype to omit --dtype from server command.
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional


DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# Keep model-specific differences in one JSON list. Values not listed here use
# the defaults below, so adding a model does not require duplicating commands.
MODEL_CONFIGS_JSON = r'''
[
    {"model": "INCModel/Qwen3-30B-A3B-Instruct-2507-MXFP8-CT-AutoRound", "gpu_memory_utilization": 0.9, "num_prompts": 200, "max_concurrency": 34},
    {"model": "INCModel/Qwen3-32B-MXFP8-CT-AutoRound", "gpu_memory_utilization": 0.85, "num_prompts": 10, "max_concurrency": 1},
    {"model": "INCModel2/Qwen3-30B-A3B-Instruct-2507-MXFP8-FP8ATTN-CT-AutoRound", "gpu_memory_utilization": 0.9, "num_prompts": 200, "max_concurrency": 34},
    {"model": "INCModel2/Qwen3-32B-MXFP8-FP8ATTN-CT-AutoRound", "gpu_memory_utilization": 0.85, "num_prompts": 10, "max_concurrency": 1},
    {"model": "Qwen/Qwen3-30B-A3B", "gpu_memory_utilization": 0.9, "num_prompts": 200, "max_concurrency": 24},
    {"model": "Qwen/Qwen3-30B-A3B-FP8", "gpu_memory_utilization": 0.9, "num_prompts": 20, "max_concurrency": 2},
    {"model": "Qwen/Qwen3-32B", "gpu_memory_utilization": 0.85, "num_prompts": 10, "max_concurrency": 1},
    {"model": "Qwen/Qwen3-32B-FP8", "gpu_memory_utilization": 0.85, "num_prompts": 10, "max_concurrency": 1}
]
'''
MODEL_CONFIGS = json.loads(MODEL_CONFIGS_JSON)

DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_MAX_NUM_BATCHED_TOKENS = 4096
DEFAULT_MAX_MODEL_LEN = 8192
DEFAULT_BLOCK_SIZE = 64
DEFAULT_MAX_NUM_SEQS = 128
DEFAULT_RANDOM_INPUT_LEN = 3500
DEFAULT_RANDOM_OUTPUT_LEN = 1500

SERVER_CMD_BASE = [
    "python3",
    "-m",
    "vllm.entrypoints.openai.api_server",
    "--port",
    "8000",
    "--host",
    "0.0.0.0",
    "--trust-remote-code",
    "--no-enable-prefix-caching",
    "--max-num-batched-tokens",
    str(DEFAULT_MAX_NUM_BATCHED_TOKENS),
    "--max-model-len",
    str(DEFAULT_MAX_MODEL_LEN),
    "--block-size",
    str(DEFAULT_BLOCK_SIZE),
    "--max-num-seqs",
    str(DEFAULT_MAX_NUM_SEQS),
    "--no-enable-log-requests",
]

CLIENT_CMD_BASE = [
    "python3",
    "-m",
    "vllm.entrypoints.cli.main",
    "bench",
    "serve",
    "--ready-check-timeout-sec",
    "1",
    "--temperature=0",
    "--dataset-name",
    "random",
    "--ignore-eos",
    "--port=8000",
    "--host",
    "0.0.0.0",
    "--request-rate",
    "inf",
    "--backend",
    "vllm",
    "--trust-remote-code",
    "--save-result",
    "--metric-percentiles",
    "95,99",
]


def get_model_config(model: str) -> dict:
    for config in MODEL_CONFIGS:
        if config["model"] == model:
            return config
    return {}


def get_model_env(model: str, model_config: dict, mode: str) -> dict[str, str]:
    env = {
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "VLLM_XPU_USE_CUSTOM_MODEL": "1",
    }
    if mode == "graph":
        env.update(
            {
                "VLLM_USE_BREAKABLE_CUDAGRAPH": "1",
                "VLLM_XPU_ENABLE_XPU_GRAPH": "1",
            }
        )
    if "FP8ATTN" in model.upper():
        env["VLLM_XPU_SUPPORT_FP8_QUERY"] = "1"
    env.update(model_config.get("env", {}))
    return env


def build_log_stem(
    model: str,
    random_input_len: int,
    random_output_len: int,
    parallel_mode: str,
    num_prompt: int,
    max_concurrency: int,
    timestamp: str,
) -> str:
    model_name = model.replace("/", "-")
    if parallel_mode == "none":
        parallel_name = "TP1-1"
    else:
        parallel_name = f"{parallel_mode[:2].upper()}{parallel_mode[2:]}-{parallel_mode[2:]}"
    return (
        f"{model_name}_Length-{random_input_len}-{random_output_len}_"
        f"{parallel_name}_Prompt-{num_prompt}_BS-_Request-inf_"
        f"Conc-{max_concurrency}_{timestamp}"
    )


def build_server_cmd(
    model: str,
    include_dtype: bool,
    dtype: str,
    mode: str,
    parallel_mode: str,
    model_config: dict,
) -> list[str]:
    cmd = SERVER_CMD_BASE[:3] + ["--model", model] + SERVER_CMD_BASE[3:]
    cmd += [
        "--gpu-memory-utilization",
        str(model_config.get("gpu_memory_utilization", DEFAULT_GPU_MEMORY_UTILIZATION)),
    ]
    if mode == "eager":
        cmd.append("--enforce-eager")

    if parallel_mode.startswith("ep"):
        cmd.append("--enable-expert-parallel")

    if parallel_mode != "none":
        tp = int(parallel_mode[2:])
        cmd.append(f"-tp={tp}")

    if include_dtype:
        cmd.append(f"--dtype={dtype}")
    return cmd


def build_client_cmd(
    model: str,
    random_input_len: int,
    random_output_len: int,
    num_prompt: int,
    max_concurrency: int,
) -> list[str]:
    return (
        CLIENT_CMD_BASE[:5]
        + ["--model", model]
        + CLIENT_CMD_BASE[5:]
        + [
            "--num-warmups",
            str(max_concurrency),
            f"--random-input-len={random_input_len}",
            f"--random-output-len={random_output_len}",
            "--num-prompts",
            str(num_prompt),
            "--max-concurrency",
            str(max_concurrency),
        ]
    )


class StreamPump:
    def __init__(
        self,
        stream,
        prefix: str,
        log_file,
        startup_pattern: Optional[str] = None,
        startup_event: Optional[threading.Event] = None,
        tail_buffer: Optional[Deque[str]] = None,
    ) -> None:
        self.stream = stream
        self.prefix = prefix
        self.log_file = log_file
        self.startup_pattern = startup_pattern
        self.startup_event = startup_event
        self.tail_buffer = tail_buffer
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def join(self) -> None:
        self.thread.join()

    def _run(self) -> None:
        for raw_line in self.stream:
            line = raw_line.rstrip("\n")
            out_line = f"[{self.prefix}] {line}"
            print(out_line, flush=True)
            self.log_file.write(out_line + "\n")
            self.log_file.flush()

            if self.tail_buffer is not None:
                self.tail_buffer.append(out_line)

            if (
                self.startup_pattern is not None
                and self.startup_event is not None
                and self.startup_pattern in line
            ):
                self.startup_event.set()


def stop_process_tree(proc: subprocess.Popen, timeout_sec: int = 10) -> None:
    if proc.poll() is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Start vLLM server, wait until startup log appears, then run client benchmark."
        )
    )
    parser.add_argument(
        "cleanup_mode",
        nargs="?",
        default="kill",
        choices=["kill", "keep"],
        help="kill: stop server when script exits; keep: leave server running",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name/path used by both server and client",
    )
    parser.add_argument(
        "--random-input-len",
        type=int,
        default=DEFAULT_RANDOM_INPUT_LEN,
        help="Random input length used by client benchmark",
    )
    parser.add_argument(
        "--random-output-len",
        type=int,
        default=DEFAULT_RANDOM_OUTPUT_LEN,
        help="Random output length used by client benchmark",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="Optionally append a server dtype value",
    )
    parser.add_argument(
        "--mode",
        default="eager",
        choices=["eager", "graph"],
        help=(
            "Server mode: graph enables XPU graph flags; eager adds --enforce-eager"
        ),
    )
    parser.add_argument(
        "--no-dtype",
        action="store_true",
        help="Do not append --dtype to server command",
    )
    parser.add_argument(
        "--parallel-mode",
        default="none",
        choices=["none", "tp2", "tp4", "ep2", "ep4", "ep8"],
        help=(
            "Single-arg parallel mode for server command: none|tp2|tp4|ep2|ep4|ep8. "
            "tp* appends -tp=<N>; ep* appends --enable-expert-parallel and -tp=<N>."
        ),
    )
    parser.add_argument(
        "--startup-timeout-sec",
        type=int,
        default=int(os.environ.get("STARTUP_TIMEOUT_SEC", "900")),
        help="Timeout in seconds while waiting for server readiness",
    )
    parser.add_argument(
        "--num-prompt",
        type=int,
        default=None,
        help="Override the model-specific number of client prompts",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Override model-specific concurrency and matching warmup count",
    )
    args = parser.parse_args()
    model_config = get_model_config(args.model)
    num_prompt = (
        args.num_prompt
        if args.num_prompt is not None
        else model_config.get("num_prompts", 4)
    )
    max_concurrency = (
        args.max_concurrency
        if args.max_concurrency is not None
        else model_config.get("max_concurrency", 1)
    )

    log_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_stem = build_log_stem(
        model=args.model,
        random_input_len=args.random_input_len,
        random_output_len=args.random_output_len,
        parallel_mode=args.parallel_mode,
        num_prompt=num_prompt,
        max_concurrency=max_concurrency,
        timestamp=log_timestamp,
    )
    default_server_log = f"./{log_stem}_server.log"
    default_client_log = f"./{log_stem}_client.log"
    server_log_path = Path(
        os.environ.get("SERVER_LOG_FILE", default_server_log)
    ).resolve()
    client_log_path = Path(
        os.environ.get("CLIENT_LOG_FILE", default_client_log)
    ).resolve()
    startup_pattern = os.environ.get("STARTUP_PATTERN", "Application startup complete")
    startup_timeout_sec = args.startup_timeout_sec

    server_log_path.parent.mkdir(parents=True, exist_ok=True)
    client_log_path.parent.mkdir(parents=True, exist_ok=True)

    startup_event = threading.Event()
    server_tail: Deque[str] = deque(maxlen=100)

    server_env = os.environ.copy()
    server_env_overrides = get_model_env(args.model, model_config, args.mode)
    server_env.update(server_env_overrides)

    server_cmd = build_server_cmd(
        model=args.model,
        include_dtype=args.dtype is not None and not args.no_dtype,
        dtype=args.dtype,
        mode=args.mode,
        parallel_mode=args.parallel_mode,
        model_config=model_config,
    )
    client_cmd = build_client_cmd(
        model=args.model,
        random_input_len=args.random_input_len,
        random_output_len=args.random_output_len,
        num_prompt=num_prompt,
        max_concurrency=max_concurrency,
    )

    server_proc: Optional[subprocess.Popen] = None
    client_exit = 1

    with server_log_path.open("w", encoding="utf-8") as server_log, client_log_path.open(
        "w", encoding="utf-8"
    ) as client_log:
        try:
            server_cmd_text = (
                " ".join(
                    f"{k}={shlex.quote(v)}" for k, v in server_env_overrides.items()
                )
                + " "
                + shlex.join(server_cmd)
            )
            client_cmd_text = shlex.join(client_cmd)

            for msg in (
                f"[INFO] Server CMD: {server_cmd_text}",
                f"[INFO] Client CMD: {client_cmd_text}",
            ):
                print(msg, flush=True)
                server_log.write(msg + "\n")
                server_log.flush()
                client_log.write(msg + "\n")
                client_log.flush()

            print("[INFO] Starting server...", flush=True)
            server_proc = subprocess.Popen(
                server_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=server_env,
                preexec_fn=os.setsid,
            )
            print(f"[INFO] Server PID: {server_proc.pid}", flush=True)
            print(f"[INFO] Server log: {server_log_path}", flush=True)

            server_pump = StreamPump(
                stream=server_proc.stdout,
                prefix="SERVER",
                log_file=server_log,
                startup_pattern=startup_pattern,
                startup_event=startup_event,
                tail_buffer=server_tail,
            )
            server_pump.start()

            print(f"[INFO] Waiting for startup pattern: {startup_pattern}", flush=True)
            deadline = time.time() + startup_timeout_sec
            while True:
                if startup_event.is_set():
                    print("[INFO] Server is ready.", flush=True)
                    break

                if server_proc.poll() is not None:
                    print("[ERROR] Server exited before becoming ready.", flush=True)
                    if server_tail:
                        print("[ERROR] Last 100 server log lines:", flush=True)
                        for line in server_tail:
                            print(line, flush=True)
                    return 1

                if time.time() >= deadline:
                    print(
                        f"[ERROR] Timeout waiting for server readiness ({startup_timeout_sec}s).",
                        flush=True,
                    )
                    if server_tail:
                        print("[ERROR] Last 100 server log lines:", flush=True)
                        for line in server_tail:
                            print(line, flush=True)
                    return 1

                time.sleep(1)

            print("[INFO] Running client...", flush=True)
            client_proc = subprocess.Popen(
                client_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=os.environ.copy(),
                preexec_fn=os.setsid,
            )
            client_pump = StreamPump(
                stream=client_proc.stdout,
                prefix="CLIENT",
                log_file=client_log,
            )
            client_pump.start()
            client_exit = client_proc.wait()
            client_pump.join()
            print(f"[INFO] Client exit code: {client_exit}", flush=True)
            print(f"[INFO] Client log: {client_log_path}", flush=True)
            return client_exit
        finally:
            if server_proc is not None and args.cleanup_mode == "kill":
                print(f"[INFO] Stopping server PID {server_proc.pid}", flush=True)
                stop_process_tree(server_proc)


if __name__ == "__main__":
    sys.exit(main())
