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
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional


DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

SERVER_CMD_BASE = [
    "python3",
    "-m",
    "vllm.entrypoints.openai.api_server",
    "--enforce-eager",
    "--max-num-seqs",
    "16",
    "--port",
    "8000",
    "--host",
    "0.0.0.0",
    "--trust-remote-code",
    "--gpu-memory-util=0.95",
    "--no-enable-prefix-caching",
    "--max-num-batched-tokens=8192",
    "--max-model-len=8192",
]

CLIENT_CMD_BASE = [
    "python3",
    "-m",
    "vllm.entrypoints.cli.main",
    "bench",
    "serve",
    "--ready-check-timeout-sec",
    "1",
    "--num-warmups",
    "1",
    "--dataset-name",
    "random",
    "--ignore-eos",
    "--port=8000",
    "--host",
    "0.0.0.0",
    "--request-rate",
    "inf",
    "--max-concurrency",
    "16",
    "--backend",
    "vllm",
    "--trust-remote-code",
]


def build_server_cmd(
    model: str,
    include_dtype: bool,
    dtype: str,
    mode: str,
    parallel_mode: str,
) -> list[str]:
    cmd = SERVER_CMD_BASE[:3] + ["--model", model] + SERVER_CMD_BASE[3:]
    if mode == "graph":
        cmd = [arg for arg in cmd if arg != "--enforce-eager"]

    if parallel_mode.startswith("ep"):
        cmd.append("--enable-expert-parallel")

    if parallel_mode != "none":
        tp = int(parallel_mode[2:])
        cmd.append(f"-tp={tp}")

    if include_dtype:
        cmd.append(f"--dtype={dtype}")
    return cmd


def build_client_cmd(
    model: str, random_input_len: int, random_output_len: int, num_prompt: int
) -> list[str]:
    return (
        CLIENT_CMD_BASE[:5]
        + ["--model", model]
        + CLIENT_CMD_BASE[5:12]
        + [f"--random-input-len={random_input_len}", f"--random-output-len={random_output_len}"]
        + CLIENT_CMD_BASE[12:15]
        + ["--num-prompt", str(num_prompt)]
        + CLIENT_CMD_BASE[15:]
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
        default=1024,
        help="Random input length used by client benchmark",
    )
    parser.add_argument(
        "--random-output-len",
        type=int,
        default=4096,
        help="Random output length used by client benchmark",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Server dtype value when dtype is enabled",
    )
    parser.add_argument(
        "--mode",
        default="eager",
        choices=["eager", "graph"],
        help=(
            "Server mode: eager keeps original server cmd; graph removes --enforce-eager "
            "and adds graph-related env flags"
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
        default=4,
        help="Number of prompts for client benchmark",
    )
    args = parser.parse_args()

    server_log_path = Path(os.environ.get("SERVER_LOG_FILE", "./vllm_server.log"))
    client_log_path = Path(os.environ.get("CLIENT_LOG_FILE", "./vllm_client.log"))
    startup_pattern = os.environ.get("STARTUP_PATTERN", "Application startup complete")
    startup_timeout_sec = args.startup_timeout_sec

    server_log_path.parent.mkdir(parents=True, exist_ok=True)
    client_log_path.parent.mkdir(parents=True, exist_ok=True)

    startup_event = threading.Event()
    server_tail: Deque[str] = deque(maxlen=100)

    server_env = os.environ.copy()
    server_env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    server_env_overrides = {
        "VLLM_USE_V2_MODEL_RUNNER": "0",
    }

    if args.mode == "graph":
        server_env["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"
        server_env["VLLM_XPU_ENABLE_XPU_GRAPH"] = "1"
        server_env_overrides["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"
        server_env_overrides["VLLM_XPU_ENABLE_XPU_GRAPH"] = "1"

    server_cmd = build_server_cmd(
        model=args.model,
        include_dtype=not args.no_dtype,
        dtype=args.dtype,
        mode=args.mode,
        parallel_mode=args.parallel_mode,
    )
    client_cmd = build_client_cmd(
        model=args.model,
        random_input_len=args.random_input_len,
        random_output_len=args.random_output_len,
        num_prompt=args.num_prompt,
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
