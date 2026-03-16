"""vLLM server lifecycle manager for phase-swapped GPU sharing.

Manages N independent vLLM instances (one per GPU, TP=1) that are
started before rollout phases and killed before training phases.
This allows the same GPUs to be used for both inference and FSDP
training, maximizing hardware utilization.

Typical lifecycle per co-evolution step::

    manager.start()              # Launch 8 × TP=1 instances
    await manager.wait_healthy() # Wait for all to respond
    # ... Phase A+B: rollouts + skill bank ...
    manager.stop()               # Kill all, free GPU memory
    # ... Phase C: FSDP GRPO training on all 8 GPUs ...

Memory budget per instance (Qwen3-14B, A100-80GB, TP=1):
  - Model weights (bf16):    ~28 GB
  - KV cache (gpu_util=0.9): ~44 GB  →  ~275 max sequences @ 1K tokens
  - Total:                   ~72 GB / 80 GB
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class VLLMServerManager:
    """Manages multiple TP=1 vLLM inference servers across GPUs."""

    def __init__(
        self,
        model_name: str,
        adapter_dir: str,
        gpu_ids: List[int],
        base_port: int = 8000,
        gpu_util: float = 0.90,
        max_num_seqs: int = 32,
        log_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.adapter_dir = adapter_dir
        self.gpu_ids = gpu_ids
        self.base_port = base_port
        self.gpu_util = gpu_util
        self.max_num_seqs = max_num_seqs
        self.log_dir = log_dir
        self._processes: List[subprocess.Popen] = []
        self._log_files: list = []

        atexit.register(self.stop)

    @property
    def n_instances(self) -> int:
        return len(self.gpu_ids)

    @property
    def base_urls(self) -> List[str]:
        return [
            f"http://localhost:{self.base_port + i}/v1"
            for i in range(self.n_instances)
        ]

    @property
    def ports(self) -> List[int]:
        return [self.base_port + i for i in range(self.n_instances)]

    def _build_lora_args(self) -> List[str]:
        """Build --lora-modules flag values from existing adapter dirs."""
        modules = []
        for sub, names in [
            ("decision", ["skill_selection", "action_taking"]),
            ("skillbank", ["segment", "contract", "curator"]),
        ]:
            for name in names:
                path = Path(self.adapter_dir) / sub / name
                if (path / "adapter_config.json").exists():
                    modules.append(f"{name}={path}")
        return modules

    def start(self) -> None:
        """Start one vLLM TP=1 instance per GPU."""
        if self._processes:
            logger.warning("Instances already running — stopping first")
            self.stop()

        lora_modules = self._build_lora_args()

        log_dir_path = Path(self.log_dir) if self.log_dir else None
        if log_dir_path:
            log_dir_path.mkdir(parents=True, exist_ok=True)

        for i, gpu_id in enumerate(self.gpu_ids):
            port = self.base_port + i

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env.setdefault("HF_HOME", "/workspace/huggingface")
            env.setdefault("HF_HUB_CACHE", os.path.join(env["HF_HOME"], "hub"))

            cmd = [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", self.model_name,
                "--tensor-parallel-size", "1",
                "--gpu-memory-utilization", str(self.gpu_util),
                "--enable-lora", "--max-loras", "5", "--max-lora-rank", "64",
                "--enable-prefix-caching",
                "--enable-chunked-prefill",
                "--max-num-seqs", str(self.max_num_seqs),
                "--max-num-batched-tokens", "8192",
                "--port", str(port),
                "--trust-remote-code",
            ]

            if lora_modules:
                cmd.extend(["--lora-modules"] + lora_modules)

            stderr_target = subprocess.DEVNULL
            log_fh = None
            if log_dir_path:
                log_path = log_dir_path / f"vllm_gpu{gpu_id}.log"
                log_fh = open(log_path, "w")
                stderr_target = log_fh

            logger.info(
                "Starting vLLM [%d/%d]: GPU %d → port %d",
                i + 1, self.n_instances, gpu_id, port,
            )

            proc = subprocess.Popen(
                cmd, env=env,
                stdout=subprocess.DEVNULL, stderr=stderr_target,
            )
            self._processes.append(proc)
            if log_fh:
                self._log_files.append(log_fh)

        if lora_modules:
            logger.info(
                "Started %d vLLM instances (ports %d–%d) with %d LoRA adapters",
                self.n_instances,
                self.base_port,
                self.base_port + self.n_instances - 1,
                len(lora_modules),
            )
        else:
            logger.info(
                "Started %d vLLM instances (ports %d–%d), no LoRA adapters yet",
                self.n_instances,
                self.base_port,
                self.base_port + self.n_instances - 1,
            )

    def stop(self) -> None:
        """Terminate all vLLM instances and free GPU memory."""
        if not self._processes:
            return

        n = len(self._processes)
        logger.info("Stopping %d vLLM instances...", n)

        # Send SIGTERM to all
        for proc in self._processes:
            if proc.poll() is None:
                try:
                    proc.terminate()
                except OSError:
                    pass

        # Wait for graceful shutdown, SIGKILL stragglers
        deadline = time.monotonic() + 15
        for proc in self._processes:
            remaining = max(0.1, deadline - time.monotonic())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass

        self._processes.clear()

        for fh in self._log_files:
            try:
                fh.close()
            except Exception:
                pass
        self._log_files.clear()

        logger.info("All %d vLLM instances stopped", n)

    async def wait_healthy(
        self,
        timeout: float = 300,
        poll_interval: float = 5,
    ) -> bool:
        """Wait for all instances to pass health checks.

        Returns True when all instances respond to /health, False on
        timeout or if any process exits unexpectedly.
        """
        import httpx as _httpx

        start = time.monotonic()
        healthy: set = set()
        n_total = self.n_instances

        async with _httpx.AsyncClient(timeout=10.0) as client:
            while time.monotonic() - start < timeout:
                # Check if any process died
                for i, proc in enumerate(self._processes):
                    if proc.poll() is not None:
                        logger.error(
                            "vLLM instance %d (GPU %d, port %d) exited "
                            "with code %d",
                            i, self.gpu_ids[i],
                            self.base_port + i, proc.returncode,
                        )
                        return False

                for i in range(n_total):
                    if i in healthy:
                        continue
                    port = self.base_port + i
                    try:
                        resp = await client.get(
                            f"http://localhost:{port}/health",
                        )
                        if resp.status_code == 200:
                            healthy.add(i)
                            elapsed = time.monotonic() - start
                            logger.info(
                                "vLLM GPU %d (port %d) healthy "
                                "[%d/%d, %.1fs]",
                                self.gpu_ids[i], port,
                                len(healthy), n_total, elapsed,
                            )
                    except (_httpx.ConnectError, _httpx.TimeoutException):
                        pass

                if len(healthy) == n_total:
                    elapsed = time.monotonic() - start
                    logger.info(
                        "All %d vLLM instances healthy (%.1fs)",
                        n_total, elapsed,
                    )
                    return True

                await asyncio.sleep(poll_interval)

        missing_ports = [
            self.base_port + i for i in range(n_total) if i not in healthy
        ]
        logger.error(
            "Timeout (%.0fs): %d/%d instances not ready (ports %s)",
            timeout, len(missing_ports), n_total, missing_ports,
        )
        return False

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass
