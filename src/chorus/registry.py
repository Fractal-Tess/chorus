from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path

from chorus.channels import ChannelPolicy, GpuQueueFullError
from chorus.devices import DevicePolicy
from chorus.engines import ENGINE_INFO
from chorus.model_worker import ModelWorker, worker_concurrency
from chorus.models import ModelCatalog, ModelSpec
from chorus.resources import (
    gpu_usage,
    process_tree_pids,
    ram_usage,
    validate_gpu_monitoring,
)
from chorus.types import AudioResult

ENGLISH = {"a", "b", "en", "en-us", "en-gb", "english"}
LOGGER = logging.getLogger("chorus")


@dataclass
class CacheEntry:
    worker: ModelWorker
    lock: threading.Semaphore = field(
        default_factory=lambda: threading.BoundedSemaphore(1)
    )
    users: int = 0
    last_used: float = field(default_factory=time.monotonic)


class EngineRegistry:
    def __init__(
        self,
        catalog: ModelCatalog | None = None,
        policy: DevicePolicy | None = None,
        enabled: list[str] | None = None,
        *,
        channel_config: Path | None = None,
        gpu_queue_size: int = 32,
        idle_timeout: float = 1800,
        ram_budget_bytes: int | None = None,
        vram_budget_bytes: dict[str, int] | None = None,
    ):
        self.catalog = catalog or ModelCatalog()
        self.policy = policy or DevicePolicy()
        self.channels = ChannelPolicy(self.catalog, self.policy, channel_config)
        if type(gpu_queue_size) is not int or gpu_queue_size < 0:
            raise ValueError("GPU queue size must be a nonnegative integer")
        self.gpu_queue_size = gpu_queue_size
        self._gpu_waiters: deque[tuple[str, str, object]] = deque()
        self.enabled = list(ENGINE_INFO) if enabled is None else enabled
        if not self.enabled or any(name not in ENGINE_INFO for name in self.enabled):
            raise ValueError("Unknown or empty engine selection")
        if not math.isfinite(idle_timeout) or idle_timeout < 0:
            raise ValueError("Idle timeout must be finite and nonnegative")
        self.idle_timeout = idle_timeout
        self.ram_budget_bytes = ram_budget_bytes
        self.vram_budget_bytes = dict(vram_budget_bytes or {})
        for limit in [ram_budget_bytes, *self.vram_budget_bytes.values()]:
            if limit is not None and (not isinstance(limit, int) or limit <= 0):
                raise ValueError("Memory budgets must be positive integer bytes")
        for device in self.vram_budget_bytes:
            if not device.startswith("cuda:") or device not in self.policy.allowed:
                raise ValueError(f"VRAM budget device {device} is not an enabled GPU")
        if self.vram_budget_bytes:
            validate_gpu_monitoring()
        self._instances: dict[tuple[str, str, str], CacheEntry] = {}
        self._condition = threading.Condition(threading.RLock())
        self._closed = False
        self._reaper: threading.Thread | None = None

    def loaded_models(self) -> list[dict]:
        with self._condition:
            return [
                {"engine": e, "model": m, "device": d}
                for (e, m, d), entry in self._instances.items()
                if entry.worker.pid is not None
            ]

    def loaded_engines(self) -> list[str]:
        return sorted({entry["engine"] for entry in self.loaded_models()})

    def _usage(self, include_gpu: bool = False) -> tuple[dict, str | None]:
        gpu_error = None
        gpu = {}
        if self._instances and (self.vram_budget_bytes or include_gpu):
            try:
                gpu = gpu_usage()
            except RuntimeError as error:
                if self.vram_budget_bytes:
                    raise
                gpu_error = str(error)
        usage = {}
        for key, entry in self._instances.items():
            pid = entry.worker.pid
            pids = process_tree_pids(pid) if pid is not None else set()
            vram = {}
            for child in pids:
                for device, size in gpu.get(child, {}).items():
                    vram[device] = vram.get(device, 0) + size
            usage[key] = {"ram_bytes": ram_usage(pids), "vram_bytes": vram}
        return usage, gpu_error

    def _evict(self, key) -> None:
        entry = self._instances[key]
        if entry.users:
            raise RuntimeError("Cannot evict an active model")
        entry.worker.close()
        del self._instances[key]

    def _gpu_model_waiting(self, key: tuple[str, str, str]) -> bool:
        return key[2].startswith("cuda:") and any(
            waiter[0] == key[0] and waiter[1] == key[1] for waiter in self._gpu_waiters
        )

    def _maintain(self) -> None:
        now = time.monotonic()
        for key, entry in list(self._instances.items()):
            if not entry.users and (
                entry.worker.pid is None
                or (
                    now - entry.last_used >= self.idle_timeout
                    and not self._gpu_model_waiting(key)
                )
            ):
                self._evict(key)
        if not self._instances or not (self.ram_budget_bytes or self.vram_budget_bytes):
            return
        usage, _ = self._usage()
        ram = sum(item["ram_bytes"] for item in usage.values())
        vram = {
            device: sum(item["vram_bytes"].get(device, 0) for item in usage.values())
            for device in self.vram_budget_bytes
        }
        for key in sorted(
            self._instances, key=lambda key: self._instances[key].last_used
        ):
            ram_over = self.ram_budget_bytes is not None and ram > self.ram_budget_bytes
            gpu_over = {
                device
                for device, limit in self.vram_budget_bytes.items()
                if vram[device] > limit
            }
            if not ram_over and not gpu_over:
                break
            entry = self._instances[key]
            item = usage[key]
            if entry.users or not (
                ram_over or any(item["vram_bytes"].get(d, 0) for d in gpu_over)
            ):
                continue
            self._evict(key)
            ram -= item["ram_bytes"]
            for device in vram:
                vram[device] -= item["vram_bytes"].get(device, 0)

    def _reap(self) -> None:
        with self._condition:
            while not self._closed:
                if not self._instances:
                    self._condition.wait()
                    continue
                idle = [
                    entry
                    for key, entry in self._instances.items()
                    if not entry.users and not self._gpu_model_waiting(key)
                ]
                delay = min(
                    (
                        max(
                            0.01,
                            self.idle_timeout - (time.monotonic() - entry.last_used),
                        )
                        for entry in idle
                    ),
                    default=None,
                )
                if self.ram_budget_bytes or self.vram_budget_bytes:
                    delay = min(5.0, delay) if delay is not None else 5.0
                self._condition.wait(timeout=delay)
                if self._closed:
                    return
                try:
                    self._maintain()
                except Exception:
                    LOGGER.exception(
                        "Cache resource monitoring failed; unloading idle workers"
                    )
                    for key, entry in list(self._instances.items()):
                        if not entry.users:
                            self._evict(key)

    def resource_status(self) -> dict[str, object]:
        with self._condition:
            usage, gpu_error = self._usage(include_gpu=True)
            return {
                "idle_timeout_seconds": self.idle_timeout,
                "ram_budget_bytes": self.ram_budget_bytes,
                "vram_budget_bytes": self.vram_budget_bytes,
                "ram_used_bytes": sum(item["ram_bytes"] for item in usage.values()),
                "vram_used_bytes": {
                    device: sum(
                        item["vram_bytes"].get(device, 0) for item in usage.values()
                    )
                    for device in self.policy.allowed
                    if device.startswith("cuda:")
                }
                if gpu_error is None
                else None,
                "gpu_monitoring_error": gpu_error,
                "gpu_queue": {
                    "capacity": self.gpu_queue_size,
                    "waiting": len(self._gpu_waiters),
                    "running": sum(
                        entry.users
                        for key, entry in self._instances.items()
                        if key[2].startswith("cuda:")
                    ),
                },
                "models": [
                    {
                        "engine": key[0],
                        "model": key[1],
                        "device": key[2],
                        "pid": entry.worker.pid,
                        "active_requests": entry.users,
                        "idle_seconds": 0
                        if entry.users
                        else max(0, time.monotonic() - entry.last_used),
                        **usage[key],
                    }
                    for key, entry in self._instances.items()
                ],
            }

    def preload(self, selector: str):
        engine, model = selector.split("/", 1)
        self.synthesize(engine, "Ready.", model=model)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            while any(entry.users for entry in self._instances.values()):
                self._condition.wait()
            for key in list(self._instances):
                self._evict(key)
        if self._reaper is not None:
            self._reaper.join()

    def channel_status(self, spec: ModelSpec) -> dict[str, object]:
        return self.channels.status(spec)

    def _available_gpu(self, spec: ModelSpec, devices: list[str]) -> str | None:
        candidates = []
        for device in devices:
            active = [
                (key, entry)
                for key, entry in self._instances.items()
                if key[2] == device and entry.users
            ]
            running = sum(entry.users for _, entry in active)
            limit = worker_concurrency(spec.engine, device)
            if any(worker_concurrency(key[0], device) == 1 for key, _ in active):
                limit = 1
            if running >= limit:
                continue
            entry = self._instances.get((spec.engine, spec.name, device))
            warm = entry is not None and entry.worker.pid is not None
            candidates.append((running, not warm, len(candidates), device))
        return min(candidates)[3] if candidates else None

    def _select_device(self, spec: ModelSpec, channel: str) -> str:
        """Reserve in FIFO order; the caller holds the condition through admission."""
        if channel == "cpu":
            return "cpu"
        devices = self.channels.physical_devices(channel)
        if not self._gpu_waiters:
            available = self._available_gpu(spec, devices)
            if available is not None:
                return available
        if len(self._gpu_waiters) >= self.gpu_queue_size:
            raise GpuQueueFullError(
                f"GPU queue is full ({self.gpu_queue_size} waiting slots); retry later"
            )
        ticket = (spec.engine, spec.name, object())
        self._gpu_waiters.append(ticket)
        self._condition.notify_all()
        try:
            while not self._closed:
                if self._gpu_waiters[0] is ticket:
                    available = self._available_gpu(spec, devices)
                    if available is not None:
                        return available
                self._condition.wait()
            raise RuntimeError("Registry is closed")
        finally:
            self._gpu_waiters.remove(ticket)
            self._condition.notify_all()

    def synthesize(
        self,
        engine: str,
        text: str,
        voice: str | None = None,
        language: str | None = None,
        speed: float = 1.0,
        lava_sr: bool = False,
        force_align: bool = False,
        model: str | None = None,
        channel: str | None = None,
    ) -> AudioResult:
        if not text.strip():
            raise ValueError("Text must contain speech")
        if not 0.5 <= speed <= 2.0:
            raise ValueError("Speed must be between 0.5 and 2.0")
        if engine not in self.enabled:
            raise ValueError(f"Engine {engine} is not enabled")
        spec = self.catalog.resolve(engine, model)
        effective_language = language or (
            voice[0]
            if engine == "kokoro" and voice
            else ENGINE_INFO[engine]["default_language"]
        )
        if force_align and effective_language.lower() not in ENGLISH:
            raise ValueError(
                "Wav2Vec2 force alignment currently supports English speech only"
            )
        started = time.perf_counter()
        with self._condition:
            if self._closed:
                raise RuntimeError("Registry is closed")
            self._maintain()
            selected_channel = self.channels.resolve(spec, channel)
            selected = self._select_device(spec, selected_channel)
            key = (engine, spec.name, selected)
            entry = self._instances.get(key)
            if entry is None:
                entry = CacheEntry(
                    ModelWorker(spec, selected),
                    lock=threading.BoundedSemaphore(
                        worker_concurrency(engine, selected)
                    ),
                )
                self._instances[key] = entry
            entry.users += 1
            if self._reaper is None:
                self._reaper = threading.Thread(
                    target=self._reap, name="chorus-cache", daemon=True
                )
                self._reaper.start()
            self._condition.notify_all()
        try:
            with entry.lock:
                queue_ms = (time.perf_counter() - started) * 1000
                audio = entry.worker.synthesize(
                    text, voice, language, speed, lava_sr, force_align
                )
        finally:
            with self._condition:
                entry.users -= 1
                entry.last_used = time.monotonic()
                try:
                    self._maintain()
                finally:
                    self._condition.notify_all()
        return replace(
            audio,
            model=spec.name,
            device=selected,
            timings_ms={
                **audio.timings_ms,
                "queue": queue_ms,
                "total": (time.perf_counter() - started) * 1000,
            },
        )
