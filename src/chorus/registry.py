from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field, replace

from chorus.devices import DevicePolicy
from chorus.engines import ENGINE_INFO
from chorus.model_worker import ModelWorker, worker_concurrency
from chorus.models import ModelCatalog
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
        idle_timeout: float = 300,
        ram_budget_bytes: int | None = None,
        vram_budget_bytes: dict[str, int] | None = None,
    ):
        self.catalog = catalog or ModelCatalog()
        self.policy = policy or DevicePolicy()
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

    def _maintain(self) -> None:
        now = time.monotonic()
        for key, entry in list(self._instances.items()):
            if not entry.users and (
                entry.worker.pid is None or now - entry.last_used >= self.idle_timeout
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
                idle = [entry for entry in self._instances.values() if not entry.users]
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

    def _select_device(self, spec, requested: str | None = None) -> str:
        requested = requested or self.policy.default
        if requested != "auto":
            return self.policy.resolve(spec.devices, requested)

        gpu_devices = [
            device
            for device in self.policy.allowed
            if device.startswith("cuda:") and "cuda" in spec.devices
        ]
        if not gpu_devices:
            return self.policy.resolve(spec.devices, "auto")

        def rank(device: str) -> tuple[int, int]:
            outstanding = sum(
                entry.users
                for key, entry in self._instances.items()
                if key[2] == device
            )
            entry = self._instances.get((spec.engine, spec.name, device))
            return (
                outstanding,
                0 if entry is not None and entry.worker.pid is not None else 1,
            )

        return min(gpu_devices, key=rank)

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
        device: str | None = None,
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
            selected = self._select_device(spec, device)
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
