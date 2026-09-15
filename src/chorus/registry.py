from __future__ import annotations

import importlib
import threading
import time
from dataclasses import replace

from chorus.devices import DevicePolicy
from chorus.engines import ENGINE_INFO
from chorus.models import ModelCatalog
from chorus.processing import Processor
from chorus.types import AudioResult

ENGLISH = {"a", "b", "en", "en-us", "en-gb", "english"}


class EngineRegistry:
    def __init__(
        self,
        catalog: ModelCatalog | None = None,
        policy: DevicePolicy | None = None,
        enabled: list[str] | None = None,
    ):
        self.catalog = catalog or ModelCatalog()
        self.policy = policy or DevicePolicy()
        self.enabled = list(ENGINE_INFO) if enabled is None else enabled
        if not self.enabled or any(name not in ENGINE_INFO for name in self.enabled):
            raise ValueError("Unknown or empty engine selection")
        self._instances = {}
        self._locks = {}
        self._lock = threading.RLock()
        self.processor = Processor()

    def loaded_models(self) -> list[dict]:
        with self._lock:
            return [
                {"engine": e, "model": m, "device": d} for e, m, d in self._instances
            ]

    def loaded_engines(self) -> list[str]:
        return sorted({entry["engine"] for entry in self.loaded_models()})

    def _resolve(self, engine, model, device):
        if engine not in self.enabled:
            raise ValueError(f"Engine {engine} is not enabled")
        spec = self.catalog.resolve(engine, model)
        selected = self.policy.resolve(spec.devices, device)
        key = (engine, spec.name, selected)
        with self._lock:
            lock = self._locks.setdefault(key, threading.Lock())
        return spec, selected, key, lock

    def _load(self, spec, device, key):
        with self._lock:
            instance = self._instances.get(key)
        if instance is None:
            module = importlib.import_module(f"chorus.engines.{spec.engine}")
            instance = module.Adapter(spec, device)
        return instance

    def preload(self, selector: str):
        engine, model = selector.split("/", 1)
        self.synthesize(engine, "Ready.", model=model)

    def close(self) -> None:
        with self._lock:
            instances = list(self._instances.items())
        for key, instance in instances:
            with self._locks[key]:
                close = getattr(instance, "close", None)
                if close is not None:
                    close()
        with self._lock:
            self._instances.clear()

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
        spec, selected, key, lock = self._resolve(engine, model, device)
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
        with lock:
            inference_started = time.perf_counter()
            instance = self._load(spec, selected, key)
            try:
                audio = instance.synthesize(text, voice, language, speed)
            except Exception:
                with self._lock:
                    cached = key in self._instances
                if not cached:
                    close = getattr(instance, "close", None)
                    if close is not None:
                        close()
                raise
            with self._lock:
                self._instances[key] = instance
            inference_ms = (time.perf_counter() - inference_started) * 1000
        audio = replace(
            audio,
            model=spec.name,
            device=selected,
            timings_ms={
                "queue": (inference_started - started) * 1000,
                "inference": inference_ms,
            },
        )
        audio = self.processor.process(
            audio,
            instance.alignment_text(text) if force_align else text,
            lava_sr=lava_sr,
            force_align=force_align,
        )
        return replace(
            audio,
            timings_ms={
                **audio.timings_ms,
                "total": (time.perf_counter() - started) * 1000,
            },
        )
