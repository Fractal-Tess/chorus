from __future__ import annotations

from typing import Any, Callable

from mini_tts.models import ModelSpec
from mini_tts.types import AudioResult


class Engine:
    """One model/device instance, accessed under the registry's instance lock."""

    def __init__(self, spec: ModelSpec, device: str):
        self.spec = spec
        self.device = device
        self._cache: dict[str, Any] = {}

    def cached(self, key: str, loader: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = loader()
        return self._cache[key]

    def synthesize(
        self, text: str, voice: str | None, language: str | None, speed: float
    ) -> AudioResult:
        raise NotImplementedError
