from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WordAlignment:
    word: str
    start_ms: int
    end_ms: int
    score: float


@dataclass(frozen=True)
class AudioResult:
    wav: bytes
    sample_rate: int
    duration: float
    timings_ms: dict[str, float] = field(default_factory=dict)
    alignment: tuple[WordAlignment, ...] | None = None
    model: str = ""
    device: str = "cpu"


def encode_wav(audio: Any, sample_rate: int) -> AudioResult:
    import numpy as np
    import soundfile as sf

    samples = np.asarray(audio, dtype=np.float32).squeeze()
    if samples.ndim != 1 or samples.size == 0:
        raise RuntimeError("Engine produced no audio")
    output = io.BytesIO()
    sf.write(output, samples, sample_rate, format="WAV", subtype="PCM_16")
    return AudioResult(output.getvalue(), sample_rate, samples.size / sample_rate)
