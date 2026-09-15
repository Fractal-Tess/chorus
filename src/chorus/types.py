from __future__ import annotations

import io
import subprocess
from dataclasses import dataclass, field
from typing import Any, Literal

ResponseFormat = Literal["wav", "mp3", "flac", "opus"]
AUDIO_MEDIA_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "opus": "audio/ogg",
}


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


def encode_response(
    audio: AudioResult, response_format: ResponseFormat
) -> tuple[bytes, int]:
    """Encode final PCM audio; return bytes and the output decoding sample rate."""
    if response_format == "wav":
        return audio.wav, audio.sample_rate
    options = {
        "mp3": ["-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3"],
        "flac": ["-c:a", "flac", "-f", "flac"],
        "opus": [
            "-c:a",
            "libopus",
            "-b:a",
            "64k",
            "-application",
            "voip",
            "-ar",
            "48000",
            "-f",
            "ogg",
        ],
    }
    if response_format not in options:
        raise ValueError(f"Unsupported audio format: {response_format}")
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-f",
                "wav",
                "-i",
                "pipe:0",
                "-map",
                "0:a:0",
                "-map_metadata",
                "-1",
                "-threads",
                "1",
                *options[response_format],
                "pipe:1",
            ],
            input=audio.wav,
            capture_output=True,
            check=False,
            timeout=120,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "FFmpeg is required for MP3, FLAC, and Opus responses"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Audio encoding timed out") from error
    if result.returncode or not result.stdout:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg {response_format} encoding failed: {detail}")
    return result.stdout, 48_000 if response_format == "opus" else audio.sample_rate
