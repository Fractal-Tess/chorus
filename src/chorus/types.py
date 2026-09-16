from __future__ import annotations

import io
import subprocess
import wave
from dataclasses import dataclass, field
from typing import Any, Literal


def _encode_mp3(wav: bytes) -> bytes:
    try:
        import lameenc
    except ImportError as error:
        raise RuntimeError("lameenc is required for MP3 responses") from error

    try:
        with wave.open(io.BytesIO(wav), "rb") as source:
            channels = source.getnchannels()
            sample_rate = source.getframerate()
            sample_width = source.getsampwidth()
            pcm = source.readframes(source.getnframes())
    except (EOFError, wave.Error) as error:
        raise RuntimeError("MP3 encoding requires a valid WAV input") from error
    if sample_width != 2 or channels not in (1, 2):
        raise RuntimeError("MP3 encoding requires mono or stereo PCM16 WAV input")

    encoder = lameenc.Encoder()
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_out_sample_rate(sample_rate)
    encoder.set_channels(channels)
    encoder.set_bit_rate(128)
    # Match FFmpeg's observed LAME quality for this 128 kbps path.
    encoder.set_quality(3)
    encoded = encoder.encode(pcm)
    encoded.extend(encoder.flush())
    if not encoded:
        raise RuntimeError("MP3 encoding produced no output")
    return bytes(encoded)


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
    if response_format == "mp3":
        return _encode_mp3(audio.wav), audio.sample_rate
    options = {
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
        raise RuntimeError("FFmpeg is required for FLAC and Opus responses") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Audio encoding timed out") from error
    if result.returncode or not result.stdout:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg {response_format} encoding failed: {detail}")
    return result.stdout, 48_000 if response_format == "opus" else audio.sample_rate
