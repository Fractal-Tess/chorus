"""Breeze TTS 2 adapter backed by an isolated CUDA worker.

Breeze requires CUDA PyTorch and NumPy 2, unlike Chorus's API environment.
The worker is started on the first request and kept alive for subsequent
requests. The ``voice`` argument is a natural-language voice description for
Breeze voice design/direction, not a named voice or a filesystem path.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import math
import os
import select
import subprocess
import threading
import uuid
from typing import Any

import soundfile as sf

from chorus.engines.base import Engine
from chorus.models import ROOT
from chorus.types import AudioResult

LOGGER = logging.getLogger("chorus.breeze")
_RUNTIME_DIR = ROOT / "runtimes" / "breeze"
_WORKER = _RUNTIME_DIR / "worker.py"
_PYTHON = _RUNTIME_DIR / ".venv" / "bin" / "python"
_ALLOWED_LANGUAGES = {
    "en",
    "en-us",
    "en-gb",
    "english",
    "zh",
    "zh-cn",
    "zh-hans",
    "chinese",
}
_CLOSE_TIMEOUT_SECONDS = 5.0


class Adapter(Engine):
    """One persistent Breeze worker for one local checkpoint and CUDA device."""

    def __init__(self, spec, device: str):
        super().__init__(spec, device)
        if not device.startswith("cuda:"):
            raise ValueError(f"Breeze TTS supports CUDA devices only, got {device!r}")
        try:
            device_index = int(device.split(":", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Invalid Breeze CUDA device: {device!r}") from exc
        if device_index < 0:
            raise ValueError(f"Invalid Breeze CUDA device: {device!r}")
        if not _WORKER.is_file():
            raise RuntimeError(f"Breeze worker is missing: {_WORKER}")
        if not _PYTHON.is_file():
            raise RuntimeError(
                f"Breeze runtime is not installed at {_PYTHON}; run the isolated "
                "Breeze setup command before selecting this engine"
            )
        for artifact in spec.manifest["artifacts"]:
            spec.artifact(artifact)
        self._process: subprocess.Popen[str] | None = None
        self._request_lock = threading.RLock()
        self._stderr_thread: threading.Thread | None = None

    def _read_stderr(self, stream: Any) -> None:
        try:
            for line in iter(stream.readline, ""):
                if line:
                    LOGGER.warning("Breeze worker: %s", line.rstrip())
        except (OSError, ValueError):
            return

    def _start_locked(self) -> subprocess.Popen[str]:
        process = self._process
        if process is not None and process.poll() is None:
            return process
        if process is not None:
            self._discard_process_locked(process)

        process = subprocess.Popen(
            [
                str(_PYTHON),
                str(_WORKER),
                "--checkpoint",
                str(self.spec.directory.resolve()),
                "--device",
                self.device,
            ],
            cwd=str(_RUNTIME_DIR),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env={
                **os.environ,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
            },
        )
        self._process = process
        assert process.stderr is not None
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            args=(process.stderr,),
            name="breeze-worker-stderr",
            daemon=True,
        )
        self._stderr_thread.start()
        return process

    @staticmethod
    def _discard_process_locked(process: subprocess.Popen[str]) -> None:
        try:
            process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        finally:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass

    @staticmethod
    def _read_line(
        process: subprocess.Popen[str], timeout: float | None = None
    ) -> dict[str, Any]:
        if process.stdout is None:
            raise RuntimeError("Breeze worker stdout is unavailable")
        if timeout is not None:
            ready, _, _ = select.select([process.stdout], [], [], timeout)
            if not ready:
                raise TimeoutError("Timed out waiting for Breeze worker shutdown")
        line = process.stdout.readline()
        if not line:
            code = process.poll()
            raise RuntimeError(
                "Breeze worker exited before returning a response"
                + (f" (exit code {code})" if code is not None else "")
            )
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Breeze worker returned invalid JSON") from exc
        if not isinstance(response, dict):
            raise RuntimeError("Breeze worker returned a non-object response")
        return response

    @staticmethod
    def _send(process: subprocess.Popen[str], payload: dict[str, Any]) -> None:
        if process.stdin is None:
            raise RuntimeError("Breeze worker stdin is unavailable")
        try:
            process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("Breeze worker pipe closed unexpectedly") from exc

    def _discard_current_locked(self) -> None:
        process = self._process
        self._process = None
        if process is not None:
            self._discard_process_locked(process)

    def _request_locked(self, payload: dict[str, Any]) -> dict[str, Any]:
        process = self._start_locked()
        try:
            self._send(process, payload)
            response = self._read_line(process)
        except Exception:
            self._discard_current_locked()
            raise
        if response.get("ok") is not True:
            error = response.get("error", "unknown Breeze worker error")
            raise RuntimeError(f"Breeze synthesis failed: {error}")
        return response

    def synthesize(
        self, text: str, voice: str | None, language: str | None, speed: float
    ) -> AudioResult:
        if voice is not None and not isinstance(voice, str):
            raise ValueError(
                "Breeze voice must be a natural-language description or null"
            )
        if language is not None:
            if not isinstance(language, str):
                raise ValueError("Breeze language must be a string or null")
            normalized_language = language.strip().lower()
            if normalized_language not in _ALLOWED_LANGUAGES:
                raise ValueError(
                    "Breeze TTS supports English and Chinese only; "
                    f"unsupported language: {language}"
                )
        if (
            isinstance(speed, bool)
            or not isinstance(speed, (int, float))
            or not math.isfinite(float(speed))
            or float(speed) != 1.0
        ):
            raise ValueError(
                "Breeze TTS does not support speed adjustment; speed must be 1.0"
            )

        request_id = uuid.uuid4().hex
        payload = {
            "op": "synthesize",
            "id": request_id,
            "text": text,
            "voice": voice,
            "language": language,
            "speed": speed,
        }
        with self._request_lock:
            response = self._request_locked(payload)

        if response.get("id") != request_id:
            raise RuntimeError("Breeze worker response id does not match request")
        encoded = response.get("audio_b64")
        if not isinstance(encoded, str):
            raise RuntimeError("Breeze worker returned no audio")
        try:
            wav = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise RuntimeError("Breeze worker returned invalid base64 audio") from exc
        if not wav:
            raise RuntimeError("Breeze worker returned empty audio")
        try:
            with sf.SoundFile(io.BytesIO(wav), mode="r") as audio_file:
                sample_rate = int(audio_file.samplerate)
                frames = int(audio_file.frames)
                channels = int(audio_file.channels)
        except Exception as exc:
            raise RuntimeError("Breeze worker returned invalid WAV audio") from exc
        if channels != 1 or frames <= 0:
            raise RuntimeError("Breeze worker returned empty or non-mono WAV audio")
        declared_rate = response.get("sample_rate")
        try:
            if declared_rate is not None and int(declared_rate) != sample_rate:
                raise RuntimeError(
                    "Breeze worker sample-rate metadata does not match WAV"
                )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Breeze worker returned invalid sample-rate metadata"
            ) from exc
        return AudioResult(
            wav=wav, sample_rate=sample_rate, duration=frames / sample_rate
        )

    def close(self) -> None:
        """Gracefully stop the persistent worker; safe to call repeatedly."""
        with self._request_lock:
            process = self._process
            if process is None:
                return
            self._process = None
            try:
                if process.poll() is None:
                    self._send(process, {"op": "close", "id": uuid.uuid4().hex})
                    self._read_line(process, timeout=_CLOSE_TIMEOUT_SECONDS)
            except (RuntimeError, TimeoutError, OSError):
                LOGGER.warning(
                    "Breeze worker did not acknowledge shutdown", exc_info=True
                )
            finally:
                for stream in (process.stdin, process.stdout):
                    if stream is not None:
                        try:
                            stream.close()
                        except OSError:
                            pass
                try:
                    process.wait(timeout=_CLOSE_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                if process.stderr is not None:
                    try:
                        process.stderr.close()
                    except OSError:
                        pass
