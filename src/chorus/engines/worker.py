"""Persistent JSONL transport for isolated, local CUDA speech runtimes."""

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

LOGGER = logging.getLogger("chorus.worker")
_CLOSE_TIMEOUT_SECONDS = 5.0


class WorkerEngine(Engine):
    """One persistent worker for one local checkpoint and CUDA device."""

    def __init__(self, spec, device: str):
        super().__init__(spec, device)
        if not device.startswith("cuda:"):
            raise ValueError(
                f"{spec.engine} supports CUDA devices only, got {device!r}"
            )
        try:
            device_index = int(device.split(":", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Invalid CUDA device: {device!r}") from exc
        if device_index < 0:
            raise ValueError(f"Invalid CUDA device: {device!r}")
        self._runtime_dir = ROOT / "runtimes" / spec.engine
        self._worker = self._runtime_dir / "worker.py"
        self._python = self._runtime_dir / ".venv" / "bin" / "python"
        if not self._worker.is_file():
            raise RuntimeError(f"{spec.engine} worker is missing: {self._worker}")
        if not self._python.is_file():
            raise RuntimeError(
                f"{spec.engine} runtime is not installed at {self._python}; "
                f"run uv sync --project runtimes/{spec.engine} --locked"
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
                    LOGGER.warning("%s worker: %s", self.spec.engine, line.rstrip())
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
                str(self._python),
                str(self._worker),
                "--checkpoint",
                str(self.spec.directory.resolve()),
                "--device",
                self.device,
            ],
            cwd=str(self._runtime_dir),
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
            name=f"{self.spec.engine}-worker-stderr",
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
            raise RuntimeError("TTS worker stdout is unavailable")
        if timeout is not None:
            ready, _, _ = select.select([process.stdout], [], [], timeout)
            if not ready:
                raise TimeoutError("Timed out waiting for TTS worker shutdown")
        line = process.stdout.readline()
        if not line:
            code = process.poll()
            raise RuntimeError(
                "TTS worker exited before returning a response"
                + (f" (exit code {code})" if code is not None else "")
            )
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError("TTS worker returned invalid JSON") from exc
        if not isinstance(response, dict):
            raise RuntimeError("TTS worker returned a non-object response")
        return response

    @staticmethod
    def _send(process: subprocess.Popen[str], payload: dict[str, Any]) -> None:
        if process.stdin is None:
            raise RuntimeError("TTS worker stdin is unavailable")
        try:
            process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("TTS worker pipe closed unexpectedly") from exc

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
            error = response.get("error", "unknown worker error")
            raise RuntimeError(f"{self.spec.engine} synthesis failed: {error}")
        return response

    def validate_request(
        self, text: str, voice: str | None, language: str | None, speed: float
    ) -> None:
        if voice is not None and not isinstance(voice, str):
            raise ValueError("Voice must be a natural-language description or null")
        if language is not None and not isinstance(language, str):
            raise ValueError("Language must be a string or null")
        if (
            isinstance(speed, bool)
            or not isinstance(speed, (int, float))
            or not math.isfinite(float(speed))
            or float(speed) != 1.0
        ):
            raise ValueError(
                f"{self.spec.engine} does not support speed adjustment; speed must be 1.0"
            )

    def synthesize(
        self, text: str, voice: str | None, language: str | None, speed: float
    ) -> AudioResult:
        self.validate_request(text, voice, language, speed)
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
            raise RuntimeError("TTS worker response id does not match request")
        encoded = response.get("audio_b64")
        if not isinstance(encoded, str):
            raise RuntimeError("TTS worker returned no audio")
        try:
            wav = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise RuntimeError("TTS worker returned invalid base64 audio") from exc
        if not wav:
            raise RuntimeError("TTS worker returned empty audio")
        try:
            with sf.SoundFile(io.BytesIO(wav), mode="r") as audio_file:
                sample_rate = int(audio_file.samplerate)
                frames = int(audio_file.frames)
                channels = int(audio_file.channels)
        except Exception as exc:
            raise RuntimeError("TTS worker returned invalid WAV audio") from exc
        if channels != 1 or frames <= 0:
            raise RuntimeError("TTS worker returned empty or non-mono WAV audio")
        declared_rate = response.get("sample_rate")
        try:
            if declared_rate is not None and int(declared_rate) != sample_rate:
                raise RuntimeError("TTS worker sample-rate metadata does not match WAV")
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "TTS worker returned invalid sample-rate metadata"
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
                    "%s worker did not acknowledge shutdown",
                    self.spec.engine,
                    exc_info=True,
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
