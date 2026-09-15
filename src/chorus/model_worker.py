"""Lazy, persistent isolated model workers.

The parent process intentionally imports only standard-library modules from this
module.  Adapter and post-processing imports happen in the child process after
its first synthesis request, keeping CUDA and PyTorch out of the API process.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import importlib
import json
import logging
import math
import os
import select
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chorus.models import ModelSpec
    from chorus.types import AudioResult

LOGGER = logging.getLogger("chorus.model_worker")
_CLOSE_TIMEOUT_SECONDS = 5.0
_KILL_GRACE_SECONDS = 1.0


def _exception_payload(error: Exception) -> dict[str, str]:
    """Return the deliberately small exception contract used over JSON."""
    error_type = "ValueError" if isinstance(error, ValueError) else "RuntimeError"
    message = str(error) or error.__class__.__name__
    return {"type": error_type, "message": message}


def _close_object(value: Any) -> None:
    close = getattr(value, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception:
        # Cleanup must continue for the other owner.  The parent will have
        # already received the request result (or the original exception).
        LOGGER.warning("model worker resource close failed", exc_info=True)


def _child_spec(payload: dict[str, Any]) -> Any:
    # Importing models in the child is cheap and, importantly, does not import
    # an adapter or initialize a CUDA runtime.
    from chorus.models import ModelSpec

    return ModelSpec(
        engine=str(payload["engine"]),
        name=str(payload["name"]),
        directory=Path(str(payload["directory"])),
        manifest=dict(payload["manifest"]),
    )


def _audio_payload(audio: Any) -> dict[str, Any]:
    """Convert AudioResult to JSON without using pickle on the local pipe."""
    from chorus.types import AudioResult

    if not isinstance(audio, AudioResult):
        raise RuntimeError("Engine returned an invalid audio result")
    alignment = None
    if audio.alignment is not None:
        alignment = [
            {
                "word": item.word,
                "start_ms": int(item.start_ms),
                "end_ms": int(item.end_ms),
                "score": float(item.score),
            }
            for item in audio.alignment
        ]
    timings: dict[str, float] = {}
    for key, value in audio.timings_ms.items():
        number = float(value)
        if not math.isfinite(number):
            raise RuntimeError("Engine returned non-finite timing metadata")
        timings[str(key)] = number
    return {
        "audio_b64": base64.b64encode(audio.wav).decode("ascii"),
        "sample_rate": int(audio.sample_rate),
        "duration": float(audio.duration),
        "timings_ms": timings,
        "alignment": alignment,
        "model": str(audio.model),
        "device": str(audio.device),
    }


def _child_main(spec_payload: dict[str, Any], device: str) -> int:
    """Serve requests in the isolated child until close or a failed request."""
    # Keep every third-party diagnostic off the JSON protocol stream.
    protocol_stdout = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    with contextlib.redirect_stdout(sys.stderr):
        module = importlib.import_module(f"chorus.engines.{spec_payload['engine']}")
        spec = _child_spec(spec_payload)
        adapter: Any = None
        processor: Any = None

        def respond(response: dict[str, Any]) -> None:
            protocol_stdout.write(
                json.dumps(response, separators=(",", ":"), allow_nan=False) + "\n"
            )
            protocol_stdout.flush()

        try:
            for line in sys.stdin:
                request: Any = None
                try:
                    request = json.loads(line)
                    if not isinstance(request, dict):
                        raise RuntimeError("Model worker request must be an object")
                    request_id = request.get("id")
                    operation = request.get("op")
                    if operation == "close":
                        respond({"ok": True, "id": request_id})
                        return 0
                    if operation != "synthesize":
                        raise RuntimeError(
                            f"Unknown model worker operation: {operation!r}"
                        )

                    # Include lazy adapter/processor construction in the cold
                    # request boundary, but retain both for warm requests.
                    started = time.perf_counter()
                    if adapter is None:
                        adapter = module.Adapter(spec, device)
                    if processor is None:
                        processor_class = importlib.import_module(
                            "chorus.processing"
                        ).Processor
                        processor = processor_class()

                    audio = adapter.synthesize(
                        request["text"],
                        request.get("voice"),
                        request.get("language"),
                        request["speed"],
                    )
                    inference_ms = (time.perf_counter() - started) * 1_000
                    force_align = bool(request.get("force_align", False))
                    lava_sr = bool(request.get("lava_sr", False))
                    processed = processor.process(
                        audio,
                        adapter.alignment_text(request["text"])
                        if force_align
                        else request["text"],
                        lava_sr=lava_sr,
                        force_align=force_align,
                    )
                    # Processor owns lava/alignment timings.  The worker owns
                    # the inference and end-to-end backend boundaries.
                    from dataclasses import replace

                    timings = dict(processed.timings_ms)
                    timings["inference"] = inference_ms
                    timings.setdefault("lava_sr", 0.0)
                    timings.setdefault("alignment", 0.0)
                    timings["total"] = (time.perf_counter() - started) * 1_000
                    processed = replace(processed, timings_ms=timings)
                    respond({"ok": True, "id": request_id, **_audio_payload(processed)})
                except Exception as error:
                    # A failed initialization/inference may leave a backend in
                    # an unknown state. Return its class where supported, then
                    # terminate so the parent cannot accidentally reuse it.
                    respond(
                        {
                            "ok": False,
                            "id": request.get("id")
                            if isinstance(request, dict)
                            else None,
                            "error": _exception_payload(error),
                        }
                    )
                    return 1
        finally:
            # Adapter first: WorkerEngine adapters own nested runtime
            # processes.
            _close_object(adapter)
            _close_object(processor)
    return 0


def _terminate_process(process: subprocess.Popen[str], timeout: float) -> None:
    """Stop a worker and descendants, bounded by ``timeout`` seconds."""
    budget = max(0.0, timeout)
    # Reserve time for escalation rather than spending the entire budget in
    # the initial wait.  Signalling the private process group also handles
    # nested runtime workers after the direct leader has already exited.
    grace = min(_KILL_GRACE_SECONDS, budget / 3)
    initial = max(0.0, budget - 2 * grace)
    try:
        process.wait(timeout=initial)
    except subprocess.TimeoutExpired:
        pass

    def signal_group(signum: int) -> None:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signum)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        elif process.poll() is None:
            try:
                if signum == signal.SIGKILL:
                    process.kill()
                else:
                    process.terminate()
            except OSError:
                pass

    signal_group(signal.SIGTERM)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # Always escalate after the grace interval on POSIX: the leader may have
    # exited while a nested child still holds the process group.
    signal_group(signal.SIGKILL)
    try:
        process.wait(timeout=max(0.0, budget - initial - grace))
    except subprocess.TimeoutExpired:
        LOGGER.warning("model worker did not exit after forced termination")


def _close_stream(stream: Any) -> None:
    if stream is None:
        return
    try:
        stream.close()
    except (OSError, ValueError):
        pass


class ModelWorker:
    """One lazily started, persistent adapter and processor pair.

    The parent uses a JSONL pipe rather than pickle.  Calls are serialized by a
    lock; registry-level locking additionally prevents close during synthesis.
    """

    def __init__(self, spec: ModelSpec, device: str):
        self.spec = spec
        self.device = device
        self._process: subprocess.Popen[str] | None = None
        self._request_lock = threading.RLock()
        self._stderr_thread: threading.Thread | None = None

    @property
    def pid(self) -> int | None:
        process = self._process
        if process is None:
            return None
        return int(process.pid)

    def _read_stderr(self, stream: Any) -> None:
        try:
            for line in iter(stream.readline, ""):
                if line:
                    LOGGER.warning(
                        "%s model worker: %s", self.spec.engine, line.rstrip()
                    )
        except (OSError, ValueError):
            return

    def _start_locked(self) -> subprocess.Popen[str]:
        process = self._process
        if process is not None and process.poll() is None:
            return process
        if process is not None:
            self._discard_current_locked(process)

        spec_payload = {
            "engine": str(self.spec.engine),
            "name": str(self.spec.name),
            "directory": str(self.spec.directory),
            "manifest": self.spec.manifest,
        }
        try:
            encoded_spec = json.dumps(
                spec_payload, separators=(",", ":"), allow_nan=False
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "Model specification is not JSON serializable"
            ) from error
        child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        source_root = str(Path(__file__).resolve().parents[1])
        if child_env.get("PYTHONPATH"):
            child_env["PYTHONPATH"] = source_root + os.pathsep + child_env["PYTHONPATH"]
        else:
            child_env["PYTHONPATH"] = source_root
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "chorus.model_worker",
                "--child",
                "--spec",
                encoded_spec,
                "--device",
                str(self.device),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=child_env,
            start_new_session=(os.name == "posix"),
        )
        self._process = process
        if process.stderr is not None:
            self._stderr_thread = threading.Thread(
                target=self._read_stderr,
                args=(process.stderr,),
                name=f"{self.spec.engine}-model-worker-stderr",
                daemon=True,
            )
            self._stderr_thread.start()
        return process

    def _discard_current_locked(
        self, process: subprocess.Popen[str] | None = None
    ) -> None:
        current = process if process is not None else self._process
        if current is self._process:
            self._process = None
        if current is None:
            return
        _terminate_process(current, _CLOSE_TIMEOUT_SECONDS)
        _close_stream(current.stdin)
        _close_stream(current.stdout)
        _close_stream(current.stderr)

    @staticmethod
    def _send(process: subprocess.Popen[str], payload: dict[str, Any]) -> None:
        if process.stdin is None:
            raise RuntimeError("Model worker stdin is unavailable")
        try:
            process.stdin.write(
                json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n"
            )
            process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as error:
            raise RuntimeError("Model worker pipe closed unexpectedly") from error

    @staticmethod
    def _read_line(
        process: subprocess.Popen[str], timeout: float | None = None
    ) -> dict[str, Any]:
        if process.stdout is None:
            raise RuntimeError("Model worker stdout is unavailable")
        if timeout is not None:
            ready, _, _ = select.select([process.stdout], [], [], timeout)
            if not ready:
                raise TimeoutError("Timed out waiting for model worker shutdown")
        line = process.stdout.readline()
        if not line:
            code = process.poll()
            raise RuntimeError(
                "Model worker exited before returning a response"
                + (f" (exit code {code})" if code is not None else "")
            )
        try:
            response = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError("Model worker returned invalid JSON") from error
        if not isinstance(response, dict):
            raise RuntimeError("Model worker returned a non-object response")
        return response

    @staticmethod
    def _raise_worker_error(response: dict[str, Any]) -> None:
        payload = response.get("error")
        if isinstance(payload, dict):
            message = str(payload.get("message", "unknown model worker error"))
            if payload.get("type") == "ValueError":
                raise ValueError(message)
            raise RuntimeError(message)
        raise RuntimeError(f"{response.get('error', 'unknown model worker error')}")

    def _request_locked(self, payload: dict[str, Any]) -> dict[str, Any]:
        process = self._start_locked()
        try:
            self._send(process, payload)
            response = self._read_line(process)
        except Exception:
            self._discard_current_locked(process)
            raise
        if response.get("ok") is not True:
            self._discard_current_locked(process)
            self._raise_worker_error(response)
        return response

    def _result_from_response(self, response: dict[str, Any]) -> AudioResult:
        from chorus.types import AudioResult, WordAlignment

        if not isinstance(response.get("audio_b64"), str):
            raise RuntimeError("Model worker returned no audio")
        try:
            wav = base64.b64decode(response["audio_b64"], validate=True)
        except (ValueError, binascii.Error) as error:
            raise RuntimeError("Model worker returned invalid base64 audio") from error
        if not wav:
            raise RuntimeError("Model worker returned empty audio")
        try:
            sample_rate = int(response["sample_rate"])
            duration = float(response["duration"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "Model worker returned invalid audio metadata"
            ) from error
        raw_timings = response.get("timings_ms", {})
        if not isinstance(raw_timings, dict):
            raise RuntimeError("Model worker returned invalid timing metadata")
        try:
            timings = {str(key): float(value) for key, value in raw_timings.items()}
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "Model worker returned invalid timing metadata"
            ) from error
        raw_alignment = response.get("alignment")
        alignment = None
        if raw_alignment is not None:
            if not isinstance(raw_alignment, list):
                raise RuntimeError("Model worker returned invalid alignment metadata")
            try:
                alignment = tuple(
                    WordAlignment(
                        word=str(item["word"]),
                        start_ms=int(item["start_ms"]),
                        end_ms=int(item["end_ms"]),
                        score=float(item["score"]),
                    )
                    for item in raw_alignment
                )
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    "Model worker returned invalid alignment metadata"
                ) from error
        return AudioResult(
            wav=wav,
            sample_rate=sample_rate,
            duration=duration,
            timings_ms=timings,
            alignment=alignment,
            model=str(response.get("model", "")),
            device=str(response.get("device", self.device)),
        )

    def synthesize(
        self,
        text: str,
        voice: str | None,
        language: str | None,
        speed: float,
        lava_sr: bool = False,
        force_align: bool = False,
    ) -> AudioResult:
        request_id = uuid.uuid4().hex
        payload = {
            "op": "synthesize",
            "id": request_id,
            "text": text,
            "voice": voice,
            "language": language,
            "speed": speed,
            "lava_sr": lava_sr,
            "force_align": force_align,
        }
        with self._request_lock:
            response = self._request_locked(payload)
        if response.get("id") != request_id:
            self.close()
            raise RuntimeError("Model worker response id does not match request")
        try:
            return self._result_from_response(response)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Gracefully stop the child and its nested runtime process tree."""
        with self._request_lock:
            process = self._process
            if process is None:
                return
            self._process = None
            try:
                if process.poll() is None:
                    try:
                        self._send(process, {"op": "close", "id": uuid.uuid4().hex})
                        self._read_line(process, timeout=_CLOSE_TIMEOUT_SECONDS)
                    except (RuntimeError, TimeoutError, OSError, ValueError):
                        LOGGER.warning(
                            "%s model worker did not acknowledge shutdown",
                            self.spec.engine,
                            exc_info=True,
                        )
            finally:
                _terminate_process(process, _CLOSE_TIMEOUT_SECONDS)
                _close_stream(process.stdin)
                _close_stream(process.stdout)
                _close_stream(process.stderr)
                stderr_thread = self._stderr_thread
                self._stderr_thread = None
                if stderr_thread is not None:
                    stderr_thread.join(timeout=0.2)


def _main() -> int:
    if "--child" not in sys.argv:
        return 2
    try:
        spec_payload = json.loads(sys.argv[sys.argv.index("--spec") + 1])
        device = sys.argv[sys.argv.index("--device") + 1]
        return _child_main(spec_payload, device)
    except Exception as error:
        # Startup failures before the request loop have no request id to
        # attach.  stderr is intentionally used; the parent turns EOF into a
        # RuntimeError while preserving the useful child diagnostic in logs.
        print(f"model worker startup failed: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
