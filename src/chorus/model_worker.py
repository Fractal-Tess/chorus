"""Lazy model processes with private, framed local RPC and bounded inference."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import logging
import os
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chorus.models import ModelSpec
    from chorus.types import AudioResult

LOGGER = logging.getLogger("chorus.model_worker")
_CLOSE_TIMEOUT_SECONDS = 5.0


def worker_concurrency(engine: str, device: str) -> int:
    """Only Kokoro's CUDA adapter supports overlapping independent runs."""
    return 2 if engine == "kokoro" and device.startswith("cuda:") else 1


def _serve(spec_payload: dict, device: str, socket_path: str) -> None:
    from chorus.models import ModelSpec
    from chorus.processing import Processor

    readiness = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    spec = ModelSpec(
        spec_payload["engine"],
        spec_payload["name"],
        Path(spec_payload["directory"]),
        spec_payload["manifest"],
    )
    adapter = None
    processor = None
    initialization = threading.Lock()
    module = importlib.import_module(f"chorus.engines.{spec.engine}")

    def synthesize(connection, request: dict) -> None:
        nonlocal adapter, processor
        try:
            started = time.perf_counter()
            with initialization:
                if adapter is None:
                    adapter = module.Adapter(spec, device)
                    processor = Processor(device)
            audio = adapter.synthesize(
                request["text"],
                request.get("voice"),
                request.get("language"),
                request["speed"],
            )
            inference_ms = (time.perf_counter() - started) * 1000
            force_align = request.get("force_align", False)
            audio = processor.process(
                audio,
                adapter.alignment_text(request["text"])
                if force_align
                else request["text"],
                lava_sr=request.get("lava_sr", False),
                force_align=force_align,
            )
            timings = {**audio.timings_ms, "inference": inference_ms}
            response = {
                "ok": True,
                "sample_rate": audio.sample_rate,
                "duration": audio.duration,
                "timings_ms": timings,
                "alignment": [asdict(word) for word in audio.alignment]
                if audio.alignment is not None
                else None,
            }
            connection.send_bytes(json.dumps(response, allow_nan=False).encode())
            connection.send_bytes(audio.wav)
        except Exception as error:
            LOGGER.exception("Model request failed")
            try:
                connection.send_bytes(
                    json.dumps(
                        {
                            "ok": False,
                            "type": "ValueError"
                            if isinstance(error, ValueError)
                            else "RuntimeError",
                            "message": str(error),
                        }
                    ).encode()
                )
            except (OSError, EOFError):
                pass
        finally:
            connection.close()

    try:
        # TemporaryDirectory in the parent makes this socket private (mode 0700).
        with Listener(socket_path, family="AF_UNIX") as listener:
            readiness.write("READY\n")
            readiness.close()
            with ThreadPoolExecutor(
                max_workers=worker_concurrency(spec.engine, device)
            ) as pool:
                while True:
                    connection = listener.accept()
                    try:
                        request = json.loads(connection.recv_bytes())
                        if request.get("op") == "close":
                            connection.send_bytes(b'{"ok":true}')
                            connection.close()
                            break
                        if request.get("op") != "synthesize":
                            raise ValueError("Unknown model worker operation")
                        pool.submit(synthesize, connection, request)
                    except Exception:
                        connection.close()
                        raise
    finally:
        if adapter is not None:
            close = getattr(adapter, "close", None)
            if close is not None:
                close()


def _terminate_process(process: subprocess.Popen) -> None:
    """Reap the leader and terminate its private group, including nested runtimes."""
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    for sig in (signal.SIGTERM, signal.SIGKILL):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, sig)
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            continue
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            stream.close()


class ModelWorker:
    """One lazily started adapter; each caller gets its own framed connection."""

    def __init__(self, spec: ModelSpec, device: str):
        self.spec = spec
        self.device = device
        self._process: subprocess.Popen | None = None
        self._directory: tempfile.TemporaryDirectory | None = None
        self._lock = threading.RLock()
        self._stderr_thread: threading.Thread | None = None

    @property
    def pid(self) -> int | None:
        process = self._process
        return process.pid if process is not None and process.poll() is None else None

    def _read_stderr(self, stream) -> None:
        try:
            for line in stream:
                LOGGER.warning("%s model worker: %s", self.spec.engine, line.rstrip())
        except (OSError, ValueError):
            pass

    def _start_locked(self) -> str:
        if self._process is not None and self._process.poll() is not None:
            self.close()
        if self._process is None:
            directory = tempfile.TemporaryDirectory(prefix="chorus-model-")
            self._directory = directory
            socket_path = str(Path(directory.name) / "rpc")
            spec_payload = {
                "engine": self.spec.engine,
                "name": self.spec.name,
                "directory": str(self.spec.directory),
                "manifest": self.spec.manifest,
            }
            child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            source_root = str(Path(__file__).resolve().parents[1])
            child_env["PYTHONPATH"] = os.pathsep.join(
                filter(None, [source_root, child_env.get("PYTHONPATH")])
            )
            try:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "chorus.model_worker",
                        "--spec",
                        json.dumps(spec_payload),
                        "--device",
                        self.device,
                        "--socket",
                        socket_path,
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    env=child_env,
                    start_new_session=True,
                )
                self._process = process
                self._stderr_thread = threading.Thread(
                    target=self._read_stderr,
                    args=(process.stderr,),
                    name=f"{self.spec.engine}-stderr",
                    daemon=True,
                )
                self._stderr_thread.start()
                ready, _, _ = select.select([process.stdout], [], [], 30)
                if not ready or process.stdout.readline().strip() != "READY":
                    raise RuntimeError("Model worker failed to start")
            except Exception:
                self.close()
                raise
        return str(Path(self._directory.name) / "rpc")

    def synthesize(
        self,
        text: str,
        voice: str | None,
        language: str | None,
        speed: float,
        lava_sr: bool = False,
        force_align: bool = False,
    ) -> AudioResult:
        from chorus.types import AudioResult, WordAlignment

        with self._lock:
            socket_path = self._start_locked()
            process = self._process
        request = {
            "op": "synthesize",
            "text": text,
            "voice": voice,
            "language": language,
            "speed": speed,
            "lava_sr": lava_sr,
            "force_align": force_align,
        }
        try:
            with Client(socket_path, family="AF_UNIX") as connection:
                connection.send_bytes(json.dumps(request, allow_nan=False).encode())
                response = json.loads(connection.recv_bytes())
                if not response.get("ok"):
                    if response.get("type") == "ValueError":
                        raise ValueError(response["message"])
                    raise RuntimeError(response["message"])
                wav = connection.recv_bytes()
        except (OSError, EOFError, RuntimeError) as error:
            with self._lock:
                # An older failed call must not close a replacement worker.
                if self._process is process:
                    self.close()
            if isinstance(error, (OSError, EOFError)):
                raise RuntimeError(
                    "Model worker connection closed unexpectedly"
                ) from error
            raise
        alignment = response["alignment"]
        return AudioResult(
            wav,
            response["sample_rate"],
            response["duration"],
            response["timings_ms"],
            tuple(WordAlignment(**word) for word in alignment)
            if alignment is not None
            else None,
            self.spec.name,
            self.device,
        )

    def close(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
            directory = self._directory
            self._directory = None
            try:
                if process is not None:
                    try:
                        if process.poll() is None and directory is not None:
                            with Client(
                                str(Path(directory.name) / "rpc"), family="AF_UNIX"
                            ) as connection:
                                connection.send_bytes(b'{"op":"close"}')
                                if connection.poll(_CLOSE_TIMEOUT_SECONDS):
                                    connection.recv_bytes()
                    except (OSError, EOFError):
                        pass
                    finally:
                        _terminate_process(process)
            finally:
                if directory is not None:
                    directory.cleanup()
                if self._stderr_thread is not None:
                    self._stderr_thread.join(timeout=0.2)
                    self._stderr_thread = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--socket", required=True)
    arguments = parser.parse_args()
    _serve(json.loads(arguments.spec), arguments.device, arguments.socket)
