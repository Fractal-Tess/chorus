"""Transport tests for WorkerEngine against a fake runtime worker.

These exercise the real JSONL protocol and process lifecycle over a stub
worker script, so no GPU, runtime virtualenv, or model weights are needed.
"""

import io
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from chorus.engines.worker import WorkerEngine
from chorus.models import ModelSpec

SAMPLE_RATE = 24_000
FRAMES = 12_000

# Modes: "ok" answers normally, "error" reports failure, "crash" exits before
# responding, "deaf" answers requests but ignores the shutdown handshake.
FAKE_WORKER = '''
import base64, io, json, sys, wave

MODE = sys.argv[sys.argv.index("--mode") + 1]


def wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate({rate})
        handle.writeframes(b"\\x00\\x00" * {frames})
    return buffer.getvalue()


def reply(payload):
    sys.stdout.write(json.dumps(payload) + "\\n")
    sys.stdout.flush()


while True:
    line = sys.stdin.readline()
    if not line:
        break
    request = json.loads(line)
    if request.get("op") == "close":
        if MODE == "deaf":
            continue
        reply({{"ok": True, "id": request["id"], "closed": True}})
        break
    if MODE == "crash":
        sys.exit(9)
    if MODE == "error":
        reply({{"ok": False, "id": request["id"], "error": "runtime exploded"}})
        continue
    reply({{
        "ok": True,
        "id": request["id"],
        "sample_rate": {rate},
        "audio_b64": base64.b64encode(wav_bytes()).decode("ascii"),
    }})
'''.format(rate=SAMPLE_RATE, frames=FRAMES)


class WorkerEngineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

        model_dir = self.root / "models/breeze/test"
        model_dir.mkdir(parents=True)
        (model_dir / "weights").write_bytes(b"not really a checkpoint")
        self.spec = ModelSpec(
            "breeze",
            "test",
            model_dir,
            {"devices": ["cuda:0"], "artifacts": {"weights": {}}},
        )
        patcher = patch("chorus.engines.worker.ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.runtime_dir = self.root / "runtimes/breeze"

    def install_runtime(self, mode: str) -> None:
        """Write a stub worker; the engine launches it on first use."""
        binaries = self.runtime_dir / ".venv/bin"
        if not binaries.exists():
            binaries.mkdir(parents=True)
            (binaries / "python").symlink_to(sys.executable)
        # The engine passes --checkpoint and --device; the stub reads --mode.
        (self.runtime_dir / "worker.py").write_text(
            f"import sys\nsys.argv += ['--mode', {mode!r}]\n{FAKE_WORKER}"
        )

    def engine(self, mode: str = "ok") -> WorkerEngine:
        self.install_runtime(mode)
        engine = WorkerEngine(self.spec, "cuda:0")
        self.addCleanup(engine.close)
        return engine

    def test_rejects_non_cuda_devices(self):
        self.install_runtime("ok")
        with self.assertRaises(ValueError):
            WorkerEngine(self.spec, "cpu")

    def test_reports_a_missing_runtime_virtualenv(self):
        self.runtime_dir.mkdir(parents=True)
        (self.runtime_dir / "worker.py").write_text("")
        with self.assertRaises(RuntimeError) as caught:
            WorkerEngine(self.spec, "cuda:0")
        self.assertIn("uv sync --project runtimes/breeze", str(caught.exception))

    def test_construction_does_not_start_the_worker(self):
        self.assertIsNone(self.engine()._process)

    def test_round_trip_returns_decoded_audio(self):
        audio = self.engine().synthesize("hello there", None, None, 1.0)
        self.assertEqual(audio.sample_rate, SAMPLE_RATE)
        self.assertAlmostEqual(audio.duration, FRAMES / SAMPLE_RATE)
        with wave.open(io.BytesIO(audio.wav)) as handle:
            self.assertEqual(handle.getnframes(), FRAMES)

    def test_worker_is_reused_across_requests(self):
        engine = self.engine()
        engine.synthesize("first", None, None, 1.0)
        pid = engine._process.pid
        engine.synthesize("second", None, None, 1.0)
        self.assertEqual(engine._process.pid, pid)

    def test_worker_errors_surface_as_runtime_errors(self):
        with self.assertRaises(RuntimeError) as caught:
            self.engine("error").synthesize("boom", None, None, 1.0)
        self.assertIn("runtime exploded", str(caught.exception))

    def test_a_crash_discards_the_process_so_the_next_call_restarts_it(self):
        engine = self.engine("crash")
        with self.assertRaises(RuntimeError):
            engine.synthesize("boom", None, None, 1.0)
        self.assertIsNone(engine._process)

        self.install_runtime("ok")
        self.assertEqual(
            engine.synthesize("recovered", None, None, 1.0).sample_rate, SAMPLE_RATE
        )

    def test_close_stops_the_worker_and_is_repeatable(self):
        engine = self.engine()
        engine.synthesize("hello", None, None, 1.0)
        process = engine._process
        engine.close()
        self.assertIsNone(engine._process)
        self.assertIsNotNone(process.poll())
        engine.close()

    def test_close_terminates_a_worker_that_ignores_the_handshake(self):
        engine = self.engine("deaf")
        engine.synthesize("hello", None, None, 1.0)
        process = engine._process
        with (
            patch("chorus.engines.worker._CLOSE_TIMEOUT_SECONDS", 0.3),
            self.assertLogs("chorus.worker", "WARNING"),
        ):
            engine.close()
        self.assertIsNone(engine._process)
        self.assertIsNotNone(process.poll())

    def test_close_without_a_started_worker_is_a_no_op(self):
        engine = self.engine()
        engine.close()
        self.assertIsNone(engine._process)

    def test_speed_other_than_one_is_rejected(self):
        with self.assertRaises(ValueError):
            self.engine().synthesize("hello", None, None, 1.5)


if __name__ == "__main__":
    unittest.main()
