import concurrent.futures
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from chorus.channels import GpuQueueFullError
from chorus.devices import DevicePolicy
from chorus.registry import EngineRegistry
from chorus.types import AudioResult


class ControlledBackend:
    """Hold inference calls so admission order is deterministic without CUDA."""

    def __init__(self):
        self.condition = threading.Condition()
        self.started = {}
        self.released = set()
        self.stopping = False

    def worker(self, spec, device):
        backend = self

        class Worker:
            pid = None

            def synthesize(self, text, *args):
                with backend.condition:
                    backend.started[text] = device
                    backend.condition.notify_all()
                    if not backend.condition.wait_for(
                        lambda: text in backend.released or backend.stopping, timeout=10
                    ):
                        raise TimeoutError("Test inference was not released")
                if text == "fail":
                    raise RuntimeError("Inference failed")
                return AudioResult(b"", 24000, 1.0, {})

            def close(self):
                pass

        return Worker()

    def wait_started(self, label):
        with self.condition:
            if not self.condition.wait_for(lambda: label in self.started, timeout=2):
                raise AssertionError(f"Request {label} did not start")
            return self.started[label]

    def release(self, label):
        with self.condition:
            self.released.add(label)
            self.condition.notify_all()

    def release_all(self):
        with self.condition:
            self.stopping = True
            self.condition.notify_all()


class GpuQueueTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        config = Path(directory.name) / "channels.toml"
        config.write_text("[models]\n")
        hardware = patch(
            "chorus.devices.available_devices",
            return_value={"cpu": "CPU", "cuda:0": "GPU", "cuda:1": "GPU"},
        )
        hardware.start()
        self.addCleanup(hardware.stop)
        telemetry = patch("chorus.registry.gpu_usage", return_value={})
        telemetry.start()
        self.addCleanup(telemetry.stop)
        self.backend = ControlledBackend()
        workers = patch("chorus.registry.ModelWorker", side_effect=self.backend.worker)
        workers.start()
        self.addCleanup(workers.stop)
        self.registry = EngineRegistry(
            policy=DevicePolicy(["cpu", "cuda:0", "cuda:1"]),
            channel_config=config,
            gpu_queue_size=2,
        )
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.addCleanup(self.pool.shutdown)
        self.addCleanup(self.registry.close)
        self.addCleanup(self.backend.release_all)

    def submit(self, label, engine="kokoro", channel="gpu"):
        return self.pool.submit(
            self.registry.synthesize, engine, label, channel=channel
        )

    def wait_queued(self, count):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if self.registry.resource_status()["gpu_queue"]["waiting"] == count:
                return
            time.sleep(0.001)
        self.fail(f"Expected {count} queued requests")

    def occupy_gpus(self):
        futures = {}
        for label in ("one", "two", "three", "four"):
            futures[label] = self.submit(label)
            self.backend.wait_started(label)
        return futures

    def test_fifo_dispatch_uses_next_free_gpu_and_rejects_overflow(self):
        active = self.occupy_gpus()
        first = self.submit("first")
        self.wait_queued(1)
        second = self.submit("second")
        self.wait_queued(2)
        with self.assertRaises(GpuQueueFullError):
            self.registry.synthesize("kokoro", "overflow", channel="gpu")
        # GPU 0 stays occupied. Preassigning queued requests to it would stall.
        self.backend.release("two")
        active["two"].result(timeout=2)
        self.assertEqual(self.backend.wait_started("first"), "cuda:1")
        self.backend.release("four")
        active["four"].result(timeout=2)
        self.assertEqual(self.backend.wait_started("second"), "cuda:1")
        self.backend.release("first")
        self.backend.release("second")
        self.assertEqual(first.result(timeout=2).device, "cuda:1")
        self.assertEqual(second.result(timeout=2).device, "cuda:1")

    def test_large_model_has_exclusive_gpu_without_blocking_other_gpu(self):
        active = self.occupy_gpus()
        large = self.submit("large", engine="breeze")
        self.wait_queued(1)
        small = self.submit("small")
        self.wait_queued(2)
        self.backend.release("two")
        active["two"].result(timeout=2)
        self.assertEqual(self.registry.resource_status()["gpu_queue"]["waiting"], 2)
        self.backend.release("four")
        active["four"].result(timeout=2)
        self.assertEqual(self.backend.wait_started("large"), "cuda:1")
        self.backend.release("one")
        active["one"].result(timeout=2)
        self.assertEqual(self.backend.wait_started("small"), "cuda:0")
        self.backend.release("large")
        self.backend.release("small")
        self.assertEqual(large.result(timeout=2).device, "cuda:1")
        self.assertEqual(small.result(timeout=2).device, "cuda:0")

    def test_shutdown_wakes_waiters_while_active_calls_drain(self):
        self.occupy_gpus()
        waiting = self.submit("waiting")
        self.wait_queued(1)
        closing = self.pool.submit(self.registry.close)
        with self.assertRaises(RuntimeError):
            waiting.result(timeout=2)
        self.backend.release_all()
        closing.result(timeout=2)
        with self.assertRaises(RuntimeError):
            self.registry.synthesize("kokoro", "after close", channel="gpu")

    def test_zero_waiting_slots_does_not_block_cpu_channel(self):
        self.registry.gpu_queue_size = 0
        self.occupy_gpus()
        with self.assertRaises(GpuQueueFullError):
            self.registry.synthesize("kokoro", "overflow", channel="gpu")
        cpu = self.submit("cpu", channel="cpu")
        self.assertEqual(self.backend.wait_started("cpu"), "cpu")
        self.backend.release("cpu")
        self.assertEqual(cpu.result(timeout=2).device, "cpu")

    def test_inference_failure_releases_gpu_for_waiting_work(self):
        failed = self.submit("fail", engine="breeze")
        self.backend.wait_started("fail")
        other = self.submit("other", engine="breeze")
        self.backend.wait_started("other")
        waiting = self.submit("waiting", engine="breeze")
        self.wait_queued(1)
        self.backend.release("fail")
        with self.assertRaises(RuntimeError):
            failed.result(timeout=2)
        self.assertEqual(self.backend.wait_started("waiting"), "cuda:0")
        self.backend.release("waiting")
        self.assertEqual(waiting.result(timeout=2).device, "cuda:0")
        self.backend.release("other")
        other.result(timeout=2)


if __name__ == "__main__":
    unittest.main()
