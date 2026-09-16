import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from chorus.engines.kokoro_fft import cuda_model


def stft_model(length=20, components=False, onesided=1, dtype=np.float32):
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(length) / length)).astype(dtype)
    tensor_type = TensorProto.FLOAT if dtype == np.float32 else TensorProto.DOUBLE
    graph = helper.make_graph(
        [
            helper.make_node(
                "STFT",
                ["signal", "step", "window", "length"],
                ["spectrum"],
                onesided=onesided,
            )
        ],
        "stft",
        [
            helper.make_tensor_value_info(
                "signal", tensor_type, [None, None, *([1] if components else [])]
            )
        ],
        [
            helper.make_tensor_value_info(
                "spectrum",
                tensor_type,
                [None, None, length // 2 + 1 if onesided else length, 2],
            )
        ],
        [
            numpy_helper.from_array(window, "window"),
            numpy_helper.from_array(np.array(5, np.int64), "step"),
            numpy_helper.from_array(np.array(length, np.int64), "length"),
        ],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9
    )


def session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])


class KokoroFftTests(unittest.TestCase):
    def compare(self, model, signals):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            onnx.save(model, path)
            original_bytes = path.read_bytes()
            reference = session(str(path))
            with cuda_model(path) as source:
                optimized = session(source)
            self.assertEqual(path.read_bytes(), original_bytes)
            for signal in signals:
                with self.subTest(shape=signal.shape, dtype=signal.dtype):
                    expected = reference.run(None, {"signal": signal})[0]
                    actual = optimized.run(None, {"signal": signal})[0]
                    # An approximate DFT passes loose tolerances but changes phase
                    # near zero enough to alter the vocoder's waveform.
                    np.testing.assert_array_equal(actual, expected)

    def test_preserves_float32_phase_at_frame_and_batch_boundaries(self):
        signals = [
            np.eye(20, dtype=np.float32),
            np.zeros((1, 21), np.float32),
            np.random.default_rng(813).normal(size=(2, 103)).astype(np.float32),
            np.sin(np.arange(127, dtype=np.float32) * np.float32(0.06))[None],
        ]
        for components in (False, True):
            with self.subTest(components=components):
                self.compare(
                    stft_model(components=components),
                    [signal[..., None] if components else signal for signal in signals],
                )

    def test_leaves_other_stft_contracts_unchanged(self):
        for length, onesided, dtype in (
            (32, 1, np.float32),
            (20, 0, np.float32),
            (20, 1, np.float64),
        ):
            with self.subTest(length=length, onesided=onesided, dtype=dtype):
                signal = np.random.default_rng(912).normal(size=(2, 83)).astype(dtype)
                self.compare(
                    stft_model(length, onesided=onesided, dtype=dtype), [signal]
                )


if __name__ == "__main__":
    unittest.main()
