"""Keep Kokoro's short STFT on CUDA without changing its phase rounding.

The float32 Bluestein/radix-2 operation order follows ONNX Runtime 1.26:
https://github.com/microsoft/onnxruntime/blob/v1.26.0/onnxruntime/core/providers/cpu/signal/dft.cc
A direct DFT is mathematically equivalent but changes near-zero signs and the
vocoder's phase features. Do not replace this with MatMul or fused arithmetic.

Adapted from ONNX Runtime, MIT License:
Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import ctypes
import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _bit_reverse(value: int, width: int) -> int:
    return int(f"{value:0{width}b}"[::-1], 2)


def _complex_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1],
            a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0],
        ),
        axis=-1,
    )


def _fft64(values: np.ndarray, twiddles: np.ndarray) -> np.ndarray:
    values = values[[_bit_reverse(index, 6) for index in range(64)]]
    unit = np.array([1, 0], np.float32)
    values = _complex_product(_complex_product(unit, values), unit)
    for level in range(1, 7):
        width = 1 << level
        midpoint = width // 2
        even = np.array(
            [(index // width) * width + index % midpoint for index in range(64)]
        )
        factors = [_bit_reverse(index % width, level) for index in range(64)]
        values = values[even] + _complex_product(
            twiddles[factors], values[even + midpoint]
        )
    return values


def _coefficients() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Use the same float libm functions as ORT, not double trig rounded to float.
    libm = ctypes.CDLL(None)
    for name in ("cosf", "sinf"):
        function = getattr(libm, name)
        function.argtypes = [ctypes.c_float]
        function.restype = ctypes.c_float
    pi = np.float32(math.pi)
    omega = np.float32(np.float32(-2) * pi / np.float32(64))
    twiddles = np.empty((64, 2), np.float32)
    for index in range(64):
        angle = np.float32(np.float32(index) * omega)
        twiddles[_bit_reverse(index, 6)] = [libm.cosf(angle), libm.sinf(angle)]
    chirp = np.zeros((64, 2), np.float32)
    for index in range(20):
        angle = np.float32(
            np.float32(np.float32(-pi * np.float32(index)) * np.float32(index))
            / np.float32(20)
        )
        chirp[index] = [libm.cosf(angle), libm.sinf(angle)]
    convolution = chirp.copy()
    convolution[:, 1] *= np.float32(-1)
    convolution[20:] = 0
    for index in range(45, 64):
        convolution[index] = convolution[64 - index]
    return twiddles, chirp, _fft64(convolution, twiddles)


def _lower_stft(node: onnx.NodeProto) -> tuple[list, list]:
    nodes, initializers = [], []
    prefix = node.output[0] + "/chorus_fft"
    serial = 0

    def constant(value):
        nonlocal serial
        serial += 1
        name = f"{prefix}/constant_{serial}"
        initializers.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    def operation(kind, *inputs, **attributes):
        nonlocal serial
        serial += 1
        name = f"{prefix}/{kind}_{serial}"
        nodes.append(
            helper.make_node(kind, list(inputs), [name], name=name, **attributes)
        )
        return name

    def multiply(a, b):
        ar, ai = a
        br, bi = b
        return (
            operation("Sub", operation("Mul", ar, br), operation("Mul", ai, bi)),
            operation("Add", operation("Mul", ar, bi), operation("Mul", ai, br)),
        )

    twiddles, chirp, convolution = _coefficients()
    zero = constant(np.float32(0))
    one = constant(np.float32(1))
    unit = one, zero

    def fft(real, imag):
        reverse = constant(np.array([_bit_reverse(i, 6) for i in range(64)], np.int64))
        values = (
            operation("Gather", real, reverse, axis=2),
            operation("Gather", imag, reverse, axis=2),
        )
        real, imag = multiply(multiply(unit, values), unit)
        for level in range(1, 7):
            width = 1 << level
            midpoint = width // 2
            even = np.array(
                [(i // width) * width + i % midpoint for i in range(64)], np.int64
            )
            factors = [_bit_reverse(i % width, level) for i in range(64)]
            even_name, odd_name = constant(even), constant(even + midpoint)
            er = operation("Gather", real, even_name, axis=2)
            ei = operation("Gather", imag, even_name, axis=2)
            product = multiply(
                (constant(twiddles[factors, 0]), constant(twiddles[factors, 1])),
                (
                    operation("Gather", real, odd_name, axis=2),
                    operation("Gather", imag, odd_name, axis=2),
                ),
            )
            real, imag = (
                operation("Add", er, product[0]),
                operation("Add", ei, product[1]),
            )
        return real, imag

    shape = operation("Shape", node.input[0])
    # A real STFT accepts both [batch, samples] and [batch, samples, 1].
    packed_shape = operation(
        "Slice",
        shape,
        constant(np.array([0], np.int64)),
        constant(np.array([2], np.int64)),
    )
    signal = operation("Reshape", node.input[0], packed_shape)
    length = operation("Gather", shape, constant(np.array(1, np.int64)), axis=0)
    end = operation("Sub", length, constant(np.array(19, np.int64)))
    starts = operation(
        "Range", constant(np.array(0, np.int64)), end, constant(np.array(5, np.int64))
    )
    starts = operation("Unsqueeze", starts, constant(np.array([1], np.int64)))
    indices = operation("Add", starts, constant(np.arange(20, dtype=np.int64)))
    frames = operation("Gather", signal, indices, axis=1)
    real = operation("Mul", frames, node.input[2])
    imag = operation("Mul", zero, node.input[2])
    real, imag = multiply(
        (real, imag), (constant(chirp[:20, 0]), constant(chirp[:20, 1]))
    )
    pads = constant(np.array([0, 0, 0, 0, 0, 44], np.int64))
    real, imag = operation("Pad", real, pads, zero), operation("Pad", imag, pads, zero)
    real, imag = fft(real, imag)
    real, imag = multiply(
        (real, imag), (constant(convolution[:, 0]), constant(convolution[:, 1]))
    )
    # ORT reuses forward twiddles for the inverse and reverses its output.
    real, imag = fft(real, imag)
    scale = constant(np.float32(64))
    real, imag = operation("Div", real, scale), operation("Div", imag, scale)
    bins = constant(np.array([0, *range(63, 53, -1)], np.int64))
    real, imag = (
        operation("Gather", real, bins, axis=2),
        operation("Gather", imag, bins, axis=2),
    )
    real, imag = multiply(
        (real, imag), (constant(chirp[:11, 0]), constant(chirp[:11, 1]))
    )
    real, imag = operation("Mul", real, one), operation("Mul", imag, one)
    axis = constant(np.array([3], np.int64))
    real, imag = operation("Unsqueeze", real, axis), operation("Unsqueeze", imag, axis)
    nodes.append(
        helper.make_node("Concat", [real, imag], list(node.output), axis=3, name=prefix)
    )
    return nodes, initializers


def _write_cuda_model(path: Path, destination: Path) -> bool:
    model = onnx.load(path)
    constants = {value.name: value for value in model.graph.initializer}
    nodes, additions, obsolete = [], [], set()
    for node in model.graph.node:
        matching = node.op_type == "STFT" and not node.domain and len(node.input) == 4
        if matching:
            step, window, length = (constants.get(name) for name in node.input[1:])
            matching = (
                step is not None
                and window is not None
                and length is not None
                and window.data_type == TensorProto.FLOAT
                and tuple(window.dims) == (20,)
                and numpy_helper.to_array(step).size == 1
                and numpy_helper.to_array(length).size == 1
                and numpy_helper.to_array(step).item() == 5
                and numpy_helper.to_array(length).item() == 20
                and all(a.name != "onesided" or a.i == 1 for a in node.attribute)
            )
        if not matching:
            nodes.append(node)
            continue
        replacement, initializers = _lower_stft(node)
        nodes.extend(replacement)
        additions.extend(initializers)
        obsolete.update((node.input[1], node.input[3]))
    if not additions:
        return False
    model.graph.ClearField("node")
    model.graph.node.extend(nodes)
    model.graph.initializer.extend(additions)
    used = set()
    pending = [model.graph]
    while pending:
        graph = pending.pop()
        used.update(value.name for value in (*graph.input, *graph.output))
        for node in graph.node:
            used.update(node.input)
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    pending.append(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    pending.extend(attribute.graphs)
    retained = [
        value
        for value in model.graph.initializer
        if value.name not in obsolete or value.name in used
    ]
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(retained)
    onnx.save(model, destination)
    return True


@contextmanager
def cuda_model(path: Path) -> Iterator[str]:
    """Load a derived graph without retaining a second serialized weight copy."""
    with TemporaryDirectory(prefix="chorus-kokoro-") as directory:
        optimized = Path(directory) / "model.onnx"
        yield str(optimized if _write_cuda_model(path, optimized) else path)
