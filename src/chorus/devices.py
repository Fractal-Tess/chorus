from __future__ import annotations

import ctypes
import json
import subprocess
import sys
from functools import lru_cache


@lru_cache(maxsize=1)
def available_devices() -> dict[str, str]:
    """Probe in a short-lived process so discovery retains no CUDA contexts."""
    result = subprocess.run(
        [sys.executable, "-m", "chorus.devices"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        raise RuntimeError(f"Device discovery failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def _probe_devices() -> dict[str, str]:
    import numpy as np
    import onnxruntime as ort

    result = {"cpu": "CPU"}
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return result
    ort.preload_dlls(directory="")
    try:
        cuda = ctypes.CDLL("libcudart.so.12")
        count = ctypes.c_int()
        if cuda.cudaGetDeviceCount(ctypes.byref(count)) != 0:
            return result
        for index in range(count.value):
            value = ort.OrtValue.ortvalue_from_numpy(
                np.zeros(1, dtype=np.float32), "cuda", index
            )
            result[f"cuda:{index}"] = f"CUDA device {index}"
            del value
    except (OSError, RuntimeError):
        return result
    return result


class DevicePolicy:
    def __init__(self, allowed: list[str] | None = None):
        available = {"cpu": "CPU"} if allowed == ["cpu"] else available_devices()
        self.allowed = list(
            dict.fromkeys(allowed if allowed is not None else available)
        )
        if not self.allowed:
            raise ValueError("At least one device must be enabled")
        for device in self.allowed:
            if device not in available:
                raise ValueError(
                    f"Device {device} is unavailable; available: {', '.join(available)}"
                )


if __name__ == "__main__":
    print(json.dumps(_probe_devices()))
