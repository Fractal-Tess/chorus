"""Process and NVIDIA resource telemetry for isolated model workers.

The functions in this module are deliberately pull-based.  A caller samples
resources when it is holding its own lifecycle lock; this module does not
start a polling thread or retain a telemetry cache.
"""

from __future__ import annotations

import csv
import errno
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

_PROC_ROOT = Path("/proc")
_NVIDIA_SMI = "nvidia-smi"
_CUDA_UUID_SCRIPT = r"""
import ctypes
import json

cuda = ctypes.CDLL("libcuda.so.1")
if cuda.cuInit(0) != 0:
    raise RuntimeError("cuInit failed")
count = ctypes.c_int()
cuda.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
cuda.cuDeviceGetCount.restype = ctypes.c_int
if cuda.cuDeviceGetCount(ctypes.byref(count)) != 0:
    raise RuntimeError("cuDeviceGetCount failed")

class UUID(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_ubyte * 16)]

cuda.cuDeviceGetUuid.argtypes = [ctypes.POINTER(UUID), ctypes.c_int]
cuda.cuDeviceGetUuid.restype = ctypes.c_int
uuids = []
for index in range(count.value):
    value = UUID()
    if cuda.cuDeviceGetUuid(ctypes.byref(value), index) != 0:
        raise RuntimeError("cuDeviceGetUuid failed")
    uuids.append("GPU-" + bytes(value.bytes).hex())
print(json.dumps(uuids))
"""
_NVIDIA_TIMEOUT_SECONDS = 5.0
_RSS_RE = re.compile(r"^VmRSS:\s*(?P<value>\d+)\s+kB\s*$", re.MULTILINE)


def _read_children(pid: int) -> list[int] | None:
    """Read children from every thread's children file for one process."""
    task_root = _PROC_ROOT / str(pid) / "task"
    try:
        task_entries = list(task_root.iterdir())
    except OSError as exc:
        if exc.errno in {errno.ENOENT, errno.ESRCH}:
            return None
        raise RuntimeError(
            f"Unable to enumerate live process PID {pid}: {exc}"
        ) from exc

    children: list[int] = []
    for task in task_entries:
        if not task.name.isdecimal():
            continue
        path = task / "children"
        try:
            with path.open(encoding="utf-8") as child_file:
                value = child_file.read()
        except OSError as exc:
            if exc.errno in {errno.ENOENT, errno.ESRCH}:
                continue
            raise RuntimeError(
                f"Unable to read live process children for PID {pid}: {exc}"
            ) from exc
        for token in value.split():
            try:
                child = int(token)
            except ValueError as exc:
                raise RuntimeError(
                    f"Malformed /proc/{pid}/task/{task.name}/children"
                ) from exc
            if child > 0:
                children.append(child)
    return children


def process_tree_pids(pid: int) -> set[int]:
    """Return *pid* and all currently living descendants.

    Linux process enumeration is inherently racy: a process which exits while
    its task files are being read is ignored.  Only files belonging to the
    requested process and its descendants are read, so unrelated unreadable
    processes cannot make a worker's accounting fail.
    """
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ValueError("pid must be a positive integer")

    result = {pid}
    pending = [pid]
    while pending:
        parent = pending.pop()
        children = _read_children(parent)
        if children is None:
            if parent == pid:
                return set()
            continue
        for child in children:
            if child not in result:
                result.add(child)
                pending.append(child)
    return result


def _read_rss(pid: int) -> int | None:
    """Read one process RSS in bytes; ``None`` means it exited during read."""
    path = _PROC_ROOT / str(pid) / "status"
    try:
        with path.open(encoding="utf-8") as status:
            value = status.read()
    except OSError as exc:
        if exc.errno in {errno.ENOENT, errno.ESRCH}:
            return None
        raise RuntimeError(
            f"Unable to read live process RSS for PID {pid}: {exc}"
        ) from exc

    state_match = re.search(r"^State:\s*(?P<state>\S+)", value, re.MULTILINE)
    match = _RSS_RE.search(value)
    if match is None:
        # A zombie's address space is gone and Linux may omit VmRSS entirely.
        if state_match is not None and state_match.group("state") == "Z":
            return 0
        raise RuntimeError(f"Malformed /proc/{pid}/status: missing VmRSS")
    return int(match.group("value")) * 1024


def ram_usage(pids: set[int]) -> int:
    """Return the sum of RSS bytes for the supplied Linux process IDs.

    A PID disappearing during a sample is normal and omitted.  Permission,
    malformed, and other read failures raise ``RuntimeError`` rather than
    silently understating a worker's resource usage.
    """
    if not isinstance(pids, set):
        raise TypeError("pids must be a set of process IDs")
    total = 0
    for pid in pids:
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            raise ValueError("pids must contain positive integer process IDs")
        rss = _read_rss(pid)
        if rss is not None:
            total += rss
    return total


def _run_nvidia_smi(*query: str) -> str:
    """Run one bounded nvidia-smi query and return stdout."""
    if shutil.which(_NVIDIA_SMI) is None:
        raise RuntimeError("NVIDIA GPU telemetry unavailable: nvidia-smi was not found")
    command = [_NVIDIA_SMI, *query]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=_NVIDIA_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "NVIDIA GPU telemetry unavailable: nvidia-smi timed out"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"NVIDIA GPU telemetry unavailable: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"NVIDIA GPU telemetry unavailable{suffix}")
    return completed.stdout


def _gpu_inventory() -> list[tuple[int, str]]:
    output = _run_nvidia_smi(
        "--query-gpu=index,uuid",
        "--format=csv,noheader,nounits",
    )
    inventory: list[tuple[int, str]] = []
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if not row or not any(cell.strip() for cell in row):
            continue
        if len(row) != 2:
            raise RuntimeError(
                f"Malformed nvidia-smi GPU inventory row: {','.join(row)!r}"
            )
        try:
            index = int(row[0].strip())
        except ValueError as exc:
            raise RuntimeError(f"Malformed nvidia-smi GPU index: {row[0]!r}") from exc
        uuid = row[1].strip()
        if index < 0 or not uuid:
            raise RuntimeError("Malformed nvidia-smi GPU inventory")
        inventory.append((index, uuid))
    if not inventory:
        raise RuntimeError("NVIDIA GPU telemetry unavailable: no GPUs reported")
    inventory.sort(key=lambda item: item[0])
    return inventory


def _normalise_uuid(value: str) -> str:
    return value.strip().lower().removeprefix("gpu-").replace("-", "")


@lru_cache(maxsize=1)
def _logical_cuda_uuids() -> tuple[str, ...]:
    """Get UUIDs in CUDA runtime logical order in an isolated process."""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _CUDA_UUID_SCRIPT],
            check=False,
            capture_output=True,
            text=True,
            timeout=_NVIDIA_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "NVIDIA GPU telemetry unavailable: CUDA UUID probe timed out"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"NVIDIA GPU telemetry unavailable: CUDA UUID probe failed: {exc}"
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(
            f"NVIDIA GPU telemetry unavailable: CUDA UUID probe failed{suffix}"
        )
    try:
        values = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "NVIDIA GPU telemetry unavailable: invalid CUDA UUID probe output"
        ) from exc
    if not isinstance(values, list) or not all(
        isinstance(value, str) and value for value in values
    ):
        raise RuntimeError(
            "NVIDIA GPU telemetry unavailable: invalid CUDA UUID probe output"
        )
    return tuple(values)


def _visible_inventory(
    inventory: list[tuple[int, str]], logical_uuids: tuple[str, ...]
) -> dict[str, int]:
    """Map physical GPU UUIDs to CUDA logical device indexes.

    CUDA's runtime order can differ from nvidia-smi's physical index order
    (notably with FASTEST_FIRST), so the order from cudaDeviceGetUuid is the
    source of truth.
    """
    by_uuid = {_normalise_uuid(uuid): uuid for _index, uuid in inventory}
    result: dict[str, int] = {}
    for logical, uuid in enumerate(logical_uuids):
        physical = by_uuid.get(_normalise_uuid(uuid))
        if physical is None:
            raise RuntimeError(
                f"NVIDIA GPU telemetry unavailable: CUDA UUID {uuid!r} was not in nvidia-smi inventory"
            )
        result[_normalise_uuid(physical)] = logical
    return result


def _memory_bytes(value: str) -> int:
    value = value.strip()
    if not value or value.upper() in {"N/A", "NA", "NOT SUPPORTED", "[N/A]"}:
        raise RuntimeError(
            "NVIDIA GPU telemetry unavailable: nvidia-smi returned unknown process memory"
        )
    parts = value.split()
    try:
        amount = int(parts[0])
    except ValueError as exc:
        raise RuntimeError(f"Malformed nvidia-smi process memory: {value!r}") from exc
    if amount < 0:
        raise RuntimeError(f"Malformed nvidia-smi process memory: {value!r}")
    unit = parts[1].lower() if len(parts) > 1 else "mib"
    multipliers = {"b": 1, "kib": 1024, "mib": 1024**2, "gib": 1024**3}
    if unit not in multipliers:
        raise RuntimeError(f"Malformed nvidia-smi process memory unit: {value!r}")
    return amount * multipliers[unit]


def _compute_apps() -> Iterable[tuple[int, str, int]]:
    output = _run_nvidia_smi(
        "--query-compute-apps=pid,gpu_uuid,used_memory",
        "--format=csv,noheader,nounits",
    )
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if not row or not any(cell.strip() for cell in row):
            continue
        if len(row) == 1 and row[0].strip().lower() == "no running processes found":
            continue
        if len(row) != 3:
            raise RuntimeError(f"Malformed nvidia-smi process row: {','.join(row)!r}")
        try:
            pid = int(row[0].strip())
        except ValueError as exc:
            raise RuntimeError(f"Malformed nvidia-smi process PID: {row[0]!r}") from exc
        if pid <= 0:
            raise RuntimeError(f"Malformed nvidia-smi process PID: {row[0]!r}")
        uuid = row[1].strip()
        if not uuid:
            raise RuntimeError("Malformed nvidia-smi process row: missing GPU UUID")
        yield pid, uuid, _memory_bytes(row[2])


def gpu_usage() -> dict[int, dict[str, int]]:
    """Return NVIDIA compute-process memory by PID and logical ``cuda:N``.

    ``nvidia-smi`` reports physical UUIDs.  CUDA UUID probing supplies the
    logical ordering, including FASTEST_FIRST and CUDA_VISIBLE_DEVICES.
    Unknown UUIDs and unavailable memory values are errors, never zeroes.
    """
    logical_by_uuid = _visible_inventory(_gpu_inventory(), _logical_cuda_uuids())
    result: dict[int, dict[str, int]] = {}
    for pid, uuid, amount in _compute_apps():
        logical = logical_by_uuid.get(_normalise_uuid(uuid))
        if logical is None:
            # A process on a hidden GPU is unrelated to this process's logical
            # view and must not be charged to its worker budget.
            continue
        per_device = result.setdefault(pid, {})
        key = f"cuda:{logical}"
        per_device[key] = per_device.get(key, 0) + amount
    return result


def validate_gpu_monitoring() -> None:
    """Raise ``RuntimeError`` unless GPU process telemetry is usable."""
    gpu_usage()
