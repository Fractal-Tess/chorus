from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from chorus.models import ModelCatalog

MIB = 1024**2


def _nonnegative_finite(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return parsed


def _positive_finite(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _mib_to_bytes(value: float) -> int:
    """Convert a validated MiB float without overflowing float arithmetic."""
    whole = int(value)
    return whole * MIB + int((value - whole) * MIB)


def _vram_budget(value: str) -> tuple[str, float]:
    device, separator, mib = value.partition("=")
    if not separator or not device.startswith("cuda:") or not device[5:].isdigit():
        raise argparse.ArgumentTypeError(
            "must use the form cuda:N=MIB (for example, cuda:0=4096)"
        )
    return device, _positive_finite(mib)


def _unique_devices(
    parser: argparse.ArgumentParser, value: str | None
) -> list[str] | None:
    if value is None:
        return None
    devices = value.split(",")
    if len(devices) != len(set(devices)):
        parser.error("--devices must not contain duplicate device IDs")
    return devices


def main() -> None:
    parser = argparse.ArgumentParser(description="Local multi-engine TTS server")
    parser.add_argument("--host", default=os.environ.get("TTS_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("TTS_PORT", "8000"))
    )
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument(
        "--devices", help="Allowed devices, e.g. cpu,cuda:0 (no duplicates)"
    )
    parser.add_argument("--default-device", default="auto")
    parser.add_argument("--engines", help="Comma-separated enabled engines")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--fetch-model", metavar="ENGINE/MODEL")
    parser.add_argument(
        "--preload", action="append", default=[], metavar="ENGINE/MODEL"
    )
    parser.add_argument(
        "--idle-timeout",
        type=_nonnegative_finite,
        default=300.0,
        metavar="SECONDS",
        help="Unload an idle model worker after SECONDS (default: 300; 0 unloads immediately).",
    )
    parser.add_argument(
        "--ram-budget-mib",
        type=_positive_finite,
        metavar="MIB",
        help="Soft aggregate RAM cache budget in MiB; eviction excludes this API process and unrelated processes.",
    )
    parser.add_argument(
        "--vram-budget-mib",
        action="append",
        type=_vram_budget,
        default=[],
        metavar="cuda:N=MIB",
        help="Soft per-device VRAM cache budget (repeatable, e.g. cuda:0=4096); duplicate devices are rejected.",
    )
    args = parser.parse_args()
    devices = _unique_devices(parser, args.devices)
    vram_budget_bytes: dict[str, int] = {}
    for device, mib in args.vram_budget_mib:
        if device in vram_budget_bytes:
            parser.error(f"--vram-budget-mib specified more than once for {device}")
        vram_budget_bytes[device] = _mib_to_bytes(mib)
    ram_budget_bytes = (
        _mib_to_bytes(args.ram_budget_mib) if args.ram_budget_mib is not None else None
    )
    catalog = ModelCatalog(args.models_dir)
    if args.fetch_model:
        try:
            catalog.fetch(args.fetch_model)
        except (ValueError, RuntimeError) as error:
            parser.error(str(error))
        return
    from chorus.devices import DevicePolicy, available_devices

    if args.list_devices:
        print(
            json.dumps(
                {
                    "available": available_devices(),
                    "models": [
                        {"engine": s.engine, "model": s.name, "supported": s.devices}
                        for s in catalog.models.values()
                    ],
                },
                indent=2,
            )
        )
        return
    from chorus.registry import EngineRegistry

    registry = None
    try:
        registry = EngineRegistry(
            catalog,
            DevicePolicy(devices, args.default_device),
            args.engines.split(",") if args.engines else None,
            idle_timeout=args.idle_timeout,
            ram_budget_bytes=ram_budget_bytes,
            vram_budget_bytes=vram_budget_bytes or None,
        )
        for selector in args.preload:
            registry.preload(selector)
        import uvicorn

        from chorus import api

        api.registry = registry
        uvicorn.run(api.app, host=args.host, port=args.port)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    finally:
        if registry is not None:
            registry.close()


if __name__ == "__main__":
    main()
