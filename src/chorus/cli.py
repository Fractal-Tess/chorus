from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from chorus.models import ModelCatalog


def main() -> None:
    parser = argparse.ArgumentParser(description="Local multi-engine TTS server")
    parser.add_argument("--host", default=os.environ.get("TTS_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("TTS_PORT", "8000"))
    )
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--devices", help="Allowed devices, e.g. cpu,cuda:0")
    parser.add_argument("--default-device", default="auto")
    parser.add_argument("--engines", help="Comma-separated enabled engines")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--fetch-model", metavar="ENGINE/MODEL")
    parser.add_argument(
        "--preload", action="append", default=[], metavar="ENGINE/MODEL"
    )
    args = parser.parse_args()
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
            DevicePolicy(
                args.devices.split(",") if args.devices else None, args.default_device
            ),
            args.engines.split(",") if args.engines else None,
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
