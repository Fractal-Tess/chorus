"""Seed the persistent model catalog, then replace this process with serve-api."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_MODELS = APP_ROOT / "models"


def _requested_models_dir(arguments: list[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(os.environ.get("TTS_MODELS_DIR") or "/models"),
    )
    options, _ = parser.parse_known_args(arguments)
    return options.models_dir


def _seed_manifests(destination_root: Path) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    for manifest in sorted(SHIPPED_MODELS.glob("*/*/manifest.json")):
        relative = manifest.relative_to(SHIPPED_MODELS)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.resolve() != manifest.resolve():
            # Manifests are refreshed, but no downloaded artifact is touched.
            shutil.copy2(manifest, destination)


def main() -> None:
    arguments = sys.argv[1:]
    models_dir = _requested_models_dir(arguments)
    _seed_manifests(models_dir)
    # Keep libraries that consult the environment aligned with CLI precedence.
    os.environ["TTS_MODELS_DIR"] = str(models_dir)
    os.execv(sys.executable, [sys.executable, "-m", "chorus.cli", *arguments])


if __name__ == "__main__":
    main()
