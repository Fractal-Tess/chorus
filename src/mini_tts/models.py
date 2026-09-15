"""Local model catalog; artifact fetching is an explicit setup operation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelSpec:
    engine: str
    name: str
    directory: Path
    manifest: dict

    @property
    def devices(self) -> list[str]:
        return self.manifest["devices"]

    def artifact(self, name: str) -> Path:
        path = (self.directory / name).resolve()
        if not path.is_relative_to(self.directory.resolve()):
            raise ValueError("Artifact path escapes model directory")
        if not path.is_file():
            raise RuntimeError(
                f"Missing model artifact: {path}. Run serve-api --fetch-model {self.engine}/{self.name}"
            )
        with path.open("rb") as stream:
            if stream.read(43).startswith(
                b"version https://git-lfs.github.com/spec/v1"
            ):
                raise RuntimeError(
                    f"Unmaterialized Git LFS pointer: {path}. Run git lfs pull"
                )
        return path


class ModelCatalog:
    def __init__(self, root: Path | None = None):
        self.root = (
            root or Path(os.environ.get("TTS_MODELS_DIR", ROOT / "models"))
        ).resolve()
        self.models = {}
        for path in sorted(self.root.glob("*/*/manifest.json")):
            manifest = json.loads(path.read_text())
            spec = ModelSpec(
                path.parent.parent.name, path.parent.name, path.parent, manifest
            )
            self.models[(spec.engine, spec.name)] = spec

    def resolve(self, engine: str, model: str | None = None) -> ModelSpec:
        candidates = [spec for (key, _), spec in self.models.items() if key == engine]
        if model is None:
            candidates = [
                spec for spec in candidates if spec.manifest.get("default", False)
            ]
        else:
            candidates = [spec for spec in candidates if spec.name == model]
        if len(candidates) != 1:
            raise ValueError(
                f"Unknown or ambiguous model for {engine}: {model or 'default'}"
            )
        return candidates[0]

    def fetch(self, selector: str) -> None:
        from huggingface_hub import hf_hub_download

        engine, name = selector.split("/", 1)
        spec = self.resolve(engine, name)
        artifacts = spec.manifest.get("artifacts", {})
        if not artifacts:
            raise ValueError(f"{selector} uses its upstream library's model management")
        import shutil

        for relative, source in artifacts.items():
            path = Path(
                hf_hub_download(
                    source["repo"], source["file"], revision=source["revision"]
                )
            )
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != source["sha256"]:
                raise RuntimeError(f"Checksum mismatch for {relative}")
            target = spec.directory / relative
            if not target.resolve().is_relative_to(spec.directory.resolve()):
                raise ValueError("Artifact path escapes model directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            print(f"Fetched {selector}/{relative}", flush=True)
