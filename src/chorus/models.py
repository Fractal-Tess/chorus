"""Local model inventory, startup checks, and explicit artifact downloads."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[2]


def model_roots(roots: list[Path] | None = None) -> tuple[Path, ...]:
    if roots is None:
        configured = os.environ.get("TTS_MODELS_DIRS")
        if configured is None:
            roots = [ROOT / "models"]
        else:
            parts = configured.split(os.pathsep)
            if any(not part for part in parts):
                raise ValueError("TTS_MODELS_DIRS must contain non-empty model paths")
            roots = [Path(part) for part in parts]
    if not roots:
        raise ValueError("At least one model directory is required")
    return tuple(dict.fromkeys(path.resolve() for path in roots))


def _file_problem(path: Path) -> str | None:
    if not path.is_file():
        return "missing or not a regular file"
    try:
        with path.open("rb") as stream:
            header = stream.read(43)
    except OSError as error:
        return f"unreadable: {error}"
    if not header:
        return "empty file"
    if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
        return "unmaterialized Git LFS pointer"
    return None


@dataclass(frozen=True)
class ModelSpec:
    engine: str
    name: str
    directory: Path
    manifest: dict

    @property
    def devices(self) -> list[str]:
        return self.manifest["devices"]

    def artifact_path(self, name: str) -> Path:
        path = (self.directory / name).resolve()
        if not path.is_relative_to(self.directory.resolve()):
            raise ValueError("Artifact path escapes model directory")
        return path

    def missing_artifacts(self) -> dict[str, str]:
        artifacts = self.manifest.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError(
                f"No artifact inventory for {self.engine}/{self.name}. "
                f"Restore a complete manifest at {self.directory / 'manifest.json'}"
            )
        return {
            name: problem
            for name in artifacts
            if (problem := _file_problem(self.artifact_path(name))) is not None
        }

    def artifact(self, name: str) -> Path:
        path = self.artifact_path(name)
        if problem := _file_problem(path):
            raise RuntimeError(
                f"{path}: {problem}. Restart serve-api with --download-missing "
                "and the same ordered --models-dir options."
            )
        return path


class ModelCatalog:
    def __init__(self, roots: list[Path] | None = None):
        self.roots = model_roots(roots)
        self.models: dict[tuple[str, str], ModelSpec] = {}
        for root in self.roots:
            for path in sorted(root.glob("*/*/manifest.json")):
                key = (path.parent.parent.name, path.parent.name)
                if key in self.models:
                    continue
                manifest = json.loads(path.read_text())
                engine, name = key
                primary = ModelSpec(
                    engine, name, self.roots[0] / engine / name, manifest
                )
                self.models[key] = primary
                for candidate_root in self.roots:
                    candidate = ModelSpec(
                        engine, name, candidate_root / engine / name, manifest
                    )
                    if not candidate.missing_artifacts():
                        self.models[key] = candidate
                        break

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

    def selected(self, engines: list[str]) -> list[ModelSpec]:
        selected = []
        for engine in engines:
            # A default must exist because requests may omit the model name.
            try:
                self.resolve(engine)
            except ValueError as error:
                raise ValueError(
                    f"{error}. Restore the manifests under "
                    f"{', '.join(str(root / engine) for root in self.roots)} "
                    "with exactly one default model, or correct --models-dir."
                ) from error
            selected.extend(
                spec for spec in self.models.values() if spec.engine == engine
            )
        return selected

    def prepare(self, engines: list[str], *, download_missing: bool = False) -> None:
        """Check every advertised model of the selected engines before serving."""
        specs = self.selected(engines)
        missing = [(spec, spec.missing_artifacts()) for spec in specs]
        diagnostics = [
            f"  {spec.artifact_path(name)}: {reason}"
            for spec, artifacts in missing
            for name, reason in artifacts.items()
        ]
        runtime_problems = []
        for engine in engines:
            if engine not in {"breeze", "fish"}:
                continue
            runtime = ROOT / "runtimes" / engine
            source = (
                "breeze_infer/runtime.py"
                if engine == "breeze"
                else "fish_speech/inference_engine/__init__.py"
            )
            requirements = (
                (
                    runtime / "worker.py",
                    "Restore the runtime worker from this checkout.",
                ),
                (
                    runtime / ".venv/bin/python",
                    f"Run uv sync --project runtimes/{engine} --locked",
                ),
                (
                    runtime / "upstream" / source,
                    f"Run git submodule update --init runtimes/{engine}/upstream",
                ),
            )
            for path, repair in requirements:
                if problem := _file_problem(path):
                    runtime_problems.append(f"  {path}: {problem}. {repair}")
            python = runtime / ".venv/bin/python"
            if python.is_file() and not os.access(python, os.X_OK):
                runtime_problems.append(
                    f"  {python}: not executable. "
                    f"Run uv sync --project runtimes/{engine} --locked"
                )
        root_options = " ".join(
            f"--models-dir {shlex.quote(str(root))}" for root in self.roots
        )
        repair = (
            "Run serve-api --engines "
            f"{shlex.quote(','.join(engines))} {root_options} --download-missing "
            "(with your other launch options), or restore the listed files."
        )
        if runtime_problems or (diagnostics and not download_missing):
            details = "\n".join(diagnostics + runtime_problems)
            if runtime_problems:
                repair += (
                    "\nComplete the runtime setup commands before downloading models."
                )
            raise RuntimeError(f"Selected engines are not ready:\n{details}\n{repair}")
        if diagnostics:
            print(
                "Downloading missing model files:\n" + "\n".join(diagnostics),
                flush=True,
            )
            for spec, artifacts in missing:
                if artifacts:
                    self.fetch(f"{spec.engine}/{spec.name}", only_missing=True)
            # Downloads must finish successfully before the listener is started.
            self.prepare(engines)

    def fetch(self, selector: str, *, only_missing: bool = False) -> None:
        engine, name = selector.split("/", 1)
        selected = self.resolve(engine, name)
        if only_missing and not selected.missing_artifacts():
            return
        spec = ModelSpec(engine, name, self.roots[0] / engine / name, selected.manifest)
        spec.directory.mkdir(parents=True, exist_ok=True)
        manifest_path = spec.directory / "manifest.json"
        if not manifest_path.exists():
            temporary_manifest = None
            try:
                with NamedTemporaryFile(
                    mode="w", dir=spec.directory, prefix=".manifest.", delete=False
                ) as output:
                    temporary_manifest = Path(output.name)
                    json.dump(spec.manifest, output, indent=2)
                    output.write("\n")
                os.replace(temporary_manifest, manifest_path)
            finally:
                if temporary_manifest is not None:
                    temporary_manifest.unlink(missing_ok=True)
        missing = spec.missing_artifacts()
        artifacts = spec.manifest["artifacts"]
        for relative, source in artifacts.items():
            if only_missing and relative not in missing:
                continue
            target = spec.artifact_path(relative)
            try:
                from huggingface_hub import hf_hub_download

                path = Path(
                    hf_hub_download(
                        source["repo"], source["file"], revision=source["revision"]
                    )
                )
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != source["sha256"]:
                    raise RuntimeError("SHA-256 checksum mismatch")
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = None
                try:
                    with NamedTemporaryFile(
                        dir=target.parent, prefix=f".{target.name}.", delete=False
                    ) as output:
                        temporary = Path(output.name)
                        with path.open("rb") as stream:
                            shutil.copyfileobj(stream, output)
                    os.replace(temporary, target)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            except Exception as error:
                raise RuntimeError(
                    f"Could not download {selector}/{relative}: {error}. "
                    "Check the source, network access, and Hugging Face credentials, "
                    "then retry --download-missing."
                ) from error
            print(f"Fetched {selector}/{relative}", flush=True)
        self.models[(engine, name)] = spec
