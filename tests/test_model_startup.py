import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chorus.models import ModelCatalog

LFS_POINTER = b"version https://git-lfs.github.com/spec/v1\noid sha256:placeholder\n"


def model(root, engine, artifacts):
    directory = root / engine / "default"
    directory.mkdir(parents=True)
    manifest = {
        "default": True,
        "devices": ["cpu"],
        "artifacts": {
            name: {
                "repo": f"test/{engine}",
                "revision": "a" * 40,
                "file": name,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in artifacts.items()
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


class ModelStartupTests(unittest.TestCase):
    def test_reports_missing_empty_and_lfs_files_without_downloading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected = model(
                root,
                "kokoro",
                {name: b"complete" for name in ("missing", "empty", "pointer")},
            )
            (selected / "empty").touch()
            (selected / "pointer").write_bytes(LFS_POINTER)
            unrelated = model(root, "breeze", {"unselected": b"complete"})
            with (
                patch(
                    "huggingface_hub.hf_hub_download",
                    side_effect=AssertionError("Unexpected download"),
                ),
                self.assertRaises(RuntimeError) as raised,
            ):
                ModelCatalog(root).prepare(["kokoro"])
            message = str(raised.exception)
            for name in ("missing", "empty", "pointer"):
                self.assertIn(str(selected / name), message)
            self.assertIn("--download-missing", message)
            self.assertNotIn(str(unrelated), message)
            self.assertEqual((selected / "pointer").read_bytes(), LFS_POINTER)

    def test_repairs_only_selected_incomplete_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = {
                "healthy": b"keep",
                "missing": b"new model",
                "pointer": b"voice",
                "empty": b"config",
            }
            selected = model(root, "kokoro", payloads)
            healthy = selected / "healthy"
            healthy.write_bytes(payloads["healthy"])
            original_stat = healthy.stat()
            (selected / "pointer").write_bytes(LFS_POINTER)
            (selected / "empty").touch()
            unrelated = model(root, "breeze", {"untouched": b"other model"})
            cache = root / "cache"
            cache.mkdir()
            for name, data in payloads.items():
                if name != "healthy":
                    (cache / name).write_bytes(data)

            def download(repo, filename, revision):
                if repo != "test/kokoro" or filename == "healthy":
                    raise AssertionError(
                        "Downloaded an unselected or complete artifact"
                    )
                return str(cache / filename)

            with patch("huggingface_hub.hf_hub_download", side_effect=download):
                ModelCatalog(root).prepare(["kokoro"], download_missing=True)
            for name, expected in payloads.items():
                self.assertEqual((selected / name).read_bytes(), expected)
            self.assertEqual(healthy.stat().st_mtime_ns, original_stat.st_mtime_ns)
            self.assertEqual(healthy.stat().st_ino, original_stat.st_ino)
            self.assertFalse((unrelated / "untouched").exists())
            with patch(
                "huggingface_hub.hf_hub_download",
                side_effect=AssertionError("Unexpected repeat download"),
            ):
                ModelCatalog(root).prepare(["kokoro"], download_missing=True)

    def test_bad_download_does_not_replace_existing_pointer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected = model(root, "kokoro", {"model.onnx": b"expected"})
            target = selected / "model.onnx"
            target.write_bytes(LFS_POINTER)
            bad_download = root / "bad-download"
            bad_download.write_bytes(b"corrupted")
            with (
                patch(
                    "huggingface_hub.hf_hub_download", return_value=str(bad_download)
                ),
                self.assertRaises(RuntimeError),
            ):
                ModelCatalog(root).prepare(["kokoro"], download_missing=True)
            self.assertEqual(target.read_bytes(), LFS_POINTER)

    def test_missing_runtime_reports_setup_before_fetching_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected = model(root, "breeze", {"weights": b"expected"})
            with (
                patch("chorus.models.ROOT", root),
                patch(
                    "huggingface_hub.hf_hub_download",
                    side_effect=AssertionError("Unexpected download"),
                ),
                self.assertRaises(RuntimeError) as raised,
            ):
                ModelCatalog(root).prepare(["breeze"], download_missing=True)
            message = str(raised.exception)
            self.assertIn(str(selected / "weights"), message)
            self.assertIn(str(root / "runtimes/breeze/.venv/bin/python"), message)
            self.assertIn("uv sync --project runtimes/breeze --locked", message)
            self.assertIn(
                "git submodule update --init runtimes/breeze/upstream", message
            )


if __name__ == "__main__":
    unittest.main()
