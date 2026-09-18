import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chorus.models import ModelCatalog, install_shipped_manifests

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
    def test_fresh_root_is_seeded_from_shipped_manifests(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            store = base / "image"
            payload = b"downloaded model"
            shipped = model(store / "models", "kokoro", {"weights": payload})
            root = base / "state/models"
            with patch("chorus.models.ROOT", store):
                install_shipped_manifests(root)
            self.assertEqual(
                (root / "kokoro/default/manifest.json").read_text(),
                (shipped / "manifest.json").read_text(),
            )
            catalog = ModelCatalog([root])
            self.assertEqual(
                catalog.resolve("kokoro").directory, root / "kokoro/default"
            )
            cached = base / "cached"
            cached.write_bytes(payload)
            with patch("huggingface_hub.hf_hub_download", return_value=str(cached)):
                catalog.prepare(["kokoro"], download_missing=True)
            self.assertEqual((root / "kokoro/default/weights").read_bytes(), payload)

            # A package upgrade refreshes manifests and leaves weights alone.
            refreshed = json.loads((shipped / "manifest.json").read_text())
            refreshed["artifacts"]["weights"]["revision"] = "b" * 40
            (shipped / "manifest.json").write_text(json.dumps(refreshed))
            with patch("chorus.models.ROOT", store):
                install_shipped_manifests(root)
                # A root that is already the shipped directory is left alone.
                install_shipped_manifests(store / "models")
            self.assertEqual(
                json.loads((root / "kokoro/default/manifest.json").read_text()),
                refreshed,
            )
            self.assertEqual((root / "kokoro/default/weights").read_bytes(), payload)

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
                ModelCatalog([root]).prepare(["kokoro"])
            message = str(raised.exception)
            for name in ("missing", "empty", "pointer"):
                self.assertIn(str(selected / name), message)
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
                ModelCatalog([root]).prepare(["kokoro"], download_missing=True)
            for name, expected in payloads.items():
                self.assertEqual((selected / name).read_bytes(), expected)
            self.assertEqual(healthy.stat().st_mtime_ns, original_stat.st_mtime_ns)
            self.assertEqual(healthy.stat().st_ino, original_stat.st_ino)
            self.assertFalse((unrelated / "untouched").exists())
            with patch(
                "huggingface_hub.hf_hub_download",
                side_effect=AssertionError("Unexpected repeat download"),
            ):
                ModelCatalog([root]).prepare(["kokoro"], download_missing=True)

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
                ModelCatalog([root]).prepare(["kokoro"], download_missing=True)
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
                ModelCatalog([root]).prepare(["breeze"], download_missing=True)
            message = str(raised.exception)
            self.assertIn(str(selected / "weights"), message)
            self.assertIn(str(root / "runtimes/breeze/.venv/bin/python"), message)

    def test_first_complete_copy_wins(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = [Path(temporary) / name for name in ("ssd", "archive")]
            for root, data in zip(roots, (b"first", b"second")):
                directory = model(root, "kokoro", {"weights": data})
                (directory / "weights").write_bytes(data)
            catalog = ModelCatalog(roots)
            self.assertEqual(
                catalog.resolve("kokoro").artifact("weights").read_bytes(), b"first"
            )
            reversed_catalog = ModelCatalog(list(reversed(roots)))
            self.assertEqual(
                reversed_catalog.resolve("kokoro").artifact("weights").read_bytes(),
                b"second",
            )

    def test_moved_model_is_found_past_incomplete_copies_without_downloading(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = [Path(temporary) / name for name in ("ssd", "disk", "archive")]
            payload = {"weights": b"complete model", "voices": b"complete voices"}
            original = model(roots[0], "kokoro", payload)
            for name, data in payload.items():
                (original / name).write_bytes(data)
            ModelCatalog(roots).prepare(["kokoro"])
            destination = roots[2] / "kokoro" / "default"
            destination.parent.mkdir(parents=True)
            shutil.move(original, destination)
            model(roots[0], "kokoro", payload)  # Launcher refreshes primary manifests.
            partial = model(roots[1], "kokoro", payload)
            (partial / "weights").write_bytes(LFS_POINTER)
            (partial / "voices").touch()
            with patch(
                "huggingface_hub.hf_hub_download",
                side_effect=AssertionError("Moved model must not be downloaded"),
            ):
                restarted = ModelCatalog(roots)
                restarted.prepare(["kokoro"], download_missing=True)
            self.assertEqual(
                restarted.resolve("kokoro").artifact("weights"), destination / "weights"
            )
            self.assertFalse((original / "weights").exists())
            self.assertEqual((partial / "weights").read_bytes(), LFS_POINTER)

    def test_missing_model_downloads_to_primary_with_metadata_only_on_secondary(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = [Path(temporary) / name for name in ("ssd", "archive")]
            payload = b"downloaded model"
            secondary = model(roots[1], "kokoro", {"weights": payload})
            cached = Path(temporary) / "cached"
            cached.write_bytes(payload)
            with patch("huggingface_hub.hf_hub_download", return_value=str(cached)):
                ModelCatalog(roots).prepare(["kokoro"], download_missing=True)
            restarted = ModelCatalog(roots)
            self.assertEqual(
                restarted.resolve("kokoro").artifact("weights").read_bytes(), payload
            )
            self.assertEqual(
                restarted.resolve("kokoro").directory, roots[0] / "kokoro/default"
            )
            self.assertFalse((secondary / "weights").exists())
            shutil.rmtree(roots[1])
            ModelCatalog([roots[0]]).prepare(["kokoro"])

    def test_partial_copies_are_not_combined_across_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = [Path(temporary) / name for name in ("ssd", "archive")]
            payload = {"weights": b"model", "voices": b"voice bank"}
            primary = model(roots[0], "kokoro", payload)
            secondary = model(roots[1], "kokoro", payload)
            (primary / "weights").write_bytes(payload["weights"])
            (secondary / "voices").write_bytes(payload["voices"])
            with self.assertRaises(RuntimeError):
                ModelCatalog(roots).prepare(["kokoro"])
            with patch(
                "huggingface_hub.hf_hub_download",
                return_value=str(secondary / "voices"),
            ):
                repaired = ModelCatalog(roots)
                repaired.prepare(["kokoro"], download_missing=True)
            self.assertEqual(
                repaired.resolve("kokoro").artifact("voices"), primary / "voices"
            )
            self.assertEqual((primary / "weights").read_bytes(), payload["weights"])
            self.assertFalse((secondary / "weights").exists())


if __name__ == "__main__":
    unittest.main()
