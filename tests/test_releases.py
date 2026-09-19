import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chorus import releases


class _Response:
    def __init__(self, payload: object):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, limit: int) -> bytes:
        return self.payload[:limit]


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        releases._cache = None

    def test_parses_versioned_changelog_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "CHANGELOG.md"
            path.write_text(
                "# Changelog\n\n## 1.2.0 - 2026-09-19\n\n- First change.\n- Second change.\n\n## 1.1.0 - Earlier\n\n- Old change.\n"
            )
            self.assertEqual(
                releases.parse_changelog(path),
                [
                    {"version": "1.2.0", "date": "2026-09-19", "changes": ["First change.", "Second change."]},
                    {"version": "1.1.0", "date": "Earlier", "changes": ["Old change."]},
                ],
            )

    @patch("chorus.releases.urllib.request.urlopen")
    def test_selects_highest_semantic_github_tag(self, urlopen):
        urlopen.return_value = _Response(
            [{"name": "v0.9.0"}, {"name": "not-a-release"}, {"name": "v1.2.3"}, {"name": "v1.10.0"}]
        )
        self.assertEqual(releases.fetch_latest_version(), "1.10.0")

    @patch("chorus.releases.fetch_latest_version", return_value="0.2.0")
    def test_reports_available_update_and_caches_check(self, fetch):
        first = releases._cached_update("0.1.8")
        second = releases._cached_update("0.1.8")
        self.assertEqual(first["status"], "update_available")
        self.assertEqual(first["latest_version"], "0.2.0")
        self.assertEqual(first, second)
        fetch.assert_called_once()

    @patch("chorus.releases.fetch_latest_version", side_effect=OSError("offline"))
    def test_update_check_degrades_when_github_is_unavailable(self, _fetch):
        self.assertEqual(releases._cached_update("0.1.8")["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
