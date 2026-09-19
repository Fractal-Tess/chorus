"""Installed release metadata and a cached public update check."""

from __future__ import annotations

import importlib.metadata
import json
import re
import threading
import time
import urllib.request
from pathlib import Path

from chorus.models import ROOT

GITHUB_TAGS_URL = "https://api.github.com/repos/Fractal-Tess/chorus/tags?per_page=100"
SOURCE_URL = "https://github.com/Fractal-Tess/chorus"
UPDATE_CACHE_SECONDS = 15 * 60
_VERSION = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")
_HEADING = re.compile(r"^##\s+(\S+)\s+-\s+(.+)$")
_cache_lock = threading.Lock()
_cache: tuple[float, dict[str, object]] | None = None


def parse_changelog(path: Path) -> list[dict[str, object]]:
    """Parse the project's deliberately simple version-and-bullets changelog."""
    entries: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if match := _HEADING.match(line):
            current = {"version": match.group(1), "date": match.group(2), "changes": []}
            entries.append(current)
        elif current is not None and line.startswith("- "):
            current["changes"].append(line[2:])
    return entries


def _version_key(version: str) -> tuple[int, int, int] | None:
    match = _VERSION.match(version)
    return tuple(map(int, match.groups())) if match else None


def fetch_latest_version(timeout: float = 3.0) -> str:
    request = urllib.request.Request(
        GITHUB_TAGS_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "chorus-update-check",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        tags = json.loads(response.read(1_000_000))
    versions = [
        (key, tag["name"].removeprefix("v"))
        for tag in tags
        if isinstance(tag, dict)
        and isinstance(tag.get("name"), str)
        and (key := _version_key(tag["name"])) is not None
    ]
    if not versions:
        raise ValueError("GitHub returned no semantic version tags")
    return max(versions)[1]


def _cached_update(current_version: str) -> dict[str, object]:
    global _cache
    now = time.monotonic()
    with _cache_lock:
        if _cache is not None and now - _cache[0] < UPDATE_CACHE_SECONDS:
            return _cache[1]
        try:
            latest = fetch_latest_version()
            current_key = _version_key(current_version)
            latest_key = _version_key(latest)
            result: dict[str, object] = {
                "status": "update_available"
                if current_key is not None and latest_key is not None and latest_key > current_key
                else "current",
                "latest_version": latest,
                "url": f"{SOURCE_URL}/tree/v{latest}",
            }
        except (OSError, ValueError, json.JSONDecodeError):
            result = {"status": "unavailable", "latest_version": None, "url": SOURCE_URL}
        _cache = (now, result)
        return result


def release_info() -> dict[str, object]:
    current_version = importlib.metadata.version("chorus")
    return {
        "version": current_version,
        "source_url": SOURCE_URL,
        "update": _cached_update(current_version),
        "changelog": parse_changelog(ROOT / "CHANGELOG.md"),
    }
