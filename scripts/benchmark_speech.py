#!/usr/bin/env python3
"""Run the canonical Chorus Kokoro throughput benchmark."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog."
DEFAULT_URL = "http://127.0.0.1:8749/v1/audio/speech"
TIMING_HEADERS = {
    "queue_ms": "X-Queue-Time-Ms",
    "inference_ms": "X-Inference-Time-Ms",
    "encoding_ms": "X-Encoding-Time-Ms",
    "backend_ms": "X-Backend-Time-Ms",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=None,
        help="Concurrent warm-up requests (default: same as --clients)",
    )
    args = parser.parse_args()
    if args.warmup_requests is None:
        args.warmup_requests = args.clients
    if args.clients < 1 or args.seconds <= 0 or args.warmup_requests < 1:
        parser.error("clients, seconds, and warm-up requests must be positive")
    return args


def request_body() -> bytes:
    return json.dumps(
        {
            "engine": "kokoro",
            "model": "82m-v1.0",
            "channel": "gpu",
            "input": DEFAULT_TEXT,
            "voice": "af_bella",
            "language": "a",
            "speed": 1.0,
            "response_format": "mp3",
        }
    ).encode()


def send_request(url: str, body: bytes) -> dict[str, float | str]:
    started = time.monotonic()
    request = urllib.request.Request(
        url,
        body,
        {"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        response.read()
        row: dict[str, float | str] = {
            "latency_ms": (time.monotonic() - started) * 1000,
            "device": response.headers["X-TTS-Device"],
        }
        for name, header in TIMING_HEADERS.items():
            row[name] = float(response.headers[header])
        return row


def error_text(error: Exception) -> str:
    if isinstance(error, urllib.error.HTTPError):
        return f"HTTP {error.code}: {error.read().decode(errors='replace')}"
    return repr(error)


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile_value * len(ordered)) - 1)
    return ordered[index]


def summarize(rows: list[dict[str, float | str]]) -> dict[str, dict[str, float]]:
    result = {}
    for name in ("latency_ms", *TIMING_HEADERS):
        values = [float(row[name]) for row in rows]
        result[name] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "p95": percentile(values, 0.95),
            "max": max(values),
        }
    return result


def main() -> int:
    args = parse_args()
    body = request_body()

    print(
        f"Warming with {args.warmup_requests} concurrent requests...",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.warmup_requests) as pool:
        warmups = [
            pool.submit(send_request, args.url, body)
            for _ in range(args.warmup_requests)
        ]
        warmup_errors = []
        for future in warmups:
            try:
                future.result()
            except Exception as error:  # noqa: BLE001 - benchmark reports failures
                warmup_errors.append(error_text(error))
    if warmup_errors:
        print(json.dumps({"warmup_errors": warmup_errors}, indent=2))
        return 1

    start_gate = threading.Barrier(args.clients + 1)
    lock = threading.Lock()
    rows: list[dict[str, float | str]] = []
    errors: list[str] = []
    deadline = 0.0

    def client() -> None:
        local_rows = []
        local_errors = []
        start_gate.wait()
        while time.monotonic() < deadline:
            try:
                local_rows.append(send_request(args.url, body))
            except Exception as error:  # noqa: BLE001 - benchmark reports failures
                local_errors.append(error_text(error))
        with lock:
            rows.extend(local_rows)
            errors.extend(local_errors)

    with ThreadPoolExecutor(max_workers=args.clients) as pool:
        futures = [pool.submit(client) for _ in range(args.clients)]
        benchmark_started = time.monotonic()
        deadline = benchmark_started + args.seconds
        start_gate.wait()
        for future in futures:
            future.result()
    benchmark_ended = time.monotonic()

    wall_seconds = benchmark_ended - benchmark_started
    result = {
        "workload": {
            "text": DEFAULT_TEXT,
            "voice": "af_bella",
            "format": "mp3",
            "clients": args.clients,
            "target_seconds": args.seconds,
            "warmup_requests": args.warmup_requests,
        },
        "wall_seconds_including_drain": wall_seconds,
        "completed": len(rows),
        "errors": len(errors),
        "requests_per_second_including_drain": len(rows) / wall_seconds,
        "requests_by_device": dict(
            sorted(Counter(str(row["device"]) for row in rows).items())
        ),
        "metrics_ms": summarize(rows) if rows else {},
        "error_samples": errors[:5],
    }
    print(json.dumps(result, indent=2))
    return int(not rows or bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
