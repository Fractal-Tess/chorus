"""Synthesize one file from the shell, through the same registry the API uses."""

from __future__ import annotations

import argparse
from pathlib import Path

from chorus.engines import ENGINE_INFO


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="chorus-tts", description="Synthesize speech with any supported engine."
    )
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--engine", required=True, choices=sorted(ENGINE_INFO))
    parser.add_argument(
        "-o", "--output", type=Path, help="WAV path (default: <engine>.wav)"
    )
    parser.add_argument("--model", help="Model version (default: the engine's default)")
    parser.add_argument("--voice", help="Engine voice, style, or description")
    parser.add_argument("--language")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--channel", choices=("cpu", "gpu"))
    args = parser.parse_args()

    from chorus.registry import EngineRegistry

    registry = EngineRegistry(enabled=[args.engine])
    try:
        audio = registry.synthesize(
            args.engine,
            args.text,
            voice=args.voice,
            language=args.language,
            speed=args.speed,
            model=args.model,
            channel=args.channel,
        )
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    finally:
        registry.close()

    output = args.output or Path(f"{args.engine}.wav")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(audio.wav)
    print(f"Wrote {output} ({audio.duration:.2f}s, {audio.model} on {audio.device})")


if __name__ == "__main__":
    main()
