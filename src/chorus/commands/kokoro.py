import argparse
from pathlib import Path

from chorus.registry import EngineRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize speech with Kokoro-82M.")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("-o", "--output", type=Path, default=Path("kokoro_output.wav"))
    parser.add_argument("--voice", default="af_heart", help="Kokoro voice name")
    parser.add_argument("--language", default="a", help="Kokoro language code")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:N")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = EngineRegistry()
    try:
        audio = registry.synthesize(
            "kokoro",
            args.text,
            voice=args.voice,
            language=args.language,
            speed=args.speed,
            device=args.device,
        )
    finally:
        registry.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(audio.wav)
    print(f"Wrote {args.output} ({audio.duration:.2f}s, {audio.device})")


if __name__ == "__main__":
    main()
