#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize speech with Kokoro-82M.")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("-o", "--output", type=Path, default=Path("kokoro_output.wav"))
    parser.add_argument("--voice", default="af_heart", help="Kokoro voice name")
    parser.add_argument("--language", default="a", help="Kokoro language code")
    parser.add_argument("--speed", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = KPipeline(lang_code=args.language)
    chunks = [audio for _, _, audio in pipeline(args.text, voice=args.voice, speed=args.speed)]
    if not chunks:
        raise RuntimeError("Kokoro produced no audio")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    audio = np.concatenate(chunks)
    sf.write(args.output, audio, 24_000)
    print(f"Wrote {args.output} ({len(audio) / 24_000:.2f}s)")


if __name__ == "__main__":
    main()
