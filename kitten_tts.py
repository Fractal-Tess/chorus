#!/usr/bin/env python3
import argparse
from pathlib import Path

import soundfile as sf
from kittentts import KittenTTS


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize speech with Kitten TTS on CPU.")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("-o", "--output", type=Path, default=Path("kitten_output.wav"))
    parser.add_argument("--voice", default="Leo")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--model", default="KittenML/kitten-tts-nano-0.8")
    args = parser.parse_args()

    model = KittenTTS(args.model)
    audio = model.generate(args.text, voice=args.voice, speed=args.speed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, audio, 24_000)
    print(f"Wrote {args.output} ({len(audio) / 24_000:.2f}s)")


if __name__ == "__main__":
    main()
