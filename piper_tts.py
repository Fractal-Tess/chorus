#!/usr/bin/env python3
import argparse
import wave
from pathlib import Path

from piper import PiperVoice


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize speech with Piper on CPU.")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("-o", "--output", type=Path, default=Path("piper_output.wav"))
    parser.add_argument("--model", type=Path, default=Path(__file__).parent / "models/piper/en_US-lessac-medium.onnx")
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(f"Piper model not found: {args.model}")

    voice = PiperVoice.load(args.model, use_cuda=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output), "wb") as wav_file:
        voice.synthesize_wav(args.text, wav_file)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
