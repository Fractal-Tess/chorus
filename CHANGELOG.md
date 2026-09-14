# Changelog

## Unreleased

- Added `serve-api` to the Nix development shell as a direct command for starting the local server.
- Enabled Pocket TTS dynamic INT8 quantization, reducing representative warm synthesis latency by about 20% and lowering model memory use.

## 0.1.0 - 2026-08-26

- Added a local CPU-only speech API and browser console for Pocket TTS, Kokoro, Piper, Kitten TTS, and Supertonic 3.
- Added per-engine voice, language, and speed controls, plus curated narration voices.
- Added WAV playback, waveform seeking, downloads, light and dark themes, and responsive layouts.
- Added optional LavaSR post-processing with 48 kHz output.
- Added optional English word alignment using Wav2Vec2, including timestamp sidecars and clickable words in the browser.
- Added millisecond timing breakdowns for queueing, inference, LavaSR, alignment, and total backend processing.
- Added a reproducible Nix and direnv development environment with CPU-only model dependencies.
