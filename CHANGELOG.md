# Changelog

## Unreleased

- Added `GET /v1/devices` with detected CPU/CUDA device IDs, enabled status, and the default device policy, including a response schema in the interactive API docs.
- Consolidated shell launchers in `bin/`, standalone Python commands in `src/chorus/commands/`, and Tailwind configuration in `static/`; launcher names remain unchanged in the development shell.
- Isolated the development flake and lock in `nix/`, with a direnv `path:` reference that keeps model weights out of Nix source snapshots.
- Added local Fish Audio S2-Pro CUDA synthesis with pinned source, checksum-verified Git LFS weights, and a separate locked runtime.
- Added Fish speaking-style controls, its multilingual catalog, and LavaSR/English alignment that excludes unspoken style and speaker markers.
- Shared the persistent worker transport with Breeze. Fish initializes checkpoint parameters on the meta device to avoid a throwaway CPU model, and reports initialization failures without hanging on upstream's queue startup event.
- Renamed Mini TTS to Chorus, including the Python package, API/browser branding, and isolated Breeze runtime metadata.
- Added Breeze TTS 2 CUDA voice design through an isolated, persistent worker with pinned inference source and a separate locked environment.
- Stored Breeze model shards, audio tokenizer, provenance checksums, and non-commercial license under `models/breeze/tts-2`; large artifacts use Git LFS.
- Added browser voice-description input and shared LavaSR/English alignment support for Breeze output, plus worker shutdown cleanup.
- Kept CPU-only server startup from probing CUDA and made invalid model-fetch selectors report CLI usage errors.
- Split the shared API into engine adapters, a versioned model catalog, per-model/device instance caching, and separate post-processing.
- Added Kokoro CUDA inference, explicit device policy, device discovery, model selection, and startup warmup. RTX 3090 warm ONNX inference measured 94 ms for 3.125 seconds of speech versus 1,513 ms on CPU.
- Added checksum-pinned Kokoro weights and 54 voices in Git LFS, with explicit model fetching and missing-LFS-pointer diagnostics.
- Preserved optional CPU LavaSR enhancement followed by English forced alignment, including resolved model/device and phase timing headers.
- Fixed forced alignment of hyphenated words and kept plain synthesis independent of queued post-processing.
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
