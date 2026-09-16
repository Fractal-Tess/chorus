# Changelog

## Unreleased

- NixOS service: Kokoro CUDA defaults, configurable engines and persistent models.

## 0.1.0 - 2026-09-16

- Seven local TTS engines, CPU/CUDA routing, and browser console.
- Engine-scoped startup checks and `--download-missing`.
- Pinned model artifacts; isolated workers with idle eviction.
- FIFO GPU queue, channel policies, and memory budgets.
- Kokoro GPU FFT: about 23 MP3 requests/s on two RTX 3090s.
- Native MP3: 20% lower latency and 46% less CPU use.
- WAV, MP3, FLAC, Opus; optional LavaSR and English alignment.
- Voice/style controls, waveform playback, timings, and Nix setup.
