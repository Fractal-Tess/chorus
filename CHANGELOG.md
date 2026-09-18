# Changelog

## 0.1.3 - 2026-09-18

- Install shipped model manifests from the CLI instead of a Docker entrypoint or Nix launcher: any writable primary root resolves and downloads the shipped models, including a fresh `--models-dir`.

## 0.1.2 - 2026-09-17

- Compare source contents during NixOS upgrades so equal-sized files with normalized Nix timestamps are refreshed, including package version metadata.

## 0.1.1 - 2026-09-17

- Keep idle model workers loaded for 30 minutes by default; retain configurable timeouts through `--idle-timeout` and `services.chorus.idleTimeout`.
- Add Docker support with persistent model/cache storage, verified Kokoro CPU and dual-GPU execution, and NVIDIA CUDA/CDI run instructions.
- Reorder README usage examples: manual setup, Docker, then the Nix flake and NixOS.
- NixOS service: Kokoro CUDA defaults, configurable engines and persistent models.
- Add a transparent amber Chorus logo and a shorter README with linked setup, API, and development guides.
- Wait for configured storage mounts and prepare writable directories before starting the NixOS service; document moving model storage to another drive.
- Add ordered model storage across CLI, Docker, and NixOS: the first complete model wins, downloads stay in the primary root, and moved model directories are rediscovered after restarting.
- Add public source-only GitHub clone tutorials and a supported-model CPU/CUDA table.
- Use support-status emojis in the README CPU/CUDA table.

## 0.1.0 - 2026-09-16

- Seven local TTS engines, CPU/CUDA routing, and browser console.
- Engine-scoped startup checks and `--download-missing`.
- Pinned model artifacts; isolated workers with idle eviction.
- FIFO GPU queue, channel policies, and memory budgets.
- Kokoro GPU FFT: about 23 MP3 requests/s on two RTX 3090s.
- Native MP3: 20% lower latency and 46% less CPU use.
- WAV, MP3, FLAC, Opus; optional LavaSR and English alignment.
- Voice/style controls, waveform playback, timings, and Nix setup.
