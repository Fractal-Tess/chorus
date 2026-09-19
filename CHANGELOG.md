# Changelog

## 0.1.6 - 2026-09-19

- Add `docker-compose.yml`: it builds the image, keeps the `models` and `cache` volumes, and publishes port `8002`. `CHORUS_ENGINES`, `CHORUS_DEVICES`, and `CHORUS_PORT` are interpolated into the CLI arguments, so a `.env` file changes the selection without a code path of its own.
- Add `compose.cuda.yml`, a CDI overlay that reserves `nvidia.com/gpu=all` for the service. Verified with Kokoro `82m-v1.0` on two RTX 3090s: CPU and GPU synthesis from one container, and concurrent GPU requests spread across both devices.
- Move the default port from `8000` to `8749`, which is unassigned and outside the ephemeral range, so the console stops colliding with other local services.
- Send `Cache-Control: no-cache` with the console assets. Browsers cached them heuristically and could pair a stale `app.js` with fresh markup after an upgrade, breaking the page until a manual hard reload.
- Style the select popups through `appearance: base-select` where the browser supports it, so voice, language, model, and execution menus follow the theme instead of falling back to the OS widget. Other browsers keep the native list.
- Fit the console into one viewport on desktop: the comparison panel takes the space the composer leaves and scrolls internally rather than pushing the page down.
- Keep the chosen voice and language when the catalog refreshes. Every render re-entered engine selection and snapped both fields back to the engine defaults.
- Start on a channel that can actually run. The shipped Kokoro policy defaults to GPU, so a CPU-only host opened with an unavailable channel selected and the generate button disabled.

## 0.1.5 - 2026-09-19

- Render into two comparison slots instead of a single "latest render" panel. Auto alternates between them so the previous take survives, a linked playhead carries the position across when switching takes, and `A`/`B`/`Space` audition from the keyboard. Slots also swap, clear, and download individually, and the header warns when the two takes came from different scripts.
- Drop Tailwind CSS for hand-written `static/app.css`, removing the generated stylesheet, its config, the `tailwindcss` dev-shell package, and the rebuild step. Browser assets are now plain HTML, CSS, and ES modules with no build.
- Move slot playback into `static/player.js` so both slots share one implementation, and build all dynamic markup through the DOM rather than interpolated HTML strings.

## 0.1.4 - 2026-09-19

- Replace the `bin/` shell launchers with `[project.scripts]` console scripts; the Nix dev shell now exposes them from the project virtualenv.
- Replace the per-engine `kokoro-tts`, `kitten-tts`, and `piper-tts` commands with a single `chorus-tts` covering every engine through the registry, so shell synthesis honours channel policy, device selection, and lazy loading. The Kitten and Piper commands previously bypassed the registry and reimplemented model loading.
- Stop re-validating requests inside the Breeze and Fish runtime workers: both checked a language they then discarded, and Breeze kept a second copy of the adapter's language allowlist. Invalid input now fails in the adapter as a clean rejection instead of surfacing as a worker failure.
- Drop the isolated worker's validation of its own responses and collapse two copies of the process shutdown escalation; the WAV header is the single source of truth for sample rate and duration, and shutdown now closes stdin first so workers exit on EOF.
- Cover the isolated worker transport with tests that exercise the real JSONL protocol against a stub worker, with no GPU or runtime virtualenv required.

## 0.1.3 - 2026-09-18

- Install shipped model manifests from the CLI instead of a Docker entrypoint or Nix launcher: any writable primary root resolves and downloads the shipped models, including a fresh `--models-dir`.
- Offer a libstdc++ at least as new as the host system in the Nix dev shell, so host binaries that inherit its `LD_LIBRARY_PATH` (Hyprland clients, editors) no longer fail with a missing `GLIBCXX_3.4.35`.

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
