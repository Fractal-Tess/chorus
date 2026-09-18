# Development

[Project overview](../README.md) · [Setup](setup.md) · [API and runtime](api.md)

Dependencies are locked in `uv.lock` and synchronized when the development shell opens. `.envrc` watches the Python dependency files and uses an explicit `path:` reference to `nix/`; model weights are not copied into the Nix flake source snapshot.

```bash
nix develop path:./nix
python -m compileall -q src/chorus
```

Run the startup, encoding, FFT, and GPU queue regressions without model downloads or CUDA:

```bash
python -m unittest discover -s tests -v
```

Rebuild the local Tailwind stylesheet after changing classes:

```bash
tailwindcss \
  -c static/tailwind.config.js \
  -i static/tailwind.input.css \
  -o static/tailwind.css \
  --minify
```

`chorus-tts` (`src/chorus/speak.py`) synthesizes a single WAV from the shell for any engine, through the same registry the API uses, so it honours channel policy, device selection, and lazy loading. Browser assets and Tailwind configuration live in `static/`. Fetch an engine's manifest files before calling it. Generated audio and smoke reports belong in the ignored `outputs/` directory.

```bash
chorus-tts --engine kokoro --voice af_heart "A quiet test." -o outputs/kokoro.wav
```

Engine code lives in `src/chorus/engines/`; versioned artifacts and manifests live in `models/<engine>/<version>/`. The registry caches isolated workers per engine, model, and device. Kokoro CUDA workers overlap up to two requests in one ONNX session without duplicating model weights; CPU and other engine workers serialize requests. Additional requests wait for a slot. Workers own optional processing from `processing.py` so eviction also releases those models. Programmatic `EngineRegistry` callers must call `close()` when finished. Add an adapter for a new engine or a manifest for another supported model version. Multi-component models can list multiple artifacts; adapters own their runtime details.

## Publishing source without model binaries

The maintainer checkout uses `origin` for Gitadel and `github` for the public
source repository. GitHub receives normal Git commits, including lightweight
LFS pointers, but no LFS objects:

```sh
git push origin main
GIT_LFS_SKIP_PUSH=1 git push github main
```

Keep Gitadel as the LFS destination. In a maintainer checkout, configure
`remote.github.lfsurl` and `remote.github.lfspushurl` to use the Gitadel remote
URL, and check `git lfs env` before publishing. These are local Git settings;
they are not inherited by new clones. Public users should clone with
`GIT_LFS_SKIP_SMUDGE=1` and use `--download-missing`, as shown in the
[tutorials](tutorials.md#clone-source-without-model-weights).

Do not run `git lfs push --all` against GitHub or upload model archives to
GitHub releases. Model usage and redistribution remain governed by upstream
licenses.

## Kokoro GPU measurement

Kokoro produces about **23 MP3 requests/s** on two RTX 3090s. Moving its short STFT from CPU to CUDA raised an earlier matched result from 10.36 to 23.52 requests/s, without extra model replicas or reduced precision. The adapter preserves ONNX Runtime 1.26's float32 Bluestein FFT operation order; an approximate DFT changed near-zero signs and caused downstream phase errors.

The original `model.onnx` and learned weights remain unchanged. Each CUDA worker loads a temporary derived graph, then deletes it after session initialization. This avoids retaining a second serialized copy of the weights in RAM. CPU execution keeps the original graph.

Native MP3 encoding removes process startup overhead. CUDA workers also block their CPU threads while waiting for GPU work, instead of spinning. Matched 60-second trials at **20 requests/s** measured the following changes on a Ryzen 7 2700, using `af_bella`, 9.875 seconds of speech per MP3, warm models, and no optional processing:

| Measurement at 20 requests/s | Before | Now |
| --- | ---: | ---: |
| Median HTTP latency | 244 ms | 195 ms |
| p95 HTTP latency | 302 ms | 250 ms |
| Mean MP3 encoding time | 89 ms | 42 ms |
| Mean Chorus CPU use, logical cores | 4.64 | 2.50 |

Both trials completed 1,200/1,200 requests without growing backlog. CPU use includes the API, its encoder subprocesses before the change, and both model workers. Peak throughput did not improve: separate eight-client trials measured 23.26 requests/s before and 23.04 after, including queue drain. These changes reduce latency and CPU cost rather than GPU inference time.

The FFT replacement matched the original CPU operation on captured speech and synthetic inputs. Full-model checks covered Bella, Heart, Adam, and speeds from 0.5 to 2.0; longer CUDA outputs have pre-existing run-to-run variation. Native 128 kbps MP3 decoded to exactly the same PCM as the former FFmpeg path in speech and synthetic checks at 22.05, 24, 44.1, and 48 kHz, including stereo and short clips. Regression checks run with `.venv/bin/python -m unittest discover -s tests`.

GPU 1 reached 88°C and reported thermal throttling during the latest fixed-rate run. Cooling can limit peak throughput. Extra ONNX sessions and higher shared-session concurrency did not produce a convincing gain, so workers still share one session with two execution slots per GPU. Different passage lengths, optional processing, and other workstation activity change these rates.
