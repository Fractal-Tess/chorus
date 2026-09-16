<p align="center">
  <img src="assets/logo.png" alt="Chorus: a faceted amber C formed from layered sound arcs" width="200" />
</p>

<p align="center">
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/version-0.1.0-d97706" alt="Version 0.1.0" /></a>
  <a href="docs/setup.md#nixos-service"><img src="https://img.shields.io/badge/NixOS-service-5277C3?logo=nixos&logoColor=white" alt="NixOS service" /></a>
  <a href="docs/api.md"><img src="https://img.shields.io/badge/API-reference-334155" alt="API reference" /></a>
</p>

<h1 align="center">Chorus</h1>

Chorus runs local text-to-speech engines behind one CPU/CUDA API and browser console.

Compare voices in the browser or generate speech from your own applications. Chorus keeps selected models warm, releases idle workers, and does not require a hosted speech API.

- Seven local engines: Pocket TTS, Kokoro, Piper, Kitten TTS, Supertonic 3, Breeze TTS 2, and Fish Audio S2-Pro.
- Logical CPU/GPU channels, a shared GPU queue, and optional memory budgets.
- WAV, MP3, FLAC, and Opus output, with optional 48 kHz enhancement and English word alignment.

## Supported models

| Engine | Model/version | CPU | GPU (CUDA) |
| --- | --- | :---: | :---: |
| Pocket TTS | `default` | Yes | No |
| Kokoro | `82m-v1.0` | Yes | Yes |
| Piper | `default` (`en_US-lessac-medium`) | Yes | No |
| Kitten TTS | `default` (`kitten-tts-nano-0.8`) | Yes | No |
| Supertonic 3 | `default` (`supertonic-3`) | Yes | No |
| Breeze TTS 2 | `tts-2` | No | Yes |
| Fish Audio S2-Pro | `s2-pro` | No | Yes |

These are the model IDs shipped in the manifests and the channels implemented by Chorus adapters. `default` is a manifest model ID, not a claim about every upstream release. Model artifacts are fetched from pinned upstream revisions and checked by SHA-256.

## Start with Kokoro

Clone without model binaries, then enter the Nix development shell:

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Fractal-Tess/chorus.git
cd chorus
nix develop path:./nix
serve-api --engines kokoro --download-missing --devices cpu,cuda:0
```

Open **[localhost:8000](http://127.0.0.1:8000)** for the console or **[localhost:8000/docs](http://127.0.0.1:8000/docs)** for interactive API documentation. `--download-missing` fetches only the selected engine's missing model artifacts.

GitHub hosts source and lightweight LFS pointers, not model binaries. `--download-missing` downloads pinned weights from upstream. CUDA requires an NVIDIA driver; for CPU-only use, pass `--devices cpu` and request `"channel": "cpu"`.

## Generate speech

```sh
curl --fail-with-body http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"engine":"kokoro","channel":"gpu","input":"The room fell quiet as the first page turned.","response_format":"mp3"}' \
  --output speech.mp3
```

Only `engine` and `input` are required. An explicit GPU request never silently falls back to CPU. See the [API guide](docs/api.md#generate-audio) for voices, formats, and processing options.

## Run as a NixOS service

The root flake provides `nixosModules.default`. After adding the input and importing the module, enable Kokoro:

```nix
services.chorus = {
  enable = true;
  engines = [ "kokoro" ];
  devices = [ "cpu" "cuda:0" ];
  downloadMissing = true;
};
```

The service runs as a dedicated user and keeps models under `/var/lib/chorus/models` by default. Set [`modelsDirectory`](docs/setup.md#store-models-on-another-drive) to load and download models on another drive. Python dependencies are provisioned at runtime with locked `uv` dependencies, not built into the Nix closure. Follow the [complete NixOS setup](docs/setup.md#nixos-service) for the flake input, driver requirements, and service options.

The API has no authentication. Keep it on localhost or behind a trusted private network. Breeze and Fish have research/non-commercial model licenses; read their [setup and license notes](docs/setup.md#breeze-setup) before use.

## Documentation

- [Tutorials](docs/tutorials.md): source-only cloning, the Nix shell, selective model downloads, CPU/GPU requests, and model storage.
- [Setup](docs/setup.md): model downloads, NVIDIA support, NixOS service, Breeze, and Fish.
- [API and runtime](docs/api.md): engines, voices, channels, GPU queue, memory budgets, and endpoints.
- [Development](docs/development.md): code layout, verification commands, and measured Kokoro performance.
- [Changelog](CHANGELOG.md): release history.
