<p align="center">
  <img src="assets/logo.png" alt="Chorus: a faceted amber C formed from layered sound arcs" width="200" />
</p>

<p align="center">
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/version-0.1.2-d97706" alt="Version 0.1.2" /></a>
  <a href="docs/docker.md"><img src="https://img.shields.io/badge/Docker-guide-2496ED?logo=docker&logoColor=white" alt="Docker guide" /></a>
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
| Pocket TTS | `default` | ✅ | ❌ |
| Kokoro | `82m-v1.0` | ✅ | ✅ |
| Piper | `default` (`en_US-lessac-medium`) | ✅ | ❌ |
| Kitten TTS | `default` (`kitten-tts-nano-0.8`) | ✅ | ❌ |
| Supertonic 3 | `default` (`supertonic-3`) | ✅ | ❌ |
| Breeze TTS 2 | `tts-2` | ❌ | ✅ |
| Fish Audio S2-Pro | `s2-pro` | ❌ | ✅ |

✅ Supported · ❌ Not supported

These are the model IDs shipped in the manifests and the channels implemented by Chorus adapters. `default` is a manifest model ID, not a claim about every upstream release. Model artifacts are fetched from pinned upstream revisions and checked by SHA-256.

Breeze and Fish have research/non-commercial model licenses; read their [license notes](docs/setup.md#breeze-setup) before use.

## Start manually

Clone the source without model binaries. Install Python 3.12, `uv`, Git, and the system audio tools FFmpeg, SoX, eSpeak NG, and libsndfile.

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Fractal-Tess/chorus.git
cd chorus
uv sync --locked --python 3.12
./bin/serve-api --engines kokoro --download-missing --devices cpu
```

GitHub hosts source and lightweight LFS pointers, not model binaries. `--download-missing` downloads pinned weights from upstream. The shipped Kokoro policy defaults to GPU, so request the CPU channel explicitly:

```sh
curl --fail-with-body http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"engine":"kokoro","model":"82m-v1.0","channel":"cpu","input":"A quiet test from the CPU.","response_format":"wav"}' \
  --output kokoro-cpu.wav
```

Open **[localhost:8000](http://127.0.0.1:8000)** for the console or **[localhost:8000/docs](http://127.0.0.1:8000/docs)** for interactive API documentation. See the [API guide](docs/api.md#generate-audio) for voices, formats, and processing options. An explicit GPU request never silently falls back to CPU.

## Run with Docker

Build `chorus:local`, create persistent model/cache volumes, and start the image with its default Kokoro CPU configuration:

```sh
docker build -t chorus:local .
docker volume create chorus-models
docker volume create chorus-cache
docker run --rm -p 127.0.0.1:8002:8000 \
  -v chorus-models:/models -v chorus-cache:/cache chorus:local
```

Use the CPU speech request above with `http://127.0.0.1:8002/v1/audio/speech`. The [Docker guide](docs/docker.md) covers NVIDIA CUDA, CDI, bind mounts, and ordered model roots. The image installs only Chorus's main Python runtime; Breeze and Fish remain separate runtimes and are not included.

## Run with Nix

The development shell supplies the launcher and system audio tools:

```sh
nix develop path:./nix
serve-api --engines kokoro --download-missing --devices cpu
```

For a persistent NixOS service, import `nixosModules.default` and configure the module:

```nix
services.chorus = {
  enable = true;
  engines = [ "kokoro" ];
  devices = [ "cpu" ];
  downloadMissing = true;
};
```

See the [NixOS setup](docs/setup.md#nixos-service) for service options and the [model storage guide](docs/setup.md#store-models-on-another-drive). The API has no authentication; keep it on localhost or behind a trusted private network.

## Documentation

- [Docker](docs/docker.md): image build, CPU/CUDA and CDI runs, persistent storage, and entrypoint behavior.
- [Tutorials](docs/tutorials.md): source-only cloning, the Nix shell, selective model downloads, CPU/GPU requests, and model storage.
- [Setup](docs/setup.md): model downloads, NVIDIA support, NixOS service, Breeze, and Fish.
- [API and runtime](docs/api.md): engines, voices, channels, GPU queue, memory budgets, and endpoints.
- [Development](docs/development.md): code layout, verification commands, and measured Kokoro performance.
- [Changelog](CHANGELOG.md): release history.
