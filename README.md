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

Compare voices in the browser or generate speech from your own applications. Choose which engines to run; Chorus checks their model files, keeps loaded models warm, and releases idle workers. No hosted speech API is required.

- Seven engines: Pocket TTS, Kokoro, Piper, Kitten TTS, Supertonic 3, Breeze TTS 2, and Fish Audio S2-Pro.
- CPU/GPU channel selection, a shared GPU queue, and optional memory budgets.
- WAV, MP3, FLAC, and Opus output, with optional 48 kHz enhancement and English word alignment.

## Start with Kokoro

From the repository root, enter the Nix development shell and start the API:

```sh
nix develop path:./nix
serve-api --engines kokoro --download-missing --devices cpu,cuda:0
```

Open **[localhost:8000](http://127.0.0.1:8000)** for the console or **[localhost:8000/docs](http://127.0.0.1:8000/docs)** for interactive API documentation.

CUDA needs a working NVIDIA driver. For CPU-only use, pass `--devices cpu` and request `"channel": "cpu"`. First startup needs network access to install dependencies and download the selected models. See [setup](docs/setup.md) for selective Git LFS cloning, channel defaults, and the separate Breeze/Fish runtimes.

## Generate speech

```sh
curl --fail-with-body http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "engine": "kokoro",
    "channel": "gpu",
    "input": "The room fell quiet as the first page turned.",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

Only `engine` and `input` are required. An explicit GPU request never silently falls back to CPU. See the [engine table](docs/api.md#engines) for supported channels and the [API guide](docs/api.md#generate-audio) for voices, formats, and processing options.

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

The service runs as a dedicated user and keeps models under `/var/lib/chorus`. Python dependencies are installed at runtime with locked `uv` dependencies, not built into the Nix closure. Follow the [complete NixOS setup](docs/setup.md#nixos-service) for the flake input, driver requirements, and service options.

The API has no authentication. Keep it on localhost or behind a trusted private network. Breeze and Fish have research/non-commercial model licenses; check their [setup and license notes](docs/setup.md#breeze-setup) before use.

## Documentation

- [Setup](docs/setup.md): model downloads, NVIDIA support, NixOS service, Breeze, and Fish.
- [API and runtime](docs/api.md): engines, voices, channels, GPU queue, memory budgets, and endpoints.
- [Development](docs/development.md): code layout, verification commands, and measured Kokoro performance.
- [Changelog](CHANGELOG.md): release history.
