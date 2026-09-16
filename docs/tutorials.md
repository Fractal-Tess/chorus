# Tutorials

[Project overview](../README.md) · [Setup](setup.md) · [API and runtime](api.md)

These steps keep the source checkout small and fetch only the model you select.

## Clone source without model weights

GitHub hosts the source and lightweight Git LFS pointers, not model binaries. The project's LFS objects remain in Gitadel. Clone with smudge disabled so Git does not try to download weights from GitHub:

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Fractal-Tess/chorus.git
cd chorus
```

`--download-missing` fetches pinned upstream artifacts directly; it does not need access to Gitadel or fetch model weights from GitHub. Do not run `git lfs pull` against the GitHub remote.

## Enter the Nix development shell

From the checkout root:

```sh
nix develop path:./nix
```

The shell provides the `serve-api` launcher and the system tools needed by the local runtime. First startup needs network access for Python dependencies and selected model artifacts.

## Download only Kokoro

Start Chorus with only Kokoro selected. The startup check covers the selected model's files, and `--download-missing` fetches only files that are absent or still LFS pointers:

```sh
serve-api --engines kokoro --download-missing --devices cpu,cuda:0
```

This enables the CPU channel and a CUDA GPU pool containing `cuda:0`. A working NVIDIA driver is required for the GPU pool. For CPU-only hardware, use `--devices cpu` instead.

## Request speech on CPU or GPU

The request names a logical channel. `cuda:0` belongs to the server's physical device pool; clients request `gpu` rather than a device number.

CPU request:

```sh
curl --fail-with-body http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"engine":"kokoro","model":"82m-v1.0","channel":"cpu","input":"A quiet test from the CPU.","response_format":"wav"}' \
  --output kokoro-cpu.wav
```

GPU request:

```sh
curl --fail-with-body http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"engine":"kokoro","model":"82m-v1.0","channel":"gpu","input":"A quiet test from the GPU.","response_format":"wav"}' \
  --output kokoro-gpu.wav
```

An explicit GPU request does not fall back to CPU. If no enabled CUDA device is available, the request returns an error instead. See the [audio endpoint reference](api.md#generate-audio) for voices, formats, and optional processing.

## Put models in another directory

The catalog contains the `ENGINE/MODEL/manifest.json` trees. From a source-only clone, copy the shipped catalog to another directory, then download selected weights there. Keep all manifests so the shipped channel policy can resolve every model it names:

```sh
mkdir -p "$HOME/.local/share/chorus-models"
cp -a models/. "$HOME/.local/share/chorus-models/"
serve-api \
  --models-dir "$HOME/.local/share/chorus-models" \
  --engines kokoro --download-missing --devices cpu,cuda:0
```

The same setting is available as `TTS_MODELS_DIR`. Preserve the manifest paths when adding another shipped model. For NixOS, set `services.chorus.modelsDirectory`; the module installs manifests and manages directory permissions for you. See [storing models on another drive](setup.md#store-models-on-another-drive).

If your checkout already contains downloaded weights, this copy includes them.
The NixOS module copies manifests automatically and never requires copying the
source checkout's model binaries.

## Model files and usage rights

Chorus source code and its download logic do not grant permission to redistribute or commercially use model weights or generated speech. `--download-missing` follows the pinned upstream revisions in each manifest and verifies their SHA-256 hashes. Read and follow the upstream license for every model you download, especially the research/non-commercial terms for Breeze TTS 2 and Fish Audio S2-Pro.
