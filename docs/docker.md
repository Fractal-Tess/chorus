# Docker

[Project overview](../README.md) · [Setup](setup.md) · [API and runtime](api.md)

The repository includes a single Linux `amd64` image for Chorus's main Python
runtime. It is built with Python 3.12, Debian slim, and uv 0.12.5. The image
provisions dependencies at build time with `uv sync --locked`; it does not
install the separate Breeze or Fish runtimes.

## Build the image

Build from the repository root:

```sh
docker build -t chorus:local .
```

The image contains source and model manifests for every model named by the
shipped `channels.toml`, but no TTS model weights. The first run with
`--download-missing` fetches the selected engine's pinned artifacts. Keep the
model and cache directories outside the container so later runs reuse them.

## Storage

The container uses `/models` for the writable model catalog and weights, and
`/cache` for download and processor caches. The entrypoint copies shipped
manifests into the selected model directory without deleting existing weights.
`TTS_MODELS_DIR` changes the default from `/models`; an explicit
`--models-dir PATH` takes precedence over that environment variable.

Named volumes are the simplest persistent setup:

```sh
docker volume create chorus-models
docker volume create chorus-cache
```

To keep data on another drive, bind mount directories there instead. The
container runs as UID/GID `10001`, so create the directories with matching
ownership before the first run:

```sh
sudo install -d -o 10001 -g 10001 -m 0750 \
  /mnt/fast/chorus/models /mnt/fast/chorus/cache
```

Use those paths in place of the named volumes:

```sh
-v /mnt/fast/chorus/models:/models \
-v /mnt/fast/chorus/cache:/cache
```

The same UID/GID requirement applies to any host bind mount used for
`TTS_MODELS_DIR`. Do not bake weights into the image or Docker build context.

## CPU

This command publishes the API and console on host loopback port `8002` and
keeps the two named volumes attached:

```sh
docker run --rm --name chorus-cpu \
  -p 127.0.0.1:8002:8000 \
  -v chorus-models:/models \
  -v chorus-cache:/cache \
  chorus:local --engines kokoro --devices cpu --download-missing
```

Request the CPU channel explicitly. A request that omits `channel` follows the
shipped Kokoro policy, whose default is GPU.

```sh
curl --fail-with-body http://127.0.0.1:8002/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"engine":"kokoro","model":"82m-v1.0","channel":"cpu","input":"A quiet test from the container CPU.","response_format":"wav"}' \
  --output kokoro-cpu.wav
```

## NVIDIA CUDA

A working host NVIDIA driver and container GPU integration are prerequisites.
The image includes ONNX Runtime's GPU wheel and pip-provided CUDA/cuDNN
runtime libraries, but it cannot provide the host driver or configure Docker's
GPU runtime. With the NVIDIA Container Toolkit configured for standard Docker,
use `--gpus all`:

```sh
docker run --rm --name chorus-cuda \
  --gpus all \
  -p 127.0.0.1:8003:8000 \
  -v chorus-models:/models \
  -v chorus-cache:/cache \
  chorus:local --engines kokoro --devices cpu,cuda:0,cuda:1 --download-missing
```

On NixOS with NVIDIA CDI configured, use the CDI device instead:

```sh
docker run --rm --name chorus-cuda \
  --device nvidia.com/gpu=all \
  -p 127.0.0.1:8003:8000 \
  -v chorus-models:/models \
  -v chorus-cache:/cache \
  chorus:local --engines kokoro --devices cpu,cuda:0,cuda:1 --download-missing
```

Ask for the GPU channel explicitly:

```sh
curl --fail-with-body http://127.0.0.1:8003/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"engine":"kokoro","model":"82m-v1.0","channel":"gpu","input":"A quiet test from CUDA.","response_format":"wav"}' \
  --output kokoro-gpu.wav
```

If Docker exposes only one host GPU, it is remapped inside the container as
`cuda:0`, even when it was host GPU 1. For example, standard NVIDIA Docker can
use `--gpus '"device=1"'`; pair that with `--devices cpu,cuda:0` and not
`cuda:1`. CDI can expose that same host ordinal with
`--device nvidia.com/gpu=1`; it also appears inside as `cuda:0`.

The image binds HTTP to `0.0.0.0:8000` inside the container. The examples
publish it only on localhost, and the API has no authentication. Keep it on
loopback or put it behind a trusted private network before publishing it to
other machines.

## Entrypoint and scope

The image provides writable `/models` and `/cache` directories. Its entrypoint
installs shipped manifests into the selected model directory, then executes
Chorus with the supplied arguments. Image defaults are `--engines kokoro --devices cpu
--download-missing`; pass explicit arguments when changing engines or device
pools. CPU and GPU are logical request channels: `--devices` declares the
available pool, while speech requests select `"channel": "cpu"` or
`"channel": "gpu"`.

Only the main Python environment is present in this image. Breeze TTS 2 and
Fish Audio S2-Pro need their separate runtimes and are not included; this
image's documented and targeted path is Kokoro.

## Verified configuration

Kokoro `82m-v1.0` was exercised in this image on Linux `amd64`, Docker 29.8,
and two RTX 3090s exposed through NVIDIA CDI:

- CPU synthesis without GPU devices exposed.
- Eight concurrent CUDA requests, with successful responses from both GPUs.
- Host GPU 1 exposed alone and used as container `cuda:0`.
- Fresh model downloads, persistent-volume reuse, and CPU synthesis after
  restarting with `--network none`.
- CLI model-directory override, UID/GID `10001`, healthy container checks,
  browser console rendering, and decoded MP3 output.

The standard `--gpus all` command requires the NVIDIA Docker runtime and was
not the injection mode used for these checks. Other engines have not been
verified in this image.
