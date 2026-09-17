# Setup

[Project overview](../README.md) · [API and runtime](api.md) · [Development](development.md)

## Run it

The development flake in `nix/` provides Python 3.12, `uv`, Git LFS, FFmpeg, SoX, eSpeak NG, libsndfile, and Tailwind CSS. Run these commands from the repository root:

```bash
direnv allow
serve-api --list-devices
serve-api --engines kokoro --download-missing --devices cpu,cuda:0
```

The development shell adds the launchers in `bin/` to `PATH`.

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Interactive API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). Set `TTS_HOST` or `TTS_PORT` to change the bind address; for example, `TTS_HOST=0.0.0.0 TTS_PORT=8787 serve-api --engines kokoro`. `serve-api --version` reports the release version.

`--engines` is required when starting the API. Before listening, Chorus checks the selected engines' model files for missing files, empty files, and unmaterialized Git LFS pointers. It exits with the affected paths and repair commands if anything is incomplete. Files belonging to unselected engines are not required.

```bash
# Check local files and refuse to start if anything is missing.
serve-api --engines kokoro,breeze --devices cpu,cuda:0

# Fetch only missing files, verify their SHA-256 hashes, then start.
serve-api --engines kokoro,breeze --download-missing --devices cpu,cuda:0
```

Existing complete files are left untouched. Downloads use pinned revisions from each model manifest; ordinary startup does not fetch models. Checks cover all advertised model versions, voices, and languages of the selected engines, including Pocket's six language checkpoints. Optional LavaSR and alignment resources are separate.

Breeze and Fish also require their isolated runtime setup below. `--download-missing` downloads model assets, not Python environments or upstream source checkouts; missing runtime files produce the corresponding setup commands. `serve-api --fetch-model kokoro/82m-v1.0` remains available for explicit model-only fetching.

For a GitHub clone, use `GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Fractal-Tess/chorus.git`, then `--download-missing`. GitHub contains lightweight LFS pointers but does not host the model binaries. The project's LFS objects remain in Gitadel; a scoped `git lfs pull --include="models/kokoro/**"` is only useful for a clone connected to that LFS server. Public users can download selected models directly from the pinned upstream manifests without Gitadel access.

`--devices` sets the physical hardware pool; speech requests choose a CPU or GPU channel rather than a GPU number. `--preload kokoro/82m-v1.0` warms a selected model on its configured default channel after the file checks. Model storage uses ordered roots: repeat `--models-dir PATH` for each root, or set `TTS_MODELS_DIRS` to a colon-separated list. Explicit CLI roots replace the environment roots. Chorus searches roots in order and selects the first complete, usable `<engine>/<model>/` directory; an incomplete earlier copy does not mask a complete later copy, and artifacts are never combined across roots. If no root has a complete model, downloads go only to the primary (first) root, reusing any partial artifacts there. Secondary roots need read and traverse access only.

CUDA uses ONNX Runtime's GPU wheel, which also supports CPU execution. The environment includes CUDA 12 and cuDNN runtime libraries; an NVIDIA driver is still required. On NixOS, the launcher includes `/run/opengl-driver/lib`. PyTorch and the optional processors remain CPU-based.

## NixOS service

The root flake exports `nixosModules.default` (also `nixosModules.chorus`) and `packages.x86_64-linux.chorus`. Add the input to your system flake:

```nix
inputs.chorus.url = "github:Fractal-Tess/chorus";
```

Include `chorus` in your flake's `outputs` arguments, then import the module in your existing `nixosSystem.modules`:

```nix
modules = [
  ./configuration.nix
  chorus.nixosModules.default
  {
    services.chorus = {
      enable = true;
      engines = [ "kokoro" ];
      devices = [ "cpu" "cuda:0" ];
      downloadMissing = true;
    };
  }
];
```

This targets x86_64 Linux with a working NVIDIA driver. The module does not change the host's driver configuration. Rebuild your system, then check `systemctl status chorus` and `journalctl -u chorus -f`. The API and console listen on `127.0.0.1:8000` by default.

Nix installs the launcher and system libraries. First startup uses `uv sync --locked` to provision Python dependencies, then downloads missing selected-engine model files before listening. **Python dependencies are provisioned at runtime, not built into the Nix closure.** First startup needs network access and several GB of disk space. State lives in `/var/lib/chorus`, models default to `/var/lib/chorus/models`, and download caches default to `/var/cache/chorus`; these paths are configurable. `modelsDirectories` is a nonempty ordered list of normalized absolute paths and defaults to `[ "${cfg.stateDirectory}/models" ]`. Model weights never enter the service package.

Set `host = "0.0.0.0"; openFirewall = true;` to serve other machines on a trusted network. The API has no authentication; do not expose it publicly. Options also cover `port`, `preload`, `idleTimeout`, `gpuQueueSize`, RAM/VRAM budgets, and `channelConfig`. Use `environmentFile` for credentials rather than putting secrets in the Nix store. See [the module](../nix/module.nix) for the full option definitions.

Idle model workers stay loaded for 30 minutes by default. Override the timeout
in seconds with `services.chorus.idleTimeout = 3600;` on NixOS, or
`--idle-timeout 3600` for manual and Docker launches. Preloading warms the
default channel at startup but does not pin the worker. A value of `0` unloads
workers immediately after pending work drains; memory budgets may also evict
idle workers before the timeout.

To enable another large engine later, add `"breeze"` or `"fish"` to `engines`. The launcher provisions only the selected engines' isolated runtimes, using their pinned upstream commits. Their model licenses restrict commercial use; read the [Breeze](#breeze-setup) and [Fish](#fish-setup) notes first.

Kokoro was verified under systemd's non-root service sandbox on an RTX 3090, including CUDA synthesis and a cached restart with external network access denied.

### Store models on another drive

Set ordered roots in your NixOS configuration. Put the fast, writable SSD first and a larger, slower disk second:

```nix
services.chorus = {
  enable = true;
  engines = [ "kokoro" ];
  devices = [ "cpu" "cuda:0" ];
  modelsDirectories = [
    "/mnt/fast/chorus/models"
    "/mnt/archive/chorus/models"
  ];
  downloadMissing = true;
};
```

The roots are searched in order. For each engine and model, Chorus uses the
first root containing a complete, usable model directory. Manifests use the
first occurrence of each model, and Chorus never merges artifacts from
different roots. If no root contains a complete copy, `downloadMissing` writes
only to the first root and reuses partial artifacts already there. Downloads
never mutate secondary roots. The module seeds the shipped manifests into the
primary root only; it does not write manifests or weights to secondary roots.

Configure both drives in NixOS `fileSystems` first. The module includes every
configured root in `RequiresMountsFor`, so Chorus waits for all of them before
starting and never downloads onto an unmounted path on the root filesystem.
Preparation, ownership changes, and the service's writable allowlist cover only
the primary root, state, and cache directories. Secondary roots only need read
and traverse access for the `chorus` user and may be mounted read-only.

Use absolute paths without spaces. Avoid `/home`, `/root`, and `/run/user`:
the service deliberately hides home directories. `cacheDirectory` independently
controls download and runtime caches; `stateDirectory` controls the application
and Python environments. Changing `modelsDirectories` leaves state and cache
paths at their defaults.
The Hugging Face download cache can retain a second copy of downloaded weights.
Put `cacheDirectory` on the larger disk too if SSD space is tight.

To move a complete model between roots, stop Chorus and move the entire
`<engine>/<model>/` directory. Do not split a model's files across roots:

```sh
sudo systemctl stop chorus
sudo install -d -o chorus -g chorus -m 0750 \
  /mnt/archive/chorus/models/kokoro
sudo mv /mnt/fast/chorus/models/kokoro/82m-v1.0 \
  /mnt/archive/chorus/models/kokoro/
sudo systemctl start chorus
journalctl -u chorus -n 30
```

After the restart, the complete copy on the second root is discovered without
a redownload. Ensure the destination has read and traverse permissions for
`chorus` before starting the service. If `downloadMissing = false`, a model
missing or incomplete in every configured root fails startup instead of
triggering a download.

## Breeze setup

[Breeze TTS 2](https://github.com/breezeblue-ai/breeze-tts) uses a separate environment under `runtimes/breeze/`: CUDA PyTorch 2.9.1, NumPy 2, Transformers, and the Qwen audio codec. Its inference source is a pinned Git submodule. The shared API keeps one worker alive per selected model/device and closes workers at shutdown.

```bash
git submodule update --init runtimes/breeze/upstream
uv sync --project runtimes/breeze --locked --python "$UV_PYTHON"
serve-api --engines kokoro,breeze --download-missing --devices cpu,cuda:0
```

Call `/v1/audio/speech` with `"engine": "breeze", "model": "tts-2", "channel": "gpu"`. For this engine, `voice` is an optional natural-language description, such as `"A calm, warm English narrator with clear diction."`; the browser exposes a text field instead of a voice list. Select `language` as `en` or `zh`. Language is inferred from the text, not translated. Speed must remain `1.0`. The existing `lava_sr` and English `force_align` options work on Breeze output.

Breeze loads its local checkpoint and audio tokenizer with Hugging Face offline mode enabled. The model bundle is about 7.7 GB. On the RTX 3090, the eager worker used about 8.2 GiB of GPU memory; one warm request generated 2.64 seconds of audio in 9.4 seconds. Cold startup plus that request took 55 seconds. Flash-attention and fast CUDA-graph paths are not enabled. The upstream runtime's context and generation limits still apply; use short passages rather than book-length requests.

The [Breeze model license](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE) restricts the weights and self-hosted outputs to research/non-commercial use. The Apache-2.0 inference code does not grant commercial rights to the model. The license is retained beside the LFS artifacts. Commercial use requires separate permission.

Large engines can use adapter packages rather than a single file and can own isolated workers like Breeze and Fish. Model manifests support multiple shards, tokenizers, and codecs. Their dependencies do not share the API environment. Reference-audio upload and cloning controls are not part of the current API.

## Fish setup

[Fish Audio S2-Pro](https://github.com/fishaudio/fish-speech) runs locally in `runtimes/fish/`, with pinned inference source and a separate CUDA PyTorch 2.9.1 environment. This is S2-Pro, not the separate [hosted S2.1-Pro offering](https://docs.fish.audio/developer-guide/models-pricing/models-overview). The upstream installation guide calls for a 24 GB GPU.

```bash
git submodule update --init runtimes/fish/upstream
uv sync --project runtimes/fish --locked --python "$UV_PYTHON"
serve-api --engines fish --download-missing --devices cuda:0
```

Use `"engine": "fish", "model": "s2-pro", "channel": "gpu"` with the shared speech endpoint. Put natural-language cues in `input`, for example `"[whisper] Close the door quietly."`. The optional `voice` field adds a leading style cue such as `"warm narration"`; it is not a named voice, reference-audio path, or voice clone. The browser labels this field **Style**.

Language is inferred from the text, not translated. The language list follows the upstream model card; English and Chinese generation were verified locally. Set `language` to `en` for English alignment. Fish's bracket cues and speaker markers are excluded from word timestamps. LavaSR still runs before alignment and returns 48 kHz audio.

Inference is local-only, eager BF16, with the full 32,768-token model context. The worker loads checkpoint parameters without allocating a throwaway FP32 CPU model. On an RTX 3090, two warm requests took a median 22.25 seconds for 3.48 seconds of audio; cold startup plus synthesis took 56.44 seconds. This configuration is not real-time, and compilation is not enabled.

The warm Fish worker occupied about 19 GiB of GPU memory. Fish and Breeze do not fit together on one 24 GB GPU. Use a Fish-only server or separate servers with disjoint `--devices` pools when isolating them. The GPU queue limits concurrent execution, but does not guarantee that multiple resident models fit in VRAM.

The bundle is about 11 GB, plus another copy in the local Git LFS object store. **Built with Fish Audio.** The [Fish Audio Research License](../models/fish/s2-pro/LICENSE.md) covers both upstream code and weights. Research and non-commercial use are permitted; commercial use requires a separate written agreement. The license and required `NOTICE` are retained with the model.
