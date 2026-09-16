<h1 align="center">Chorus</h1>

<p align="center"><strong>Local text-to-speech engines behind one CPU/CUDA API and browser console.</strong></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/inference-CPU%20%2B%20CUDA-0f172a" alt="CPU and CUDA inference">
  <img src="https://img.shields.io/badge/environment-Nix%20flake-5277C3?logo=nixos&logoColor=white" alt="Nix flake">
</p>

Chorus runs Pocket TTS, Kokoro, Piper, Kitten TTS, Supertonic 3, Breeze TTS 2, and Fish Audio S2-Pro on one machine. Use the browser to compare voices and render waveforms, or call the shared HTTP endpoint from another application.

- No hosted speech API required. Kokoro supports CPU/CUDA; Breeze and Fish require CUDA.
- Models load on demand, stay warm between requests, and unload after five idle minutes.
- Narration picks are grouped in the console, with optional LavaSR enhancement to 48 kHz.

## Run it

The development flake in `nix/` provides Python 3.12, `uv`, Git LFS, FFmpeg, SoX, eSpeak NG, libsndfile, and Tailwind CSS. Run these commands from the repository root:

```bash
direnv allow
serve-api --list-devices
serve-api --engines kokoro --download-missing --devices cpu,cuda:0
```

The development shell adds `bin/` to `PATH`, so launcher names are unchanged. Their files now live at `bin/serve-api`, `bin/kokoro-tts`, and the other `bin/*-tts` paths.

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

For a new clone, set `GIT_LFS_SKIP_SMUDGE=1` when cloning if you only want selected engines' weights, then use `--download-missing` or a scoped `git lfs pull --include="models/kokoro/**"`. LFS keeps every committed weight version; remote storage and download quotas still apply.

`--devices` sets the physical hardware pool; speech requests choose a CPU or GPU channel rather than a GPU number. `--preload kokoro/82m-v1.0` warms a selected model on its configured default channel after the file checks. `--models-dir PATH` (or `TTS_MODELS_DIR`) selects another model catalog.

CUDA uses ONNX Runtime's GPU wheel, which also supports CPU execution. The environment includes CUDA 12 and cuDNN runtime libraries; an NVIDIA driver is still required. On NixOS, the launcher includes `/run/opengl-driver/lib`. PyTorch and the optional processors remain CPU-based.

### Execution channels

[channels.toml](channels.toml) controls which channels each model may use and its default:

```toml
[models."kokoro/82m-v1.0"]
channels = ["cpu", "gpu"]
default_channel = "gpu"
```

Use `["cpu"]` with `default_channel = "cpu"` for CPU-only Kokoro, or `["gpu"]` for GPU-only. Breeze and Fish are GPU-only in the supplied configuration and cannot enable unsupported CPU execution. Enabling both channels does not preload extra model copies.

Set `--channel-config PATH` or `TTS_CHANNEL_CONFIG` to use another file. The explicit CLI path takes precedence. Policy is loaded at startup; restart after editing. Unknown models, unsupported channels, misspelled keys, and invalid defaults are errors. Custom catalogs need a matching policy file. Models without an entry allow their supported channels and prefer an available GPU, then CPU.

Omitting `channel` uses the model's default. An explicit or configured GPU request never silently falls back to CPU, including when GPUs are busy or unavailable. Disabled or unsupported channels return HTTP 400; unavailable hardware returns 503. The console shows channel availability and the configured default. Physical GPU IDs remain under **Physical diagnostics**.

### GPU queue

`--gpu-queue-size 32` sets the maximum number of **waiting** GPU requests, separate from active inference slots. Choose `5`, `10`, or another nonnegative count; `0` rejects requests whenever all compatible execution slots are occupied.

The shared FIFO queue assigns a physical GPU only when a slot opens, avoiding requests stranded behind a busy GPU while another is free. Dispatch prefers the least-busy GPU, then an already-warm model. Kokoro overlaps two runs per GPU in one ONNX session; other GPU engines run exclusively on their selected GPU. No microbatching or extra model replicas are used. A full queue returns HTTP 429 with `Retry-After: 1`; clients control whether to retry.

Start with 4–8 concurrent clients and leave queue capacity at 32. More waiting slots absorb bursts, but do not increase inference capacity. The [Kokoro measurements](#kokoro-gpu-measurement) show about 23 requests/s at saturation and a steady 20 requests/s without growing backlog on two RTX 3090s.

### Memory and idle unloading

Startup loads no models unless `--preload` is supplied. Each model/device pair runs in its own worker process, retaining weights and optional post-processing models between requests. `--idle-timeout` sets the idle lifetime in seconds (default `300`; `0` unloads once pending work drains). Eviction exits the worker and its nested runtimes, releasing their RAM and GPU allocations. The API process remains running with a small RAM footprint.

Set an aggregate worker RAM budget and separate VRAM budgets for enabled GPUs:

```bash
serve-api --engines kokoro --devices cpu,cuda:0,cuda:1 \
  --idle-timeout 300 \
  --ram-budget-mib 8192 \
  --vram-budget-mib cuda:0=4096 \
  --vram-budget-mib cuda:1=4096
```

Budgets are optional, soft cache targets, not hard allocation limits. Loading and active requests may exceed them. Least-recently-used idle workers are evicted under pressure; workers serving or already reserved for requests are protected. GPU queue entries are not bound to workers until dispatch. A model larger than its budget can serve a request but is unloaded afterward. RAM accounting sums worker-tree RSS (shared pages may be counted more than once); budgets exclude the API process and unrelated applications. VRAM budgets require working `nvidia-smi` telemetry and respect CUDA's logical device ordering.

`GET /v1/resources` reports budgets, worker RAM/VRAM usage, active requests, idle age, and `gpu_queue` counts (`capacity`, `waiting`, `running`). With no workers loaded, model usage is zero and background cache monitoring sleeps. First requests after eviction pay the cold-load cost again.

## Engines

| Engine | CPU channel | GPU channel | Default voice | Sample rate | Notes |
| --- | :---: | :---: | --- | ---: | --- |
| Pocket TTS | Yes | No | `alba` | 24 kHz | INT8-optimized voice cloning and multilingual models |
| Kokoro | Yes | Yes | `af_heart` | 24 kHz | 54 voices; FP32 ONNX |
| Piper | Yes | No | `en_US-lessac-medium` | 22.05 kHz | Small, dependable English model |
| Kitten TTS | Yes | No | `Leo` | 24 kHz | Eight lightweight English voices |
| Supertonic 3 | Yes | No | `M1` | 44.1 kHz | Ten voices and multilingual synthesis |
| Breeze TTS 2 | No | Yes | Voice description | 24 kHz | English/Chinese; research/non-commercial license |
| Fish Audio S2-Pro | No | Yes | Optional speaking style | 44.1 kHz | Multilingual; research/non-commercial license |

Channel support reflects the adapters shipped with Chorus, not every capability of the upstream projects. Per-model policy can disable a supported channel; it cannot enable an unsupported one. GPU inference still uses CPU work, and optional LavaSR and alignment run on CPU.

For English narration, start with Kokoro `af_heart`. The console also marks expressive, audiobook, documentary, and British narration alternatives.

Pocket TTS dynamically quantizes its transformer attention and feed-forward layers to INT8; its flow network and audio decoder remain FP32. On the target Ryzen 3 7320U, this reduced representative warm synthesis latency by about 20%. For the lowest latency without voice cloning, Piper is the fastest included engine, followed by Kitten TTS. LavaSR and word alignment are separate optional stages and add their own processing time.

## Generate audio

```bash
curl --fail-with-body \
  -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "engine": "kokoro",
    "channel": "gpu",
    "response_format": "wav",
    "input": "The room fell quiet as the first page turned.",
    "voice": "af_heart",
    "language": "a",
    "speed": 1.0,
    "lava_sr": true,
    "force_align": true
  }' \
  --output speech.wav
```

Only `engine` and `input` are required. Set `lava_sr` to `true` to post-process the generated speech at 48 kHz. Set `force_align` to `true` for English word timestamps. Engine-specific defaults are listed by `GET /v1/engines`.

Optional `model` and `channel` fields select a model version and execution channel, for example `"model": "82m-v1.0", "channel": "gpu"`. Responses report `X-TTS-Model` and `X-TTS-Channel`; `X-TTS-Device` identifies the physical device for diagnostics. The old request `device` field and `auto` channel are not accepted. Enhancement runs before alignment so word timestamps match the returned audio.

Set `response_format` to choose the output. WAV is returned directly. MP3 uses the bundled LAME encoder through `lameenc`, without starting a subprocess. FLAC and Opus use FFmpeg. Encoding runs after enhancement and alignment.

| `response_format` | Output | Content type |
| --- | --- | --- |
| `wav` | Lossless 16-bit PCM WAV | `audio/wav` |
| `mp3` | MP3 at 128 kbps | `audio/mpeg` |
| `flac` | Lossless FLAC | `audio/flac` |
| `opus` | Ogg Opus at 64 kbps, 48 kHz | `audio/ogg` |

For example, send `"response_format": "mp3"` and save the response as `speech.mp3`. `Content-Disposition` supplies the matching extension; `X-Sample-Rate` reports the output decoding rate. `X-Audio-Duration` and alignment timestamps describe the generated speech before encoding; MP3 can add a small amount of codec delay and padding. Unsupported formats return HTTP 422. Outside the Nix shell, install FFmpeg with `flac` and `libopus` encoders for those formats; MP3 needs only the Python dependencies.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Runtime and loaded-engine status |
| `GET` | `/v1/devices` | Detected physical CPU/CUDA devices and enabled status for diagnostics |
| `GET` | `/v1/resources` | Worker RAM/VRAM usage, GPU queue occupancy, soft cache budgets, and idle state |
| `GET` | `/v1/engines` | Engines, voices, languages, and defaults |
| `GET` | `/v1/models` | Model versions, channel support/policy/availability, defaults, and loaded instances |
| `POST` | `/v1/audio/speech` | Generate WAV, MP3, FLAC, or Ogg Opus audio |
| `GET` | `/v1/audio/alignments/{id}` | Retrieve a generated word-alignment sidecar |
| `GET` | `/docs` | OpenAPI console |

`GET /v1/models` reports each model's `default_channel` and a `channels` list with `id`, `supported`, `enabled`, and `available` fields. `GET /v1/devices` returns physical `devices` with `id`, `type`, and `enabled` fields. Physical detection respects `CUDA_VISIBLE_DEVICES` and is cached for the server process; `--devices` restricts which detected devices the scheduler may use. There is no server-wide `default_device`; defaults are per model.

The speech endpoint accepts text up to 10,000 characters and a speed from `0.5` to `2.0`. Pocket TTS, Breeze, and Fish use a fixed speed of `1.0`. LavaSR and force alignment are disabled unless the request explicitly enables them.

English force alignment uses the permissively licensed `WAV2VEC2_ASR_BASE_960H` model. Its 378 MB weights download on the first aligned request and stay loaded until their model worker is evicted. An aligned audio response includes `X-Alignment-Id` and `X-Alignment-Url`; fetch that URL for word-level `start_ms`, `end_ms`, and confidence scores. The in-memory sidecar cache retains the 32 most recently accessed alignments and resets with the API process. Words without supported English letters are omitted.

Every successful speech response reports backend phase durations in milliseconds through `X-Queue-Time-Ms`, `X-Inference-Time-Ms`, `X-LavaSR-Time-Ms`, `X-Alignment-Time-Ms`, `X-Encoding-Time-Ms`, and `X-Backend-Time-Ms`. `Server-Timing` includes the same phases, and backend total includes encoding. The browser console shows synthesis and processing timing. Inference and alignment timings include lazy model loading on a cold request.

## NixOS service

The root flake exports `nixosModules.default` (also `nixosModules.chorus`) and `packages.x86_64-linux.chorus`. Add the input to your system flake:

```nix
inputs.chorus.url = "git+ssh://git@neo.netbird.cloud:2222/fractal-tess/chorus.git";
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

Nix installs the launcher and system libraries. First startup uses `uv sync --locked` to provision Python dependencies, then downloads missing selected-engine model files before listening. **Python dependencies are provisioned at runtime, not built into the Nix closure.** First startup needs network access and several GB of disk space. State lives in `/var/lib/chorus`, models in `/var/lib/chorus/models`, and download caches in `/var/cache/chorus`; these paths are configurable. Model weights never enter the service package.

Set `host = "0.0.0.0"; openFirewall = true;` to serve other machines on a trusted network. The API has no authentication; do not expose it publicly. Options also cover `port`, `preload`, `idleTimeout`, `gpuQueueSize`, RAM/VRAM budgets, and `channelConfig`. Use `environmentFile` for credentials rather than putting secrets in the Nix store. See [the module](nix/module.nix) for the full option definitions.

To enable another large engine later, add `"breeze"` or `"fish"` to `engines`. The launcher provisions only the selected engines' isolated runtimes, using their pinned upstream commits. Their model licenses restrict commercial use; read the [Breeze](#breeze-setup) and [Fish](#fish-setup) notes first.

Kokoro was verified under systemd's non-root service sandbox on an RTX 3090, including CUDA synthesis and a cached restart with external network access denied.

## Development

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

Standalone Python commands live in `src/chorus/commands/`, their shell launchers in `bin/`, and browser assets and Tailwind configuration in `static/`. Piper and Kitten commands use the versioned local model directories; fetch their manifests' files first. Kitten's `--model` accepts a local directory, not a Hub repository. Generated audio and smoke reports belong in the ignored `outputs/` directory.

Engine code lives in `src/chorus/engines/`; versioned artifacts and manifests live in `models/<engine>/<version>/`. The registry caches isolated workers per engine, model, and device. Kokoro CUDA workers overlap up to two requests in one ONNX session without duplicating model weights; CPU and other engine workers serialize requests. Additional requests wait for a slot. Workers own optional processing from `processing.py` so eviction also releases those models. Programmatic `EngineRegistry` callers must call `close()` when finished. Add an adapter for a new engine or a manifest for another supported model version. Multi-component models can list multiple artifacts; adapters own their runtime details.

### Kokoro GPU measurement

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

### Breeze setup

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

### Fish setup

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

The bundle is about 11 GB, plus another copy in the local Git LFS object store. **Built with Fish Audio.** The [Fish Audio Research License](models/fish/s2-pro/LICENSE.md) covers both upstream code and weights. Research and non-commercial use are permitted; commercial use requires a separate written agreement. The license and required `NOTICE` are retained with the model.
