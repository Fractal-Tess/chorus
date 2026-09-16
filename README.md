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
git lfs install
git lfs pull --include="models/kokoro/**"
serve-api --list-devices
serve-api --devices cpu,cuda:0
```

The development shell adds `bin/` to `PATH`, so launcher names are unchanged. Their files now live at `bin/serve-api`, `bin/kokoro-tts`, and the other `bin/*-tts` paths.

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Interactive API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). Set `TTS_HOST` or `TTS_PORT` to change the bind address; for example, `TTS_HOST=0.0.0.0 TTS_PORT=8787 serve-api`.

Kokoro weights and all 54 voice tensors live in `models/kokoro/82m-v1.0/`, tracked with Git LFS. Missing files can also be fetched from checksum-verified, pinned upstream revisions with `serve-api --fetch-model kokoro/82m-v1.0`. Inference never downloads Kokoro weights or voice tensors. Language-processing resources and the other engines' upstream libraries retain their existing download behavior.

For a new clone, set `GIT_LFS_SKIP_SMUDGE=1` when cloning if you only want selected engines' weights, then use the scoped `git lfs pull` commands. LFS keeps every committed weight version; remote storage and download quotas still apply.

`--devices` sets the physical hardware pool; speech requests choose a CPU or GPU channel rather than a GPU number. `--engines kokoro` restricts the engine list. `--preload kokoro/82m-v1.0` warms the model on its configured default channel before accepting requests. `--models-dir PATH` (or `TTS_MODELS_DIR`) selects another model catalog.

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

Start with 4–8 concurrent clients and leave queue capacity at 32. More waiting slots absorb bursts, but do not increase inference capacity. The [Kokoro measurements](#kokoro-gpu-measurement) show 23.5 requests/s at saturation and a steady 20 requests/s without growing backlog on two RTX 3090s.

### Memory and idle unloading

Startup loads no models unless `--preload` is supplied. Each model/device pair runs in its own worker process, retaining weights and optional post-processing models between requests. `--idle-timeout` sets the idle lifetime in seconds (default `300`; `0` unloads once pending work drains). Eviction exits the worker and its nested runtimes, releasing their RAM and GPU allocations. The API process remains running with a small RAM footprint.

Set an aggregate worker RAM budget and separate VRAM budgets for enabled GPUs:

```bash
serve-api --devices cpu,cuda:0,cuda:1 \
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

Set `response_format` to choose the output. WAV is the default and needs no encoder process; the other formats use the bundled FFmpeg after enhancement and alignment.

| `response_format` | Output | Content type |
| --- | --- | --- |
| `wav` | Lossless 16-bit PCM WAV | `audio/wav` |
| `mp3` | MP3 at 128 kbps | `audio/mpeg` |
| `flac` | Lossless FLAC | `audio/flac` |
| `opus` | Ogg Opus at 64 kbps, 48 kHz | `audio/ogg` |

For example, send `"response_format": "mp3"` and save the response as `speech.mp3`. `Content-Disposition` supplies the matching extension; `X-Sample-Rate` reports the output decoding rate. `X-Audio-Duration` and alignment timestamps describe the generated speech before encoding; MP3 can add a small amount of codec delay and padding. Unsupported formats return HTTP 422. Outside the Nix shell, install FFmpeg with `libmp3lame`, `flac`, and `libopus` encoders.

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

## Development

Dependencies are locked in `uv.lock` and synchronized when the development shell opens. `.envrc` watches the Python dependency files and uses an explicit `path:` reference to `nix/`; model weights are not copied into the Nix flake source snapshot.

```bash
nix develop path:./nix
python -m compileall -q src/chorus
```

Run the GPU queue regression checks without loading models or requiring CUDA:

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

Standalone Python commands live in `src/chorus/commands/`, their shell launchers in `bin/`, and browser assets and Tailwind configuration in `static/`. Generated audio and smoke reports belong in the ignored `outputs/` directory.

Engine code lives in `src/chorus/engines/`; versioned artifacts and manifests live in `models/<engine>/<version>/`. The registry caches isolated workers per engine, model, and device. Kokoro CUDA workers overlap up to two requests in one ONNX session without duplicating model weights; CPU and other engine workers serialize requests. Additional requests wait for a slot. Workers own optional processing from `processing.py` so eviction also releases those models. Programmatic `EngineRegistry` callers must call `close()` when finished. Add an adapter for a new engine or a manifest for another supported model version. Multi-component models can list multiple artifacts; adapters own their runtime details.

### Kokoro GPU measurement

Moving Kokoro's short STFT from CPU to CUDA increased warm MP3 throughput **2.27×**, without additional model replicas. The adapter preserves ONNX Runtime 1.26's float32 Bluestein FFT operation order. A simpler, mathematically equivalent DFT changed near-zero signs and caused large downstream phase differences.

The original `model.onnx` and learned weights remain unchanged. Each CUDA worker loads a temporary derived graph, then deletes it after session initialization. This avoids retaining a second serialized copy of the weights in RAM. CPU execution keeps the original graph.

Matched 60-second trials used eight concurrent clients, two RTX 3090s, a Ryzen 7 2700, ONNX Runtime 1.26, and `af_bella` producing 9.875 seconds of English speech per MP3. Model loading and warmup are excluded; throughput includes queue drain. Optional processing was disabled.

| Measurement | Original graph | GPU FFT |
| --- | ---: | ---: |
| Completed requests/s | 10.36 | 23.52 |
| Median HTTP latency | 765 ms | 333 ms |
| p95 HTTP latency | 887 ms | 409 ms |
| Mean GPU utilization, GPU 0 / GPU 1 | 40% / 38% | 84% / 93% |
| Chorus VRAM per GPU | 1.30 GiB | 1.30 GiB |

All 628 original-graph and 1,418 optimized-graph requests succeeded. A separate fixed-rate trial completed 1,200/1,200 requests at 20 requests/s, with 227 ms median latency, 292 ms p95, and no growing queue. After reducing graph-loading memory, another 600 requests at 20 requests/s passed; the two GPU workers used 3.16 GiB of RAM in total.

The FFT replacement matched the original CPU operation bit-for-bit on captured speech and synthetic inputs. Full-model checks covered Bella, Heart, Adam, and speeds from 0.5 to 2.0. Deterministic cases matched exactly or at float32 rounding noise. Longer CUDA outputs have pre-existing run-to-run variation, confirmed with repeated original-graph runs rather than assuming every difference came from the optimization. Regression checks run with `.venv/bin/python -m unittest discover -s tests`.

Utilization is board-wide, including the desktop on GPU 1. During saturation, GPU 1 reached 87°C and reported thermal throttling. Cooling can limit peak throughput. Extra ONNX sessions did not improve throughput after the FFT change, so workers still share one session per GPU. Longer passages and optional processing change these rates.

### Breeze setup

[Breeze TTS 2](https://github.com/breezeblue-ai/breeze-tts) uses a separate environment under `runtimes/breeze/`: CUDA PyTorch 2.9.1, NumPy 2, Transformers, and the Qwen audio codec. Its inference source is a pinned Git submodule. The shared API keeps one worker alive per selected model/device and closes workers at shutdown.

```bash
git submodule update --init runtimes/breeze/upstream
uv sync --project runtimes/breeze --locked --python "$UV_PYTHON"
git lfs pull --include="models/breeze/**"
# Alternatively: serve-api --fetch-model breeze/tts-2
serve-api --engines kokoro,breeze --devices cpu,cuda:0
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
git lfs pull --include="models/fish/**"
# Alternatively: serve-api --fetch-model fish/s2-pro
serve-api --engines fish --devices cuda:0
```

Use `"engine": "fish", "model": "s2-pro", "channel": "gpu"` with the shared speech endpoint. Put natural-language cues in `input`, for example `"[whisper] Close the door quietly."`. The optional `voice` field adds a leading style cue such as `"warm narration"`; it is not a named voice, reference-audio path, or voice clone. The browser labels this field **Style**.

Language is inferred from the text, not translated. The language list follows the upstream model card; English and Chinese generation were verified locally. Set `language` to `en` for English alignment. Fish's bracket cues and speaker markers are excluded from word timestamps. LavaSR still runs before alignment and returns 48 kHz audio.

Inference is local-only, eager BF16, with the full 32,768-token model context. The worker loads checkpoint parameters without allocating a throwaway FP32 CPU model. On an RTX 3090, two warm requests took a median 22.25 seconds for 3.48 seconds of audio; cold startup plus synthesis took 56.44 seconds. This configuration is not real-time, and compilation is not enabled.

The warm Fish worker occupied about 19 GiB of GPU memory. Fish and Breeze do not fit together on one 24 GB GPU. Use a Fish-only server or separate servers with disjoint `--devices` pools when isolating them. The GPU queue limits concurrent execution, but does not guarantee that multiple resident models fit in VRAM.

The bundle is about 11 GB, plus another copy in the local Git LFS object store. **Built with Fish Audio.** The [Fish Audio Research License](models/fish/s2-pro/LICENSE.md) covers both upstream code and weights. Research and non-commercial use are permitted; commercial use requires a separate written agreement. The license and required `NOTICE` are retained with the model.
