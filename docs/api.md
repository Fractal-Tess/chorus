# API and runtime

[Project overview](../README.md) · [Setup](setup.md) · [Development](development.md)

## Execution channels

[channels.toml](../channels.toml) controls which channels each model may use and its default:

```toml
[models."kokoro/82m-v1.0"]
channels = ["cpu", "gpu"]
default_channel = "gpu"
```

Use `["cpu"]` with `default_channel = "cpu"` for CPU-only Kokoro, or `["gpu"]` for GPU-only. Breeze and Fish are GPU-only in the supplied configuration and cannot enable unsupported CPU execution. Enabling both channels does not preload extra model copies.

Set `--channel-config PATH` or `TTS_CHANNEL_CONFIG` to use another file. The explicit CLI path takes precedence. Policy is loaded at startup; restart after editing. Unknown models, unsupported channels, misspelled keys, and invalid defaults are errors. Custom catalogs need a matching policy file. Models without an entry allow their supported channels and prefer an available GPU, then CPU.

Omitting `channel` uses the model's default. An explicit or configured GPU request never silently falls back to CPU, including when GPUs are busy or unavailable. Disabled or unsupported channels return HTTP 400; unavailable hardware returns 503. The console shows channel availability and the configured default. Physical GPU IDs remain under **Physical diagnostics**.

## GPU queue

`--gpu-queue-size 32` sets the maximum number of **waiting** GPU requests, separate from active inference slots. Choose `5`, `10`, or another nonnegative count; `0` rejects requests whenever all compatible execution slots are occupied.

The shared FIFO queue assigns a physical GPU only when a slot opens, avoiding requests stranded behind a busy GPU while another is free. Dispatch prefers the least-busy GPU, then an already-warm model. Kokoro overlaps two runs per GPU in one ONNX session; other GPU engines run exclusively on their selected GPU. No microbatching or extra model replicas are used. A full queue returns HTTP 429 with `Retry-After: 1`; clients control whether to retry.

Start with 4–8 concurrent clients and leave queue capacity at 32. More waiting slots absorb bursts, but do not increase inference capacity. The [Kokoro measurements](development.md#kokoro-gpu-measurement) show about 23 requests/s at saturation and a steady 20 requests/s without growing backlog on two RTX 3090s.

## Memory and idle unloading

Startup loads no models unless `--preload` is supplied. Each model/device pair runs in its own worker process, retaining weights and optional post-processing models between requests. `--idle-timeout` sets the idle lifetime in seconds (default `1800`, or 30 minutes; `0` unloads once pending work drains). Eviction exits the worker and its nested runtimes, releasing their RAM and GPU allocations. The API process remains running with a small RAM footprint.

Set an aggregate worker RAM budget and separate VRAM budgets for enabled GPUs:

```bash
serve-api --engines kokoro --devices cpu,cuda:0,cuda:1 \
  --idle-timeout 1800 \
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
  -X POST http://127.0.0.1:8749/v1/audio/speech \
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
