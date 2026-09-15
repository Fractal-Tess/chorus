<h1 align="center">Mini TTS</h1>

<p align="center"><strong>Local text-to-speech engines behind one CPU/CUDA API and browser console.</strong></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/inference-CPU%20%2B%20CUDA-0f172a" alt="CPU and CUDA inference">
  <img src="https://img.shields.io/badge/environment-Nix%20flake-5277C3?logo=nixos&logoColor=white" alt="Nix flake">
</p>

Mini TTS runs Pocket TTS, Kokoro, Piper, Kitten TTS, Supertonic 3, and Breeze TTS 2 on one machine. Use the browser to compare voices and render waveforms, or call the shared HTTP endpoint from another application.

- No hosted speech API required. Kokoro supports CPU/CUDA; Breeze requires CUDA.
- Models load on demand and stay warm for later requests.
- Narration picks are grouped in the console, with optional LavaSR enhancement to 48 kHz.

## Run it

The included Nix flake provides Python 3.12, `uv`, Git LFS, FFmpeg, SoX, eSpeak NG, libsndfile, and Tailwind CSS.

```bash
direnv allow
git lfs install
git lfs pull --include="models/kokoro/**"
serve-api --list-devices
serve-api --devices cpu,cuda:0
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Interactive API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). Set `TTS_HOST` or `TTS_PORT` to change the bind address; for example, `TTS_HOST=0.0.0.0 TTS_PORT=8787 serve-api`.

Kokoro weights and all 54 voice tensors live in `models/kokoro/82m-v1.0/`, tracked with Git LFS. Missing files can also be fetched from checksum-verified, pinned upstream revisions with `serve-api --fetch-model kokoro/82m-v1.0`. Inference never downloads Kokoro weights or voice tensors. Language-processing resources and the other engines' upstream libraries retain their existing download behavior.

For a new clone, set `GIT_LFS_SKIP_SMUDGE=1` when cloning if you only want selected engines' weights, then use the scoped `git lfs pull` commands. LFS keeps every committed weight version; remote storage and download quotas still apply.

`--devices` restricts execution to listed devices. `--default-device auto` prefers an enabled GPU when the selected model supports it, otherwise CPU. Explicit unavailable or unsupported devices are errors; Kokoro refuses whole-session CPU fallback. `--engines kokoro` restricts the engine list. `--preload kokoro/82m-v1.0` loads and warms the model before accepting requests. `--models-dir PATH` (or `TTS_MODELS_DIR`) selects another model catalog.

CUDA uses ONNX Runtime's GPU wheel, which also supports CPU execution. The environment includes CUDA 12 and cuDNN runtime libraries; an NVIDIA driver is still required. On NixOS, the launcher includes `/run/opengl-driver/lib`. PyTorch and the optional processors remain CPU-based.

## Engines

| Engine | Default voice | Sample rate | Notes |
| --- | --- | ---: | --- |
| Pocket TTS | `alba` | 24 kHz | INT8-optimized voice cloning and multilingual models |
| Kokoro | `af_heart` | 24 kHz | 54 voices; FP32 ONNX on CPU or CUDA |
| Piper | `en_US-lessac-medium` | 22.05 kHz | Small, dependable English model |
| Kitten TTS | `Leo` | 24 kHz | Eight lightweight English voices |
| Supertonic 3 | `M1` | 44.1 kHz | Ten voices and multilingual synthesis |
| Breeze TTS 2 | Voice description | 24 kHz | English/Chinese; CUDA; research/non-commercial license |

For English narration, start with Kokoro `af_heart`. The console also marks expressive, audiobook, documentary, and British narration alternatives.

Pocket TTS dynamically quantizes its transformer attention and feed-forward layers to INT8; its flow network and audio decoder remain FP32. On the target Ryzen 3 7320U, this reduced representative warm synthesis latency by about 20%. For the lowest latency without voice cloning, Piper is the fastest included engine, followed by Kitten TTS. LavaSR and word alignment are separate optional stages and add their own processing time.

## Generate audio

```bash
curl --fail-with-body \
  -X POST http://127.0.0.1:8787/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "engine": "kokoro",
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

Optional `model` and `device` fields select a model version and device, for example `"model": "82m-v1.0", "device": "cuda:0"`. Responses report the resolved choice in `X-TTS-Model` and `X-TTS-Device`. Enhancement runs before alignment so word timestamps match the returned audio.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Runtime and loaded-engine status |
| `GET` | `/v1/engines` | Engines, voices, languages, and defaults |
| `GET` | `/v1/models` | Model versions, supported/allowed devices, loaded instances |
| `POST` | `/v1/audio/speech` | Generate a WAV response |
| `GET` | `/v1/audio/alignments/{id}` | Retrieve a generated word-alignment sidecar |
| `GET` | `/docs` | OpenAPI console |

The speech endpoint accepts text up to 10,000 characters and a speed from `0.5` to `2.0`. Pocket TTS uses a fixed speed. LavaSR and force alignment are disabled unless the request explicitly enables them.

English force alignment uses the permissively licensed `WAV2VEC2_ASR_BASE_960H` model. Its 378 MB weights download on the first aligned request and remain loaded afterward. An aligned WAV response includes `X-Alignment-Id` and `X-Alignment-Url`; fetch that URL for word-level `start_ms`, `end_ms`, and confidence scores. The in-memory sidecar cache retains the 32 most recently accessed alignments and resets with the API process. Words without supported English letters are omitted.

Every successful speech response reports backend phase durations in milliseconds through `X-Queue-Time-Ms`, `X-Inference-Time-Ms`, `X-LavaSR-Time-Ms`, `X-Alignment-Time-Ms`, and `X-Backend-Time-Ms`. The same values are included in the standard `Server-Timing` header and shown with the latest render in the browser console. Inference and alignment timings include lazy model loading on a cold request.

## Development

Dependencies are locked in `uv.lock` and synchronized when the development shell opens.

```bash
nix develop
python -m compileall -q src/mini_tts
```

Rebuild the local Tailwind stylesheet after changing classes:

```bash
tailwindcss \
  -c tailwind.config.js \
  -i static/tailwind.input.css \
  -o static/tailwind.css \
  --minify
```

Engine code lives in `src/mini_tts/engines/`; versioned artifacts and manifests live in `models/<engine>/<version>/`. The registry caches separate instances per engine, model, and device, with a lock per instance. Optional processing is separate in `processing.py`. Add an adapter for a new engine or a manifest for another supported model version. Multi-component models can list multiple artifacts; adapters own their runtime details.

### Kokoro GPU measurement

On an RTX 3090 with ONNX Runtime 1.26, five warm inference runs gave median latencies of 94 ms for 3.125 seconds of speech and 295 ms for 10.725 seconds. CPU medians on the Ryzen 7 2700 (four ONNX threads) were 1,513 ms and 5,214 ms. These exclude model loading, text processing, WAV encoding, and optional processing. An HTTP smoke request including text processing and WAV encoding took 101 ms warm.

CUDA profiling recorded 1,895 GPU nodes and 182 CPU nodes. Some operations, including STFT and indexing, still use CPU. Heuristic convolution selection performed as well as exhaustive selection without extra tuning work. The upstream FP16 candidate produced non-finite audio and was not selected.

### Breeze setup

[Breeze TTS 2](https://github.com/breezeblue-ai/breeze-tts) uses a separate environment under `runtimes/breeze/`: CUDA PyTorch 2.9.1, NumPy 2, Transformers, and the Qwen audio codec. Its inference source is a pinned Git submodule. The shared API keeps one worker alive per selected model/device and closes workers at shutdown.

```bash
git submodule update --init runtimes/breeze/upstream
uv sync --project runtimes/breeze --locked --python "$UV_PYTHON"
git lfs pull --include="models/breeze/**"
# Alternatively: serve-api --fetch-model breeze/tts-2
serve-api --engines kokoro,breeze --devices cpu,cuda:0
```

Call `/v1/audio/speech` with `"engine": "breeze", "model": "tts-2", "device": "cuda:0"`. For this engine, `voice` is an optional natural-language description, such as `"A calm, warm English narrator with clear diction."`; the browser exposes a text field instead of a voice list. Select `language` as `en` or `zh`. Language is inferred from the text, not translated. Speed must remain `1.0`. The existing `lava_sr` and English `force_align` options work on Breeze output.

Breeze loads its local checkpoint and audio tokenizer with Hugging Face offline mode enabled. The model bundle is about 7.7 GB. On the RTX 3090, the eager worker used about 8.2 GiB of GPU memory; one warm request generated 2.64 seconds of audio in 9.4 seconds. Cold startup plus that request took 55 seconds. Flash-attention and fast CUDA-graph paths are not enabled. The upstream runtime's context and generation limits still apply; use short passages rather than book-length requests.

The [Breeze model license](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE) restricts the weights and self-hosted outputs to research/non-commercial use. The Apache-2.0 inference code does not grant commercial rights to the model. The license is retained beside the LFS artifacts. Commercial use requires separate permission.

Large engines can use adapter packages rather than a single file and can own isolated workers like Breeze. Model manifests support multiple shards, tokenizers, and codecs. Adding Index TTS or Fish TTS does not require their dependencies to share the API environment. Reference-audio upload and cloning controls are not part of the current API.
