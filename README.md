<h1 align="center">Mini TTS</h1>

<p align="center"><strong>Five local text-to-speech engines behind one CPU-only API and browser console.</strong></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/inference-CPU%20only-0f172a" alt="CPU-only inference">
  <img src="https://img.shields.io/badge/environment-Nix%20flake-5277C3?logo=nixos&logoColor=white" alt="Nix flake">
</p>

Mini TTS runs Pocket TTS, Kokoro, Piper, Kitten TTS, and Supertonic 3 on one machine. Use the browser to compare voices and render waveforms, or call the shared HTTP endpoint from another application.

- No hosted speech API or GPU required.
- Models load on demand and stay warm for later requests.
- Narration picks are grouped in the console, with optional LavaSR enhancement to 48 kHz.

## Run it

The included Nix flake provides Python 3.12, `uv`, FFmpeg, eSpeak NG, libsndfile, and Tailwind CSS.

```bash
direnv allow
serve-api
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Interactive API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). Set `TTS_HOST` or `TTS_PORT` to change the bind address; for example, `TTS_HOST=0.0.0.0 TTS_PORT=8787 serve-api`.

Model files download on first use. The first render for an engine is therefore slower than later renders.

## Engines

| Engine | Default voice | Sample rate | Notes |
| --- | --- | ---: | --- |
| Pocket TTS | `alba` | 24 kHz | INT8-optimized voice cloning and multilingual models |
| Kokoro | `af_heart` | 24 kHz | 54 voices; FP32 ONNX inference |
| Piper | `en_US-lessac-medium` | 22.05 kHz | Small, dependable English model |
| Kitten TTS | `Leo` | 24 kHz | Eight lightweight English voices |
| Supertonic 3 | `M1` | 44.1 kHz | Ten voices and multilingual synthesis |

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

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Runtime and loaded-engine status |
| `GET` | `/v1/engines` | Engines, voices, languages, and defaults |
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
python -m py_compile tts_api.py tts_engines.py
```

Rebuild the local Tailwind stylesheet after changing classes:

```bash
tailwindcss \
  -c tailwind.config.js \
  -i static/tailwind.input.css \
  -o static/tailwind.css \
  --minify
```
