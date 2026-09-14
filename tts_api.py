from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tts_engines import ENGINE_INFO, EngineRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mini-tts")

EngineName = Literal["pocket", "kokoro", "piper", "kitten", "supertonic"]


class SpeechRequest(BaseModel):
    engine: EngineName
    input: str = Field(min_length=1, max_length=10_000)
    voice: str | None = None
    language: str | None = None
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    lava_sr: bool = Field(default=False, description="Post-process generated speech with LavaSR and return 48 kHz audio")
    force_align: bool = Field(default=False, description="Generate English word timestamps with Wav2Vec2")


registry = EngineRegistry()
alignment_cache: OrderedDict[str, dict[str, object]] = OrderedDict()
alignment_cache_lock = threading.Lock()
ALIGNMENT_CACHE_LIMIT = 32
ENGLISH_ALIGNMENT_LANGUAGES = {None, "a", "b", "en", "en-us", "en-gb", "english"}


def store_alignment(payload: dict[str, object]) -> None:
    alignment_id = str(payload["id"])
    with alignment_cache_lock:
        alignment_cache[alignment_id] = payload
        alignment_cache.move_to_end(alignment_id)
        while len(alignment_cache) > ALIGNMENT_CACHE_LIMIT:
            alignment_cache.popitem(last=False)


app = FastAPI(
    title="Mini TTS API",
    version="1.0.0",
    description="Local CPU-only API for Pocket, Kokoro, Piper, Kitten, and Supertonic 3.",
)

static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
def landing_page() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "device": "cpu",
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "loaded_engines": registry.loaded_engines(),
    }


@app.get("/v1/engines")
def engines() -> dict[str, object]:
    loaded = set(registry.loaded_engines())
    return {
        "engines": [
            {"id": name, "loaded": name in loaded, **details}
            for name, details in ENGINE_INFO.items()
        ]
    }

@app.get("/v1/audio/alignments/{alignment_id}")
def get_alignment(alignment_id: str) -> dict[str, object]:
    with alignment_cache_lock:
        alignment = alignment_cache.get(alignment_id)
        if alignment is None:
            raise HTTPException(status_code=404, detail="Alignment not found or expired")
        alignment_cache.move_to_end(alignment_id)
        return alignment


@app.post("/v1/audio/speech")
def create_speech(request: SpeechRequest) -> Response:
    if request.force_align and request.language is not None and request.language.lower() not in ENGLISH_ALIGNMENT_LANGUAGES:
        raise HTTPException(status_code=400, detail="Wav2Vec2 force alignment currently supports English speech only")

    try:
        audio = registry.synthesize(
            engine=request.engine,
            text=request.input,
            voice=request.voice,
            language=request.language,
            speed=request.speed,
            lava_sr=request.lava_sr,
            force_align=request.force_align,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("Synthesis failed for %s", request.engine)
        raise HTTPException(status_code=500, detail=f"{request.engine} synthesis failed: {error}") from error

    timings = audio.timings_ms
    headers = {
        "Content-Disposition": f'attachment; filename="{request.engine}.wav"',
        "X-TTS-Engine": request.engine,
        "X-Sample-Rate": str(audio.sample_rate),
        "X-Audio-Duration": f"{audio.duration:.3f}",
        "X-LavaSR-Applied": str(request.lava_sr).lower(),
        "X-Force-Alignment-Applied": str(request.force_align).lower(),
        "X-Queue-Time-Ms": f"{timings['queue']:.1f}",
        "X-Inference-Time-Ms": f"{timings['inference']:.1f}",
        "X-LavaSR-Time-Ms": f"{timings['lava_sr']:.1f}",
        "X-Alignment-Time-Ms": f"{timings['alignment']:.1f}",
        "X-Backend-Time-Ms": f"{timings['total']:.1f}",
        "Server-Timing": (
            f"queue;dur={timings['queue']:.1f}, "
            f"inference;dur={timings['inference']:.1f}, "
            f"lavasr;dur={timings['lava_sr']:.1f}, "
            f"alignment;dur={timings['alignment']:.1f}, "
            f"total;dur={timings['total']:.1f}"
        ),
    }
    if audio.alignment is not None:
        alignment_id = uuid.uuid4().hex
        alignment_url = f"/v1/audio/alignments/{alignment_id}"
        store_alignment(
            {
                "id": alignment_id,
                "engine": request.engine,
                "language": "en",
                "model": "WAV2VEC2_ASR_BASE_960H",
                "duration_ms": round(audio.duration * 1_000),
                "words": [asdict(word) for word in audio.alignment],
            }
        )
        headers["X-Alignment-Id"] = alignment_id
        headers["X-Alignment-Url"] = alignment_url

    return Response(content=audio.wav, media_type="audio/wav", headers=headers)
