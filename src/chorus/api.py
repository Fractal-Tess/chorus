from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import asdict

import torch
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from chorus.engines import ENGINE_INFO
from chorus.devices import available_devices
from chorus.models import ROOT
from chorus.registry import EngineRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chorus")


class DeviceInfo(BaseModel):
    id: str = Field(description="Device ID accepted by speech requests, e.g. cpu or cuda:0.")
    type: str = Field(description="Execution device type: cpu or cuda.")
    enabled: bool = Field(description="Whether server policy permits this device.")


class DevicesResponse(BaseModel):
    default_device: str = Field(
        description="Configured default: auto prefers an enabled GPU supported by the model, otherwise CPU."
    )
    devices: list[DeviceInfo]


class SpeechRequest(BaseModel):
    engine: str
    model: str | None = None
    device: str | None = None
    input: str = Field(min_length=1, max_length=10_000)
    voice: str | None = Field(
        default=None,
        description="Named voice for most engines; natural-language voice description for Breeze or speaking style for Fish. Fish reference-audio cloning is not exposed.",
    )
    language: str | None = None
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    lava_sr: bool = Field(
        default=False,
        description="Post-process generated speech with LavaSR and return 48 kHz audio",
    )
    force_align: bool = Field(
        default=False, description="Generate English word timestamps with Wav2Vec2"
    )


registry: EngineRegistry | None = None
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global registry
    if registry is None:
        registry = EngineRegistry()
    try:
        yield
    finally:
        registry.close()


app = FastAPI(
    title="Chorus API",
    version="1.0.0",
    description="Local multi-engine TTS with CPU/CUDA synthesis and optional post-processing.",
    lifespan=lifespan,
)

static_dir = ROOT / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
def landing_page() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "device": registry.policy.default,
        "allowed_devices": registry.policy.allowed,
        "loaded_models": registry.loaded_models(),
        "torch": torch.__version__,
        "cuda_available": any(d.startswith("cuda:") for d in registry.policy.allowed),
        "loaded_engines": registry.loaded_engines(),
    }


@app.get("/v1/devices", response_model=DevicesResponse)
def devices() -> dict[str, object]:
    """List detected devices, including those disabled by server policy.

    Detection respects CUDA_VISIBLE_DEVICES and is cached for this process.
    Use /v1/models to check which enabled devices a model supports.
    """
    return {
        "default_device": registry.policy.default,
        "devices": [
            {
                "id": device,
                "type": device.split(":")[0],
                "enabled": device in registry.policy.allowed,
            }
            for device in available_devices()
        ],
    }


@app.get("/v1/engines")
def engines() -> dict[str, object]:
    loaded = set(registry.loaded_engines())
    return {
        "engines": [
            {
                "id": name,
                "loaded": name in loaded,
                **details,
                "models": [
                    spec.name
                    for spec in registry.catalog.models.values()
                    if spec.engine == name
                ],
            }
            for name, details in ENGINE_INFO.items()
            if name in registry.enabled
        ]
    }


@app.get("/v1/models")
def models() -> dict[str, object]:
    return {
        "models": [
            {
                "engine": spec.engine,
                "model": spec.name,
                "supported_devices": spec.devices,
                "allowed_devices": [
                    d
                    for d in registry.policy.allowed
                    if d.split(":")[0] in spec.devices
                ],
                "default": spec.manifest.get("default", False),
            }
            for spec in registry.catalog.models.values()
            if spec.engine in registry.enabled
        ],
        "loaded": registry.loaded_models(),
    }


@app.get("/v1/audio/alignments/{alignment_id}")
def get_alignment(alignment_id: str) -> dict[str, object]:
    with alignment_cache_lock:
        alignment = alignment_cache.get(alignment_id)
        if alignment is None:
            raise HTTPException(
                status_code=404, detail="Alignment not found or expired"
            )
        alignment_cache.move_to_end(alignment_id)
        return alignment


@app.post("/v1/audio/speech")
def create_speech(request: SpeechRequest) -> Response:
    if (
        request.force_align
        and request.language is not None
        and request.language.lower() not in ENGLISH_ALIGNMENT_LANGUAGES
    ):
        raise HTTPException(
            status_code=400,
            detail="Wav2Vec2 force alignment currently supports English speech only",
        )

    try:
        audio = registry.synthesize(
            engine=request.engine,
            text=request.input,
            model=request.model,
            device=request.device,
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
        raise HTTPException(
            status_code=500, detail=f"{request.engine} synthesis failed: {error}"
        ) from error

    timings = audio.timings_ms
    headers = {
        "Content-Disposition": f'attachment; filename="{request.engine}.wav"',
        "X-TTS-Engine": request.engine,
        "X-TTS-Model": audio.model,
        "X-TTS-Device": audio.device,
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
