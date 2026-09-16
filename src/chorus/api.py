from __future__ import annotations

import importlib.metadata
import logging
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import asdict
from enum import Enum

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from chorus.channels import ChannelUnavailableError, GpuQueueFullError
from chorus.devices import available_devices
from chorus.engines import ENGINE_INFO
from chorus.models import ROOT
from chorus.registry import EngineRegistry
from chorus.types import AUDIO_MEDIA_TYPES, ResponseFormat, encode_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chorus")


class ExecutionChannel(str, Enum):
    cpu = "cpu"
    gpu = "gpu"


class DeviceInfo(BaseModel):
    id: str = Field(description="Physical device identifier for diagnostics only.")
    type: str = Field(description="Physical device type: cpu or cuda.")
    enabled: bool = Field(description="Whether server policy permits this device.")


class DevicesResponse(BaseModel):
    devices: list[DeviceInfo]


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: str
    model: str | None = None
    channel: ExecutionChannel | None = None
    response_format: ResponseFormat = Field(
        default="wav",
        description="Output format: WAV PCM, MP3 (128 kbps), lossless FLAC, or Ogg Opus (64 kbps, 48 kHz).",
    )
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
    try:
        if registry is None:
            registry = EngineRegistry()
            registry.catalog.prepare(registry.enabled)
        yield
    finally:
        if registry is not None:
            registry.close()


app = FastAPI(
    title="Chorus API",
    version=importlib.metadata.version("chorus"),
    description="Local multi-engine TTS with CPU/CUDA synthesis and optional post-processing.",
    lifespan=lifespan,
)

static_dir = ROOT / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
def landing_page() -> FileResponse:
    return FileResponse(static_dir / "index.html")


def _health_channels() -> list[dict[str, object]]:
    """Summarize logical channel capabilities without exposing physical IDs."""
    statuses = {
        channel: {
            "id": channel,
            "supported": False,
            "enabled": False,
            "available": False,
        }
        for channel in ("cpu", "gpu")
    }
    for spec in registry.catalog.models.values():
        if spec.engine not in registry.enabled:
            continue
        status = registry.channel_status(spec)
        for channel_status in status["channels"]:
            channel = channel_status["id"]
            current = statuses[channel]
            current["supported"] |= bool(channel_status["supported"])
            current["enabled"] |= bool(channel_status["enabled"])
            current["available"] |= bool(channel_status["available"])
    return list(statuses.values())


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "channels": _health_channels(),
        "allowed_devices": registry.policy.allowed,
        "loaded_models": registry.loaded_models(),
        "torch": importlib.metadata.version("torch"),
        "cuda_available": any(d.startswith("cuda:") for d in registry.policy.allowed),
        "loaded_engines": registry.loaded_engines(),
    }


@app.get("/v1/devices", response_model=DevicesResponse)
def devices() -> dict[str, object]:
    """List detected physical devices for diagnostics.

    Detection respects CUDA_VISIBLE_DEVICES and is cached for this process.
    Logical CPU/GPU routing policy is reported by /v1/models.
    """
    detected = available_devices()
    return {
        "devices": [
            {
                "id": device,
                "type": device.split(":")[0],
                "enabled": device in registry.policy.allowed,
            }
            for device in detected
        ],
    }


@app.get("/v1/resources")
def resources() -> dict[str, object]:
    """Report worker resource usage and soft cache budgets.

    RAM and VRAM budgets are soft eviction targets for Chorus worker process
    trees. They exclude this parent API process and unrelated processes.
    Workers are evicted after the configured idle timeout; a timeout of zero
    unloads a worker immediately after its request.
    """
    return registry.resource_status()


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
                **registry.channel_status(spec),
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


@app.post(
    "/v1/audio/speech",
    response_class=Response,
    responses={
        400: {"description": "Invalid request or unsupported/disabled model channel."},
        429: {
            "description": "The GPU waiting queue is full. No inference was started.",
            "headers": {
                "Retry-After": {
                    "description": "Suggested delay in seconds before retrying.",
                    "schema": {"type": "integer"},
                }
            },
        },
        503: {
            "description": "The selected channel has no enabled compatible hardware."
        },
        200: {
            "description": "Audio in the requested response_format (WAV by default).",
            "content": {
                media_type: {"schema": {"type": "string", "format": "binary"}}
                for media_type in AUDIO_MEDIA_TYPES.values()
            },
        },
    },
)
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

    started = time.perf_counter()
    try:
        audio = registry.synthesize(
            engine=request.engine,
            text=request.input,
            model=request.model,
            channel=request.channel.value if request.channel is not None else None,
            voice=request.voice,
            language=request.language,
            speed=request.speed,
            lava_sr=request.lava_sr,
            force_align=request.force_align,
        )
        encoding_started = time.perf_counter()
        content, sample_rate = encode_response(audio, request.response_format)
        encoding_ms = (time.perf_counter() - encoding_started) * 1_000
    except GpuQueueFullError as error:
        raise HTTPException(
            status_code=429,
            detail=str(error),
            headers={"Retry-After": "1"},
        ) from error
    except ChannelUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("Synthesis failed for %s", request.engine)
        raise HTTPException(
            status_code=500, detail=f"{request.engine} synthesis failed: {error}"
        ) from error

    timings = {
        **audio.timings_ms,
        "encoding": encoding_ms,
        "total": (time.perf_counter() - started) * 1_000,
    }
    headers = {
        "Content-Disposition": f'attachment; filename="{request.engine}.{request.response_format}"',
        "X-TTS-Engine": request.engine,
        "X-TTS-Model": audio.model,
        "X-TTS-Device": audio.device,
        "X-Sample-Rate": str(sample_rate),
        "X-Audio-Duration": f"{audio.duration:.3f}",
        "X-LavaSR-Applied": str(request.lava_sr).lower(),
        "X-TTS-Channel": (
            request.channel.value
            if request.channel is not None
            else ("gpu" if audio.device.startswith("cuda:") else "cpu")
        ),
        "X-Force-Alignment-Applied": str(request.force_align).lower(),
        "X-Queue-Time-Ms": f"{timings['queue']:.1f}",
        "X-Inference-Time-Ms": f"{timings['inference']:.1f}",
        "X-LavaSR-Time-Ms": f"{timings['lava_sr']:.1f}",
        "X-Alignment-Time-Ms": f"{timings['alignment']:.1f}",
        "X-Encoding-Time-Ms": f"{timings['encoding']:.1f}",
        "X-Backend-Time-Ms": f"{timings['total']:.1f}",
        "Server-Timing": (
            f"queue;dur={timings['queue']:.1f}, "
            f"inference;dur={timings['inference']:.1f}, "
            f"lavasr;dur={timings['lava_sr']:.1f}, "
            f"alignment;dur={timings['alignment']:.1f}, "
            f"encoding;dur={timings['encoding']:.1f}, "
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

    return Response(
        content=content,
        media_type=AUDIO_MEDIA_TYPES[request.response_format],
        headers=headers,
    )
