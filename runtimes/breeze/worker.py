"""Persistent, local-only Breeze TTS 2 worker.

The parent process starts this worker from the Breeze-specific virtual
environment. Requests and responses are JSON lines; generated WAV bytes are
base64 encoded in the response. The checkpoint is a command-line argument,
never request-controlled, and Hugging Face is forced into offline mode.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

# Keep third-party import and generation prints off the JSON response stream.
PROTOCOL = sys.stdout
sys.stdout = sys.stderr

# The inference repository is intentionally a pinned git submodule rather than
# a package dependency. Keep it ahead of site packages so its local modules
# (`models` and `breeze_infer`) resolve exactly as upstream expects.
UPSTREAM = Path(__file__).resolve().parent / "upstream"
if not UPSTREAM.is_dir():
    raise RuntimeError(
        f"Missing Breeze inference source at {UPSTREAM}; initialize the pinned "
        "runtimes/breeze/upstream git submodule"
    )
sys.path.insert(0, str(UPSTREAM))

# load_runtime ultimately calls from_pretrained. A supplied checkpoint must be
# complete and local; never allow a missing file to trigger a Hub lookup.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"

import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from breeze_infer.runtime import (  # noqa: E402
    load_runtime,
    set_all_seeds,
    update_generation_config_for_breeze,
)
from breeze_infer.templates import (  # noqa: E402
    get_template,
    prepare_inputs,
    select_template_name,
)
from models.fast_streaming import (  # noqa: E402
    FastBreezeStreamingRuntime,
    FastStreamingConfig,
)

LOGGER = logging.getLogger("chorus.breeze-worker")
_ALLOWED_LANGUAGES = {
    "en",
    "en-us",
    "en-gb",
    "english",
    "zh",
    "zh-cn",
    "zh-hans",
    "chinese",
}
_MAX_NEW_TOKENS = 1500
_MAX_SEQ_LEN = 2048


def _response(payload: dict[str, Any]) -> None:
    PROTOCOL.write(json.dumps(payload, separators=(",", ":")) + "\n")
    PROTOCOL.flush()


def _error(request_id: Any, message: str) -> None:
    _response({"ok": False, "id": request_id, "error": message})


def _validate_request(
    message: dict[str, Any],
) -> tuple[str, str, str | None, str | None]:
    request_id = message.get("id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("Request id must be a non-empty string")

    text = message.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must contain speech")

    voice = message.get("voice")
    if voice is not None and not isinstance(voice, str):
        raise ValueError("Voice must be a text description or null")
    voice = voice.strip() if voice else None

    language = message.get("language")
    if language is not None and not isinstance(language, str):
        raise ValueError("Language must be a string or null")
    language = language.strip().lower() if language else None
    if language is not None and language not in _ALLOWED_LANGUAGES:
        raise ValueError(
            "Breeze TTS supports English and Chinese only; "
            f"unsupported language: {language}"
        )

    speed = message.get("speed", 1.0)
    if isinstance(speed, bool) or not isinstance(speed, (int, float)):
        raise ValueError(
            "Speed must be exactly 1.0; Breeze TTS does not support speed adjustment"
        )
    if not math.isfinite(float(speed)) or float(speed) != 1.0:
        raise ValueError(
            "Speed must be exactly 1.0; Breeze TTS does not support speed adjustment"
        )

    return request_id, text, voice, language


def _synthesize(
    runtime: FastBreezeStreamingRuntime,
    tokenizer: Any,
    model: Any,
    audio_tokenizer: Any,
    *,
    request_id: str,
    text: str,
    voice: str | None,
) -> tuple[bytes, int]:
    # Breeze's voice-design/direction instruction is represented by this
    # adapter's `voice` field. No reference-audio paths are accepted here.
    request: dict[str, Any] = {
        "id": request_id,
        "text": text,
        "speaker": "S0",
    }
    if voice:
        request["instruction"] = voice

    template_name = select_template_name(request)
    inputs = prepare_inputs(
        tokenizer,
        audio_tokenizer,
        model,
        [request],
        get_template(template_name),
        guidance_scale=4.0 if voice else 1.0,
        guidance_scale_ref=None,
        guidance_scale_ins=None,
    )

    output = io.BytesIO()
    with sf.SoundFile(
        output,
        mode="w",
        samplerate=runtime.sample_rate,
        channels=1,
        subtype="PCM_16",
        format="WAV",
    ) as wav_file:
        for chunk in runtime.iter_audio_chunks(
            inputs,
            request_id=request_id,
            seed=42,
        ):
            wav_file.write(chunk.audio)
    wav = output.getvalue()
    if not wav:
        raise RuntimeError("Breeze TTS produced no WAV data")
    return wav, runtime.sample_rate


def _serve(checkpoint: Path, device: str) -> None:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Breeze checkpoint directory not found: {checkpoint}")
    if not device.startswith("cuda:"):
        raise ValueError(f"Breeze TTS requires a CUDA device, got {device!r}")
    try:
        device_index = int(device.split(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid CUDA device: {device!r}") from exc
    if device_index < 0:
        raise ValueError(f"Invalid CUDA device: {device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("Breeze TTS requires CUDA, but no CUDA device is available")

    LOGGER.info("Loading Breeze TTS checkpoint from %s on %s", checkpoint, device)
    tokenizer, model, audio_tokenizer = load_runtime(
        checkpoint,
        device=device,
        attn_implementation="eager",
    )
    update_generation_config_for_breeze(model)
    # Keep all graph/flash acceleration disabled. Eager mode is portable across
    # supported CUDA architectures and avoids a runtime flash-attn build.
    runtime = FastBreezeStreamingRuntime(
        model,
        audio_tokenizer,
        FastStreamingConfig(
            max_new_tokens=_MAX_NEW_TOKENS,
            max_seq_len=_MAX_SEQ_LEN,
            fast_all=False,
            repetition_penalty=1.1,
        ),
        tokenizer=tokenizer,
    )
    LOGGER.info("Breeze TTS worker ready (sample_rate=%s)", runtime.sample_rate)

    for raw_line in sys.stdin:
        if not raw_line.strip():
            continue
        request_id: Any = None
        try:
            message = json.loads(raw_line)
            if not isinstance(message, dict):
                raise ValueError("Request must be a JSON object")
            request_id = message.get("id")
            operation = message.get("op", "synthesize")
            if operation == "close":
                _response({"ok": True, "id": request_id, "closed": True})
                return
            if operation != "synthesize":
                raise ValueError(f"Unknown worker operation: {operation!r}")
            request_id, text, voice, _language = _validate_request(message)
            set_all_seeds(42)
            wav, sample_rate = _synthesize(
                runtime,
                tokenizer,
                model,
                audio_tokenizer,
                request_id=request_id,
                text=text,
                voice=voice,
            )
            _response(
                {
                    "ok": True,
                    "id": request_id,
                    "sample_rate": sample_rate,
                    "audio_b64": base64.b64encode(wav).decode("ascii"),
                }
            )
        except (BrokenPipeError, EOFError):
            return
        except (
            Exception
        ) as exc:  # keep the persistent worker alive for the next request
            LOGGER.exception("Breeze request failed")
            _error(request_id, str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persistent local Breeze TTS 2 CUDA worker"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    _serve(args.checkpoint, args.device)


if __name__ == "__main__":
    main()
