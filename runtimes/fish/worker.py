"""Persistent local-only Fish Audio S2-Pro JSONL worker.

The parent process owns request validation and WAV decoding. This process owns the
pinned Fish Speech inference engine and emits only JSON responses on stdout.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import queue
import sys
import threading
import wave
from pathlib import Path
from typing import Any

# Third-party libraries and upstream code can log during model setup. Keep the
# protocol stream pristine; diagnostics belong on stderr.
PROTOCOL = sys.stdout
sys.stdout = sys.stderr

UPSTREAM = Path(__file__).resolve().parent / "upstream"
if not UPSTREAM.is_dir():
    raise RuntimeError(f"Fish Speech upstream submodule is missing: {UPSTREAM}")
sys.path.insert(0, str(UPSTREAM))

# The model and tokenizer must be complete local files. These settings prevent
# any accidental Hub/model download from an upstream helper.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fish_speech.inference_engine import TTSInferenceEngine  # noqa: E402
from fish_speech.models.dac.inference import load_model as load_dac_model  # noqa: E402
from fish_speech.models.text2semantic.inference import (  # noqa: E402
    GenerateRequest,
    WrappedGenerateResponse,
    decode_one_token_ar,
    generate_long,
)
from fish_speech.models.text2semantic.llama import (  # noqa: E402
    DualARTransformer,
    precompute_freqs_cis,
)
from fish_speech.utils.schema import ServeTTSRequest  # noqa: E402

LOGGER = logging.getLogger("chorus.fish-worker")
_SEED = 42
_CHUNK_LENGTH = 300


class _SynchronousLlamaQueue:
    """Queue bridge with synchronous upstream model initialization.

    Fish's ``launch_thread_safe_queue`` waits on an event that is never set when
    initialization raises in its daemon thread. Initialize in this thread so
    startup failures reach the parent. Meta parameters avoid the upstream
    constructor's throwaway FP32 CPU model; native generation stays unchanged.
    """

    def __init__(self, checkpoint: Path, device: str) -> None:
        self._requests: queue.Queue[GenerateRequest | None] = queue.Queue()
        self._closed = False
        with torch.device("meta"):
            self._model = DualARTransformer.from_pretrained(
                str(checkpoint), load_weights=True
            )
        missing = [
            name
            for name, parameter in self._model.named_parameters()
            if parameter.is_meta
        ]
        if missing:
            raise RuntimeError(f"Fish checkpoint is missing weights: {missing}")
        config = self._model.config
        # These non-persistent buffers are not in the checkpoint. Recreate them
        # with the pinned upstream formulas before moving the loaded model.
        self._model.freqs_cis = precompute_freqs_cis(
            config.max_seq_len, config.head_dim, config.rope_base
        )
        self._model.fast_freqs_cis = precompute_freqs_cis(
            config.num_codebooks, config.fast_head_dim, config.rope_base
        )
        self._model.causal_mask = torch.ones(
            (config.max_seq_len, config.max_seq_len),
            dtype=torch.bool,
            device=device,
        ).tril_()
        self._model = self._model.to(device=device, dtype=torch.bfloat16).eval()
        with torch.device(device):
            self._model.setup_caches(
                max_batch_size=1,
                max_seq_len=self._model.config.max_seq_len,
                dtype=next(self._model.parameters()).dtype,
            )
        self._model._cache_setup_done = True
        self._thread = threading.Thread(
            target=self._run,
            name="fish-semantic-worker",
            daemon=False,
        )
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._requests.get()
            if item is None:
                return
            try:
                torch.cuda.set_device(item.request["device"])
                for chunk in generate_long(
                    model=self._model,
                    decode_one_token=decode_one_token_ar,
                    **item.request,
                ):
                    item.response_queue.put(
                        WrappedGenerateResponse(status="success", response=chunk)
                    )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                LOGGER.exception("Fish semantic generation failed")
                item.response_queue.put(
                    WrappedGenerateResponse(status="error", response=exc)
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def put(self, item: GenerateRequest | None) -> None:
        self._requests.put(item)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._requests.put(None)
        self._thread.join()


def _response(payload: dict[str, Any]) -> None:
    PROTOCOL.write(json.dumps(payload, separators=(",", ":")) + "\n")
    PROTOCOL.flush()


def _error(request_id: Any, message: str) -> None:
    _response({"ok": False, "id": request_id, "error": message})


def _read_request(message: dict[str, Any]) -> tuple[Any, str, str | None]:
    """Text, speed, and language are validated by the adapter that spawns us."""
    return message["id"], message["text"], message.get("voice")


def _wav_pcm16(audio: Any, sample_rate: int) -> bytes:
    values = np.asarray(audio)
    if values.ndim == 1:
        mono = values
    elif values.ndim == 2 and 1 in values.shape:
        mono = values.reshape(-1)
    else:
        raise RuntimeError(f"Fish Speech returned non-mono audio shape {values.shape}")
    if mono.size == 0:
        raise RuntimeError("Fish Speech returned empty audio")
    values = np.asarray(mono, dtype=np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError("Fish Speech returned non-finite audio")
    pcm = np.rint(np.clip(values, -1.0, 1.0) * 32767.0).astype("<i2", copy=False)

    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm.tobytes())
    return output.getvalue()


def _synthesize(
    engine: TTSInferenceEngine,
    text: str,
    voice: str | None,
) -> tuple[bytes, int]:
    # S2's style controls are inline natural-language bracket cues. Keep the
    # caller's existing cues untouched and put an optional voice direction first.
    prompt = f"[{voice.strip()}] {text}" if voice is not None else text
    req = ServeTTSRequest(
        text=prompt,
        chunk_length=_CHUNK_LENGTH,
        format="wav",
        streaming=False,
        max_new_tokens=1024,
        top_p=0.8,
        repetition_penalty=1.1,
        temperature=0.8,
        seed=_SEED,
    )

    final_audio: Any = None
    sample_rate: int | None = None
    for result in engine.inference(req):
        if result.code == "error":
            raise RuntimeError(str(result.error))
        if result.code == "final" and isinstance(result.audio, tuple):
            sample_rate, final_audio = int(result.audio[0]), result.audio[1]
            break
    if sample_rate is None or final_audio is None:
        raise RuntimeError("Fish Speech produced no final audio")
    return _wav_pcm16(final_audio, sample_rate), sample_rate


def _validate_checkpoint(checkpoint: Path) -> tuple[Path, Path]:
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Fish checkpoint directory not found: {checkpoint}")
    required = checkpoint / "codec.pth"
    if not required.is_file():
        raise FileNotFoundError(f"Fish codec checkpoint not found: {required}")
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(
            f"Fish model config not found: {checkpoint / 'config.json'}"
        )
    model_files = (
        checkpoint / "model.safetensors",
        checkpoint / "model.safetensors.index.json",
        checkpoint / "model.pth",
    )
    if not any(path.is_file() for path in model_files):
        raise FileNotFoundError(
            f"Fish model weights not found in {checkpoint} (expected model.safetensors, "
            "model.safetensors.index.json, or model.pth)"
        )
    return checkpoint, required


def _validate_device(device: str) -> None:
    if not device.startswith("cuda"):
        raise ValueError("Fish S2-Pro requires an explicit CUDA device")
    parsed = torch.device(device)
    if parsed.type != "cuda" or parsed.index is None:
        raise ValueError("device must be an explicit CUDA device such as cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; Fish S2-Pro has no CPU fallback")
    if parsed.index < 0 or parsed.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index out of range: {parsed.index}")
    torch.cuda.set_device(parsed)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Fish S2-Pro requires BF16-capable CUDA hardware")


def _serve(checkpoint_arg: Path, device: str) -> None:
    _validate_device(device)
    checkpoint, codec_checkpoint = _validate_checkpoint(checkpoint_arg.resolve())
    LOGGER.info(
        "Loading Fish S2-Pro from local checkpoint %s on %s", checkpoint, device
    )

    # Use upstream model/generation functions with a local queue bridge. Unlike
    # launch_thread_safe_queue, semantic model initialization is synchronous, so
    # checkpoint/OOM failures propagate to the parent instead of wedging startup.
    llama_queue = _SynchronousLlamaQueue(checkpoint, device)
    try:
        decoder_model = load_dac_model(
            config_name="modded_dac_vq",
            checkpoint_path=str(codec_checkpoint),
            device=device,
        )
        engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=torch.bfloat16,
            compile=False,
        )
        sample_rate = int(decoder_model.sample_rate)
        LOGGER.info("Fish S2-Pro ready; codec sample rate=%d Hz", sample_rate)
        for raw_line in sys.stdin:
            if not raw_line.strip():
                continue
            request_id: Any = None
            try:
                message = json.loads(raw_line)
                if not isinstance(message, dict):
                    raise ValueError("JSON request must be an object")
                if message.get("op") == "close":
                    request_id = message.get("id")
                    if request_id is None:
                        raise ValueError("close request id is required")
                    _response({"ok": True, "id": request_id, "closed": True})
                    break
                request_id, text, voice = _read_request(message)
                audio, generated_rate = _synthesize(engine, text, voice)
                _response(
                    {
                        "ok": True,
                        "id": request_id,
                        "audio_b64": base64.b64encode(audio).decode("ascii"),
                        "sample_rate": generated_rate,
                    }
                )
            except Exception as exc:
                LOGGER.exception("Fish request failed")
                _error(request_id, str(exc))
    finally:
        # Stop the local semantic worker and release its CUDA model/cache.
        llama_queue.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chorus Fish S2-Pro JSONL worker")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    _serve(args.checkpoint, args.device)


if __name__ == "__main__":
    main()
