from __future__ import annotations

import json

import numpy as np
import onnxruntime as ort

from mini_tts.engines.base import Engine
from mini_tts.types import AudioResult, encode_wav

LANGUAGES = {
    "en": "a",
    "en-us": "a",
    "en-gb": "b",
    "es": "e",
    "fr": "f",
    "hi": "h",
    "it": "i",
    "ja": "j",
    "pt": "p",
    "zh": "z",
}


class Adapter(Engine):
    def __init__(self, spec, device: str):
        super().__init__(spec, device)
        self.vocab = json.loads(spec.artifact("config.json").read_text())["vocab"]
        options = ort.SessionOptions()
        # Avoid thread-pool spinning competing with phonemization and other engines.
        options.intra_op_num_threads = 4 if device == "cpu" else 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda:"):
            ort.preload_dlls(directory="")
            providers.insert(
                0,
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": int(device.split(":")[1]),
                        "cudnn_conv_algo_search": "HEURISTIC",
                        "cudnn_conv1d_pad_to_nc1d": "1",
                    },
                ),
            )
        self.session = ort.InferenceSession(
            str(spec.artifact("model.onnx")), sess_options=options, providers=providers
        )
        self.session.disable_fallback()
        if (
            device.startswith("cuda:")
            and "CUDAExecutionProvider" not in self.session.get_providers()
        ):
            raise RuntimeError(
                f"Kokoro CUDA provider failed to initialize on {device}; refusing CPU fallback"
            )

    def synthesize(
        self, text: str, voice: str | None, language: str | None, speed: float
    ) -> AudioResult:
        from kokoro import KPipeline
        from mini_tts.engines import ENGINE_INFO

        voice = voice or "af_heart"
        if voice not in ENGINE_INFO["kokoro"]["voices"]:
            raise ValueError(f"Unknown Kokoro voice: {voice}")
        language = LANGUAGES.get(
            (language or voice[0]).lower(), (language or voice[0]).lower()
        )
        if language not in ENGINE_INFO["kokoro"]["languages"]:
            raise ValueError(f"Unsupported Kokoro language: {language}")
        pipeline = self.cached(
            f"pipeline:{language}",
            lambda: KPipeline(
                lang_code=language, repo_id="hexgrad/Kokoro-82M", model=False
            ),
        )
        voice_pack = self.cached(
            f"voice:{voice}",
            lambda: np.fromfile(
                self.spec.artifact(f"voices/{voice}.bin"), dtype=np.float32
            ).reshape(-1, 1, 256),
        )
        chunks = []
        for result in pipeline(text):
            tokens = [self.vocab[p] for p in result.phonemes if p in self.vocab]
            if not tokens:
                continue
            inputs = {
                "input_ids": np.array([[0, *tokens, 0]], dtype=np.int64),
                "style": voice_pack[min(len(tokens), voice_pack.shape[0] - 1)],
                "speed": np.array([speed], dtype=np.float32),
            }
            chunks.append(self.session.run(None, inputs)[0].reshape(-1))
        if not chunks:
            raise ValueError("Text contains no pronounceable speech")
        return encode_wav(
            chunks[0] if len(chunks) == 1 else np.concatenate(chunks), 24_000
        )
