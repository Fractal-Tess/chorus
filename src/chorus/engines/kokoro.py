from __future__ import annotations

import ctypes
import json
import threading
from contextlib import nullcontext

import numpy as np
import onnxruntime as ort

from chorus.engines.base import Engine
from chorus.types import AudioResult, encode_wav

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


def _configure_cuda_wait(device_id: int) -> None:
    """Let CPU threads sleep while their CUDA work finishes, rather than spin."""
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuDevicePrimaryCtxGetState.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_int),
    ]
    driver.cuDevicePrimaryCtxSetFlags_v2.argtypes = [ctypes.c_int, ctypes.c_uint]

    def check(result: int) -> None:
        if result:
            raise RuntimeError(
                f"Cannot configure CUDA synchronization on cuda:{device_id}: "
                f"driver error {result}"
            )

    check(driver.cuInit(0))
    flags, active = ctypes.c_uint(), ctypes.c_int()
    check(
        driver.cuDevicePrimaryCtxGetState(
            device_id, ctypes.byref(flags), ctypes.byref(active)
        )
    )
    # Replace only CU_CTX_SCHED_MASK with CU_CTX_SCHED_BLOCKING_SYNC.
    # This changes host waiting, not GPU kernels, precision, or hardware settings.
    check(driver.cuDevicePrimaryCtxSetFlags_v2(device_id, (flags.value & ~7) | 4))


class Adapter(Engine):
    def __init__(self, spec, device: str):
        super().__init__(spec, device)
        self._prepare_lock = threading.Lock()
        self.vocab = json.loads(spec.artifact("config.json").read_text())["vocab"]
        options = ort.SessionOptions()
        # Avoid thread-pool spinning competing with phonemization and other engines.
        options.intra_op_num_threads = 4 if device == "cpu" else 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda:"):
            ort.preload_dlls(directory="")
            device_id = int(device.split(":")[1])
            _configure_cuda_wait(device_id)
            providers.insert(
                0,
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": device_id,
                        "cudnn_conv_algo_search": "HEURISTIC",
                        "cudnn_conv1d_pad_to_nc1d": "1",
                    },
                ),
            )
        model = spec.artifact("model.onnx")
        if device.startswith("cuda:"):
            from chorus.engines.kokoro_fft import cuda_model

            model_context = cuda_model(model)
        else:
            model_context = nullcontext(str(model))
        with model_context as source:
            self.session = ort.InferenceSession(
                source, sess_options=options, providers=providers
            )
            self.session.disable_fallback()
        if (
            device.startswith("cuda:")
            and "CUDAExecutionProvider" not in self.session.get_providers()
        ):
            raise RuntimeError(
                f"Kokoro CUDA provider failed to initialize on {device}; refusing CPU fallback"
            )

    def _prepare(
        self, text: str, voice: str | None, language: str | None, speed: float
    ) -> list[dict[str, np.ndarray]]:
        from kokoro import KPipeline

        from chorus.engines import ENGINE_INFO

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
        feeds = []
        for result in pipeline(text):
            tokens = [self.vocab[p] for p in result.phonemes if p in self.vocab]
            if not tokens:
                continue
            inputs = {
                "input_ids": np.array([[0, *tokens, 0]], dtype=np.int64),
                "style": voice_pack[min(len(tokens), voice_pack.shape[0] - 1)],
                "speed": np.array([speed], dtype=np.float32),
            }
            feeds.append(inputs)
        if not feeds:
            raise ValueError("Text contains no pronounceable speech")
        return feeds

    def synthesize(
        self, text: str, voice: str | None, language: str | None, speed: float
    ) -> AudioResult:
        # Language pipelines and lazy caches are shared; ORT runs use independent inputs.
        with self._prepare_lock:
            feeds = self._prepare(text, voice, language, speed)
        chunks = [self.session.run(None, inputs)[0].reshape(-1) for inputs in feeds]
        return encode_wav(
            chunks[0] if len(chunks) == 1 else np.concatenate(chunks), 24_000
        )
