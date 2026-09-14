from __future__ import annotations

import io
import json
import math
import re
import threading
import time
import unicodedata
import wave
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
PIPER_MODEL = ROOT / "models/piper/en_US-lessac-medium.onnx"
KOKORO_ONNX_REPO = "onnx-community/Kokoro-82M-v1.0-ONNX"
KOKORO_ONNX_MODEL = "onnx/model.onnx"


@dataclass(frozen=True)
class WordAlignment:
    word: str
    start_ms: int
    end_ms: int
    score: float


@dataclass(frozen=True)
class AudioResult:
    wav: bytes
    sample_rate: int
    duration: float
    timings_ms: dict[str, float] = field(default_factory=dict)
    alignment: tuple[WordAlignment, ...] | None = None


@dataclass(frozen=True)
class KokoroOnnxRuntime:
    session: Any
    vocab: dict[str, int]


ENGINE_INFO = {
    "pocket": {
        "label": "Pocket TTS",
        "summary": "Compact voice cloning with INT8-optimized CPU inference.",
        "default_voice": "alba",
        "default_language": "english",
        "sample_rate": 24_000,
        "languages": ["english", "french_24l", "spanish_24l", "german_24l", "italian_24l", "portuguese_24l"],
        "voices": ["alba", "giovanni", "lola", "juergen", "rafael", "estelle", "anna", "azelma", "bill_boerst", "caro_davy", "charles", "cosette", "eponine", "eve", "fantine", "george", "jane", "javert", "jean", "marius", "mary", "michael", "paul", "peter_yearsley", "stuart_bell", "vera"],
    },
    "kokoro": {
        "label": "Kokoro",
        "summary": "Expressive 82M-parameter voices across nine accents and languages.",
        "default_voice": "af_heart",
        "default_language": "a",
        "sample_rate": 24_000,
        "languages": ["a", "b", "e", "f", "h", "i", "j", "p", "z"],
        "voices": ["af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa", "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis", "ef_dora", "em_alex", "em_santa", "ff_siwis", "hf_alpha", "hf_beta", "hm_omega", "hm_psi", "if_sara", "im_nicola", "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo", "pf_dora", "pm_alex", "pm_santa", "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang"],
    },
    "piper": {
        "label": "Piper",
        "summary": "Fast, dependable ONNX synthesis with a bundled US voice.",
        "default_voice": "en_US-lessac-medium",
        "default_language": "en-US",
        "sample_rate": 22_050,
        "languages": ["en-US"],
        "voices": ["en_US-lessac-medium"],
    },
    "kitten": {
        "label": "Kitten TTS",
        "summary": "Tiny 15M-parameter ONNX model with eight English voices.",
        "default_voice": "Leo",
        "default_language": "en",
        "sample_rate": 24_000,
        "languages": ["en"],
        "voices": ["Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"],
    },
    "supertonic": {
        "label": "Supertonic 3",
        "summary": "High-speed multilingual synthesis at studio-grade 44.1 kHz.",
        "default_voice": "M1",
        "default_language": "en",
        "sample_rate": 44_100,
        "languages": ["en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi", "na"],
        "voices": ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"],
    },
}

_KOKORO_LANGUAGES = {
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

_POCKET_LANGUAGES = {
    "en": "english",
    "en-us": "english",
    "en-gb": "english",
    "es": "spanish_24l",
    "fr": "french_24l",
    "de": "german_24l",
    "it": "italian_24l",
    "pt": "portuguese_24l",
}


def _encode_wav(audio: Any, sample_rate: int) -> AudioResult:
    samples = np.asarray(audio, dtype=np.float32).squeeze()
    if samples.ndim != 1 or samples.size == 0:
        raise RuntimeError("Engine produced no audio")
    output = io.BytesIO()
    sf.write(output, samples, sample_rate, format="WAV", subtype="PCM_16")
    return AudioResult(output.getvalue(), sample_rate, samples.size / sample_rate)

def _load_kokoro_onnx() -> KokoroOnnxRuntime:
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(KOKORO_ONNX_REPO, KOKORO_ONNX_MODEL)
    config_path = hf_hub_download("hexgrad/Kokoro-82M", "config.json")
    with open(config_path, encoding="utf-8") as config_file:
        vocab = json.load(config_file)["vocab"]
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return KokoroOnnxRuntime(session, vocab)


class EngineRegistry:
    """Lazily loads CPU models and serializes inference on the shared CPU."""

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._load_lock = threading.RLock()
        self._cpu_lock = threading.Lock()

    def loaded_engines(self) -> list[str]:
        with self._load_lock:
            return [
                engine
                for engine in ENGINE_INFO
                if any(key == engine or key.startswith(f"{engine}:") for key in self._models)
            ]

    def synthesize(
        self,
        engine: str,
        text: str,
        voice: str | None = None,
        language: str | None = None,
        speed: float = 1.0,
        lava_sr: bool = False,
        force_align: bool = False,
    ) -> AudioResult:
        if engine not in ENGINE_INFO:
            raise ValueError(f"Unknown engine: {engine}")

        total_started = time.perf_counter()
        with self._cpu_lock:
            inference_started = time.perf_counter()
            queue_ms = (inference_started - total_started) * 1_000
            method = getattr(self, f"_synthesize_{engine}")
            audio = method(text, voice, language, speed)
            inference_ms = (time.perf_counter() - inference_started) * 1_000

            lava_sr_ms = 0.0
            if lava_sr:
                lava_sr_started = time.perf_counter()
                audio = self._enhance_lava(audio)
                lava_sr_ms = (time.perf_counter() - lava_sr_started) * 1_000

            alignment_ms = 0.0
            alignment = None
            if force_align:
                alignment_started = time.perf_counter()
                alignment = self._align_english(audio, text)
                alignment_ms = (time.perf_counter() - alignment_started) * 1_000

        return replace(
            audio,
            alignment=alignment,
            timings_ms={
                "queue": queue_ms,
                "inference": inference_ms,
                "lava_sr": lava_sr_ms,
                "alignment": alignment_ms,
                "total": (time.perf_counter() - total_started) * 1_000,
            },
        )

    def _get_or_load(self, key: str, loader: Any) -> Any:
        with self._load_lock:
            if key not in self._models:
                self._models[key] = loader()
            return self._models[key]

    def _enhance_lava(self, audio: AudioResult) -> AudioResult:
        import torch
        import torchaudio
        from LavaSR.enhancer.linkwitz_merge import FastLRMerge
        from LavaSR.model import LavaEnhance2

        def load_model() -> LavaEnhance2:
            model = LavaEnhance2("YatharthS/LavaSR", device="cpu")
            model.bwe_model.lr_refiner = FastLRMerge(device="cpu", cutoff=8_000, transition_bins=1_024)
            return model

        model = self._get_or_load("lavasr", load_model)
        samples, sample_rate = sf.read(io.BytesIO(audio.wav), dtype="float32")
        input_audio = torch.from_numpy(np.asarray(samples)).reshape(1, -1)
        input_audio = torchaudio.functional.resample(input_audio, sample_rate, 16_000)
        enhanced = model.enhance(input_audio, denoise=False).detach().cpu().numpy()
        return _encode_wav(enhanced, 48_000)

    def _align_english(self, audio: AudioResult, text: str) -> tuple[WordAlignment, ...]:
        import torch
        import torchaudio

        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

        def load_model() -> Any:
            return bundle.get_model().eval()

        model = self._get_or_load("force-aligner:english", load_model)
        dictionary = {label: index for index, label in enumerate(bundle.get_labels())}
        words: list[tuple[str, str]] = []
        for raw_word in re.findall(r"\S+", text):
            ascii_word = unicodedata.normalize("NFKD", raw_word).encode("ascii", "ignore").decode()
            normalized = "".join(character for character in ascii_word.upper() if character in dictionary and character != "|")
            if normalized:
                words.append((raw_word, normalized))
        if not words:
            raise ValueError("Force alignment requires English words containing letters")

        transcript = "|".join(normalized for _, normalized in words)
        targets = torch.tensor([[dictionary[character] for character in transcript]], dtype=torch.int32)
        samples, sample_rate = sf.read(io.BytesIO(audio.wav), dtype="float32")
        samples = np.asarray(samples)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        waveform = torch.from_numpy(samples).reshape(1, -1)
        if sample_rate != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

        with torch.inference_mode():
            emissions, _ = model(waveform)
            emissions = torch.log_softmax(emissions, dim=-1)
            try:
                path, scores = torchaudio.functional.forced_align(emissions, targets, blank=0)
            except RuntimeError as error:
                raise ValueError(f"Force alignment failed: {error}") from error
        token_spans = torchaudio.functional.merge_tokens(path[0], scores[0])
        if len(token_spans) != len(transcript):
            raise RuntimeError("Force alignment returned an unexpected token sequence")

        milliseconds_per_frame = audio.duration * 1_000 / emissions.size(1)
        aligned_words: list[WordAlignment] = []
        cursor = 0
        for index, (raw_word, normalized) in enumerate(words):
            spans = token_spans[cursor : cursor + len(normalized)]
            cursor += len(normalized)
            if index < len(words) - 1:
                cursor += 1
            frame_count = sum(span.end - span.start for span in spans)
            mean_log_score = sum(float(span.score) * (span.end - span.start) for span in spans) / frame_count
            aligned_words.append(
                WordAlignment(
                    word=raw_word,
                    start_ms=round(spans[0].start * milliseconds_per_frame),
                    end_ms=round(spans[-1].end * milliseconds_per_frame),
                    score=round(math.exp(mean_log_score), 4),
                )
            )
        return tuple(aligned_words)

    def _synthesize_pocket(self, text: str, voice: str | None, language: str | None, speed: float) -> AudioResult:
        from pocket_tts import TTSModel

        if speed != 1.0:
            raise ValueError("Pocket TTS does not support speed adjustment")
        language = (language or "english").lower()
        language = _POCKET_LANGUAGES.get(language, language)
        model = self._get_or_load(
            f"pocket:{language}",
            lambda: TTSModel.load_model(language=language, quantize=True),
        )
        voice = voice or "alba"
        state = self._get_or_load(f"pocket-voice:{language}:{voice}", lambda: model.get_state_for_audio_prompt(voice))
        audio = model.generate_audio(state, text)
        return _encode_wav(audio.detach().cpu().numpy(), model.sample_rate)

    def _synthesize_kokoro(self, text: str, voice: str | None, language: str | None, speed: float) -> AudioResult:
        from huggingface_hub import hf_hub_download
        from kokoro import KPipeline

        language = (language or "a").lower()
        language = _KOKORO_LANGUAGES.get(language, language)
        pipeline = self._get_or_load(
            f"kokoro:pipeline:{language}",
            lambda: KPipeline(lang_code=language, repo_id="hexgrad/Kokoro-82M", model=False),
        )
        runtime = self._get_or_load("kokoro:onnx", _load_kokoro_onnx)
        voice = voice or "af_heart"

        def load_voice() -> np.ndarray:
            path = hf_hub_download(KOKORO_ONNX_REPO, f"voices/{voice}.bin")
            return np.fromfile(path, dtype=np.float32).reshape(-1, 1, 256)

        voice_pack = self._get_or_load(f"kokoro:voice:{voice}", load_voice)
        chunks = []
        for result in pipeline(text):
            tokens = [runtime.vocab[phoneme] for phoneme in result.phonemes if phoneme in runtime.vocab]
            if not tokens:
                continue
            inputs = {
                "input_ids": np.array([[0, *tokens, 0]], dtype=np.int64),
                "style": voice_pack[min(len(tokens), voice_pack.shape[0] - 1)],
                "speed": np.array([speed], dtype=np.float32),
            }
            chunks.append(runtime.session.run(None, inputs)[0].squeeze())
        if not chunks:
            raise RuntimeError("Kokoro produced no audio")
        return _encode_wav(np.concatenate(chunks), 24_000)

    def _synthesize_piper(self, text: str, voice: str | None, language: str | None, speed: float) -> AudioResult:
        from piper import PiperVoice, SynthesisConfig

        if voice not in (None, "en_US-lessac-medium"):
            raise ValueError("Only the bundled Piper voice en_US-lessac-medium is available")
        if language and language.lower() not in {"en", "en-us"}:
            raise ValueError("The bundled Piper model only supports en-US")
        if not PIPER_MODEL.is_file():
            raise RuntimeError(f"Piper model is missing: {PIPER_MODEL}")
        model = self._get_or_load("piper", lambda: PiperVoice.load(PIPER_MODEL, use_cuda=False))
        output = io.BytesIO()
        with wave.open(output, "wb") as wav_file:
            model.synthesize_wav(text, wav_file, SynthesisConfig(length_scale=1.0 / speed))
        data = output.getvalue()
        with sf.SoundFile(io.BytesIO(data)) as audio_file:
            sample_rate = audio_file.samplerate
            duration = audio_file.frames / sample_rate
        return AudioResult(data, sample_rate, duration)

    def _synthesize_kitten(self, text: str, voice: str | None, language: str | None, speed: float) -> AudioResult:
        from kittentts import KittenTTS

        if language and language.lower() not in {"en", "en-us", "en-gb"}:
            raise ValueError("Kitten TTS 0.8 supports English")
        model = self._get_or_load("kitten", lambda: KittenTTS("KittenML/kitten-tts-nano-0.8"))
        audio = model.generate(text, voice=voice or "Leo", speed=speed)
        return _encode_wav(audio, 24_000)

    def _synthesize_supertonic(self, text: str, voice: str | None, language: str | None, speed: float) -> AudioResult:
        from supertonic import TTS

        model = self._get_or_load("supertonic", lambda: TTS(model="supertonic-3", auto_download=True))
        voice = voice or "M1"
        style = self._get_or_load(f"supertonic-style:{voice}", lambda: model.get_voice_style(voice))
        audio, _ = model.synthesize(text, voice_style=style, speed=speed, lang=language or "en", verbose=False)
        return _encode_wav(audio, model.sample_rate)
