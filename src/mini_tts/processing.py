from __future__ import annotations

import io
import math
import re
import threading
import time
import unicodedata
from dataclasses import replace
from typing import Any

import numpy as np
import soundfile as sf

from .types import AudioResult, WordAlignment, encode_wav


class Processor:
    """Apply optional CPU audio enhancement and English forced alignment."""

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._load_lock = threading.RLock()
        self._cpu_lock = threading.Lock()

    def process(
        self,
        audio: AudioResult,
        text: str,
        lava_sr: bool = False,
        force_align: bool = False,
    ) -> AudioResult:
        if not lava_sr and not force_align:
            return replace(
                audio,
                timings_ms={
                    **audio.timings_ms,
                    "lava_sr": 0.0,
                    "alignment": 0.0,
                    "total": 0.0,
                },
            )
        total_started = time.perf_counter()
        with self._cpu_lock:
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

        timings_ms = dict(audio.timings_ms)
        timings_ms.update(
            {
                "lava_sr": lava_sr_ms,
                "alignment": alignment_ms,
                "total": (time.perf_counter() - total_started) * 1_000,
            }
        )
        return replace(audio, alignment=alignment, timings_ms=timings_ms)

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
            model.bwe_model.lr_refiner = FastLRMerge(
                device="cpu", cutoff=8_000, transition_bins=1_024
            )
            return model

        model = self._get_or_load("lavasr", load_model)
        samples, sample_rate = sf.read(io.BytesIO(audio.wav), dtype="float32")
        input_audio = torch.from_numpy(np.asarray(samples)).reshape(1, -1)
        input_audio = torchaudio.functional.resample(input_audio, sample_rate, 16_000)
        enhanced = model.enhance(input_audio, denoise=False).detach().cpu().numpy()
        enhanced_audio = encode_wav(enhanced, 48_000)
        return replace(
            audio,
            wav=enhanced_audio.wav,
            sample_rate=enhanced_audio.sample_rate,
            duration=enhanced_audio.duration,
        )

    def _align_english(
        self, audio: AudioResult, text: str
    ) -> tuple[WordAlignment, ...]:
        import torch
        import torchaudio

        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

        def load_model() -> Any:
            return bundle.get_model().eval()

        model = self._get_or_load("force-aligner:english", load_model)
        dictionary = {label: index for index, label in enumerate(bundle.get_labels())}
        words: list[tuple[str, str]] = []
        for raw_word in re.findall(r"\S+", text):
            ascii_word = (
                unicodedata.normalize("NFKD", raw_word)
                .encode("ascii", "ignore")
                .decode()
            )
            normalized = "".join(
                character
                for character in ascii_word.upper()
                if character in dictionary
                and dictionary[character] != 0
                and character != "|"
            )
            if normalized:
                words.append((raw_word, normalized))
        if not words:
            raise ValueError(
                "Force alignment requires English words containing letters"
            )

        transcript = "|".join(normalized for _, normalized in words)
        targets = torch.tensor(
            [[dictionary[character] for character in transcript]], dtype=torch.int32
        )
        samples, sample_rate = sf.read(io.BytesIO(audio.wav), dtype="float32")
        samples = np.asarray(samples)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        waveform = torch.from_numpy(samples).reshape(1, -1)
        if sample_rate != bundle.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, bundle.sample_rate
            )

        with torch.inference_mode():
            emissions, _ = model(waveform)
            emissions = torch.log_softmax(emissions, dim=-1)
            try:
                path, scores = torchaudio.functional.forced_align(
                    emissions, targets, blank=0
                )
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
            mean_log_score = (
                sum(float(span.score) * (span.end - span.start) for span in spans)
                / frame_count
            )
            aligned_words.append(
                WordAlignment(
                    word=raw_word,
                    start_ms=round(spans[0].start * milliseconds_per_frame),
                    end_ms=round(spans[-1].end * milliseconds_per_frame),
                    score=round(math.exp(mean_log_score), 4),
                )
            )
        return tuple(aligned_words)
