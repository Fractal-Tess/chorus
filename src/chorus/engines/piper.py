import io
import wave

import soundfile as sf

from chorus.engines.base import Engine
from chorus.types import AudioResult


class Adapter(Engine):
    def synthesize(
        self, text: str, voice: str | None, language: str | None, speed: float
    ) -> AudioResult:
        from piper import PiperVoice, SynthesisConfig

        if voice not in (None, "en_US-lessac-medium"):
            raise ValueError(
                "Only the bundled Piper voice en_US-lessac-medium is available"
            )
        if language and language.lower() not in {"en", "en-us"}:
            raise ValueError("The bundled Piper model only supports en-US")
        model_path = self.spec.directory.parent / "en_US-lessac-medium.onnx"
        if not model_path.is_file():
            raise RuntimeError(f"Piper model is missing: {model_path}")
        model = self.cached(
            "piper", lambda: PiperVoice.load(model_path, use_cuda=False)
        )
        output = io.BytesIO()
        with wave.open(output, "wb") as wav_file:
            model.synthesize_wav(
                text, wav_file, SynthesisConfig(length_scale=1.0 / speed)
            )
        data = output.getvalue()
        with sf.SoundFile(io.BytesIO(data)) as audio_file:
            sample_rate = audio_file.samplerate
            duration = audio_file.frames / sample_rate
        return AudioResult(data, sample_rate, duration)
