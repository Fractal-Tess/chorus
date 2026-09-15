from mini_tts.engines.base import Engine
from mini_tts.types import AudioResult, encode_wav


class Adapter(Engine):
    def synthesize(
        self, text: str, voice: str | None, language: str | None, speed: float
    ) -> AudioResult:
        from supertonic import TTS

        model = self.cached(
            "supertonic", lambda: TTS(model="supertonic-3", auto_download=True)
        )
        voice = voice or "M1"
        style = self.cached(
            f"supertonic-style:{voice}", lambda: model.get_voice_style(voice)
        )
        audio, _ = model.synthesize(
            text, voice_style=style, speed=speed, lang=language or "en", verbose=False
        )
        return encode_wav(audio, model.sample_rate)
