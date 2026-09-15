from chorus.engines.base import Engine
from chorus.types import AudioResult, encode_wav


class Adapter(Engine):
    def synthesize(
        self, text: str, voice: str | None, language: str | None, speed: float
    ) -> AudioResult:
        from kittentts import KittenTTS

        if language and language.lower() not in {"en", "en-us", "en-gb"}:
            raise ValueError("Kitten TTS 0.8 supports English")
        model = self.cached("kitten", lambda: KittenTTS("KittenML/kitten-tts-nano-0.8"))
        audio = model.generate(text, voice=voice or "Leo", speed=speed)
        return encode_wav(audio, 24_000)
