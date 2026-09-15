from chorus.engines.base import Engine
from chorus.types import AudioResult, encode_wav

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


class Adapter(Engine):
    def synthesize(
        self, text: str, voice: str | None, language: str | None, speed: float
    ) -> AudioResult:
        from pocket_tts import TTSModel

        if speed != 1.0:
            raise ValueError("Pocket TTS does not support speed adjustment")
        language = (language or "english").lower()
        language = _POCKET_LANGUAGES.get(language, language)
        model = self.cached(
            f"pocket:{language}",
            lambda: TTSModel.load_model(language=language, quantize=True),
        )
        voice = voice or "alba"
        state = self.cached(
            f"pocket-voice:{language}:{voice}",
            lambda: model.get_state_for_audio_prompt(voice),
        )
        audio = model.generate_audio(state, text)
        return encode_wav(audio.detach().cpu().numpy(), model.sample_rate)
