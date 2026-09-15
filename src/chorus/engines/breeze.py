"""Breeze TTS 2 voice design through an isolated CUDA worker."""

from chorus.engines.worker import WorkerEngine

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


class Adapter(WorkerEngine):
    def validate_request(self, text, voice, language, speed) -> None:
        super().validate_request(text, voice, language, speed)
        if language is not None and language.strip().lower() not in _ALLOWED_LANGUAGES:
            raise ValueError(
                "Breeze TTS supports English and Chinese only; "
                f"unsupported language: {language}"
            )
