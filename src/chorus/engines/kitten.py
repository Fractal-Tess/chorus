import json

from chorus.engines.base import Engine
from chorus.models import ModelSpec
from chorus.types import AudioResult, encode_wav


def _load_model(spec: ModelSpec):
    from kittentts.onnx_model import KittenTTS_1_Onnx

    config = json.loads(spec.artifact("config.json").read_text())
    model_path = spec.artifact(config["model_file"])
    voices_path = spec.artifact(config["voices"])
    return KittenTTS_1_Onnx(
        model_path=model_path,
        voices_path=voices_path,
        speed_priors=config.get("speed_priors", {}),
        voice_aliases=config.get("voice_aliases", {}),
    )


class Adapter(Engine):
    def synthesize(
        self, text: str, voice: str | None, language: str | None, speed: float
    ) -> AudioResult:
        if language and language.lower() not in {"en", "en-us", "en-gb"}:
            raise ValueError("Kitten TTS 0.8 supports English")
        model = self.cached("kitten", lambda: _load_model(self.spec))
        audio = model.generate(text, voice=voice or "Leo", speed=speed)
        return encode_wav(audio, 24_000)
