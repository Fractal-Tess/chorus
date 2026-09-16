from __future__ import annotations

import tempfile
from pathlib import Path

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
    def _load_model(self, language: str):
        import yaml
        from pocket_tts import TTSModel
        from pocket_tts.utils.config import CONFIGS_DIR

        model_path = self.spec.artifact(f"languages/{language}/model.safetensors")
        tokenizer_path = self.spec.artifact(f"languages/{language}/tokenizer.model")
        config = yaml.safe_load(
            (CONFIGS_DIR / f"{language}.yaml").read_text(encoding="utf-8")
        )
        # TTSModel.load_model accepts local YAML configs. Override every
        # upstream-managed path so loading cannot trigger a network fetch.
        config["weights_path"] = str(model_path)
        config["weights_path_without_voice_cloning"] = str(model_path)
        config["flow_lm"]["lookup_table"]["tokenizer_path"] = str(tokenizer_path)

        with tempfile.TemporaryDirectory(prefix="chorus-pocket-") as directory:
            config_path = Path(directory) / f"{language}.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            return TTSModel.load_model(config=config_path, quantize=True)

    def synthesize(
        self, text: str, voice: str | None, language: str | None, speed: float
    ) -> AudioResult:
        from chorus.engines import ENGINE_INFO

        if speed != 1.0:
            raise ValueError("Pocket TTS does not support speed adjustment")
        language = (language or "english").lower()
        language = _POCKET_LANGUAGES.get(language, language)
        voice = voice or "alba"
        if language not in ENGINE_INFO["pocket"]["languages"]:
            raise ValueError(f"Unsupported Pocket TTS language: {language}")
        if voice not in ENGINE_INFO["pocket"]["voices"]:
            raise ValueError(f"Unknown Pocket TTS voice: {voice}")
        model = self.cached(
            f"pocket:{language}",
            lambda: self._load_model(language),
        )
        state = self.cached(
            f"pocket-voice:{language}:{voice}",
            lambda: model.get_state_for_audio_prompt(
                self.spec.artifact(
                    f"languages/{language}/embeddings/{voice}.safetensors"
                )
            ),
        )
        audio = model.generate_audio(state, text)
        return encode_wav(audio.detach().cpu().numpy(), model.sample_rate)
