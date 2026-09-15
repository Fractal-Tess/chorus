"""Fish Audio S2-Pro with natural-language style cues and local CUDA inference."""

import re

from chorus.engines import ENGINE_INFO
from chorus.engines.worker import WorkerEngine

_CONTROL_CUES = re.compile(r"\[[^\[\]]*\]|<\|speaker:\d+\|>")


class Adapter(WorkerEngine):
    def alignment_text(self, text: str) -> str:
        """Align spoken words, not Fish's style and speaker instructions."""
        return " ".join(_CONTROL_CUES.sub(" ", text).split())

    def validate_request(self, text, voice, language, speed) -> None:
        super().validate_request(text, voice, language, speed)
        if voice is not None and not voice.strip():
            raise ValueError("Fish style must be non-empty when provided")
        if language is not None:
            code = language.strip().lower().replace("_", "-").split("-", 1)[0]
            if code not in ENGINE_INFO["fish"]["languages"]:
                raise ValueError(f"Unsupported Fish S2-Pro language: {language}")
        if not self.alignment_text(text):
            raise ValueError("Fish input must contain speech, not only control cues")
