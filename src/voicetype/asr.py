"""faster-whisper transcription wrapper with a warm, lazily-loaded model."""

from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel


class Transcriber:
    def __init__(self, model: str, device: str, compute_type: str):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self._model: WhisperModel | None = None

    def _load(self) -> WhisperModel:
        device = self.device if self.device != "auto" else "cuda"
        try:
            return WhisperModel(self.model_name, device, self.compute_type)
        except RuntimeError as exc:
            if "cuda" in str(exc).lower() or "out of memory" in str(exc).lower():
                # Fall back to CPU int8.
                self.device = "cpu"
                self.compute_type = "int8"
                return WhisperModel(self.model_name, "cpu", "int8")
            raise

    def ensure_loaded(self) -> None:
        if self._model is None:
            self._model = self._load()

    def transcribe(self, audio: np.ndarray, language: str) -> str:
        self.ensure_loaded()
        lang = None if language == "auto" else language
        segments, _info = self._model.transcribe(audio, language=lang)
        return " ".join(seg.text.strip() for seg in segments).strip()
