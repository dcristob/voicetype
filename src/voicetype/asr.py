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
        primary = self.device if self.device != "auto" else "cuda"
        # Try progressively more conservative options:
        #   1. the configured device + compute type
        #   2. the same GPU with plain int8 (older GPUs like Pascal/GTX 10xx
        #      reject int8_float16/float16 but support int8)
        #   3. CPU int8 as a universal fallback
        # NOTE: device/compute_type MUST be keyword args — WhisperModel's 3rd
        # positional parameter is device_index, not compute_type.
        attempts = []
        if primary != "cpu":
            attempts.append((primary, self.compute_type))
            if self.compute_type != "int8":
                attempts.append((primary, "int8"))
        attempts.append(("cpu", "int8"))

        last_error: Exception | None = None
        for device, compute_type in attempts:
            try:
                model = WhisperModel(
                    self.model_name, device=device, compute_type=compute_type
                )
                self.device = device
                self.compute_type = compute_type
                return model
            except Exception as exc:
                last_error = exc
        raise last_error

    def ensure_loaded(self) -> None:
        if self._model is None:
            self._model = self._load()

    def transcribe(self, audio: np.ndarray, language: str) -> str:
        self.ensure_loaded()
        lang = None if language == "auto" else language
        segments, _info = self._model.transcribe(audio, language=lang)
        return " ".join(seg.text.strip() for seg in segments).strip()
