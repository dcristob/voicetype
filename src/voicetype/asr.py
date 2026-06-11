"""faster-whisper transcription wrapper with a warm, lazily-loaded model."""

from __future__ import annotations

import logging

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


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
        try:
            return self._run(audio, lang)
        except Exception as exc:
            # _load tiers GPU->CPU only at load time; the GPU can still run out
            # of memory (or otherwise fail) DURING inference. If that happens,
            # reload on CPU int8 and retry once. On CPU there is no lower tier,
            # so re-raise and let the controller report it.
            if self.device == "cpu":
                raise
            logger.warning(
                "GPU inference failed (%s); reloading on CPU int8 and retrying",
                exc,
            )
            self._model = WhisperModel(
                self.model_name, device="cpu", compute_type="int8"
            )
            self.device = "cpu"
            self.compute_type = "int8"
            return self._run(audio, lang)

    def _run(self, audio: np.ndarray, lang: str | None) -> str:
        segments, _info = self._model.transcribe(audio, language=lang)
        return " ".join(seg.text.strip() for seg in segments).strip()
