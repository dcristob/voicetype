"""Audio capture: a pure accumulation buffer plus a sounddevice recorder."""

from __future__ import annotations

from typing import Callable

import numpy as np

SAMPLE_RATE = 16000


class AudioBuffer:
    """Accumulates mono float32 frames into one contiguous array."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._frames: list[np.ndarray] = []

    def add(self, frame: np.ndarray) -> None:
        self._frames.append(frame.astype(np.float32, copy=False))

    def get_audio(self) -> np.ndarray:
        if not self._frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._frames)

    def clear(self) -> None:
        self._frames.clear()

    def duration_seconds(self) -> float:
        return sum(len(f) for f in self._frames) / self.sample_rate
