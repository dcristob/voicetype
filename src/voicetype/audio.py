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


SILERO_FRAME_SAMPLES = 512


class Recorder:
    """Captures mono 16 kHz audio, delivering 512-sample float32 frames."""

    def __init__(self, mic_device: str | None = None):
        self.mic_device = mic_device
        self._stream = None
        self._buffer = AudioBuffer()
        self._on_frame: Callable[[np.ndarray], None] | None = None

    def warm_up(self) -> None:
        """Initialise PortAudio ahead of the first recording.

        Importing sounddevice runs Pa_Initialize, which enumerates every audio
        device and can take several seconds. Done lazily in start() it would
        freeze the event loop on the first hotkey press after daemon start, and
        swallow the first words spoken. The daemon calls this off-loop at boot.
        """
        import sounddevice as sd

        sd.query_devices()

    def start(self, on_frame: Callable[[np.ndarray], None]) -> None:
        import sounddevice as sd

        self._on_frame = on_frame
        self._buffer.clear()

        def callback(indata, frames, time_info, status):  # noqa: ARG001
            frame = indata[:, 0].copy()
            self._buffer.add(frame)
            if self._on_frame is not None:
                self._on_frame(frame)

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=SILERO_FRAME_SAMPLES,
            device=self.mic_device,
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        return self._buffer.get_audio()
