"""Voice-activity detection: pure silence timing + silero model wrapper."""

from __future__ import annotations


class SilenceTracker:
    """Reports end-of-utterance from a stream of per-frame speech probabilities.

    An utterance "ends" once total speech has exceeded ``min_speech_seconds``
    and ``silence_seconds`` of continuous silence has followed.
    """

    def __init__(
        self,
        silence_seconds: float,
        frame_seconds: float,
        min_speech_seconds: float = 0.3,
        threshold: float = 0.5,
    ):
        self.silence_seconds = silence_seconds
        self.frame_seconds = frame_seconds
        self.min_speech_seconds = min_speech_seconds
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self._speech_total = 0.0
        self._silence_run = 0.0
        self._spoke = False

    def update(self, speech_prob: float) -> bool:
        is_speech = speech_prob >= self.threshold
        if is_speech:
            self._speech_total += self.frame_seconds
            self._silence_run = 0.0
            if self._speech_total >= self.min_speech_seconds:
                self._spoke = True
            return False
        # silence frame
        if not self._spoke:
            return False
        self._silence_run += self.frame_seconds
        return self._silence_run >= self.silence_seconds


# Silero operates on fixed 512-sample frames at 16 kHz.
SILERO_FRAME_SAMPLES = 512
SAMPLE_RATE = 16000


class SileroVad:
    """Wraps silero-vad to score a 512-sample frame in [0, 1]."""

    def __init__(self) -> None:
        import torch  # noqa: F401  (silero pulls torch)
        from silero_vad import load_silero_vad

        self._model = load_silero_vad()

    def prob(self, frame) -> float:
        """frame: float32 numpy array of length 512, mono, 16 kHz."""
        import torch

        tensor = torch.from_numpy(frame)
        return float(self._model(tensor, SAMPLE_RATE).item())
