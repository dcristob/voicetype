from unittest.mock import MagicMock, patch

import numpy as np

from voicetype.asr import Transcriber


class _Seg:
    def __init__(self, text):
        self.text = text


def test_auto_language_passes_none_to_model():
    tr = Transcriber(model="small", device="cpu", compute_type="int8")
    fake = MagicMock()
    fake.transcribe.return_value = ([_Seg(" hello")], None)
    with patch("voicetype.asr.WhisperModel", return_value=fake):
        text = tr.transcribe(np.zeros(16000, dtype=np.float32), language="auto")
    assert text == "hello"
    assert fake.transcribe.call_args.kwargs["language"] is None


def test_forced_language_passed_through():
    tr = Transcriber(model="small", device="cpu", compute_type="int8")
    fake = MagicMock()
    fake.transcribe.return_value = ([_Seg("hola")], None)
    with patch("voicetype.asr.WhisperModel", return_value=fake):
        tr.transcribe(np.zeros(16000, dtype=np.float32), language="es")
    assert fake.transcribe.call_args.kwargs["language"] == "es"


def test_segments_joined_and_stripped():
    tr = Transcriber(model="small", device="cpu", compute_type="int8")
    fake = MagicMock()
    fake.transcribe.return_value = ([_Seg(" one"), _Seg(" two")], None)
    with patch("voicetype.asr.WhisperModel", return_value=fake):
        text = tr.transcribe(np.zeros(16000, dtype=np.float32), language="auto")
    assert text == "one two"


def test_cuda_oom_falls_back_to_cpu():
    tr = Transcriber(model="small", device="cuda", compute_type="int8_float16")
    fake = MagicMock()
    fake.transcribe.return_value = ([_Seg("ok")], None)

    calls = []

    def construct(model, device, compute_type):
        calls.append(device)
        if device == "cuda":
            raise RuntimeError("CUDA out of memory")
        return fake

    with patch("voicetype.asr.WhisperModel", side_effect=construct):
        text = tr.transcribe(np.zeros(16000, dtype=np.float32), language="auto")
    assert text == "ok"
    assert calls == ["cuda", "cpu"]  # tried CUDA, fell back to CPU
