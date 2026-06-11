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

    attempts = []

    def construct(model, device, compute_type):
        attempts.append((device, compute_type))
        if device == "cuda":
            raise RuntimeError("CUDA error")
        return fake

    with patch("voicetype.asr.WhisperModel", side_effect=construct):
        text = tr.transcribe(np.zeros(16000, dtype=np.float32), language="auto")
    assert text == "ok"
    # Tried the GPU (both compute types) then fell back to CPU int8.
    assert ("cuda", "int8_float16") in attempts
    assert ("cuda", "int8") in attempts
    assert attempts[-1] == ("cpu", "int8")
    assert tr.device == "cpu"


def test_unsupported_gpu_compute_downgrades_to_int8_on_gpu():
    # Pascal/GTX 10xx case: int8_float16 is rejected on the GPU, but plain int8
    # works on the GPU. We must NOT fall all the way back to CPU.
    tr = Transcriber(model="small", device="cuda", compute_type="int8_float16")
    fake = MagicMock()
    fake.transcribe.return_value = ([_Seg("ok")], None)

    def construct(model, device, compute_type):
        if device == "cuda" and compute_type == "int8_float16":
            raise ValueError("int8_float16 not supported on this device")
        return fake  # cuda/int8 succeeds

    with patch("voicetype.asr.WhisperModel", side_effect=construct):
        text = tr.transcribe(np.zeros(16000, dtype=np.float32), language="auto")
    assert text == "ok"
    assert tr.device == "cuda"
    assert tr.compute_type == "int8"


def test_model_constructed_with_keyword_args():
    # Regression: WhisperModel's 3rd positional arg is device_index, NOT
    # compute_type. device and compute_type must be passed as keywords or the
    # compute_type string lands in device_index and the model fails to load.
    tr = Transcriber(model="small", device="cpu", compute_type="int8")
    fake = MagicMock()
    fake.transcribe.return_value = ([_Seg("x")], None)
    with patch("voicetype.asr.WhisperModel", return_value=fake) as wm:
        tr.transcribe(np.zeros(16000, dtype=np.float32), language="auto")
    args, kwargs = wm.call_args
    assert args == ("small",)  # only the model name is positional
    assert kwargs["device"] == "cpu"
    assert kwargs["compute_type"] == "int8"
