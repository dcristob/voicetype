import numpy as np

from voicetype.audio import AudioBuffer


def test_empty_returns_zero_length():
    buf = AudioBuffer()
    out = buf.get_audio()
    assert out.dtype == np.float32
    assert len(out) == 0


def test_accumulates_frames_in_order():
    buf = AudioBuffer()
    buf.add(np.array([1, 2], dtype=np.float32))
    buf.add(np.array([3, 4], dtype=np.float32))
    out = buf.get_audio()
    assert out.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_clear_empties_buffer():
    buf = AudioBuffer()
    buf.add(np.array([1, 2], dtype=np.float32))
    buf.clear()
    assert len(buf.get_audio()) == 0


def test_duration_seconds():
    buf = AudioBuffer(sample_rate=16000)
    buf.add(np.zeros(16000, dtype=np.float32))
    assert buf.duration_seconds() == 1.0
