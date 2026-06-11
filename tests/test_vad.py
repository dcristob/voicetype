from voicetype.vad import SilenceTracker

FRAME = 0.032  # 32 ms frames (512 samples @ 16 kHz)


def make_tracker():
    return SilenceTracker(
        silence_seconds=2.0,
        frame_seconds=FRAME,
        min_speech_seconds=0.1,
        threshold=0.5,
    )


def test_no_end_before_any_speech():
    tr = make_tracker()
    assert not any(tr.update(0.0) for _ in range(100))


def test_ends_after_speech_then_silence():
    tr = make_tracker()
    for _ in range(7):
        assert tr.update(0.9) is False
    ended = [tr.update(0.0) for _ in range(70)]
    assert True in ended
    first_end = ended.index(True)
    assert first_end == 62


def test_silence_resets_on_renewed_speech():
    tr = make_tracker()
    for _ in range(7):
        tr.update(0.9)
    for _ in range(30):  # ~0.96 s silence, not enough
        assert tr.update(0.0) is False
    tr.update(0.9)  # speech again resets the silence counter
    for _ in range(30):
        assert tr.update(0.0) is False  # still not enough after reset


def test_reset_clears_state():
    tr = make_tracker()
    for _ in range(7):
        tr.update(0.9)
    tr.reset()
    assert not any(tr.update(0.0) for _ in range(100))  # no prior speech after reset
