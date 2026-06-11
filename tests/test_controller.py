import numpy as np

from voicetype.config import Config
from voicetype.controller import Controller
from voicetype.state import State


class BoomTranscriber:
    def ensure_loaded(self):
        pass

    def transcribe(self, audio, language):
        raise RuntimeError("kaboom")


class FakeRecorder:
    def __init__(self):
        self.started = False
        self.audio = np.ones(16000, dtype=np.float32)

    def start(self, on_frame):
        self.started = True
        self._on_frame = on_frame

    def stop(self):
        self.started = False
        return self.audio


class FakeVad:
    def prob(self, frame):
        return 0.0


class FakeTranscriber:
    def __init__(self, text="hello world"):
        self.text = text
        self.calls = []

    def ensure_loaded(self):
        pass

    def transcribe(self, audio, language):
        self.calls.append(language)
        return self.text


class FakeInjector:
    def __init__(self):
        self.typed = []

    def type_text(self, text):
        self.typed.append(text)


def make_controller(transcriber=None):
    return Controller(
        config=Config(silence_seconds=2.0, min_speech_seconds=0.1),
        recorder=FakeRecorder(),
        vad=FakeVad(),
        transcriber=transcriber or FakeTranscriber(),
        injector=FakeInjector(),
    )


def test_toggle_from_idle_starts_listening():
    c = make_controller()
    c.toggle()
    assert c.state is State.LISTENING
    assert c.recorder.started is True


def test_full_cycle_types_text_and_returns_idle():
    inj = FakeInjector()
    tr = FakeTranscriber("hola")
    c = Controller(
        config=Config(min_speech_seconds=0.1),
        recorder=FakeRecorder(),
        vad=FakeVad(),
        transcriber=tr,
        injector=inj,
    )
    c.toggle()        # start listening
    c.finish()        # stop + transcribe + inject
    assert inj.typed == ["hola"]
    assert c.state is State.IDLE


def test_feed_frame_detects_end_of_utterance():
    c = make_controller()
    c.toggle()
    frame = np.zeros(512, dtype=np.float32)
    # Mark that speech happened so silence can trigger an end.
    c._tracker._spoke = True
    results = [c.feed_frame(frame) for _ in range(70)]
    assert True in results


def test_auto_language_by_default():
    tr = FakeTranscriber()
    c = make_controller(transcriber=tr)
    c.toggle()
    c.finish()
    assert tr.calls == ["auto"]


def test_set_language_forces_language():
    tr = FakeTranscriber()
    c = make_controller(transcriber=tr)
    c.set_language("es")
    c.toggle()
    c.finish()
    assert tr.calls == ["es"]


def test_transcription_failure_is_logged_with_detail(caplog):
    # The actual exception must be written to the log (journal), not just shown
    # as an ephemeral notification — otherwise failures are undebuggable.
    notices = []
    c = Controller(
        config=Config(min_speech_seconds=0.1),
        recorder=FakeRecorder(),
        vad=FakeVad(),
        transcriber=BoomTranscriber(),
        injector=FakeInjector(),
        notifier=lambda title, body: notices.append((title, body)),
    )
    c.toggle()
    with caplog.at_level("ERROR", logger="voicetype.controller"):
        c.finish()
    assert "kaboom" in caplog.text
    assert notices and notices[0][0] == "Transcription failed"
    assert c.state is State.IDLE


def test_empty_transcription_skips_injection():
    inj = FakeInjector()
    c = Controller(
        config=Config(min_speech_seconds=0.1),
        recorder=FakeRecorder(),
        vad=FakeVad(),
        transcriber=FakeTranscriber(""),
        injector=inj,
    )
    c.toggle()
    c.finish()
    assert inj.typed == []
    assert c.state is State.IDLE


def test_toggle_while_listening_finishes():
    inj = FakeInjector()
    c = Controller(
        config=Config(min_speech_seconds=0.1),
        recorder=FakeRecorder(),
        vad=FakeVad(),
        transcriber=FakeTranscriber("done"),
        injector=inj,
    )
    c.toggle()  # listening
    c.toggle()  # should finish
    assert inj.typed == ["done"]
    assert c.state is State.IDLE
