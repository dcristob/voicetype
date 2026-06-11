"""Orchestrates capture → VAD → transcription → injection."""

from __future__ import annotations

from typing import Callable

from .config import Config
from .state import State, StateMachine
from .vad import SilenceTracker

FRAME_SECONDS = 512 / 16000  # 0.032 s


class Controller:
    def __init__(
        self,
        *,
        config: Config,
        recorder,
        vad,
        transcriber,
        injector,
        notifier: Callable[[str, str], None] | None = None,
        on_state_change: Callable[[State, State], None] | None = None,
    ):
        self.config = config
        self.recorder = recorder
        self.vad = vad
        self.transcriber = transcriber
        self.injector = injector
        self.notifier = notifier
        self.language = config.language_default
        self._sm = StateMachine(on_change=on_state_change)
        self._tracker = SilenceTracker(
            silence_seconds=config.silence_seconds,
            frame_seconds=FRAME_SECONDS,
            min_speech_seconds=config.min_speech_seconds,
        )

    @property
    def state(self) -> State:
        return self._sm.state

    def set_language(self, lang: str) -> None:
        self.language = lang

    def toggle(self) -> None:
        if self._sm.state is State.IDLE:
            self._start()
        elif self._sm.state is State.LISTENING:
            self.finish()

    def _start(self) -> None:
        self._tracker.reset()
        self._sm.transition(State.LISTENING)
        self.recorder.start(self._noop)

    def _noop(self, frame) -> None:
        # The daemon feeds frames via feed_frame(); the recorder callback only
        # needs to accumulate audio, which it does internally.
        pass

    def feed_frame(self, frame) -> bool:
        if self._sm.state is not State.LISTENING:
            return False
        return self._tracker.update(self.vad.prob(frame))

    def finish(self) -> None:
        if self._sm.state is not State.LISTENING:
            return
        self._sm.transition(State.TRANSCRIBING)
        audio = self.recorder.stop()
        try:
            text = self.transcriber.transcribe(audio, self.language)
        except Exception as exc:  # transcription failure -> notify, reset
            self._sm.transition(State.ERROR)
            self._notify("Transcription failed", str(exc))
            self._sm.transition(State.IDLE)
            return
        if text:
            try:
                self.injector.type_text(text)
            except Exception as exc:
                self._sm.transition(State.ERROR)
                self._notify("Injection failed", str(exc))
                self._sm.transition(State.IDLE)
                return
        self._sm.transition(State.IDLE)

    def _notify(self, title: str, body: str) -> None:
        if self.notifier is not None:
            self.notifier(title, body)
