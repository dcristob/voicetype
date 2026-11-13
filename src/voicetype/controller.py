"""Orchestrates capture → VAD → transcription → injection."""

from __future__ import annotations

import logging
from typing import Callable

from .config import Config
from .state import State, StateMachine
from .vad import SilenceTracker

logger = logging.getLogger(__name__)

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
        self._on_state_change = on_state_change
        self._sm = StateMachine(on_change=self._log_and_forward_change)
        self._tracker = SilenceTracker(
            silence_seconds=config.silence_seconds,
            frame_seconds=FRAME_SECONDS,
            min_speech_seconds=config.min_speech_seconds,
        )

    def _log_and_forward_change(self, old: State, new: State) -> None:
        logger.info("state %s -> %s", old.value, new.value)
        if self._on_state_change is not None:
            self._on_state_change(old, new)

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
        else:
            logger.info("toggle ignored while %s", self._sm.state.value)

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
        """Stop listening, transcribe, and inject — synchronously.

        Composed of ``begin_transcribe`` + ``complete_transcribe`` so the daemon
        can run the fast state transition on its event loop and hand the blocking
        transcription off to a worker thread (keeping the loop free to flush the
        tray icon and answer socket commands). Callers wanting it all in one
        thread — tests, headless use — just call ``finish()``.
        """
        audio = self.begin_transcribe()
        if audio is None:
            return
        self.complete_transcribe(audio)

    def begin_transcribe(self):
        """Transition LISTENING→TRANSCRIBING and return captured audio.

        Fast and non-blocking; meant to run on the daemon's event loop so the
        amber TRANSCRIBING icon is pushed out immediately. Returns ``None`` if we
        weren't listening (closing the double-finish race: once we're in
        TRANSCRIBING any concurrent finish sees the changed state and bails) or if
        stopping the recorder failed.
        """
        if self._sm.state is not State.LISTENING:
            return None
        self._sm.transition(State.TRANSCRIBING)
        # recorder.stop() is guarded so a failure still routes back to IDLE
        # rather than leaving the daemon stuck in TRANSCRIBING.
        try:
            return self.recorder.stop()
        except Exception as exc:
            self._fail("Transcription failed", exc)
            return None

    def complete_transcribe(self, audio) -> None:
        """Transcribe + inject + return to IDLE. May block; run off the loop."""
        try:
            text = self.transcriber.transcribe(audio, self.language)
        except Exception as exc:
            self._fail("Transcription failed", exc)
            return
        if text:
            # Length only: dictated text is private and stays out of the journal.
            logger.info("transcribed %d chars; injecting", len(text))
            try:
                self.injector.type_text(text)
            except Exception as exc:
                self._fail("Injection failed", exc)
                return
        else:
            logger.info("empty transcription; nothing to inject")
        self._sm.transition(State.IDLE)

    def _fail(self, title: str, exc: Exception) -> None:
        # Log the full exception (with traceback) so it lands in the journal —
        # the notification is ephemeral and otherwise the only record.
        logger.error("%s: %s", title, exc, exc_info=exc)
        self._sm.transition(State.ERROR)
        self._notify(title, str(exc))
        self._sm.transition(State.IDLE)

    def _notify(self, title: str, body: str) -> None:
        if self.notifier is not None:
            self.notifier(title, body)
