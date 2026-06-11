"""Activation state machine."""

from __future__ import annotations

from enum import Enum
from typing import Callable


class State(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    ERROR = "error"


class InvalidTransition(Exception):
    pass


_ALLOWED: dict[State, set[State]] = {
    State.IDLE: {State.LISTENING, State.ERROR},
    State.LISTENING: {State.TRANSCRIBING, State.IDLE, State.ERROR},
    State.TRANSCRIBING: {State.IDLE, State.ERROR},
    State.ERROR: {State.IDLE},
}


class StateMachine:
    def __init__(self, on_change: Callable[[State, State], None] | None = None):
        self._state = State.IDLE
        self._on_change = on_change

    @property
    def state(self) -> State:
        return self._state

    def transition(self, target: State) -> None:
        if target not in _ALLOWED[self._state]:
            raise InvalidTransition(f"{self._state.value} -> {target.value}")
        old = self._state
        self._state = target
        if self._on_change is not None:
            self._on_change(old, target)
