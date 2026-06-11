import pytest

from voicetype.state import State, StateMachine, InvalidTransition


def test_starts_idle():
    sm = StateMachine()
    assert sm.state is State.IDLE


def test_full_cycle():
    sm = StateMachine()
    sm.transition(State.LISTENING)
    sm.transition(State.TRANSCRIBING)
    sm.transition(State.IDLE)
    assert sm.state is State.IDLE


def test_invalid_transition_raises():
    sm = StateMachine()
    with pytest.raises(InvalidTransition):
        sm.transition(State.TRANSCRIBING)  # IDLE -> TRANSCRIBING not allowed


def test_error_from_any_state_then_idle():
    sm = StateMachine()
    sm.transition(State.LISTENING)
    sm.transition(State.ERROR)
    sm.transition(State.IDLE)
    assert sm.state is State.IDLE


def test_on_change_callback_fires():
    seen = []
    sm = StateMachine(on_change=lambda old, new: seen.append((old, new)))
    sm.transition(State.LISTENING)
    assert seen == [(State.IDLE, State.LISTENING)]
