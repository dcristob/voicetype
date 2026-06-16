from unittest.mock import patch

import pytest

from voicetype.inject import (
    ClipboardInjector,
    YdotoolInjector,
    InjectionError,
    make_injector,
)
from voicetype.config import Config


def test_type_text_builds_ydotool_command():
    inj = YdotoolInjector()
    with patch("voicetype.inject.subprocess.run") as run:
        run.return_value.returncode = 0
        inj.type_text("hola mundo")
    run.assert_called_once()
    args = run.call_args[0][0]
    assert args[:2] == ["ydotool", "type"]
    assert args[-1] == "hola mundo"


def test_empty_text_does_not_call_ydotool():
    inj = YdotoolInjector()
    with patch("voicetype.inject.subprocess.run") as run:
        inj.type_text("   ")
    run.assert_not_called()


def test_missing_ydotool_raises_injection_error():
    inj = YdotoolInjector()
    with patch("voicetype.inject.subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(InjectionError):
            inj.type_text("hi")


# --- ClipboardInjector --------------------------------------------------


def _ok(stdout=b""):
    r = type("R", (), {})()
    r.returncode = 0
    r.stdout = stdout
    r.stderr = ""
    return r


def test_clipboard_injector_copies_text_then_pastes():
    inj = ClipboardInjector(restore_clipboard=False)
    calls = []

    def run(cmd, *args, **kwargs):
        calls.append((cmd, kwargs.get("input")))
        return _ok()

    with patch("voicetype.inject.subprocess.run", side_effect=run):
        inj.type_text("áéí — don't")

    # First: wl-copy with the text on stdin (no arg escaping needed).
    assert calls[0][0][0] == "wl-copy"
    assert calls[0][1] == "áéí — don't"
    # Then: ydotool key for the paste shortcut.
    assert calls[1][0][:2] == ["ydotool", "key"]


def test_clipboard_ctrl_v_shortcut():
    inj = ClipboardInjector(paste_shortcut="ctrl+v", restore_clipboard=False)
    calls = []
    with patch(
        "voicetype.inject.subprocess.run",
        side_effect=lambda cmd, *a, **k: calls.append(cmd) or _ok(),
    ):
        inj.type_text("x")
    # Ctrl down (29:1), V down (47:1), V up (47:0), Ctrl up (29:0)
    assert calls[1][2:] == ["29:1", "47:1", "47:0", "29:0"]


def test_clipboard_ctrl_shift_v_shortcut():
    inj = ClipboardInjector(paste_shortcut="ctrl+shift+v", restore_clipboard=False)
    calls = []
    with patch(
        "voicetype.inject.subprocess.run",
        side_effect=lambda cmd, *a, **k: calls.append(cmd) or _ok(),
    ):
        inj.type_text("x")
    # Press ctrl, shift, v; release in reverse: v, shift, ctrl.
    assert calls[1][2:] == ["29:1", "42:1", "47:1", "47:0", "42:0", "29:0"]


def test_clipboard_empty_text_is_noop():
    inj = ClipboardInjector()
    with patch("voicetype.inject.subprocess.run") as run:
        inj.type_text("   ")
    run.assert_not_called()


def test_clipboard_saves_and_restores_previous_contents():
    inj = ClipboardInjector(restore_clipboard=True, restore_delay=0)
    calls = []

    def run(cmd, *args, **kwargs):
        calls.append((cmd, kwargs.get("input")))
        # wl-paste returns the existing clipboard contents.
        return _ok(b"previous" if cmd[0] == "wl-paste" else b"")

    with patch("voicetype.inject.subprocess.run", side_effect=run):
        with patch("voicetype.inject.time.sleep"):
            inj.type_text("new text")

    cmds = [c[0][0] for c in calls]
    assert cmds[0] == "wl-paste"  # save
    assert cmds[1] == "wl-copy"  # set new
    assert cmds[2] == "ydotool"  # paste
    # Last call restores the saved bytes.
    assert calls[-1][0][0] == "wl-copy"
    assert calls[-1][1] == b"previous"


def test_clipboard_missing_tool_raises_injection_error():
    inj = ClipboardInjector(restore_clipboard=False)
    with patch("voicetype.inject.subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(InjectionError):
            inj.type_text("hi")


def test_make_injector_selects_clipboard():
    cfg = Config(inject_method="clipboard")
    inj = make_injector(cfg)
    assert isinstance(inj, ClipboardInjector)
    # Restore is off by default (avoids the paste-vs-restore race).
    assert inj.restore_clipboard is False


def test_make_injector_passes_restore_flag():
    cfg = Config(inject_method="clipboard", restore_clipboard=True)
    assert make_injector(cfg).restore_clipboard is True


def test_make_injector_selects_ydotool():
    cfg = Config(inject_method="ydotool")
    assert isinstance(make_injector(cfg), YdotoolInjector)
