import socket
from unittest.mock import patch

import pytest

from voicetype.inject import (
    ClipboardInjector,
    YdotoolInjector,
    InjectionError,
    make_injector,
    wayland_env,
)
from voicetype.config import Config


def _bind(directory, name):
    """Create a real AF_UNIX socket, the way a compositor does."""
    sock = socket.socket(socket.AF_UNIX)
    sock.bind(str(directory / name))
    return sock


@pytest.fixture
def runtime_dir(tmp_path):
    """An empty XDG_RUNTIME_DIR, separate from the autouse fixture's socket."""
    d = tmp_path / "run"
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def fake_wayland_session(tmp_path, monkeypatch):
    """Run every test as if a compositor socket were present.

    ClipboardInjector resolves the Wayland display from XDG_RUNTIME_DIR at call
    time; without this the suite would depend on a live compositor. Tests that
    care about resolution override these env vars in their own body.
    """
    sock = _bind(tmp_path, "wayland-0")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    yield tmp_path
    sock.close()


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


# --- Wayland display resolution ----------------------------------------
#
# systemd starts voicetyped at login, which on COSMIC happens *before* the
# compositor pushes WAYLAND_DISPLAY into the user manager's environment, so the
# daemon inherits an environment without it and every wl-copy call fails with
# "Failed to connect to a Wayland server". Resolve the socket at injection time
# instead of trusting the environment we were launched with.


def test_wayland_env_keeps_display_whose_socket_exists():
    assert wayland_env()["WAYLAND_DISPLAY"] == "wayland-0"


def test_wayland_env_discovers_socket_when_display_is_unset(runtime_dir, monkeypatch):
    sock = _bind(runtime_dir, "wayland-1")
    # The compositor also drops a plain lock file next to the socket.
    (runtime_dir / "wayland-1.lock").write_text("")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    try:
        assert wayland_env()["WAYLAND_DISPLAY"] == "wayland-1"
    finally:
        sock.close()


def test_wayland_env_replaces_a_stale_display(runtime_dir, monkeypatch):
    sock = _bind(runtime_dir, "wayland-1")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-7")  # no such socket
    try:
        assert wayland_env()["WAYLAND_DISPLAY"] == "wayland-1"
    finally:
        sock.close()


def test_wayland_env_raises_when_no_compositor_socket(runtime_dir, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with pytest.raises(InjectionError, match="Wayland"):
        wayland_env()


def test_clipboard_passes_resolved_display_to_wl_copy(runtime_dir, monkeypatch):
    sock = _bind(runtime_dir, "wayland-2")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    inj = ClipboardInjector(restore_clipboard=False)
    calls = []

    def run(cmd, *args, **kwargs):
        calls.append((cmd, kwargs.get("env")))
        return _ok()

    try:
        with patch("voicetype.inject.subprocess.run", side_effect=run):
            inj.type_text("hola")
    finally:
        sock.close()

    cmd, env = next(c for c in calls if c[0][0] == "wl-copy")
    assert env["WAYLAND_DISPLAY"] == "wayland-2"


def test_clipboard_passes_resolved_display_to_wl_paste(runtime_dir, monkeypatch):
    sock = _bind(runtime_dir, "wayland-2")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    inj = ClipboardInjector(restore_clipboard=True, restore_delay=0)
    calls = []

    def run(cmd, *args, **kwargs):
        calls.append((cmd, kwargs.get("env")))
        return _ok(b"previous" if cmd[0] == "wl-paste" else b"")

    try:
        with patch("voicetype.inject.subprocess.run", side_effect=run):
            with patch("voicetype.inject.time.sleep"):
                inj.type_text("hola")
    finally:
        sock.close()

    cmd, env = next(c for c in calls if c[0][0] == "wl-paste")
    assert env["WAYLAND_DISPLAY"] == "wayland-2"


# --- diagnostics: injection steps are logged (lengths/results, never text) ----


def test_clipboard_injection_steps_are_logged_without_text(caplog):
    inj = ClipboardInjector(restore_clipboard=False)
    with patch("voicetype.inject.subprocess.run", return_value=_ok()):
        with caplog.at_level("INFO", logger="voicetype.inject"):
            inj.type_text("top secret")
    assert "wl-copy" in caplog.text
    assert "ydotool key" in caplog.text
    assert "rc=0" in caplog.text
    assert "top secret" not in caplog.text


def test_paste_logs_ydotool_output(caplog):
    # ydotool exits 0 even when it can't reach ydotoold, printing the problem
    # instead; that output must reach the journal.
    inj = ClipboardInjector(restore_clipboard=False)
    noisy = _ok(stdout="failed to connect socket")
    with patch("voicetype.inject.subprocess.run", return_value=noisy):
        with caplog.at_level("INFO", logger="voicetype.inject"):
            inj.type_text("x")
    assert "failed to connect socket" in caplog.text
