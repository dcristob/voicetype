"""Type transcribed text into the focused window.

Two strategies:

- ``YdotoolInjector`` emulates physical key presses via ``ydotool type``. It is
  ASCII-only: ydotool maps characters to keycodes through an internal US-layout
  table, so non-ASCII output (Spanish accents á é í ó ú ñ, and the typographic
  apostrophe ’ Whisper emits in English contractions) is silently dropped.
- ``ClipboardInjector`` puts the text on the Wayland clipboard (``wl-copy``) and
  sends a paste keystroke (``ydotool key``). This carries arbitrary Unicode
  faithfully and works on any compositor via the existing uinput setup. It saves
  and restores the previous clipboard contents.
"""

from __future__ import annotations

import subprocess
import time


class InjectionError(Exception):
    pass


class YdotoolInjector:
    def type_text(self, text: str) -> None:
        if not text.strip():
            return
        try:
            result = subprocess.run(
                ["ydotool", "type", "--", text],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise InjectionError(
                "ydotool not found; is it installed and is ydotoold running?"
            ) from exc
        if result.returncode != 0:
            raise InjectionError(f"ydotool failed: {result.stderr.strip()}")


# Linux input-event keycodes (see linux/input-event-codes.h) used to build
# ydotool key sequences for the paste shortcut.
_KEYCODES = {
    "ctrl": 29,
    "control": 29,
    "shift": 42,
    "alt": 56,
    "super": 125,
    "meta": 125,
    "v": 47,
}


class ClipboardInjector:
    def __init__(
        self,
        paste_shortcut: str = "ctrl+shift+v",
        restore_clipboard: bool = False,
        restore_delay: float = 0.3,
    ):
        self.restore_clipboard = restore_clipboard
        # Give the focused app time to read the clipboard before we restore the
        # previous contents, otherwise the paste can race the restore.
        self.restore_delay = restore_delay
        self._key_events = self._build_key_events(paste_shortcut)

    @staticmethod
    def _build_key_events(shortcut: str) -> list[str]:
        keys = [k.strip().lower() for k in shortcut.split("+") if k.strip()]
        try:
            codes = [_KEYCODES[k] for k in keys]
        except KeyError as exc:
            raise InjectionError(f"unknown key in paste shortcut: {exc}") from exc
        # Press in order, release in reverse.
        events = [f"{c}:1" for c in codes]
        events += [f"{c}:0" for c in reversed(codes)]
        return events

    def type_text(self, text: str) -> None:
        if not text.strip():
            return
        saved = self._read_clipboard() if self.restore_clipboard else None
        self._set_clipboard(text)
        self._paste()
        if self.restore_clipboard:
            time.sleep(self.restore_delay)
            self._restore_clipboard(saved)

    def _read_clipboard(self) -> bytes | None:
        try:
            result = subprocess.run(
                ["wl-paste", "--no-newline"], capture_output=True
            )
        except FileNotFoundError as exc:
            raise InjectionError(
                "wl-paste not found; install wl-clipboard"
            ) from exc
        # Non-zero typically means the clipboard is empty.
        return result.stdout if result.returncode == 0 else None

    def _set_clipboard(self, text: str) -> None:
        # wl-copy double-forks a server that holds the selection; that server
        # inherits our stdout/stderr, so capturing output would make run() block
        # until the server exits (i.e. forever). Discard its output instead.
        try:
            result = subprocess.run(
                ["wl-copy"],
                input=text,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError as exc:
            raise InjectionError(
                "wl-copy not found; install wl-clipboard"
            ) from exc
        if result.returncode != 0:
            raise InjectionError("wl-copy failed to set the clipboard")

    def _restore_clipboard(self, saved: bytes | None) -> None:
        try:
            if saved:
                subprocess.run(
                    ["wl-copy"],
                    input=saved,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                subprocess.run(
                    ["wl-copy", "--clear"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except FileNotFoundError:
            pass  # best effort; the paste already succeeded

    def _paste(self) -> None:
        try:
            result = subprocess.run(
                ["ydotool", "key", *self._key_events],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise InjectionError(
                "ydotool not found; is it installed and is ydotoold running?"
            ) from exc
        if result.returncode != 0:
            raise InjectionError(f"ydotool failed: {result.stderr.strip()}")


def make_injector(config):
    """Build the injector named by ``config.inject_method``."""
    if config.inject_method == "clipboard":
        return ClipboardInjector(
            paste_shortcut=config.paste_shortcut,
            restore_clipboard=config.restore_clipboard,
        )
    return YdotoolInjector()
