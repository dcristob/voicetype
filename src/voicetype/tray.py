"""StatusNotifierItem tray icon with a control menu (pystray)."""

from __future__ import annotations

from importlib.resources import files
from typing import Callable

import pystray
from PIL import Image

from .state import State

_ICONS = {
    State.IDLE: "idle.png",
    State.LISTENING: "listening.png",
    State.TRANSCRIBING: "transcribing.png",
    State.ERROR: "idle.png",
}


def _load_icon(state: State) -> Image.Image:
    resource = files("voicetype").joinpath("icons", _ICONS[state])
    with resource.open("rb") as fh:
        img = Image.open(fh)
        img.load()
    return img


class Tray:
    def __init__(
        self,
        *,
        on_toggle: Callable[[], None],
        on_set_language: Callable[[str], None],
        on_quit: Callable[[], None],
        get_language: Callable[[], str],
    ):
        self._on_toggle = on_toggle
        self._on_set_language = on_set_language
        self._on_quit = on_quit
        self._get_language = get_language
        self._icon = pystray.Icon(
            "voicetype",
            icon=_load_icon(State.IDLE),
            title="VoiceType",
            menu=self._build_menu(),
        )

    def _lang_item(self, label: str, value: str) -> pystray.MenuItem:
        return pystray.MenuItem(
            label,
            lambda: self._on_set_language(value),
            checked=lambda item: self._get_language() == value,
            radio=True,
        )

    def _build_menu(self) -> pystray.Menu:
        return pystray.Menu(
            pystray.MenuItem("Toggle dictation", lambda: self._on_toggle()),
            pystray.Menu.SEPARATOR,
            self._lang_item("Auto", "auto"),
            self._lang_item("English", "en"),
            self._lang_item("Español", "es"),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", lambda: self._on_quit()),
        )

    def set_state(self, state: State) -> None:
        self._icon.icon = _load_icon(state)

    def run_detached(self) -> None:
        self._icon.run_detached()

    def stop(self) -> None:
        self._icon.stop()
