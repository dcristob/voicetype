"""Type transcribed text into the focused window via ydotool."""

from __future__ import annotations

import subprocess


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
