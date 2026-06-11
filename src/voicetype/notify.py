"""Desktop notifications via notify-send (best-effort)."""

from __future__ import annotations

import subprocess


def notify(title: str, body: str = "") -> None:
    try:
        subprocess.run(
            ["notify-send", "-a", "VoiceType", title, body],
            capture_output=True,
        )
    except FileNotFoundError:
        pass  # no notify-send; silently skip
