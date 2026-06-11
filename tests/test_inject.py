from unittest.mock import patch

import pytest

from voicetype.inject import YdotoolInjector, InjectionError


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
