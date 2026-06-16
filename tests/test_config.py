from pathlib import Path

from voicetype.config import Config, load_config, save_config


def test_defaults():
    cfg = Config()
    assert cfg.model == "medium"
    assert cfg.device == "auto"
    assert cfg.compute_type == "int8_float16"
    assert cfg.language_default == "auto"
    assert cfg.mic_device is None
    assert cfg.silence_seconds == 2.0
    assert cfg.min_speech_seconds == 0.3
    assert cfg.inject_method == "clipboard"
    assert cfg.paste_shortcut == "ctrl+shift+v"


def test_load_missing_file_returns_defaults_and_writes(tmp_path):
    path = tmp_path / "config.toml"
    cfg = load_config(path)
    assert cfg == Config()
    assert path.exists()  # defaults written on first run


def test_save_then_load_roundtrip(tmp_path):
    path = tmp_path / "config.toml"
    cfg = Config(model="small", language_default="es", silence_seconds=1.5)
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded == cfg


def test_load_ignores_unknown_keys(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('model = "small"\nbogus_key = 42\n')
    cfg = load_config(path)
    assert cfg.model == "small"
    assert cfg.silence_seconds == 2.0  # default preserved
