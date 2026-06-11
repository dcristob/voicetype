"""Configuration: a dataclass persisted as TOML."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import tomli_w

CONFIG_PATH = Path.home() / ".config" / "voicetype" / "config.toml"


@dataclass
class Config:
    model: str = "medium"
    device: str = "auto"  # auto | cuda | cpu
    compute_type: str = "int8_float16"
    language_default: str = "auto"  # auto | en | es
    mic_device: str | None = None
    silence_seconds: float = 2.0
    min_speech_seconds: float = 0.3
    inject_method: str = "ydotool"


def load_config(path: Path = CONFIG_PATH) -> Config:
    if not path.exists():
        cfg = Config()
        save_config(cfg, path)
        return cfg
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    known = {f.name for f in fields(Config)}
    filtered = {k: v for k, v in data.items() if k in known}
    return Config(**filtered)


def save_config(cfg: Config, path: Path = CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {k: v for k, v in asdict(cfg).items() if v is not None}
    with path.open("wb") as fh:
        tomli_w.dump(data, fh)
