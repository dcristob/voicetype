# VoiceType Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Wayland voice-dictation utility: a hotkey opens the mic, faster-whisper transcribes ES/EN locally after 2 s of silence, and the text is typed into the focused app, with a tray icon showing state.

**Architecture:** A long-running `voicetyped` daemon holds the warm Whisper model, owns mic capture, runs a state machine, draws an SNI tray icon, and listens on a Unix socket. A thin `voicetype` CLI (bound to a DE hotkey) sends commands over that socket. Pure logic (config, state, IPC, VAD timing, audio buffer, ASR text-join, inject command, controller orchestration) is unit-tested with TDD; hardware/DE glue (silero model, sounddevice stream, pystray tray, systemd) is implemented with manual verification.

**Tech Stack:** Python 3.14, `uv`, faster-whisper (CTranslate2), silero-vad, sounddevice (PipeWire), ydotool, pystray, dbus, pytest.

---

## File Structure

```
voicetype/
  pyproject.toml
  src/voicetype/
    __init__.py
    config.py      # Config dataclass, TOML load/save
    state.py       # State enum + StateMachine
    ipc.py         # socket path + JSON line encode/decode
    vad.py         # SilenceTracker (pure timing) + SileroVad (model wrapper)
    audio.py       # AudioBuffer (pure) + Recorder (sounddevice)
    asr.py         # Transcriber (faster-whisper wrapper)
    inject.py      # YdotoolInjector
    controller.py  # Controller: wires audio→vad→asr→inject, drives state
    notify.py      # desktop notifications
    daemon.py      # asyncio socket server + tray + main()
    tray.py        # SNI tray icon + menu (pystray)
    cli.py         # client commands
  tests/
    test_config.py
    test_state.py
    test_ipc.py
    test_vad.py
    test_audio.py
    test_asr.py
    test_inject.py
    test_controller.py
    test_cli.py
  data/icons/        # idle.png listening.png transcribing.png
  systemd/voicetype.service
  packaging/install.sh
  packaging/70-uinput.rules
  README.md
```

---

## Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/voicetype/__init__.py`
- Create: `tests/__init__.py`
- Create: `.gitignore`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "voicetype"
version = "0.1.0"
description = "Local voice dictation for Wayland (COSMIC/Budgie/GNOME)"
requires-python = ">=3.11"
dependencies = [
    "faster-whisper>=1.0",
    "sounddevice>=0.4",
    "numpy>=1.24",
    "silero-vad>=5.1",
    "pystray>=0.19",
    "pillow>=10.0",
    "tomli-w>=1.0",
]

[project.scripts]
voicetype = "voicetype.cli:main"
voicetyped = "voicetype.daemon:main"

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/voicetype"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package and test init files**

`src/voicetype/__init__.py`:
```python
"""VoiceType: local voice dictation for Wayland."""

__version__ = "0.1.0"
```

`tests/__init__.py`:
```python
```

- [ ] **Step 3: Create `.gitignore`**

```
__pycache__/
*.pyc
.venv/
.pytest_cache/
dist/
*.egg-info/
```

- [ ] **Step 4: Create the environment and verify install**

Run: `uv venv && uv pip install -e ".[dev]"`
Expected: installs without error. (faster-whisper/silero pull torch; this is a large download — expected.)

- [ ] **Step 5: Verify pytest runs (collects zero tests)**

Run: `uv run pytest -q`
Expected: "no tests ran" (exit 5) or similar — confirms pytest + pythonpath work.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: scaffold voicetype project"
```

---

## Task 2: Config (TOML load/save)

**Files:**
- Create: `src/voicetype/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_config.py`:
```python
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
    assert cfg.inject_method == "ydotool"


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voicetype.config'`

- [ ] **Step 3: Write the implementation**

`src/voicetype/config.py`:
```python
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
```

Note: `mic_device=None` is dropped on save (TOML has no null); it reappears as the `None` default on load.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/voicetype/config.py tests/test_config.py
git commit -m "feat: config load/save with TOML"
```

---

## Task 3: State machine

**Files:**
- Create: `src/voicetype/state.py`
- Test: `tests/test_state.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_state.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voicetype.state'`

- [ ] **Step 3: Write the implementation**

`src/voicetype/state.py`:
```python
"""Activation state machine."""

from __future__ import annotations

from enum import Enum
from typing import Callable


class State(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    ERROR = "error"


class InvalidTransition(Exception):
    pass


_ALLOWED: dict[State, set[State]] = {
    State.IDLE: {State.LISTENING, State.ERROR},
    State.LISTENING: {State.TRANSCRIBING, State.IDLE, State.ERROR},
    State.TRANSCRIBING: {State.IDLE, State.ERROR},
    State.ERROR: {State.IDLE},
}


class StateMachine:
    def __init__(self, on_change: Callable[[State, State], None] | None = None):
        self._state = State.IDLE
        self._on_change = on_change

    @property
    def state(self) -> State:
        return self._state

    def transition(self, target: State) -> None:
        if target not in _ALLOWED[self._state]:
            raise InvalidTransition(f"{self._state.value} -> {target.value}")
        old = self._state
        self._state = target
        if self._on_change is not None:
            self._on_change(old, target)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_state.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/voicetype/state.py tests/test_state.py
git commit -m "feat: activation state machine"
```

---

## Task 4: IPC protocol

**Files:**
- Create: `src/voicetype/ipc.py`
- Test: `tests/test_ipc.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_ipc.py`:
```python
from pathlib import Path

from voicetype import ipc


def test_socket_path_uses_runtime_dir(monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
    assert ipc.socket_path() == Path("/run/user/1000/voicetype.sock")


def test_socket_path_falls_back_to_tmp(monkeypatch):
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    assert ipc.socket_path() == Path("/tmp/voicetype.sock")


def test_encode_decode_roundtrip():
    msg = {"cmd": "lang", "value": "es"}
    line = ipc.encode(msg)
    assert line.endswith(b"\n")
    assert ipc.decode(line) == msg


def test_decode_strips_newline():
    assert ipc.decode(b'{"cmd": "toggle"}\n') == {"cmd": "toggle"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ipc.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voicetype.ipc'`

- [ ] **Step 3: Write the implementation**

`src/voicetype/ipc.py`:
```python
"""Unix-socket control protocol shared by daemon and CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path


def socket_path() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
    return Path(runtime) / "voicetype.sock"


def encode(msg: dict) -> bytes:
    return (json.dumps(msg) + "\n").encode("utf-8")


def decode(line: bytes) -> dict:
    return json.loads(line.decode("utf-8"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ipc.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/voicetype/ipc.py tests/test_ipc.py
git commit -m "feat: IPC socket protocol"
```

---

## Task 5: VAD silence timing (pure)

**Files:**
- Create: `src/voicetype/vad.py`
- Test: `tests/test_vad.py`

The `SilenceTracker` is pure timing logic: fed per-frame speech probabilities, it reports when an utterance has ended (speech happened, then `silence_seconds` of continuous silence). The silero model wrapper is added in Task 6.

- [ ] **Step 1: Write the failing tests**

`tests/test_vad.py`:
```python
from voicetype.vad import SilenceTracker

FRAME = 0.032  # 32 ms frames (512 samples @ 16 kHz)


def make_tracker():
    return SilenceTracker(
        silence_seconds=2.0,
        frame_seconds=FRAME,
        min_speech_seconds=0.1,
        threshold=0.5,
    )


def test_no_end_before_any_speech():
    tr = make_tracker()
    # 100 frames of silence with no prior speech -> never ends
    assert not any(tr.update(0.0) for _ in range(100))


def test_ends_after_speech_then_silence():
    tr = make_tracker()
    # ~0.2 s of speech (well past min_speech)
    for _ in range(7):
        assert tr.update(0.9) is False
    # silence: needs >= 2.0 s = 63 frames; should fire on the frame that crosses 2 s
    ended = [tr.update(0.0) for _ in range(70)]
    assert True in ended
    first_end = ended.index(True)
    # 2.0 s / 0.032 s = 62.5 -> ends on the 63rd silent frame (index 62)
    assert first_end == 62


def test_silence_resets_on_renewed_speech():
    tr = make_tracker()
    for _ in range(7):
        tr.update(0.9)
    for _ in range(30):  # ~0.96 s silence, not enough
        assert tr.update(0.0) is False
    tr.update(0.9)  # speech again resets the silence counter
    for _ in range(30):
        assert tr.update(0.0) is False  # still not enough after reset


def test_reset_clears_state():
    tr = make_tracker()
    for _ in range(7):
        tr.update(0.9)
    tr.reset()
    assert not any(tr.update(0.0) for _ in range(100))  # no prior speech after reset
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_vad.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voicetype.vad'`

- [ ] **Step 3: Write the implementation**

`src/voicetype/vad.py`:
```python
"""Voice-activity detection: pure silence timing + silero model wrapper."""

from __future__ import annotations


class SilenceTracker:
    """Reports end-of-utterance from a stream of per-frame speech probabilities.

    An utterance "ends" once total speech has exceeded ``min_speech_seconds``
    and ``silence_seconds`` of continuous silence has followed.
    """

    def __init__(
        self,
        silence_seconds: float,
        frame_seconds: float,
        min_speech_seconds: float = 0.3,
        threshold: float = 0.5,
    ):
        self.silence_seconds = silence_seconds
        self.frame_seconds = frame_seconds
        self.min_speech_seconds = min_speech_seconds
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self._speech_total = 0.0
        self._silence_run = 0.0
        self._spoke = False

    def update(self, speech_prob: float) -> bool:
        is_speech = speech_prob >= self.threshold
        if is_speech:
            self._speech_total += self.frame_seconds
            self._silence_run = 0.0
            if self._speech_total >= self.min_speech_seconds:
                self._spoke = True
            return False
        # silence frame
        if not self._spoke:
            return False
        self._silence_run += self.frame_seconds
        return self._silence_run >= self.silence_seconds
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_vad.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/voicetype/vad.py tests/test_vad.py
git commit -m "feat: VAD silence tracker"
```

---

## Task 6: Silero VAD model wrapper

**Files:**
- Modify: `src/voicetype/vad.py`

This wraps the silero-vad model to turn a 512-sample float32 frame into a speech probability. It needs the model weights (downloaded by the library), so it is verified manually rather than unit-tested.

- [ ] **Step 1: Add the wrapper to `src/voicetype/vad.py`**

Append to `src/voicetype/vad.py`:
```python
# Silero operates on fixed 512-sample frames at 16 kHz.
SILERO_FRAME_SAMPLES = 512
SAMPLE_RATE = 16000


class SileroVad:
    """Wraps silero-vad to score a 512-sample frame in [0, 1]."""

    def __init__(self) -> None:
        import torch  # noqa: F401  (silero pulls torch)
        from silero_vad import load_silero_vad

        self._model = load_silero_vad()

    def prob(self, frame) -> float:
        """frame: float32 numpy array of length 512, mono, 16 kHz."""
        import torch

        tensor = torch.from_numpy(frame)
        return float(self._model(tensor, SAMPLE_RATE).item())
```

- [ ] **Step 2: Manually verify the model loads and scores**

Run:
```bash
uv run python -c "
import numpy as np
from voicetype.vad import SileroVad, SILERO_FRAME_SAMPLES
v = SileroVad()
silence = np.zeros(SILERO_FRAME_SAMPLES, dtype=np.float32)
print('silence prob:', v.prob(silence))
loud = (np.random.randn(SILERO_FRAME_SAMPLES) * 0.3).astype(np.float32)
print('noise prob:', v.prob(loud))
"
```
Expected: prints two probabilities in [0, 1]; the silence one should be very low (near 0).

- [ ] **Step 3: Commit**

```bash
git add src/voicetype/vad.py
git commit -m "feat: silero VAD model wrapper"
```

---

## Task 7: Audio buffer (pure)

**Files:**
- Create: `src/voicetype/audio.py`
- Test: `tests/test_audio.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_audio.py`:
```python
import numpy as np

from voicetype.audio import AudioBuffer


def test_empty_returns_zero_length():
    buf = AudioBuffer()
    out = buf.get_audio()
    assert out.dtype == np.float32
    assert len(out) == 0


def test_accumulates_frames_in_order():
    buf = AudioBuffer()
    buf.add(np.array([1, 2], dtype=np.float32))
    buf.add(np.array([3, 4], dtype=np.float32))
    out = buf.get_audio()
    assert out.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_clear_empties_buffer():
    buf = AudioBuffer()
    buf.add(np.array([1, 2], dtype=np.float32))
    buf.clear()
    assert len(buf.get_audio()) == 0


def test_duration_seconds():
    buf = AudioBuffer(sample_rate=16000)
    buf.add(np.zeros(16000, dtype=np.float32))
    assert buf.duration_seconds() == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_audio.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voicetype.audio'`

- [ ] **Step 3: Write the implementation**

`src/voicetype/audio.py`:
```python
"""Audio capture: a pure accumulation buffer plus a sounddevice recorder."""

from __future__ import annotations

from typing import Callable

import numpy as np

SAMPLE_RATE = 16000


class AudioBuffer:
    """Accumulates mono float32 frames into one contiguous array."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._frames: list[np.ndarray] = []

    def add(self, frame: np.ndarray) -> None:
        self._frames.append(frame.astype(np.float32, copy=False))

    def get_audio(self) -> np.ndarray:
        if not self._frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._frames)

    def clear(self) -> None:
        self._frames.clear()

    def duration_seconds(self) -> float:
        return sum(len(f) for f in self._frames) / self.sample_rate
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_audio.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/voicetype/audio.py tests/test_audio.py
git commit -m "feat: audio accumulation buffer"
```

---

## Task 8: Recorder (sounddevice capture)

**Files:**
- Modify: `src/voicetype/audio.py`

The `Recorder` opens a PipeWire/Pulse input stream via sounddevice and delivers fixed 512-sample frames to a callback. It needs a real mic, so it is verified manually.

- [ ] **Step 1: Add the Recorder to `src/voicetype/audio.py`**

Append to `src/voicetype/audio.py`:
```python
SILERO_FRAME_SAMPLES = 512


class Recorder:
    """Captures mono 16 kHz audio, delivering 512-sample float32 frames."""

    def __init__(self, mic_device: str | None = None):
        self.mic_device = mic_device
        self._stream = None
        self._buffer = AudioBuffer()
        self._on_frame: Callable[[np.ndarray], None] | None = None

    def start(self, on_frame: Callable[[np.ndarray], None]) -> None:
        import sounddevice as sd

        self._on_frame = on_frame
        self._buffer.clear()

        def callback(indata, frames, time_info, status):  # noqa: ARG001
            frame = indata[:, 0].copy()
            self._buffer.add(frame)
            if self._on_frame is not None:
                self._on_frame(frame)

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=SILERO_FRAME_SAMPLES,
            device=self.mic_device,
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        return self._buffer.get_audio()
```

- [ ] **Step 2: Manually verify capture works**

Run (speak during the 3-second window):
```bash
uv run python -c "
import time, numpy as np
from voicetype.audio import Recorder
r = Recorder()
r.start(lambda f: None)
time.sleep(3)
audio = r.stop()
print('captured seconds:', len(audio)/16000, 'peak:', float(np.abs(audio).max()))
"
```
Expected: ~3 captured seconds and a non-trivial peak (e.g. > 0.01) when you spoke.

- [ ] **Step 3: Commit**

```bash
git add src/voicetype/audio.py
git commit -m "feat: sounddevice recorder"
```

---

## Task 9: Text injection (ydotool)

**Files:**
- Create: `src/voicetype/inject.py`
- Test: `tests/test_inject.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_inject.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_inject.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voicetype.inject'`

- [ ] **Step 3: Write the implementation**

`src/voicetype/inject.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_inject.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/voicetype/inject.py tests/test_inject.py
git commit -m "feat: ydotool text injection"
```

---

## Task 10: ASR (faster-whisper wrapper)

**Files:**
- Create: `src/voicetype/asr.py`
- Test: `tests/test_asr.py`

The model is lazy-loaded so tests can exercise the language-mapping and segment-joining logic with a mocked `WhisperModel`. Device fallback (CUDA→CPU) is also unit-tested via mock.

- [ ] **Step 1: Write the failing tests**

`tests/test_asr.py`:
```python
from unittest.mock import MagicMock, patch

import numpy as np

from voicetype.asr import Transcriber


class _Seg:
    def __init__(self, text):
        self.text = text


def test_auto_language_passes_none_to_model():
    tr = Transcriber(model="small", device="cpu", compute_type="int8")
    fake = MagicMock()
    fake.transcribe.return_value = ([_Seg(" hello")], None)
    with patch("voicetype.asr.WhisperModel", return_value=fake):
        text = tr.transcribe(np.zeros(16000, dtype=np.float32), language="auto")
    assert text == "hello"
    assert fake.transcribe.call_args.kwargs["language"] is None


def test_forced_language_passed_through():
    tr = Transcriber(model="small", device="cpu", compute_type="int8")
    fake = MagicMock()
    fake.transcribe.return_value = ([_Seg("hola")], None)
    with patch("voicetype.asr.WhisperModel", return_value=fake):
        tr.transcribe(np.zeros(16000, dtype=np.float32), language="es")
    assert fake.transcribe.call_args.kwargs["language"] == "es"


def test_segments_joined_and_stripped():
    tr = Transcriber(model="small", device="cpu", compute_type="int8")
    fake = MagicMock()
    fake.transcribe.return_value = ([_Seg(" one"), _Seg(" two")], None)
    with patch("voicetype.asr.WhisperModel", return_value=fake):
        text = tr.transcribe(np.zeros(16000, dtype=np.float32), language="auto")
    assert text == "one two"


def test_cuda_oom_falls_back_to_cpu():
    tr = Transcriber(model="small", device="cuda", compute_type="int8_float16")
    fake = MagicMock()
    fake.transcribe.return_value = ([_Seg("ok")], None)

    calls = []

    def construct(model, device, compute_type):
        calls.append(device)
        if device == "cuda":
            raise RuntimeError("CUDA out of memory")
        return fake

    with patch("voicetype.asr.WhisperModel", side_effect=construct):
        text = tr.transcribe(np.zeros(16000, dtype=np.float32), language="auto")
    assert text == "ok"
    assert calls == ["cuda", "cpu"]  # tried CUDA, fell back to CPU
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_asr.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voicetype.asr'`

- [ ] **Step 3: Write the implementation**

`src/voicetype/asr.py`:
```python
"""faster-whisper transcription wrapper with a warm, lazily-loaded model."""

from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel


class Transcriber:
    def __init__(self, model: str, device: str, compute_type: str):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self._model: WhisperModel | None = None

    def _load(self) -> WhisperModel:
        device = self.device if self.device != "auto" else "cuda"
        try:
            return WhisperModel(self.model_name, device, self.compute_type)
        except RuntimeError as exc:
            if "cuda" in str(exc).lower() or "out of memory" in str(exc).lower():
                # Fall back to CPU int8.
                self.device = "cpu"
                self.compute_type = "int8"
                return WhisperModel(self.model_name, "cpu", "int8")
            raise

    def ensure_loaded(self) -> None:
        if self._model is None:
            self._model = self._load()

    def transcribe(self, audio: np.ndarray, language: str) -> str:
        self.ensure_loaded()
        lang = None if language == "auto" else language
        segments, _info = self._model.transcribe(audio, language=lang)
        return " ".join(seg.text.strip() for seg in segments).strip()
```

Note on the fallback test: `_load` constructs CUDA, which raises; the `except` swaps to CPU and constructs again — matching the `["cuda", "cpu"]` call sequence. With `device="auto"` the first attempt also uses `"cuda"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_asr.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/voicetype/asr.py tests/test_asr.py
git commit -m "feat: faster-whisper transcriber with CPU fallback"
```

---

## Task 11: Controller orchestration

**Files:**
- Create: `src/voicetype/controller.py`
- Test: `tests/test_controller.py`

The Controller wires recorder → VAD → ASR → inject and drives the state machine. Collaborators are injected so it can be tested with fakes. It exposes:
- `toggle()` — IDLE→start listening; LISTENING→finish now.
- `feed_frame(frame)` — score a frame; returns True if end-of-utterance (caller then schedules `finish`).
- `finish()` — stop recording, transcribe, inject, return to IDLE.
- `set_language(lang)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_controller.py`:
```python
import numpy as np

from voicetype.config import Config
from voicetype.controller import Controller
from voicetype.state import State


class FakeRecorder:
    def __init__(self):
        self.started = False
        self.audio = np.ones(16000, dtype=np.float32)

    def start(self, on_frame):
        self.started = True
        self._on_frame = on_frame

    def stop(self):
        self.started = False
        return self.audio


class FakeVad:
    def prob(self, frame):
        return 0.0


class FakeTranscriber:
    def __init__(self, text="hello world"):
        self.text = text
        self.calls = []

    def ensure_loaded(self):
        pass

    def transcribe(self, audio, language):
        self.calls.append(language)
        return self.text


class FakeInjector:
    def __init__(self):
        self.typed = []

    def type_text(self, text):
        self.typed.append(text)


def make_controller(transcriber=None):
    return Controller(
        config=Config(silence_seconds=2.0, min_speech_seconds=0.1),
        recorder=FakeRecorder(),
        vad=FakeVad(),
        transcriber=transcriber or FakeTranscriber(),
        injector=FakeInjector(),
    )


def test_toggle_from_idle_starts_listening():
    c = make_controller()
    c.toggle()
    assert c.state is State.LISTENING
    assert c.recorder.started is True


def test_full_cycle_types_text_and_returns_idle():
    inj = FakeInjector()
    tr = FakeTranscriber("hola")
    c = Controller(
        config=Config(min_speech_seconds=0.1),
        recorder=FakeRecorder(),
        vad=FakeVad(),
        transcriber=tr,
        injector=inj,
    )
    c.toggle()        # start listening
    c.finish()        # stop + transcribe + inject
    assert inj.typed == ["hola"]
    assert c.state is State.IDLE


def test_feed_frame_detects_end_of_utterance():
    c = make_controller()
    c.toggle()
    frame = np.zeros(512, dtype=np.float32)
    # Mark that speech happened so silence can trigger an end.
    c._tracker._spoke = True
    results = [c.feed_frame(frame) for _ in range(70)]
    assert True in results


def test_auto_language_by_default():
    tr = FakeTranscriber()
    c = make_controller(transcriber=tr)
    c.toggle()
    c.finish()
    assert tr.calls == ["auto"]


def test_set_language_forces_language():
    tr = FakeTranscriber()
    c = make_controller(transcriber=tr)
    c.set_language("es")
    c.toggle()
    c.finish()
    assert tr.calls == ["es"]


def test_empty_transcription_skips_injection():
    inj = FakeInjector()
    c = Controller(
        config=Config(min_speech_seconds=0.1),
        recorder=FakeRecorder(),
        vad=FakeVad(),
        transcriber=FakeTranscriber(""),
        injector=inj,
    )
    c.toggle()
    c.finish()
    assert inj.typed == []
    assert c.state is State.IDLE


def test_toggle_while_listening_finishes():
    inj = FakeInjector()
    c = Controller(
        config=Config(min_speech_seconds=0.1),
        recorder=FakeRecorder(),
        vad=FakeVad(),
        transcriber=FakeTranscriber("done"),
        injector=inj,
    )
    c.toggle()  # listening
    c.toggle()  # should finish
    assert inj.typed == ["done"]
    assert c.state is State.IDLE
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_controller.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voicetype.controller'`

- [ ] **Step 3: Write the implementation**

`src/voicetype/controller.py`:
```python
"""Orchestrates capture → VAD → transcription → injection."""

from __future__ import annotations

from typing import Callable

from .config import Config
from .state import State, StateMachine
from .vad import SilenceTracker

FRAME_SECONDS = 512 / 16000  # 0.032 s


class Controller:
    def __init__(
        self,
        *,
        config: Config,
        recorder,
        vad,
        transcriber,
        injector,
        notifier: Callable[[str, str], None] | None = None,
        on_state_change: Callable[[State, State], None] | None = None,
    ):
        self.config = config
        self.recorder = recorder
        self.vad = vad
        self.transcriber = transcriber
        self.injector = injector
        self.notifier = notifier
        self.language = config.language_default
        self._sm = StateMachine(on_change=on_state_change)
        self._tracker = SilenceTracker(
            silence_seconds=config.silence_seconds,
            frame_seconds=FRAME_SECONDS,
            min_speech_seconds=config.min_speech_seconds,
        )

    @property
    def state(self) -> State:
        return self._sm.state

    def set_language(self, lang: str) -> None:
        self.language = lang

    def toggle(self) -> None:
        if self._sm.state is State.IDLE:
            self._start()
        elif self._sm.state is State.LISTENING:
            self.finish()

    def _start(self) -> None:
        self._tracker.reset()
        self._sm.transition(State.LISTENING)
        self.recorder.start(self._noop)

    def _noop(self, frame) -> None:
        # The daemon feeds frames via feed_frame(); the recorder callback only
        # needs to accumulate audio, which it does internally.
        pass

    def feed_frame(self, frame) -> bool:
        if self._sm.state is not State.LISTENING:
            return False
        return self._tracker.update(self.vad.prob(frame))

    def finish(self) -> None:
        if self._sm.state is not State.LISTENING:
            return
        self._sm.transition(State.TRANSCRIBING)
        audio = self.recorder.stop()
        try:
            text = self.transcriber.transcribe(audio, self.language)
        except Exception as exc:  # transcription failure -> notify, reset
            self._sm.transition(State.ERROR)
            self._notify("Transcription failed", str(exc))
            self._sm.transition(State.IDLE)
            return
        if text:
            try:
                self.injector.type_text(text)
            except Exception as exc:
                self._sm.transition(State.ERROR)
                self._notify("Injection failed", str(exc))
                self._sm.transition(State.IDLE)
                return
        self._sm.transition(State.IDLE)

    def _notify(self, title: str, body: str) -> None:
        if self.notifier is not None:
            self.notifier(title, body)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_controller.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/voicetype/controller.py tests/test_controller.py
git commit -m "feat: controller orchestration"
```

---

## Task 12: Desktop notifications

**Files:**
- Create: `src/voicetype/notify.py`

A thin wrapper over `notify-send`, verified manually.

- [ ] **Step 1: Write the implementation**

`src/voicetype/notify.py`:
```python
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
```

- [ ] **Step 2: Manually verify**

Run: `uv run python -c "from voicetype.notify import notify; notify('VoiceType', 'hello')"`
Expected: a desktop notification appears.

- [ ] **Step 3: Commit**

```bash
git add src/voicetype/notify.py
git commit -m "feat: desktop notifications"
```

---

## Task 13: Tray icon

**Files:**
- Create: `src/voicetype/tray.py`
- Create: `data/icons/idle.png`, `data/icons/listening.png`, `data/icons/transcribing.png`

The tray uses pystray (AppIndicator/SNI backend). It needs a running DE tray, so it is verified manually. Icons are generated programmatically to avoid binary assets in git.

- [ ] **Step 1: Generate the three state icons**

Run:
```bash
uv run python -c "
from PIL import Image
import os
os.makedirs('data/icons', exist_ok=True)
colors = {'idle': (130,130,130), 'listening': (220,40,40), 'transcribing': (235,170,30)}
for name, rgb in colors.items():
    img = Image.new('RGBA', (64, 64), (0,0,0,0))
    px = img.load()
    cx, cy, r = 32, 32, 26
    for y in range(64):
        for x in range(64):
            if (x-cx)**2 + (y-cy)**2 <= r*r:
                px[x, y] = rgb + (255,)
    img.save(f'data/icons/{name}.png')
    print('wrote', name)
"
```
Expected: writes three PNGs (grey/red/amber filled circles).

- [ ] **Step 2: Write the tray implementation**

`src/voicetype/tray.py`:
```python
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
    path = files("voicetype").joinpath("..", "..", "data", "icons", _ICONS[state])
    return Image.open(str(path))


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
```

- [ ] **Step 3: Manually verify the tray appears**

Run: `uv run python -c "
import time
from voicetype.tray import Tray
t = Tray(on_toggle=lambda: print('toggle'), on_set_language=lambda l: print('lang', l), on_quit=lambda: print('quit'), get_language=lambda: 'auto')
t.run_detached()
time.sleep(15)
t.stop()
"`
Expected: a grey circle appears in the tray for 15 s; clicking the menu prints toggle/lang/quit. (On GNOME you must have the AppIndicator extension enabled.)

- [ ] **Step 4: Commit**

```bash
git add src/voicetype/tray.py data/icons
git commit -m "feat: tray icon with state and language menu"
```

---

## Task 14: Daemon (socket server + glue + main)

**Files:**
- Create: `src/voicetype/daemon.py`

The daemon ties Controller, Tray, Recorder frame-feeding, and the asyncio socket server together. End-of-utterance from the audio thread is handed to the event loop via `call_soon_threadsafe`. Verified by an integration run.

- [ ] **Step 1: Write the daemon**

`src/voicetype/daemon.py`:
```python
"""voicetyped: socket server + tray + controller wiring."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from . import ipc
from .audio import Recorder
from .asr import Transcriber
from .config import load_config
from .controller import Controller
from .inject import YdotoolInjector
from .notify import notify
from .state import State
from .tray import Tray
from .vad import SileroVad


class Daemon:
    def __init__(self):
        self.config = load_config()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.recorder = Recorder(self.config.mic_device)
        self.controller = Controller(
            config=self.config,
            recorder=self.recorder,
            vad=SileroVad(),
            transcriber=Transcriber(
                self.config.model, self.config.device, self.config.compute_type
            ),
            injector=YdotoolInjector(),
            notifier=notify,
            on_state_change=self._on_state_change,
        )
        self.tray = Tray(
            on_toggle=self._toggle_from_tray,
            on_set_language=self.controller.set_language,
            on_quit=self._quit_from_tray,
            get_language=lambda: self.controller.language,
        )
        self._stopping = asyncio.Event()

    # --- state/tray ---
    def _on_state_change(self, old: State, new: State) -> None:
        self.tray.set_state(new)

    # --- audio frame feeding (runs in the sounddevice thread) ---
    def _on_frame(self, frame) -> None:
        if self.controller.feed_frame(frame) and self._loop is not None:
            self._loop.call_soon_threadsafe(self.controller.finish)

    def _start_listening(self) -> None:
        # Re-route the recorder callback through the daemon so frames hit VAD.
        self.controller._start()  # transitions + starts recorder
        # Replace the no-op callback with our frame feeder.
        self.recorder._on_frame = self._on_frame

    def toggle(self) -> None:
        if self.controller.state is State.IDLE:
            self._start_listening()
        elif self.controller.state is State.LISTENING:
            self.controller.finish()

    def _toggle_from_tray(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self.toggle)

    def _quit_from_tray(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._stopping.set)

    # --- socket server ---
    async def _handle(self, reader, writer):
        data = await reader.readline()
        if not data:
            writer.close()
            return
        msg = ipc.decode(data)
        cmd = msg.get("cmd")
        if cmd == "toggle":
            self.toggle()
            reply = {"ok": True, "state": self.controller.state.value}
        elif cmd == "stop":
            if self.controller.state is State.LISTENING:
                self.controller.finish()
            reply = {"ok": True, "state": self.controller.state.value}
        elif cmd == "lang":
            self.controller.set_language(msg.get("value", "auto"))
            reply = {"ok": True, "language": self.controller.language}
        elif cmd == "status":
            reply = {
                "ok": True,
                "state": self.controller.state.value,
                "language": self.controller.language,
            }
        else:
            reply = {"ok": False, "error": f"unknown command: {cmd}"}
        writer.write(ipc.encode(reply))
        await writer.drain()
        writer.close()

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        path = ipc.socket_path()
        if path.exists():
            path.unlink()
        server = await asyncio.start_unix_server(self._handle, path=str(path))
        os.chmod(path, 0o600)
        self.tray.run_detached()
        async with server:
            await self._stopping.wait()
        self.tray.stop()
        if path.exists():
            path.unlink()


def main() -> None:
    daemon = Daemon()
    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
```

Note: `Controller._start` sets the recorder callback to `_noop`; the daemon overrides `recorder._on_frame` immediately after so VAD frames are fed. This keeps the Controller unit-testable (no daemon needed) while the daemon supplies real frame routing.

- [ ] **Step 2: Integration-verify the daemon end-to-end**

Prerequisite: `ydotoold` running and uinput permission set up (Task 15). Then run the daemon in one terminal:
```bash
uv run voicetyped
```
In a second terminal, focus a text editor, then:
```bash
uv run voicetype toggle   # speak a sentence, pause 2 s
```
Expected: tray icon goes red while you speak, amber while transcribing, then your words are typed into the focused editor and the icon returns to grey.

- [ ] **Step 3: Commit**

```bash
git add src/voicetype/daemon.py
git commit -m "feat: voicetyped daemon (socket + tray + glue)"
```

---

## Task 15: CLI client

**Files:**
- Create: `src/voicetype/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py`:
```python
import asyncio
import json

import pytest

from voicetype import ipc
from voicetype.cli import send_command


@pytest.fixture
def fake_server(tmp_path, monkeypatch):
    sock = tmp_path / "voicetype.sock"
    monkeypatch.setattr(ipc, "socket_path", lambda: sock)
    received = {}

    async def run():
        async def handle(reader, writer):
            line = await reader.readline()
            received["msg"] = ipc.decode(line)
            writer.write(ipc.encode({"ok": True, "state": "listening"}))
            await writer.drain()
            writer.close()

        server = await asyncio.start_unix_server(handle, path=str(sock))
        async with server:
            await asyncio.sleep(0.5)

    return run, received


def test_send_command_toggle(fake_server):
    run, received = fake_server

    async def scenario():
        task = asyncio.create_task(run())
        await asyncio.sleep(0.05)
        reply = await send_command({"cmd": "toggle"})
        task.cancel()
        return reply

    reply = asyncio.run(scenario())
    assert reply["ok"] is True
    assert received["msg"] == {"cmd": "toggle"}


def test_send_command_daemon_not_running(tmp_path, monkeypatch):
    monkeypatch.setattr(ipc, "socket_path", lambda: tmp_path / "nope.sock")
    with pytest.raises(ConnectionError):
        asyncio.run(send_command({"cmd": "toggle"}))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voicetype.cli'`

- [ ] **Step 3: Write the implementation**

`src/voicetype/cli.py`:
```python
"""voicetype: thin client that sends commands to the daemon."""

from __future__ import annotations

import argparse
import asyncio
import sys

from . import ipc


async def send_command(msg: dict) -> dict:
    path = ipc.socket_path()
    try:
        reader, writer = await asyncio.open_unix_connection(str(path))
    except (FileNotFoundError, ConnectionRefusedError) as exc:
        raise ConnectionError(
            f"voicetyped not running (no socket at {path})"
        ) from exc
    writer.write(ipc.encode(msg))
    await writer.drain()
    line = await reader.readline()
    writer.close()
    return ipc.decode(line)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voicetype")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("toggle", help="start/stop dictation")
    sub.add_parser("stop", help="stop dictation now")
    sub.add_parser("status", help="print daemon state")
    lang = sub.add_parser("lang", help="set language")
    lang.add_argument("value", choices=["auto", "en", "es"])
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    msg: dict = {"cmd": args.cmd}
    if args.cmd == "lang":
        msg["value"] = args.value
    try:
        reply = asyncio.run(send_command(msg))
    except ConnectionError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    print(reply)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/voicetype/cli.py tests/test_cli.py
git commit -m "feat: voicetype CLI client"
```

---

## Task 16: Packaging — systemd, ydotool/uinput, install

**Files:**
- Create: `systemd/voicetype.service`
- Create: `packaging/70-uinput.rules`
- Create: `packaging/install.sh`

- [ ] **Step 1: Create the user systemd unit**

`systemd/voicetype.service`:
```ini
[Unit]
Description=VoiceType dictation daemon
After=graphical-session.target
PartOf=graphical-session.target

[Service]
Type=simple
ExecStart=%h/.local/bin/voicetyped
Restart=on-failure
RestartSec=2

[Install]
WantedBy=graphical-session.target
```

- [ ] **Step 2: Create the uinput udev rule**

`packaging/70-uinput.rules`:
```
KERNEL=="uinput", GROUP="input", MODE="0660", OPTIONS+="static_node=uinput"
```

- [ ] **Step 3: Write the install script**

`packaging/install.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail

echo "==> Installing voicetype (user)"
uv tool install --force . || pipx install --force .

echo "==> Installing systemd user unit"
mkdir -p "$HOME/.config/systemd/user"
cp systemd/voicetype.service "$HOME/.config/systemd/user/voicetype.service"
systemctl --user daemon-reload

echo "==> Configuring uinput permission for ydotool (needs sudo)"
sudo cp packaging/70-uinput.rules /etc/udev/rules.d/70-uinput.rules
sudo groupadd -f input
sudo usermod -aG input "$USER"
sudo udevadm control --reload-rules
sudo udevadm trigger /dev/uinput || true

echo "==> ydotool"
if ! command -v ydotool >/dev/null; then
  echo "Please install ydotool with your package manager (e.g. 'sudo pacman -S ydotool')."
fi

cat <<'EOF'

Next steps:
  1. Log out/in (so your 'input' group membership applies).
  2. Start ydotoold:   systemctl --user enable --now ydotool   (or run 'ydotoold &')
  3. Enable voicetype: systemctl --user enable --now voicetype
  4. Bind a hotkey in your DE keyboard settings to:  voicetype toggle
  5. GNOME only: enable the AppIndicator/Ayatana extension for the tray icon.
EOF
```

- [ ] **Step 4: Verify the script is syntactically valid**

Run: `bash -n packaging/install.sh && chmod +x packaging/install.sh && echo OK`
Expected: prints `OK`.

- [ ] **Step 5: Commit**

```bash
git add systemd packaging
git commit -m "feat: packaging (systemd unit, udev rule, install script)"
```

---

## Task 17: Documentation

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write the README**

`README.md`:
```markdown
# VoiceType

Local voice dictation for Wayland (COSMIC, Budgie, GNOME). Press a hotkey, speak,
and your words are typed into whatever app has focus. Transcription is fully local
via faster-whisper, with automatic Spanish/English detection.

## How it works

- `voicetyped` is a background daemon: it holds the Whisper model, captures the
  mic, shows a tray icon, and listens on a Unix socket.
- `voicetype toggle` (bound to a hotkey in your DE) tells the daemon to start
  listening. After 2 s of silence it transcribes and types the result.
- The tray icon is grey (idle), red (listening), amber (transcribing). Its menu
  switches language (Auto / English / Español) and quits.

## Requirements

- Wayland session on COSMIC, Budgie, or GNOME (GNOME needs the AppIndicator
  extension for the tray).
- PipeWire or PulseAudio, a microphone.
- `ydotool` + `ydotoold` (text injection).
- NVIDIA GPU optional; falls back to CPU.

## Install

```bash
./packaging/install.sh
```

Then follow the printed steps (log out/in for `input` group, start `ydotoold`,
enable the service, bind `voicetype toggle` to a key).

## Configuration

`~/.config/voicetype/config.toml` (created on first run):

| Key | Default | Meaning |
|---|---|---|
| `model` | `medium` | Whisper model (`small`/`medium`/...). |
| `device` | `auto` | `auto`/`cuda`/`cpu`. |
| `compute_type` | `int8_float16` | CTranslate2 compute type. |
| `language_default` | `auto` | `auto`/`en`/`es`. |
| `mic_device` | (system default) | sounddevice device name. |
| `silence_seconds` | `2.0` | Silence before transcription. |
| `min_speech_seconds` | `0.3` | Minimum speech to count as an utterance. |
| `inject_method` | `ydotool` | Text-injection backend. |

## Commands

- `voicetype toggle` — start/stop dictation
- `voicetype stop` — stop now
- `voicetype lang auto|en|es` — set language
- `voicetype status` — show daemon state

## Per-desktop hotkey binding

- **COSMIC:** Settings → Keyboard → Custom shortcuts → command `voicetype toggle`.
- **Budgie:** Budgie Control Center → Keyboard → custom shortcut → `voicetype toggle`.
- **GNOME:** Settings → Keyboard → Custom Shortcuts → `voicetype toggle`; enable
  the AppIndicator extension for the tray.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with install, config, per-DE hotkey notes"
```

---

## Final verification

- [ ] **Run the full unit suite**

Run: `uv run pytest -q`
Expected: all tests pass (config, state, ipc, vad, audio, asr, inject, controller, cli).

- [ ] **End-to-end smoke test**

With `ydotoold` running and the service enabled: focus a text field, run
`voicetype toggle`, speak a Spanish and an English sentence in separate
activations, and confirm each is typed correctly and the tray icon cycles
grey → red → amber → grey.

---

## Notes on remaining trade-offs

- **Model default (`medium` vs `small`):** benchmark both on the GTX 1060 during
  the end-to-end test; if `medium` latency is unpleasant, set `small` in the
  shipped default config.
- **Tray backend:** if pystray's icon-state updates or COSMIC support misbehave,
  replace `tray.py` internals with a hand-rolled `dbus-fast` StatusNotifierItem —
  the `Tray` interface (`set_state`, `run_detached`, `stop`, callbacks) stays the
  same, so no other module changes.
```
