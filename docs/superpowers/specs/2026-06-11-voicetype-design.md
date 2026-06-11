# VoiceType — Design Spec

**Date:** 2026-06-11
**Status:** Approved (design phase)

## 1. Summary

VoiceType is a Linux voice-dictation utility. A keybind opens the microphone,
transcribes speech locally, and types the result into whatever application
currently has focus. A tray icon shows activation state and exposes a small
menu. It targets the **COSMIC**, **Budgie**, and **GNOME** desktop environments
on **Wayland**.

## 2. Goals

- Press a hotkey → mic opens → speak → after **2 s of silence** the speech is
  transcribed and typed into the focused app.
- Fully **local** speech-to-text (no cloud).
- Automatic **Spanish/English** detection, with a manual override.
- Tray icon that changes appearance by state, with a control menu.
- Work across COSMIC, Budgie, and GNOME on Wayland.
- Shareable / open-source: clean install path and packaging.

## 3. Non-goals (initial scope)

- No X11-specific code paths (Wayland is the target).
- No real-time streaming/partial transcription (transcribe once on stop).
- No Flatpak in the first cut (uinput access makes it hard; deferred).
- No languages beyond ES/EN configured by default (Whisper supports more, but
  ES/EN is what we tune and document).

## 4. Core decisions

| Area | Decision | Rationale |
|---|---|---|
| Runtime | **Python** (managed with `uv`) | Best local-ASR ecosystem; fast to iterate. |
| ASR engine | **faster-whisper** (CTranslate2) | Native ES+EN, automatic language detection, runs on GTX 1060 or CPU. |
| Model | **medium**, configurable | Best ES accuracy that fits 3 GB VRAM at `int8_float16`. |
| Text injection | **ydotool** (type) | Works across COSMIC/Budgie/GNOME via the uinput kernel device. |
| Activation | **Toggle + auto-stop** | Press once to start, auto-stops on 2 s silence (or press again). |
| Language UX | **Auto-detect + manual override** | Auto by default; tray radio forces EN/ES when auto is wrong. |
| VAD | **silero-vad** | Robust silence detection in noise; cheap on CPU. |
| Tray | **pystray** (AppIndicator backend) to start | Fast to build; fall back to hand-rolled `dbus-fast` SNI if needed. |
| Hotkey | **DE-native keybind → CLI → daemon socket** | No app can reliably grab a global hotkey on Wayland across these DEs. |

## 5. Architecture

Three pieces around one long-running process:

- **`voicetyped` (daemon)** — a `systemd --user` service that stays running.
  It holds the warm Whisper model in VRAM, owns mic capture, runs the state
  machine, draws the tray icon, and listens on a control socket.
- **`voicetype` (CLI client)** — a thin command bound to a hotkey in each DE's
  own keyboard settings: `voicetype toggle` (also `stop`, `lang auto|en|es`,
  `status`). It sends a message to the daemon over a Unix socket and exits.
- **Tray icon (StatusNotifierItem)** — lives inside the daemon; changes
  appearance by state and exposes a menu (language, mic device, pause, quit).

### Flow

```
hotkey → `voicetype toggle` → socket → daemon
  IDLE ──toggle──▶ LISTENING ──2s silence (or toggle)──▶ TRANSCRIBING ──▶ inject ──▶ IDLE
```

1. Hotkey → daemon starts capturing immediately (no clipped first word),
   icon → **listening**.
2. Audio (16 kHz mono) streams into a ring buffer; the VAD watches it. After
   speech, once **2 s of continuous silence** is seen (or the hotkey is pressed
   again), capture stops, icon → **transcribing**.
3. Buffered audio → faster-whisper → text.
4. Text is typed into the focused app via **ydotool**; icon → **idle**.

## 6. Modules (`src/voicetype/`)

| Module | Responsibility | Key dependency |
|---|---|---|
| `daemon.py` | main loop, state machine driver, socket server, orchestration | asyncio |
| `cli.py` | client commands → socket | — |
| `ipc.py` | socket path + JSON line protocol (shared by daemon and CLI) | — |
| `audio.py` | mic capture, ring buffer | `sounddevice` (PipeWire/Pulse) |
| `vad.py` | speech/silence detection, 2 s silence timeout | `silero-vad` |
| `asr.py` | faster-whisper wrapper, warm model, auto/forced language | `faster-whisper` |
| `inject.py` | type text via ydotool | `ydotoold` |
| `tray.py` | SNI icon states + menu | `pystray` / `dbus-fast` |
| `config.py` | TOML at `~/.config/voicetype/config.toml` | — |
| `state.py` | state enum + transitions | — |

### State machine (`state.py`)

States: `IDLE`, `LISTENING`, `TRANSCRIBING`, `ERROR`.

Transitions:
- `IDLE` --toggle--> `LISTENING` (start capture)
- `LISTENING` --2s silence / toggle--> `TRANSCRIBING` (stop capture, run ASR)
- `TRANSCRIBING` --done--> `IDLE` (inject text)
- any --error--> `ERROR` --> `IDLE` (notify, reset)

## 7. Model, language, hardware

- faster-whisper **medium**, `device=cuda`, `compute_type=int8_float16`
  (~2 GB, fits 3 GB). Auto-fallback to CPU `int8` on CUDA OOM. All configurable.
- Language: **auto-detect per utterance** by default; tray radio menu
  (Auto / English / Español) forces a language. CLI `lang` mirrors this.
- Model stays loaded between uses so transcription starts instantly.

## 8. IPC protocol (`ipc.py`)

- Unix domain socket at `$XDG_RUNTIME_DIR/voicetype.sock`.
- JSON line protocol. Client sends `{"cmd": "toggle"}`, `{"cmd": "stop"}`,
  `{"cmd": "lang", "value": "auto|en|es"}`, `{"cmd": "status"}`.
- Daemon replies with one JSON line (e.g. current state / ack / error).
- If the socket is absent/unconnectable, the CLI prints a clear hint that the
  daemon isn't running.

## 9. Configuration (`config.py`)

TOML at `~/.config/voicetype/config.toml`. Keys (initial):

- `model` (default `medium`), `device` (`cuda`/`cpu`/`auto`), `compute_type`
- `language_default` (`auto`)
- `mic_device` (default system default)
- `silence_seconds` (default `2.0`)
- `inject_method` (default `ydotool`)
- `min_speech_seconds` (guard against empty/accidental triggers)

Missing file → sensible defaults written on first run.

## 10. Error handling

- **Mic busy / no device** → desktop notification, stay `IDLE`.
- **CUDA OOM** → fall back to CPU `int8`, warn once.
- **Empty / silent transcription** → no-op + brief notification.
- **ydotool / uinput not set up** → notification pointing at the fix.
- **Daemon not running when CLI fires** → CLI prints a clear hint.
- All failures return the icon to **IDLE**.

## 11. Setup & packaging

- `pyproject.toml` managed with `uv`.
- `systemd/voicetype.service` (user unit).
- `install.sh`: install deps, enable the service, configure **ydotool** and a
  udev rule for `/dev/uinput` (the one step needing elevated permission, done
  once).
- Per-DE docs: how to bind the hotkey; on **GNOME**, the AppIndicator/Ayatana
  extension is required for the tray to appear.
- Later: AUR PKGBUILD + PyPI publication. Flatpak deferred (uinput access).

### Project layout

```
voicetype/
  pyproject.toml
  src/voicetype/
    __init__.py
    daemon.py
    cli.py
    ipc.py
    audio.py
    vad.py
    asr.py
    inject.py
    tray.py
    config.py
    state.py
  data/icons/        # idle / listening / transcribing
  systemd/voicetype.service
  packaging/         # install.sh, udev rule, AUR PKGBUILD
  docs/
  tests/
```

## 12. Testing (TDD)

- **Unit:** VAD silence logic on synthetic frames; IPC protocol round-trip;
  config load/save with defaults; state-machine transitions; ydotool command
  construction (mocked subprocess); faster-whisper wrapper on a tiny fixture WAV.
- **Integration:** socket `toggle` drives the full state machine with a fake
  audio source and a fake ASR, asserting `inject` is called with the expected
  text and the icon returns to `IDLE`.

## 13. Open trade-offs (revisit during implementation)

- **Tray library:** `pystray` first; switch to hand-rolled `dbus-fast` SNI only
  if COSMIC support or dynamic icon-state updates misbehave.
- **Model default:** `medium` vs `small` — benchmark on the GTX 1060 during
  implementation and confirm the shipped default.
