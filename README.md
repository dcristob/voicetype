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
  switches language (Auto / English / Español) and quits. The icon uses the
  StatusNotifierItem D-Bus protocol, so it works natively on Wayland.

## Requirements

- Wayland session on COSMIC, Budgie, or GNOME (GNOME needs the AppIndicator
  extension, which also provides the StatusNotifierWatcher the tray registers with).
- PipeWire or PulseAudio, a microphone.
- `ydotool` + `ydotoold` (text injection).
- NVIDIA GPU optional; falls back to CPU automatically.

### GPU acceleration (NVIDIA)

Install the CUDA runtime libraries CTranslate2 needs:

```bash
uv pip install -e ".[cuda]"   # nvidia-cublas-cu12 + nvidia-cudnn-cu12
```

The transcriber picks the best working option automatically: it tries the
configured compute type on the GPU, then plain `int8` on the GPU (older cards
like the GTX 10xx / Pascal series support `int8` but not `int8_float16`), then
falls back to CPU `int8`. So `compute_type = "int8_float16"` is safe everywhere
— a Pascal GPU quietly runs `int8` on the GPU instead of dropping to CPU.

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
