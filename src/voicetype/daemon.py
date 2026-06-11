"""voicetyped: socket server + tray + controller wiring."""

from __future__ import annotations

import asyncio
import os

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
