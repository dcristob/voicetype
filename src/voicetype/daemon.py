"""voicetyped: socket server + tray + controller wiring."""

from __future__ import annotations

import asyncio
import logging
import os
import signal

from . import ipc
from .audio import Recorder
from .asr import Transcriber
from .config import load_config
from .controller import Controller
from .inject import make_injector
from .notify import notify
from .state import State
from .tray import Tray
from .vad import SileroVad

logger = logging.getLogger(__name__)


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
            injector=make_injector(self.config),
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
        self._finishing = False

    # --- state/tray ---
    def _on_state_change(self, old: State, new: State) -> None:
        # Transitions can fire from a worker thread (complete_transcribe runs in
        # an executor), so marshal the dbus-fast tray update onto the loop thread.
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self.tray.set_state, new)
        else:
            self.tray.set_state(new)

    def _finish(self) -> None:
        # Runs on the loop thread. Do the fast transition (icon -> amber) inline,
        # then hand the blocking transcribe+inject to a worker thread so the loop
        # stays free to flush that icon update and answer socket/tray commands.
        audio = self.controller.begin_transcribe()
        if audio is None:
            return
        if self._loop is not None:
            self._loop.run_in_executor(
                None, self.controller.complete_transcribe, audio
            )
        else:
            self.controller.complete_transcribe(audio)

    # --- audio frame feeding (runs in the sounddevice thread) ---
    def _on_frame(self, frame) -> None:
        # Once end-of-utterance is detected the silence tracker keeps reporting
        # True for every subsequent silent frame; the flag ensures finish() is
        # scheduled exactly once per utterance.
        if self._finishing:
            return
        if self.controller.feed_frame(frame) and self._loop is not None:
            self._finishing = True
            self._loop.call_soon_threadsafe(self._finish)

    def _start_listening(self) -> None:
        # Re-route the recorder callback through the daemon so frames hit VAD.
        self._finishing = False
        self.controller._start()  # transitions + starts recorder
        # Replace the no-op callback with our frame feeder.
        self.recorder._on_frame = self._on_frame

    def toggle(self) -> None:
        if self.controller.state is State.IDLE:
            self._start_listening()
        elif self.controller.state is State.LISTENING:
            self._finish()

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
        # status is polled by tooling; keep it out of the journal at INFO.
        logger.log(
            logging.DEBUG if cmd == "status" else logging.INFO,
            "command %r received in state %s",
            cmd,
            self.controller.state.value,
        )
        if cmd == "toggle":
            self.toggle()
            reply = {"ok": True, "state": self.controller.state.value}
        elif cmd == "stop":
            if self.controller.state is State.LISTENING:
                self._finish()
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
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._loop.add_signal_handler(sig, self._stopping.set)
        path = ipc.socket_path()
        if path.exists():
            path.unlink()
        server = await asyncio.start_unix_server(self._handle, path=str(path))
        os.chmod(path, 0o600)
        try:
            await self.tray.setup()
        except Exception as exc:  # no SNI host available -> run headless
            notify("VoiceType: tray unavailable", str(exc))
        async with server:
            await self._stopping.wait()
        await self.tray.stop()
        if path.exists():
            path.unlink()


def main() -> None:
    # systemd captures stdout/stderr into the journal; surface INFO+ there so
    # transcription/injection failures and GPU->CPU fallbacks are debuggable.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    daemon = Daemon()
    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
