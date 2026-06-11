import asyncio

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
