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
