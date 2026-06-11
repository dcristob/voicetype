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
