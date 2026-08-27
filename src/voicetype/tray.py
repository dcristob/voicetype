"""Wayland-native tray icon via the StatusNotifierItem D-Bus protocol.

Implements org.kde.StatusNotifierItem plus a com.canonical.dbusmenu menu using
dbus-fast. This works across COSMIC, Budgie and GNOME (with the AppIndicator
extension) without any X11 systray or GTK dependency.

The public ``Tray`` interface (constructor callbacks, ``set_state``, async
``setup``/``stop``) is what the daemon depends on.

NOTE: this module deliberately does NOT use ``from __future__ import
annotations``. dbus-fast's classic API reads method/property annotations at
runtime to derive D-Bus signatures (e.g. ``-> "a(iiay)"``); PEP 563 string-izing
would corrupt those signatures.
"""

import asyncio
import os
from importlib.resources import files
from typing import Callable

from dbus_fast import BusType, PropertyAccess, Variant
from dbus_fast.aio import MessageBus
from dbus_fast.service import ServiceInterface, dbus_property, method, signal
from PIL import Image

from .state import State

SNI_PATH = "/StatusNotifierItem"
MENU_PATH = "/MenuBar"
WATCHER_NAME = "org.kde.StatusNotifierWatcher"
WATCHER_PATH = "/StatusNotifierWatcher"

_ICON_FILES = {
    State.IDLE: "idle.png",
    State.LISTENING: "listening.png",
    State.TRANSCRIBING: "transcribing.png",
    State.ERROR: "idle.png",
}

# Menu item ids are stable so Event() can map a click back to an action.
TOGGLE_ID = 1
LANG_AUTO_ID = 3
LANG_EN_ID = 4
LANG_ES_ID = 5
QUIT_ID = 7

_MENU_ITEMS = [
    {"id": TOGGLE_ID, "label": "Toggle dictation"},
    {"id": 2, "separator": True},
    {"id": LANG_AUTO_ID, "label": "Auto", "lang": "auto"},
    {"id": LANG_EN_ID, "label": "English", "lang": "en"},
    {"id": LANG_ES_ID, "label": "Español", "lang": "es"},
    {"id": 6, "separator": True},
    {"id": QUIT_ID, "label": "Quit"},
]


def load_pixmap(state: State) -> list:
    """Return the SNI IconPixmap value (``a(iiay)``) for a state.

    The wire format is a list of (width, height, ARGB32-big-endian bytes).
    """
    resource = files("voicetype").joinpath("icons", _ICON_FILES[state])
    with resource.open("rb") as fh:
        img = Image.open(fh).convert("RGBA")
        img.load()
    width, height = img.size
    rgba = img.tobytes()  # R, G, B, A per pixel
    argb = bytearray(len(rgba))
    for i in range(0, len(rgba), 4):
        r, g, b, a = rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]
        argb[i] = a
        argb[i + 1] = r
        argb[i + 2] = g
        argb[i + 3] = b
    return [[width, height, bytes(argb)]]


def menu_item_props(item: dict, current_language: str) -> dict:
    """Build the ``a{sv}`` property map for one dbusmenu item."""
    if item.get("separator"):
        return {"type": Variant("s", "separator")}
    props = {
        "label": Variant("s", item["label"]),
        "enabled": Variant("b", True),
        "visible": Variant("b", True),
    }
    if "lang" in item:
        props["toggle-type"] = Variant("s", "radio")
        props["toggle-state"] = Variant("i", 1 if current_language == item["lang"] else 0)
    return props


class _StatusNotifierItem(ServiceInterface):
    def __init__(self, tray: "Tray"):
        super().__init__("org.kde.StatusNotifierItem")
        self._tray = tray

    @dbus_property(access=PropertyAccess.READ)
    def Category(self) -> "s":  # noqa: F821
        return "ApplicationStatus"

    @dbus_property(access=PropertyAccess.READ)
    def Id(self) -> "s":  # noqa: F821
        return "voicetype"

    @dbus_property(access=PropertyAccess.READ)
    def Title(self) -> "s":  # noqa: F821
        return "VoiceType"

    @dbus_property(access=PropertyAccess.READ)
    def Status(self) -> "s":  # noqa: F821
        return "Active"

    @dbus_property(access=PropertyAccess.READ)
    def WindowId(self) -> "u":  # noqa: F821
        return 0

    @dbus_property(access=PropertyAccess.READ)
    def IconThemePath(self) -> "s":  # noqa: F821
        return ""

    @dbus_property(access=PropertyAccess.READ)
    def IconName(self) -> "s":  # noqa: F821
        return ""

    @dbus_property(access=PropertyAccess.READ)
    def IconPixmap(self) -> "a(iiay)":  # noqa: F821
        return self._tray._pixmap

    @dbus_property(access=PropertyAccess.READ)
    def OverlayIconName(self) -> "s":  # noqa: F821
        return ""

    @dbus_property(access=PropertyAccess.READ)
    def OverlayIconPixmap(self) -> "a(iiay)":  # noqa: F821
        return []

    @dbus_property(access=PropertyAccess.READ)
    def AttentionIconName(self) -> "s":  # noqa: F821
        return ""

    @dbus_property(access=PropertyAccess.READ)
    def AttentionIconPixmap(self) -> "a(iiay)":  # noqa: F821
        return []

    @dbus_property(access=PropertyAccess.READ)
    def AttentionMovieName(self) -> "s":  # noqa: F821
        return ""

    @dbus_property(access=PropertyAccess.READ)
    def ToolTip(self) -> "(sa(iiay)ss)":  # noqa: F821
        return ["", [], "VoiceType", self._tray._tooltip]

    @dbus_property(access=PropertyAccess.READ)
    def ItemIsMenu(self) -> "b":  # noqa: F821
        # False -> left click triggers Activate (toggle); menu still available.
        return False

    @dbus_property(access=PropertyAccess.READ)
    def Menu(self) -> "o":  # noqa: F821
        return MENU_PATH

    @method()
    def Activate(self, x: "i", y: "i"):  # noqa: F821
        self._tray._on_toggle()

    @method()
    def SecondaryActivate(self, x: "i", y: "i"):  # noqa: F821
        self._tray._on_toggle()

    @method()
    def ContextMenu(self, x: "i", y: "i"):  # noqa: F821
        # Hosts that read the Menu property render the menu themselves.
        pass

    @method()
    def Scroll(self, delta: "i", orientation: "s"):  # noqa: F821
        pass

    @signal()
    def NewIcon(self):
        pass

    @signal()
    def NewToolTip(self):
        pass

    @signal()
    def NewTitle(self):
        pass

    @signal()
    def NewStatus(self) -> "s":  # noqa: F821
        return "Active"


class _DBusMenu(ServiceInterface):
    def __init__(self, tray: "Tray"):
        super().__init__("com.canonical.dbusmenu")
        self._tray = tray
        self._revision = 1

    @dbus_property(access=PropertyAccess.READ)
    def Version(self) -> "u":  # noqa: F821
        return 3

    @dbus_property(access=PropertyAccess.READ)
    def Status(self) -> "s":  # noqa: F821
        return "normal"

    @dbus_property(access=PropertyAccess.READ)
    def TextDirection(self) -> "s":  # noqa: F821
        return "ltr"

    @dbus_property(access=PropertyAccess.READ)
    def IconThemePath(self) -> "as":  # noqa: F821
        return []

    def _props(self, item: dict) -> dict:
        return menu_item_props(item, self._tray._get_language())

    @method()
    def GetLayout(
        self, parentId: "i", recursionDepth: "i", propertyNames: "as"  # noqa: F821
    ) -> "u(ia{sv}av)":  # noqa: F821
        children = [
            Variant("(ia{sv}av)", [item["id"], self._props(item), []])
            for item in self._tray._menu_items
        ]
        root = [0, {"children-display": Variant("s", "submenu")}, children]
        return [self._revision, root]

    @method()
    def GetGroupProperties(
        self, ids: "ai", propertyNames: "as"  # noqa: F821
    ) -> "a(ia{sv})":  # noqa: F821
        return [
            [item["id"], self._props(item)]
            for item in self._tray._menu_items
            if not ids or item["id"] in ids
        ]

    @method()
    def GetProperty(self, id: "i", name: "s") -> "v":  # noqa: F821
        for item in self._tray._menu_items:
            if item["id"] == id:
                props = self._props(item)
                if name in props:
                    return props[name]
        return Variant("s", "")

    @method()
    def Event(self, id: "i", eventId: "s", data: "v", timestamp: "u"):  # noqa: F821
        if eventId == "clicked":
            self._tray._on_menu_click(id)

    @method()
    def EventGroup(self, events: "a(isvu)") -> "ai":  # noqa: F821
        for ev in events:
            item_id, event_id, _data, _ts = ev
            if event_id == "clicked":
                self._tray._on_menu_click(item_id)
        return []

    @method()
    def AboutToShow(self, id: "i") -> "b":  # noqa: F821
        return False

    @method()
    def AboutToShowGroup(self, ids: "ai") -> "(aiai)":  # noqa: F821
        return [[], []]

    @signal()
    def LayoutUpdated(self) -> "ui":  # noqa: F821
        return [self._revision, 0]

    def notify_changed(self) -> None:
        """Bump the revision and tell hosts to re-fetch the layout."""
        self._revision += 1
        self.LayoutUpdated()


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
        self._menu_items = _MENU_ITEMS
        self._pixmap = load_pixmap(State.IDLE)
        self._tooltip = State.IDLE.value
        self._bus: MessageBus | None = None
        self._sni: _StatusNotifierItem | None = None
        self._menu: _DBusMenu | None = None
        self._name = ""

    def _on_menu_click(self, item_id: int) -> None:
        item = next((i for i in self._menu_items if i["id"] == item_id), None)
        if item is None:
            return
        if "lang" in item:
            self._on_set_language(item["lang"])
            if self._menu is not None:
                self._menu.notify_changed()
        elif item_id == TOGGLE_ID:
            self._on_toggle()
        elif item_id == QUIT_ID:
            self._on_quit()

    async def setup(self) -> None:
        """Connect to the session bus, export the item + menu, register.

        Registration is resilient to the StatusNotifierWatcher not being up at
        the moment we start (the common case at login: the daemon is pulled in
        by default.target before the panel's tray applet owns the watcher name)
        and to the watcher restarting later. We (re)register on every
        NameOwnerChanged for the watcher name, so the icon reappears instead of
        being lost forever after a one-shot boot-time race.
        """
        self._bus = await MessageBus(bus_type=BusType.SESSION).connect()
        self._sni = _StatusNotifierItem(self)
        self._menu = _DBusMenu(self)
        self._bus.export(SNI_PATH, self._sni)
        self._bus.export(MENU_PATH, self._menu)
        self._name = f"org.kde.StatusNotifierItem-{os.getpid()}-1"
        await self._bus.request_name(self._name)
        await self._watch_watcher()
        await self._register()

    async def _watch_watcher(self) -> None:
        """Re-register whenever the StatusNotifierWatcher (re)appears."""
        introspection = await self._bus.introspect(
            "org.freedesktop.DBus", "/org/freedesktop/DBus"
        )
        dbus_obj = self._bus.get_proxy_object(
            "org.freedesktop.DBus", "/org/freedesktop/DBus", introspection
        )
        dbus_iface = dbus_obj.get_interface("org.freedesktop.DBus")

        def on_name_owner_changed(name, old_owner, new_owner):
            # new_owner == "" means the watcher went away; only re-register when
            # it gains an owner (appeared or restarted).
            if name == WATCHER_NAME and new_owner:
                asyncio.ensure_future(self._register())

        dbus_iface.on_name_owner_changed(on_name_owner_changed)

    async def _register(self) -> None:
        """(Re)register our item with the watcher; a no-op if it is absent.

        Swallows errors so a missing/restarting watcher doesn't crash the
        daemon — on_name_owner_changed retries once the watcher is back.
        """
        if self._bus is None:
            return
        try:
            introspection = await self._bus.introspect(WATCHER_NAME, WATCHER_PATH)
            obj = self._bus.get_proxy_object(WATCHER_NAME, WATCHER_PATH, introspection)
            watcher = obj.get_interface(WATCHER_NAME)
            await watcher.call_register_status_notifier_item(self._name)
        except Exception:
            pass

    def set_state(self, state: State) -> None:
        self._pixmap = load_pixmap(state)
        self._tooltip = state.value
        if self._sni is not None:
            self._sni.NewIcon()
            self._sni.NewToolTip()

    async def stop(self) -> None:
        if self._bus is not None:
            self._bus.disconnect()
            self._bus = None
