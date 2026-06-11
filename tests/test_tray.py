from voicetype import tray
from voicetype.state import State


def test_load_pixmap_shape_and_argb_order():
    # idle icon is a 64x64 grey (130,130,130) filled circle on transparent bg.
    pixmaps = tray.load_pixmap(State.IDLE)
    assert len(pixmaps) == 1
    width, height, data = pixmaps[0]
    assert (width, height) == (64, 64)
    assert len(data) == 64 * 64 * 4  # ARGB, 4 bytes/pixel

    # The centre pixel is opaque grey; in ARGB order that is A=255 then 130,130,130.
    centre = (32 * 64 + 32) * 4
    a, r, g, b = data[centre], data[centre + 1], data[centre + 2], data[centre + 3]
    assert a == 255
    assert (r, g, b) == (130, 130, 130)

    # A corner pixel is outside the circle -> fully transparent.
    corner = 0
    assert data[corner] == 0  # alpha


def test_menu_item_props_separator():
    item = {"id": 2, "separator": True}
    props = tray.menu_item_props(item, "auto")
    assert props["type"].value == "separator"


def test_menu_item_props_radio_reflects_current_language():
    auto = {"id": tray.LANG_AUTO_ID, "label": "Auto", "lang": "auto"}
    es = {"id": tray.LANG_ES_ID, "label": "Español", "lang": "es"}
    # current language is "es"
    assert tray.menu_item_props(auto, "es")["toggle-state"].value == 0
    assert tray.menu_item_props(es, "es")["toggle-state"].value == 1
    assert tray.menu_item_props(auto, "es")["toggle-type"].value == "radio"


def test_plain_item_has_label():
    item = {"id": tray.TOGGLE_ID, "label": "Toggle dictation"}
    props = tray.menu_item_props(item, "auto")
    assert props["label"].value == "Toggle dictation"
    assert "toggle-type" not in props


def test_on_menu_click_routes_actions():
    events = []
    t = tray.Tray(
        on_toggle=lambda: events.append("toggle"),
        on_set_language=lambda lang: events.append(("lang", lang)),
        on_quit=lambda: events.append("quit"),
        get_language=lambda: "auto",
    )
    t._on_menu_click(tray.TOGGLE_ID)
    t._on_menu_click(tray.LANG_ES_ID)
    t._on_menu_click(tray.QUIT_ID)
    assert events == ["toggle", ("lang", "es"), "quit"]
