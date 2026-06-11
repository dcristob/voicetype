#!/usr/bin/env python3
"""Generate the tray state icons as a microphone glyph, one per state colour.

Run from the repo root:  python scripts/make_icons.py
Writes src/voicetype/icons/{idle,listening,transcribing}.png (64x64 RGBA).
Drawn at 4x and downsampled for smooth edges.
"""

from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 64
SS = 4  # supersample factor

COLORS = {
    "idle": (130, 130, 130),
    "listening": (220, 40, 40),
    "transcribing": (235, 170, 30),
}

OUT_DIR = Path(__file__).resolve().parent.parent / "src" / "voicetype" / "icons"


def draw_mic(color: tuple[int, int, int]) -> Image.Image:
    s = SIZE * SS
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    fill = color + (255,)

    def sc(v: float) -> int:
        return round(v * SS)

    cx = 32
    line_w = sc(3)

    # Mic capsule (rounded pill) — the head.
    cap_left, cap_right, cap_top, cap_bottom = 24, 40, 9, 41
    draw.rounded_rectangle(
        [sc(cap_left), sc(cap_top), sc(cap_right), sc(cap_bottom)],
        radius=sc((cap_right - cap_left) / 2),
        fill=fill,
    )

    # Cradle: a U-shaped bracket hugging the lower half of the capsule.
    draw.arc(
        [sc(17), sc(24), sc(47), sc(46)],
        start=0,
        end=180,
        fill=fill,
        width=line_w,
    )

    # Stem from the bottom of the cradle down to the base.
    draw.line([sc(cx), sc(46), sc(cx), sc(54)], fill=fill, width=line_w)

    # Base foot.
    draw.line([sc(23), sc(54), sc(41), sc(54)], fill=fill, width=line_w)

    return img.resize((SIZE, SIZE), Image.Resampling.LANCZOS)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rgb in COLORS.items():
        draw_mic(rgb).save(OUT_DIR / f"{name}.png")
        print("wrote", OUT_DIR / f"{name}.png")


if __name__ == "__main__":
    main()
