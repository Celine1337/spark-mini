#!/usr/bin/env python3
"""Generate pink profile-style README card assets (hero, pills, sections, chips, stats).

Style reference: kawaii GitHub profile cards - near-transparent page, rounded
maroon cards, pink scalloped flower badges, mauve/brown blobs, pill buttons,
outlined chips. Requires Pillow. Usage: python make_style.py style.json out_dir
"""
import json
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont

CARD = "#2b2126"
INK = "#f4eef1"
MUTED = "#c9b8bf"
LABEL = "#b7a6ad"
PINK = "#eba7b9"
PINK_PALE = "#f8e3ea"
GLYPH = "#7b5563"
MAUVE = "#8e4a63"
BROWN = "#8a6a2f"
PILL = "#7b5563"
CHIP_BORDER = "#66525a"
CHIP_BG = "#241a1f"
TILE = "#3a2b32"

W = 1280
F_BOLD = "C:/Windows/Fonts/segoeuib.ttf"
F_REG = "C:/Windows/Fonts/segoeui.ttf"
F_MONO = "C:/Windows/Fonts/consola.ttf"


def font(path, size):
    return ImageFont.truetype(path, size)


def rounded_mask(size, radius):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius, fill=255)
    return m


def clip_round(img, radius):
    img.putalpha(ImageChops_multiply(img.getchannel("A"), rounded_mask(img.size, radius)))
    return img


def ImageChops_multiply(a, b):
    from PIL import ImageChops
    return ImageChops.multiply(a, b)


def flower(d, cx, cy, r, fill, bumps=12, bump_r=None):
    bump_r = bump_r or r * 0.30
    ring = r * 0.82
    for k in range(bumps):
        a = 2 * math.pi * k / bumps
        x, y = cx + ring * math.cos(a), cy + ring * math.sin(a)
        d.ellipse([x - bump_r, y - bump_r, x + bump_r, y + bump_r], fill=fill)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)


def spaced(d, xy, text, f, fill, spacing=3):
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=f, fill=fill)
        x += d.textlength(ch, font=f) + spacing
    return x


def glyph(d, kind, cx, cy, s, color):
    """Small vector icon centered at (cx, cy), size s."""
    h = s / 2
    if kind == "code":
        d.line([(cx - h * 0.5, cy - h * 0.6), (cx - h, cy), (cx - h * 0.5, cy + h * 0.6)], fill=color, width=max(2, s // 8))
        d.line([(cx + h * 0.5, cy - h * 0.6), (cx + h, cy), (cx + h * 0.5, cy + h * 0.6)], fill=color, width=max(2, s // 8))
    elif kind == "check":
        d.line([(cx - h * 0.8, cy), (cx - h * 0.2, cy + h * 0.6), (cx + h * 0.9, cy - h * 0.6)], fill=color, width=max(2, s // 7))
    elif kind == "book":
        d.rounded_rectangle([cx - h * 0.8, cy - h * 0.7, cx + h * 0.8, cy + h * 0.7], s // 8, outline=color, width=max(2, s // 9))
        d.line([(cx, cy - h * 0.7), (cx, cy + h * 0.7)], fill=color, width=max(2, s // 9))
    elif kind == "tag":
        d.polygon([(cx - h * 0.9, cy - h * 0.5), (cx + h * 0.1, cy - h * 0.5), (cx + h * 0.9, cy),
                   (cx + h * 0.1, cy + h * 0.5), (cx - h * 0.9, cy + h * 0.5)], outline=color, width=max(2, s // 9))
        d.ellipse([cx - h * 0.55, cy - s // 10, cx - h * 0.25, cy + s // 10], fill=color)
    elif kind == "doc":
        d.rounded_rectangle([cx - h * 0.65, cy - h * 0.8, cx + h * 0.65, cy + h * 0.8], s // 8, outline=color, width=max(2, s // 9))
        for i in range(3):
            y = cy - h * 0.35 + i * h * 0.35
            d.line([(cx - h * 0.35, y), (cx + h * 0.35, y)], fill=color, width=max(2, s // 10))
    elif kind == "gear":
        flower(d, cx, cy, h, color, bumps=8)
        d.ellipse([cx - h * 0.35, cy - h * 0.35, cx + h * 0.35, cy + h * 0.35], fill=CARD)
    elif kind == "graph":
        pts = [(cx - h * 0.7, cy + h * 0.5), (cx, cy - h * 0.6), (cx + h * 0.7, cy + h * 0.3)]
        d.line([pts[0], pts[1], pts[2], pts[0]], fill=color, width=max(2, s // 9))
        for x, y in pts:
            d.ellipse([x - s // 6, y - s // 6, x + s // 6, y + s // 6], fill=color)
    elif kind == "play":
        d.polygon([(cx - h * 0.5, cy - h * 0.7), (cx + h * 0.8, cy), (cx - h * 0.5, cy + h * 0.7)], fill=color)
    elif kind == "image":
        d.rounded_rectangle([cx - h * 0.85, cy - h * 0.65, cx + h * 0.85, cy + h * 0.65], s // 8, outline=color, width=max(2, s // 9))
        d.ellipse([cx - h * 0.45, cy - h * 0.35, cx - h * 0.15, cy - h * 0.05], fill=color)
        d.line([(cx - h * 0.6, cy + h * 0.4), (cx - h * 0.1, cy - h * 0.1), (cx + h * 0.3, cy + h * 0.3), (cx + h * 0.6, cy + h * 0.0)], fill=color, width=max(2, s // 9))
    elif kind == "star":
        d.polygon([(cx, cy - h), (cx + h * 0.25, cy - h * 0.25), (cx + h, cy), (cx + h * 0.25, cy + h * 0.25),
                   (cx, cy + h), (cx - h * 0.25, cy + h * 0.25), (cx - h, cy), (cx - h * 0.25, cy - h * 0.25)], fill=color)
    elif kind == "pin":
        d.ellipse([cx - h * 0.6, cy - h * 0.8, cx + h * 0.6, cy + h * 0.2], outline=color, width=max(2, s // 8))
        d.polygon([(cx - h * 0.35, cy + h * 0.05), (cx + h * 0.35, cy + h * 0.05), (cx, cy + h * 0.8)], fill=color)
    elif kind == "hex":
        f = font(F_MONO, int(s * 0.5))
        d.text((cx - s * 0.6, cy - s * 0.62), "0x", font=f, fill=color)
        for i in range(3):
            y = cy - s * 0.02 + i * s * 0.24
            d.rounded_rectangle([cx - s * 0.6, y, cx - s * 0.1, y + s * 0.12], 2, fill=color)
            d.rounded_rectangle([cx + s * 0.05, y, cx + s * 0.6, y + s * 0.12], 2, fill=color)
    elif kind == "at":
        f = font(F_BOLD, int(s * 1.1))
        tw = d.textlength("@", font=f)
        d.text((cx - tw / 2, cy - s * 0.62), "@", font=f, fill=color)


def card(w, h, radius=24):
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius, fill=CARD)
    return img, d


def blobs(d, w, h):
    for cx, cy, r in [(w - 90, 30, 110), (w - 210, -30, 90), (w - 20, 130, 80)]:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=MAUVE)
    for cx, cy, r in [(w - 190, h + 10, 80), (w - 90, h - 20, 60)]:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=BROWN)
    flower(d, w - 120, h - 90, 42, MAUVE, bumps=8)
    d.ellipse([w - 138, h - 108, w - 102, h - 72], fill=CARD)


def hero(cfg, out):
    h = 420
    img, d = card(W, h)
    blobs(d, W, h)
    flower(d, 170, 210, 108, PINK)
    d.ellipse([170 - 88, 210 - 88, 170 + 88, 210 + 88], fill=PINK_PALE)
    glyph(d, cfg["avatar"], 170, 210, 96, GLYPH)
    x = 340
    d.text((x, 96), cfg["kicker"], font=font(F_BOLD, 22), fill=INK)
    d.text((x, 130), cfg["name"], font=font(F_BOLD, 68), fill=INK)
    sub_f = font(F_REG, 24)
    d.text((x, 228), cfg["subtitle"], font=sub_f, fill=MUTED)
    cw = d.textlength(cfg["subtitle"], font=sub_f)
    d.rectangle([x + cw + 6, 230, x + cw + 10, 256], fill=PINK)
    px = x
    for icon, label in cfg["hero_pills"]:
        f = font(F_BOLD, 16)
        tw = d.textlength(label, font=f)
        pw = int(tw) + 66
        d.rounded_rectangle([px, 292, px + pw, 338], 23, fill=TILE)
        glyph(d, icon, px + 26, 315, 18, PINK)
        d.text((px + 44, 306), label, font=f, fill=INK)
        px += pw + 14
    clip_round(img, 24).save(os.path.join(out, "hero.png"))


def pill(cfg_pill, out):
    f = font(F_BOLD, 17)
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    tw = tmp.textlength(cfg_pill["label"], font=f)
    w = int(tw) + 76
    h = 58
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w - 1, h - 1], 29, fill=PILL)
    glyph(d, cfg_pill["icon"], 30, h // 2, 20, INK)
    d.text((50, 18), cfg_pill["label"], font=f, fill=INK)
    img.save(os.path.join(out, f"pill-{cfg_pill['id']}.png"))


def section(title, icon, out, fname):
    h = 96
    img, d = card(W, h, 20)
    flower(d, 56, 48, 24, PINK, bumps=10)
    glyph(d, icon, 56, 48, 22, GLYPH)
    d.text((100, 32), title, font=font(F_BOLD, 27), fill=INK)
    clip_round(img, 20).save(os.path.join(out, fname))


def chips(cfg, out):
    groups = cfg["chips"]
    row_h, gap, pad = 50, 14, 28
    h = pad
    for label, items in groups.items():
        h += 26 + row_h + 18
    h += pad - 18
    img, d = card(W, h, 20)
    y = pad
    for label, items in groups.items():
        spaced(d, (pad, y), label.upper(), font(F_BOLD, 13), LABEL, 3)
        y += 26
        x = pad
        for item in items:
            f = font(F_REG, 16)
            tw = d.textlength(item, font=f)
            cw = int(tw) + 62
            d.rounded_rectangle([x, y, x + cw, y + row_h], 12, fill=CHIP_BG, outline=CHIP_BORDER, width=2)
            glyph(d, "code" if label == "MAIN" else "gear", x + 26, y + row_h // 2, 18, PINK)
            d.text((x + 44, y + 15), item, font=f, fill=INK)
            x += cw + gap
        y += row_h + 18
    clip_round(img, 20).save(os.path.join(out, "chips.png"))


def stats(cfg, out):
    tiles = cfg["stats"]
    h = 150
    img, d = card(W, h, 20)
    x = 28
    for value, label in tiles:
        tw = 300
        d.rounded_rectangle([x, 26, x + tw, h - 26], 16, fill=TILE)
        flower(d, x + 40, 62, 17, PINK, bumps=10)
        glyph(d, "star", x + 40, 62, 14, GLYPH)
        d.text((x + 70, 42), value, font=font(F_BOLD, 34), fill=INK)
        d.text((x + 26, 96), label, font=font(F_REG, 14), fill=MUTED)
        x += tw + 20
    clip_round(img, 20).save(os.path.join(out, "stats.png"))


def social(cfg, out):
    img = Image.new("RGBA", (W, 640), "#0d1117")
    hero_path = os.path.join(out, "hero.png")
    img.paste(Image.open(hero_path), (0, 110), Image.open(hero_path))
    d = ImageDraw.Draw(img)
    d.text((64, 566), f"github.com/Celine1337/{cfg.get('slug', cfg['name'])}", font=font(F_MONO, 20), fill=LABEL)
    img.convert("RGB").save(os.path.join(out, "social-preview.png"))


def main():
    cfg = json.load(open(sys.argv[1], encoding="utf-8"))
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    hero(cfg, out)
    for p in cfg["pills"]:
        pill(p, out)
    for lang, secs in cfg["sections"].items():
        for sid, title, icon in secs:
            section(title, icon, out, f"sec-{sid}-{lang}.png")
    chips(cfg, out)
    stats(cfg, out)
    social(cfg, out)
    print(f"{out}: hero, {len(cfg['pills'])} pills, "
          f"{sum(len(s) for s in cfg['sections'].values())} sections, chips, stats, social")


if __name__ == "__main__":
    main()