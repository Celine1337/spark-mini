#!/usr/bin/env python3
"""Render tinybin ANSI output as a terminal-window PNG (for README screenshots).

Usage: python render_terminal.py input.ansi output.png --title "..." --cmd "tinybin ..."
Requires Pillow (pip install pillow). Windows: uses Consolas / Segoe UI.
"""
import argparse
import re

from PIL import Image, ImageDraw, ImageFilter, ImageFont

PALETTE = {
    "30": "#0c0c0c", "31": "#c50f1f", "32": "#13a10e", "33": "#c19c00",
    "34": "#0037da", "35": "#881798", "36": "#3a96dd", "37": "#cccccc",
    "90": "#767676", "91": "#e74856", "92": "#16c60c", "93": "#f9f1a5",
    "94": "#3b78ff", "95": "#b4009e", "96": "#61d6d6", "97": "#f2f2f2",
}
FG = "#cccccc"
BG = "#0c0c0c"
FONT_SIZE = 14
LINE_H = 21
PAD_X, PAD_Y = 22, 18
TITLE_H = 38
MARGIN = 36
RADIUS = 10


def parse_ansi(text):
    """Return list of lines; each line is a list of (text, color, bold, dim)."""
    lines = [[]]
    color, bold, dim = None, False, False
    for part in re.split(r"(\x1b\[[0-9;]*m)", text):
        m = re.fullmatch(r"\x1b\[([0-9;]*)m", part)
        if m:
            for code in (m.group(1) or "0").split(";"):
                if code == "0":
                    color, bold, dim = None, False, False
                elif code == "1":
                    bold = True
                elif code == "2":
                    dim = True
                elif code in PALETTE:
                    color = PALETTE[code]
            continue
        chunks = part.split("\n")
        for i, chunk in enumerate(chunks):
            if i > 0:
                lines.append([])
            if chunk:
                lines[-1].append((chunk, color, bold, dim))
    return lines


def blend(hex_color, other, t):
    a = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(other[i:i + 2], 16) for i in (1, 3, 5))
    return tuple(int(a[i] * t + b[i] * (1 - t)) for i in range(3))


def render(lines, title, cmd, out_path):
    reg = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", FONT_SIZE)
    boldf = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", FONT_SIZE)
    uif = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 12)
    cw = reg.getlength("M")

    if cmd:
        lines = [[(">", "#16c60c", True, False), (" " + cmd, None, False, False)], []] + lines
    cols = max(sum(len(t) for t, _, _, _ in ln) for ln in lines)
    win_w = int(cols * cw) + 2 * PAD_X
    win_h = TITLE_H + 2 * PAD_Y + len(lines) * LINE_H
    W, H = win_w + 2 * MARGIN, win_h + 2 * MARGIN

    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    top, bot = (26, 27, 38), (15, 16, 21)
    for y in range(H):
        t = y / max(H - 1, 1)
        d.line([(0, y), (W, y)], fill=tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3)))

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle([MARGIN, MARGIN + 12, MARGIN + win_w, MARGIN + 12 + win_h],
                         RADIUS, fill=(0, 0, 0, 110))
    shadow = shadow.filter(ImageFilter.GaussianBlur(16))
    img = Image.alpha_composite(img.convert("RGBA"), shadow).convert("RGB")
    d = ImageDraw.Draw(img)

    x0, y0 = MARGIN, MARGIN
    d.rounded_rectangle([x0, y0, x0 + win_w, y0 + win_h], RADIUS, fill=BG, outline="#2a2a2a")
    d.rounded_rectangle([x0, y0, x0 + win_w, y0 + TITLE_H + RADIUS], RADIUS, fill="#1f1f1f")
    d.rectangle([x0, y0 + TITLE_H - RADIUS, x0 + win_w, y0 + TITLE_H], fill="#1f1f1f")
    d.line([(x0, y0 + TITLE_H), (x0 + win_w, y0 + TITLE_H)], fill="#2a2a2a")
    for i, c in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        cx = x0 + 20 + i * 20
        cy = y0 + TITLE_H // 2
        d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=c)
    tw = d.textlength(title, font=uif)
    d.text((x0 + (win_w - tw) / 2, y0 + TITLE_H / 2 - 8), title, font=uif, fill="#9a9a9a")

    y = y0 + TITLE_H + PAD_Y
    for ln in lines:
        x = x0 + PAD_X
        for text, color, bold, dim in ln:
            f = boldf if bold else reg
            fill = blend(color or FG, BG, 0.55) if dim else (color or FG)
            d.text((x, y), text, font=f, fill=fill)
            x += f.getlength(text)
        y += LINE_H
    img.save(out_path)
    print(f"{out_path}: {W}x{H}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--title", default="tinybin")
    ap.add_argument("--cmd", default=None)
    args = ap.parse_args()
    with open(args.input, "r", encoding="utf-8") as f:
        lines = parse_ansi(f.read().rstrip("\n"))
    render(lines, args.title, args.cmd, args.output)


if __name__ == "__main__":
    main()