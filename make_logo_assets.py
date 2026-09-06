#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PREDICT-logó: vektoros (SVG) és raszteres (PNG) változatok egy geometriából.

Az eredeti logó: sötétkék „P”, amelyet egy fehér hullám vág át, alatta a
ritkított, félkövér PREDICT felirat. A szkript ugyanabból a geometriából írja:

    static/predict-logo.svg   teljes logó (jel + felirat), web
    static/predict-mark.svg   csak a jel, négyzetes, fejléc és favicon
    static/predict-logo.png   teljes logó, levelekhez és Word-űrlaphoz
    static/predict-mark.png   csak a jel

Ha az eredeti PNG rendelkezésre áll, elég ugyanezekre a nevekre bemásolni.
"""

import math
import os

from PIL import Image, ImageDraw, ImageFont

NAVY = "#0B2B5C"
W, H = 1000, 1240          # a teljes logó koordinátarendszere
STEM = (261, 190, 434, 854)  # x0, y0, x1, y1
BOWL_X0, BOWL_Y0, BOWL_Y1 = 261, 190, 606
BOWL_ARC_X = 560           # innen indul a jobb oldali ív
BOWL_RX, BOWL_RY = 215, 208
WAVE = [(200, 400), (480, 400), (560, 400), (585, 478), (660, 478), (735, 478), (770, 455), (860, 448)]
WAVE_WIDTH = 84
TEXT_Y = 1000
TEXT_SIZE = 118
LETTER_SPACING = 0.30      # em


def bezier(p0, p1, p2, p3, n=40):
    pts = []
    for i in range(n + 1):
        t = i / n
        x = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0]
        y = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1]
        pts.append((x, y))
    return pts


def wave_points():
    """A hullám középvonala: vízszintes szakasz, majd két egymásba érő Bézier-ív."""
    a, b, c1, c2, c3, c4, c5, end = WAVE
    pts = [a, b]
    pts += bezier(b, c1, c2, c3)[1:]
    pts += bezier(c3, c4, c5, end)[1:]
    return pts


def wave_svg_path():
    a, b, c1, c2, c3, c4, c5, end = WAVE
    return (f"M {a[0]} {a[1]} L {b[0]} {b[1]} "
            f"C {c1[0]} {c1[1]}, {c2[0]} {c2[1]}, {c3[0]} {c3[1]} "
            f"C {c4[0]} {c4[1]}, {c5[0]} {c5[1]}, {end[0]} {end[1]}")


def silhouette_svg_path():
    x0, y0, x1, y1 = STEM
    return (f"M {x0} {y0} H {BOWL_ARC_X} A {BOWL_RX} {BOWL_RY} 0 0 1 {BOWL_ARC_X} {BOWL_Y1} "
            f"H {x1} V {y1} H {x0} Z")


def svg_mark(padding=60):
    x0, y0, x1, y1 = STEM
    right = BOWL_ARC_X + BOWL_RX
    size = max(right - x0, y1 - y0) + 2 * padding
    ox = x0 - padding - (size - (right - x0) - 2 * padding) / 2
    oy = y0 - padding
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{ox:.0f} {oy:.0f} {size:.0f} {size:.0f}" role="img" aria-label="PREDICT">'
            f'<path d="{silhouette_svg_path()}" fill="{NAVY}"/>'
            f'<path d="{wave_svg_path()}" fill="none" stroke="#ffffff" stroke-width="{WAVE_WIDTH}" stroke-linecap="round" stroke-linejoin="round"/>'
            f'</svg>\n')


def svg_full():
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-label="PREDICT">'
            f'<path d="{silhouette_svg_path()}" fill="{NAVY}"/>'
            f'<path d="{wave_svg_path()}" fill="none" stroke="#ffffff" stroke-width="{WAVE_WIDTH}" stroke-linecap="round" stroke-linejoin="round"/>'
            f'<text x="{W / 2}" y="{TEXT_Y}" text-anchor="middle" fill="{NAVY}" font-family="Montserrat, Poppins, \'Avenir Next\', Arial, sans-serif" '
            f'font-weight="700" font-size="{TEXT_SIZE}" letter-spacing="{LETTER_SPACING * TEXT_SIZE:.0f}">PREDICT</text>'
            f'</svg>\n')


def load_font(size):
    candidates = [
        os.path.expanduser("~/Library/Fonts/Montserrat-VariableFont_wght.ttf"),
        os.path.expanduser("~/Library/Fonts/Montserrat-Bold.ttf"),
        "/System/Library/Fonts/Avenir Next.ttc",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            font = ImageFont.truetype(path, size)
            try:
                names = [n.decode() if isinstance(n, bytes) else n for n in font.get_variation_names()]
                if "Bold" in names:
                    font.set_variation_by_name("Bold")
            except (OSError, AttributeError):
                pass
            return font
        except OSError:
            continue
    return ImageFont.load_default()


def draw_mark(draw, scale, offset=(0, 0)):
    ox, oy = offset
    tr = lambda p: (p[0] * scale + ox, p[1] * scale + oy)
    x0, y0, x1, y1 = STEM
    draw.rectangle([tr((x0, y0)), tr((x1, y1))], fill=NAVY)
    # a „has” (bowl): téglalap a stem tetejétől az ív kezdetéig + ellipszis jobb fele
    draw.rectangle([tr((x0, BOWL_Y0)), tr((BOWL_ARC_X, BOWL_Y1))], fill=NAVY)
    cx, cy = BOWL_ARC_X, (BOWL_Y0 + BOWL_Y1) / 2
    draw.ellipse([tr((cx - BOWL_RX, cy - BOWL_RY)), tr((cx + BOWL_RX, cy + BOWL_RY))], fill=NAVY)
    draw.rectangle([tr((cx - BOWL_RX, cy - BOWL_RY)), tr((cx, cy + BOWL_RY))], fill=NAVY)  # bal fél takarás → csak a jobb ív marad kerek
    # A hullám: a középvonal két oldalára tolt pontokból kitöltött sokszög
    # (a vastag vonalrajzolás csíkos élt adna), lekerekített végekkel.
    pts = [tr(p) for p in wave_points()]
    half = WAVE_WIDTH * scale / 2
    left, right = [], []
    for i, (x, y) in enumerate(pts):
        x_prev, y_prev = pts[max(i - 1, 0)]
        x_next, y_next = pts[min(i + 1, len(pts) - 1)]
        dx, dy = x_next - x_prev, y_next - y_prev
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length
        left.append((x + nx * half, y + ny * half))
        right.append((x - nx * half, y - ny * half))
    draw.polygon(left + right[::-1], fill="white")
    for p in (pts[0], pts[-1]):
        draw.ellipse([p[0] - half, p[1] - half, p[0] + half, p[1] + half], fill="white")


def png_full(width=1200, supersample=3):
    width_ss = width * supersample
    scale = width_ss / W
    img = Image.new("RGBA", (width_ss, int(H * scale)), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw_mark(draw, scale)
    font = load_font(int(TEXT_SIZE * scale))
    text = "PREDICT"
    spacing = LETTER_SPACING * TEXT_SIZE * scale
    widths = [draw.textlength(ch, font=font) for ch in text]
    total = sum(widths) + spacing * (len(text) - 1)
    x = (width_ss - total) / 2
    y = TEXT_Y * scale - TEXT_SIZE * scale * 0.82
    for ch, w in zip(text, widths):
        draw.text((x, y), ch, font=font, fill=NAVY)
        x += w + spacing
    return img.resize((width, int(H * width / W)), Image.LANCZOS)


def png_mark(size=512, padding=44, supersample=3):
    x0, y0, x1, y1 = STEM
    right = BOWL_ARC_X + BOWL_RX
    span = max(right - x0, y1 - y0)
    size_ss, padding_ss = size * supersample, padding * supersample
    scale = (size_ss - 2 * padding_ss) / span
    img = Image.new("RGBA", (size_ss, size_ss), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    ox = padding_ss + ((size_ss - 2 * padding_ss) - (right - x0) * scale) / 2 - x0 * scale
    oy = padding_ss - y0 * scale
    draw_mark(draw, scale, (ox, oy))
    return img.resize((size, size), Image.LANCZOS)


if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    with open("static/predict-logo.svg", "w", encoding="utf-8") as handle:
        handle.write(svg_full())
    with open("static/predict-mark.svg", "w", encoding="utf-8") as handle:
        handle.write(svg_mark())
    png_full().save("static/predict-logo.png")
    png_mark().save("static/predict-mark.png")
    png_mark(180, 16).save("static/apple-touch-icon.png")
    print("static/predict-logo.svg, predict-mark.svg, predict-logo.png, predict-mark.png, apple-touch-icon.png")
