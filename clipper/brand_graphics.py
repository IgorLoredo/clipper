"""Draw reusable transparent ViraCortes artwork with OpenCV."""
from functools import lru_cache
from pathlib import Path
import unicodedata

import cv2
import numpy as np

from clipper.branding import Branding


def bgr(color: str) -> tuple[int, int, int]:
    return tuple(int(color[index:index + 2], 16) for index in (5, 3, 1))


def rounded(mask: np.ndarray, box: tuple[int, int, int, int], radius: int) -> None:
    x, y, right, bottom = box
    if x > right or y > bottom:
        return
    r = min(radius, (right - x) // 2, (bottom - y) // 2)
    cv2.rectangle(mask, (x + r, y), (right - r, bottom), 255, -1)
    cv2.rectangle(mask, (x, y + r), (right, bottom - r), 255, -1)
    for cx in (x + r, right - r):
        for cy in (y + r, bottom - r):
            cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)


def composite(canvas: np.ndarray, mask: np.ndarray, color: str, opacity: float = 1) -> None:
    a = mask.astype(np.float32) * (opacity / 255)
    retained = canvas[:, :, 3].astype(np.float32) / 255 * (1 - a)
    alpha = a + retained
    visible = alpha > 0
    if not visible.any():
        return
    colors = canvas[:, :, :3][visible].astype(np.float32)
    canvas[:, :, :3][visible] = np.rint(
        (colors * retained[visible, None] + np.array(bgr(color)) * a[visible, None]) / alpha[visible, None]
    ).astype(np.uint8)
    canvas[:, :, 3] = np.rint(alpha * 255).astype(np.uint8)


def letter(mask: np.ndarray, text: str, center: tuple[int, int], height: float, face: int,
           max_width: float, tracking: int = 0) -> None:
    if not text or height <= 0 or max_width <= 0:
        return
    accents = {}
    plain = []
    for char in unicodedata.normalize("NFKD", text):
        if char.isascii():
            plain.append(char)
        elif plain and unicodedata.combining(char):
            accents[len(plain) - 1] = char
    text = "".join(plain)
    if not text:
        return
    weight = max(1, round(height * 0.094))
    scale = cv2.getFontScaleFromHeight(face, max(1, round(height)), weight)
    width = cv2.getTextSize(text, face, scale, weight)[0][0]
    scale *= min(1, max_width / max(1, width + tracking * (len(text) - 1)))
    tracking = min(tracking, max(0, int((max_width - cv2.getTextSize(text, face, scale, weight)[0][0]) / len(text))))
    chunks = list(text) if tracking else [text]
    widths = [cv2.getTextSize(chunk, face, scale, weight)[0][0] for chunk in chunks]
    glyph_height = cv2.getTextSize(text, face, scale, weight)[0][1]
    x = center[0] - (sum(widths) + tracking * (len(chunks) - 1)) // 2
    left = x
    baseline = center[1] + glyph_height // 2
    for chunk, width in zip(chunks, widths):
        cv2.putText(mask, chunk, (x, baseline), face, scale, 255, weight, cv2.LINE_AA)
        x += width + tracking
    for index, accent in accents.items():
        if tracking:
            ax = left + sum(widths[:index]) + index * tracking + widths[index] // 2
        else:
            glyph_width = cv2.getTextSize(text[:index + 1], face, scale, weight)[0][0]
            char_width = cv2.getTextSize(text[index], face, scale, weight)[0][0]
            ax = left + glyph_width - char_width // 2
        ascender = 1.0 if text[index].isupper() else 0.75
        ay = baseline - round(glyph_height * ascender) - max(1, weight)
        step = max(1, round(glyph_height * 0.12))
        marks = {
            "\u0300": [(-1, -1), (1, 0)],
            "\u0301": [(-1, 0), (1, -1)],
            "\u0302": [(-1, 0), (0, -1), (1, 0)],
            "\u0303": [(-1, 0), (0, -1), (1, 0), (2, -1)],
            "\u0327": [(0, 0), (1, 1), (0, 2)],
        }
        if accent not in marks:
            continue
        if accent == "\u0327":
            ay = baseline
        points = np.array([(ax + dx * step, ay + dy * step) for dx, dy in marks[accent]], dtype=np.int32)
        cv2.polylines(mask, [points], False, 255, max(1, weight // 2), cv2.LINE_AA)


def draw_panels(canvas: np.ndarray, brand: Branding, opening: np.ndarray) -> None:
    w, h = brand.size
    panels = 255 - opening
    if brand.background_enabled:
        composite(canvas, panels, brand.background_color, brand.background_opacity)
    if brand.theme == "minimal":
        return
    y, x = np.ogrid[:h, :w]
    glow = np.maximum(
        np.exp(-((x / w / 0.38) ** 2 + ((y / h - 0.015) / 0.14) ** 2)),
        np.exp(-(((x / w - 1) / 0.38) ** 2 + ((y / h - 0.985) / 0.14) ** 2)),
    )
    composite(canvas, np.uint8(panels * glow), brand.neon_color, 0.55)
    details = np.zeros_like(opening)
    for points in (
        [(0.02, 0), (0.115, 0), (0.28, 0.15), (0.17, 0.15)],
        [(0.73, 1), (0.64, 1), (1, 0.75), (1, 0.80)],
    ):
        cv2.fillConvexPoly(details, np.array([(round(px * w), round(py * h)) for px, py in points]),
                           255, cv2.LINE_AA)
    composite(canvas, cv2.bitwise_and(details, panels), brand.neon_color, 0.2)


def draw_play(canvas: np.ndarray, brand: Branding, x: int, y: int, size: float) -> None:
    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    size = max(1, round(size))
    points = np.array([(x + size // 5, y), (x + size, y + size // 2), (x, y + size)], dtype=np.int32)
    stroke = max(1, round(size * 0.18))
    cv2.polylines(mask, [points], True, 255, stroke, cv2.LINE_AA)
    for point in points:
        cv2.circle(mask, tuple(point), stroke // 2, 255, -1, cv2.LINE_AA)
    cut_y = y + round(size * 0.52)
    cv2.line(mask, (x - stroke, cut_y), (x + size // 3, cut_y - stroke // 4),
             0, max(1, stroke // 3), cv2.LINE_AA)
    composite(canvas, mask, brand.neon_color)
    shine = np.zeros_like(mask)
    cv2.line(shine, (x + size // 9, y + size), (x + size, y + size // 2),
             255, stroke, cv2.LINE_AA)
    composite(canvas, cv2.bitwise_and(shine, mask), brand.border_color, 0.8)


def draw_titles(canvas: np.ndarray, brand: Branding) -> None:
    w, h = brand.size
    s = w * brand.scale
    name = "SCRIPT_SIMPLEX" if brand.logo_font == "script" else brand.logo_font.upper()
    face = getattr(cv2, "FONT_HERSHEY_" + name)
    white = np.zeros(canvas.shape[:2], dtype=np.uint8)
    purple = np.zeros_like(white)
    x, y = round(w * brand.logo_x), round(h * brand.logo_y)
    title_size = s * brand.logo_size
    label_y = y + round(title_size * 0.75)
    pill_w = round(s * 0.141)
    pill_h = round(title_size * 0.18)
    rounded(purple, (x - pill_w, label_y - pill_h, x + pill_w, label_y + pill_h), pill_h)
    composite(canvas, purple, brand.neon_color)
    purple.fill(0)
    letter(white, brand.logo_text, (x, y - round(title_size * 0.13)), title_size, face | cv2.FONT_ITALIC,
           s * 0.42)
    letter(white, brand.logo_label, (x, label_y), title_size * 0.23, face, pill_w * 1.75, round(s * 0.008))
    if brand.logo_icon:
        draw_play(canvas, brand, x - round(s * 0.253), y - round(s * 0.048), s * 0.085)
    if brand.footer_visible:
        fy = round(h * brand.footer_y)
        size = s * brand.footer_size
        bar = round(w * brand.footer_width / 2)
        label_y = fy + round(size * 2.55)
        letter(white, brand.footer_text, (w // 2, fy), size, face, w * 0.87, round(s * 0.006))
        letter(white, brand.footer_label, (w // 2, label_y), size * 0.67, face, w * 0.72, round(s * 0.003))
        cv2.line(purple, (w // 2 - bar, fy + round(size * 1.38)), (w // 2 + bar, fy + round(size * 1.38)),
                 255, max(1, round(s * 0.004)), cv2.LINE_AA)
        for sign in (-1, 1):
            for i, amplitude in enumerate((0.010, 0.016, 0.022, 0.013, 0.007)):
                bx = round(w * (0.5 + sign * (0.491 - 0.016 * i)))
                dy = round(h * amplitude)
                cv2.line(purple, (bx, fy - dy), (bx, fy + dy), 255, max(1, round(s * 0.005)), cv2.LINE_AA)
    composite(canvas, purple, brand.neon_color, 0.76)
    composite(canvas, white, brand.logo_color)


def draw_headline(canvas: np.ndarray, brand: Branding, text: str, frame_top: int) -> None:
    if not brand.title_visible or not text:
        return
    w, h = brand.size
    mask = np.zeros((h, w), dtype=np.uint8)
    outline = np.zeros_like(mask)
    letters = np.zeros_like(mask)
    half_width = round(w * brand.title_width / 2)
    size = w * brand.scale * brand.title_size
    face = cv2.FONT_HERSHEY_DUPLEX | cv2.FONT_ITALIC
    max_width = max(1, 2 * half_width - w * 0.14)
    weight = max(1, round(size * 0.094))
    scale = cv2.getFontScaleFromHeight(face, max(1, round(size)), weight)
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    lines = [text]
    words = text.split()
    if cv2.getTextSize(plain, face, scale, weight)[0][0] > max_width and len(words) > 1:
        split = min(range(1, len(words)),
                    key=lambda index: abs(len(" ".join(words[:index])) - len(" ".join(words[index:]))))
        lines = [" ".join(words[:split]), " ".join(words[split:])]
    line_step = round(size * 1.34)
    half_height = round(max(h * 0.025, size * 0.85) + line_step * (len(lines) - 1) / 2)
    x = w // 2
    y = frame_top - round(h * brand.title_gap) - half_height
    bevel = round(half_height * 0.65)
    polygon = np.array([
        (x - half_width + bevel, y - half_height), (x + half_width, y - half_height),
        (x + half_width - bevel, y + half_height), (x - half_width, y + half_height),
    ], dtype=np.int32)
    cv2.fillConvexPoly(mask, polygon, 255, cv2.LINE_AA)
    cv2.polylines(outline, [polygon], True, 255, max(1, round(w * 0.003)), cv2.LINE_AA)
    if brand.neon_blur:
        glow = cv2.GaussianBlur(outline, (0, 0), max(0.1, w * brand.neon_blur))
        composite(canvas, glow, brand.neon_color, brand.neon_opacity)
    composite(canvas, mask, brand.title_color, 0.96)
    composite(canvas, outline, brand.border_color)
    for index, line in enumerate(lines):
        line_y = round(y + (index - (len(lines) - 1) / 2) * line_step)
        letter(letters, line, (x, line_y), size, face, max_width)
    composite(canvas, letters, brand.logo_color)


@lru_cache(maxsize=8)
def overlay_png(brand: Branding, source_aspect: float = 16 / 9, title: str = "") -> bytes:
    w, h = brand.size
    outer = np.zeros((h, w), dtype=np.uint8)
    inner = np.zeros_like(outer)
    canvas = np.zeros((h, w, 4), dtype=np.uint8)
    window = brand.window
    if brand.mode == "horizontal":
        left, top, width, height = brand.horizontal_frame(source_aspect)
        window = left, top, left + width - 1, top + height - 1
    x, y, right, bottom = window
    radius = round(w * brand.border_radius)
    t = max(1, round(w * brand.border_width)) if brand.border_width else 0
    rounded(outer, window, radius)
    rounded(inner, (x + t, y + t, right - t, bottom - t), max(0, radius - t))
    draw_panels(canvas, brand, outer)
    stroke = cv2.subtract(outer, inner)
    if brand.neon_blur and brand.neon_opacity and brand.neon_intensity:
        glow = cv2.GaussianBlur(stroke, (0, 0), max(0.1, w * brand.neon_blur))
        glow = np.uint8(np.minimum(255, glow.astype(float) * 3 * brand.neon_intensity))
        composite(canvas, glow, brand.neon_color, brand.neon_opacity)
    composite(canvas, stroke, brand.border_color, brand.border_opacity)
    draw_titles(canvas, brand)
    if brand.mode == "horizontal":
        draw_headline(canvas, brand, brand.title_text or title, y)
    canvas[:, :, 3] = np.rint(canvas[:, :, 3] * (1 - inner.astype(np.float32) / 255)).astype(np.uint8)
    ok, png = cv2.imencode(".png", canvas)
    if not ok:
        raise ValueError("could not encode ViraCortes artwork")
    return png.tobytes()


def write_overlay(brand: Branding, output: Path, source_aspect: float = 16 / 9, title: str = "") -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(overlay_png(brand, source_aspect, title))
    return output
