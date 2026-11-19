from dataclasses import replace

import av
import cv2
import numpy as np
import pytest

from clipper import ffmpeg
from clipper.brand_graphics import overlay_png, write_overlay
from clipper.branding import Branding, load_branding
from clipper.captions import CaptionStyle, build_ass
from clipper.reframe import Framing
from clipper.render import CPU_ENCODER, HARDWARE_ENCODERS, preferred_encoder, render_clip, render_preview
from clipper.transcribe import Word


def test_render_retries_with_cpu_when_detected_encoder_fails(tmp_path, monkeypatch):
    attempted = []

    def run(args, cwd=None):
        encoder = args[args.index("-c:v") + 1]
        attempted.append(encoder)
        if cwd is not None and encoder != CPU_ENCODER[0]:
            raise ffmpeg.FFmpegError("GPU unavailable")

    monkeypatch.setattr(ffmpeg, "run", run)
    preferred_encoder.cache_clear()
    try:
        render_clip(tmp_path / "input.mp4", 0, 1, Framing("fit", 320, 180),
                    None, tmp_path / "clip.mp4")
    finally:
        preferred_encoder.cache_clear()
    nvenc = HARDWARE_ENCODERS[0][0]
    assert attempted == [nvenc, nvenc, CPU_ENCODER[0]]


def test_template_configuration_and_alpha(tmp_path):
    configuration = tmp_path / "template.json"
    brand = load_branding(configuration)
    assert brand.mode == "horizontal"
    assert brand.size == (1080, 1920)
    configuration.write_text(
        '{"width": 270, "height": 480, "mode": "horizontal", "logo_font": "script",'
        ' "footer_visible": false, "border_color": "#00FF00"}', encoding="utf-8")
    customized = load_branding(configuration)
    png = overlay_png(customized)
    pixels = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    assert pixels.shape == (480, 270, 4)
    assert pixels[240, 135, 3] == 0
    x, _, _, _ = customized.horizontal_frame(16 / 9)
    assert pixels[240, x, 1] > 180
    assert overlay_png(customized) is png
    assert png != overlay_png(replace(customized, logo_text="Canal"))
    assert png != overlay_png(customized, 16 / 9, "Um título chamativo")
    configuration.write_text('{"width": 271}', encoding="utf-8")
    with pytest.raises(ValueError, match="even integers"):
        load_branding(configuration)


@pytest.mark.parametrize(("dimensions", "mode"), [
    ((320, 180), "horizontal"),
    ((320, 180), "existing"),
    ((320, 180), "fill"),
    ((180, 320), "horizontal"),
    ((180, 320), "fill"),
])
def test_branded_render_and_preview(tmp_path, dimensions, mode):
    src_w, src_h = dimensions
    source = tmp_path / "source.mp4"
    ffmpeg.run(["-f", "lavfi", "-i", f"color=c=blue:s={src_w}x{src_h}:r=25",
                "-f", "lavfi", "-i", "sine=frequency=480:sample_rate=48000", "-t", "2",
                "-c:v", "libx264", "-c:a", "aac", str(source)])
    framing = Framing("gameplay", src_w, src_h)
    brand = Branding(width=270, height=480, mode=mode)
    overlay = write_overlay(brand, tmp_path / "overlay.png", src_w / src_h, "Um título chamativo")
    sub = tmp_path / "captions.ass"
    sub.write_text(build_ass([Word(text="Teste", start=0.5, end=1.1)], 0.25, 1.25,
                             CaptionStyle(size=26, margin_v=120), 270, 480), encoding="utf-8")
    start, end = 0.25, 1.25
    preview = render_preview(source, start, end, framing, sub, tmp_path / "preview.png",
                             brand=brand, overlay=overlay, at=0.5)
    clean = render_preview(source, start, end, framing, None, tmp_path / "clean.png",
                           brand=brand, overlay=overlay, at=0.5)
    preview_img = cv2.imread(str(preview))
    clean_img = cv2.imread(str(clean))
    assert preview_img.shape[:2] == (480, 270)
    assert np.any(preview_img[315:380] != clean_img[315:380])
    assert np.array_equal(preview_img[:60], clean_img[:60])
    assert np.array_equal(preview_img[440:], clean_img[440:])
    assert clean_img[240, 135, 0] > 200
    result = render_clip(source, start, end, framing, sub, tmp_path / "clip.mp4",
                         brand=brand, overlay=overlay)
    with av.open(str(result)) as media:
        stream = media.streams.video[0]
        assert (stream.width, stream.height) == brand.size
        assert stream.average_rate == 25
        assert stream.codec_context.name == "h264"
        assert media.streams.audio[0].codec_context.name == "aac"
        for output_stream in (stream, media.streams.audio[0]):
            assert float(output_stream.duration * output_stream.time_base) == pytest.approx(1, abs=0.05)
    if mode == "existing":
        disabled = replace(brand, enabled=False)
        legacy = render_preview(source, start, end, framing, None, tmp_path / "legacy.png", brand=disabled)
        assert cv2.imread(str(legacy)).shape[:2] == (1920, 1080)
        assert clean_img[80, 135, 0] > 200
    if mode == "horizontal" and src_w > src_h:
        assert 80 < clean_img[95, 135, 0] < clean_img[240, 135, 0]
        assert clean_img[240, 20, 0] > 200
        _, top, _, _ = brand.horizontal_frame(src_w / src_h)
        title_y = top - round(480 * brand.title_gap) - round(max(480 * 0.025, 270 * brand.title_size * 0.85))
        blue, green, red = map(int, clean_img[title_y, 33])
        assert red > green + 20 and blue > red + 20
