from functools import lru_cache
from pathlib import Path
import re

from clipper import ffmpeg
from clipper.branding import Branding
from clipper.reframe import Framing, x_expression

OUT_W, OUT_H = 1080, 1920
SPLIT_WEBCAM_HEIGHT = 720
HARDWARE_ENCODERS = (
    ("h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "20", "-b:v", "0"),
    ("h264_qsv", "-preset", "fast", "-global_quality", "20"),
    ("h264_amf", "-quality", "balanced", "-rc", "cqp", "-qp_i", "20", "-qp_p", "20"),
)
CPU_ENCODER = ("libx264", "-preset", "veryfast", "-crf", "20")
_FAILED_ENCODERS: set[str] = set()
_ASS_SPECIAL = re.compile(r"[':\\\[\],;]")


@lru_cache(maxsize=1)
def preferred_encoder() -> tuple[str, ...]:
    _FAILED_ENCODERS.clear()
    for encoder in HARDWARE_ENCODERS:
        try:
            ffmpeg.run(["-f", "lavfi", "-i", "color=size=320x240", "-frames:v", "1",
                        "-c:v", *encoder, "-pix_fmt", "yuv420p", "-f", "null", "-"])
        except ffmpeg.FFmpegError:
            continue
        return encoder
    return CPU_ENCODER


def _escape_ass_path(name: str) -> str:
    if not _ASS_SPECIAL.search(name):
        return name
    escaped = (name.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
               .replace("[", "\\[").replace("]", "\\]").replace(",", "\\,").replace(";", "\\;"))
    return f"'{escaped}'"


def blurred_source(width: int, height: int) -> str:
    radius = min(24, width // 4, height // 4)
    return (f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},boxblur={radius}:4")


def rounded_foreground(width: int, height: int, radius: int) -> str:
    scale = f"[b]scale={width}:{height},setsar=1"
    if not radius:
        return f"{scale}[fg]"
    # Compute the alpha mask from a single still, then reuse it across the clip.
    dx = f"max({radius}-X,max(0,X-(W-1-{radius})))"
    dy = f"max({radius}-Y,max(0,Y-(H-1-{radius})))"
    alpha = f"clip(({radius}+0.5-hypot({dx},{dy}))*255,0,255)"
    return (f"{scale},format=yuva420p[sharp];"
            f"nullsrc=s={width}x{height}:r=1:d=1,format=gray,geq=lum='{alpha}'[mask];"
            "[sharp][mask]alphamerge[fg]")


def video_filter(framing: Framing, subtitles: str | None, width: int = OUT_W,
                 height: int = OUT_H, background_mode: str = "blur",
                 frame_color: str = "0x00E5FF@0.95", *, output: str = "v") -> str:
    subs = f",ass={_escape_ass_path(subtitles)}" if subtitles else ""
    if framing.mode == "gameplay-split":
        if framing.webcam is None:
            raise ValueError("split gameplay requires a webcam crop")
        x, y, cam_w, cam_h = framing.webcam
        top = round(height * SPLIT_WEBCAM_HEIGHT / OUT_H / 2) * 2
        bottom = height - top
        return (f"[0:v]split[camera][game];"
                f"[camera]crop={cam_w}:{cam_h}:{x}:{y}:exact=1,"
                f"scale={width}:{top}:force_original_aspect_ratio=increase:force_divisible_by=2,"
                f"crop={width}:{top},setsar=1[upper];"
                f"[game]scale={width}:{bottom}:force_original_aspect_ratio=increase:force_divisible_by=2,"
                f"crop={width}:{bottom},setsar=1[lower];"
                f"[upper][lower]vstack{subs}[{output}]")
    if framing.mode == "podcast-split":
        if framing.boxes is None:
            raise ValueError("podcast split requires two face boxes")
        (x1, y1, w1, h1), (x2, y2, w2, h2) = framing.boxes
        top = (height // 2) - ((height // 2) % 2)
        bottom = height - top
        return (f"[0:v]split[p1][p2];"
                f"[p1]crop={w1}:{h1}:{x1}:{y1}:exact=1,"
                f"scale={width}:{top}:force_original_aspect_ratio=increase:force_divisible_by=2,"
                f"crop={width}:{top},setsar=1[upper];"
                f"[p2]crop={w2}:{h2}:{x2}:{y2}:exact=1,"
                f"scale={width}:{bottom}:force_original_aspect_ratio=increase:force_divisible_by=2,"
                f"crop={width}:{bottom},setsar=1[lower];"
                f"[upper][lower]vstack{subs}[{output}]")
    if framing.mode == "track":
        x = x_expression(framing.points)
        return (f"[0:v]crop=w={framing.crop_w}:h={framing.src_h}:x='{x}':y=0,"
                f"scale={width}:{height},setsar=1{subs}[{output}]")
    if framing.mode == "fit":
        return (f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1{subs}[{output}]")
    if framing.mode in {"gameplay", "anime", "mass"}:
        if background_mode == "framed":
            frame_x = max(8, int(width * 0.018))
            frame_y = int(height * 0.32)
            frame_w = width - frame_x * 2
            frame_h = int(height * 0.34)
            return (f"[0:v]split[bg][src];"
                    f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},boxblur=24:4[blur];"
                    f"[src]scale={frame_w}:{frame_h}:force_original_aspect_ratio=decrease[fg];"
                    f"[blur][fg]overlay={frame_x}+({frame_w}-overlay_w)/2:{frame_y}+({frame_h}-overlay_h)/2,"
                    f"drawbox=x={frame_x-6}:y={frame_y-6}:w={frame_w+12}:h={frame_h+12}:color={frame_color}:t=4,"
                    f"drawbox=x=48:y={int(height*0.115)}:w={width-96}:h={int(height*0.105)}:color=0x031626@0.88:t=fill,"
                    f"drawbox=x=48:y={int(height*0.695)}:w={width-96}:h={int(height*0.17)}:color=0x031626@0.88:t=fill,"
                    f"setsar=1{subs}[{output}]")
        if background_mode == "cover":
            return (f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
                    f"crop={width}:{height},setsar=1{subs}[{output}]")
        # Full source frame on a blurred vertical canvas: all of a 16:9 game capture stays visible.
        return (f"[0:v]split[a][b];"
                f"[a]{blurred_source(width, height)}[bg];"
                f"[b]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1{subs}[{output}]")
    return (f"[0:v]split[a][b];"
            f"[a]{blurred_source(width, height)}[bg];"
            f"[b]scale={width}:-2[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1{subs}[{output}]")


def branded_filter(framing: Framing, subtitles: str | None, brand: Branding) -> str:
    w, h = brand.size
    subs = f",ass={_escape_ass_path(subtitles)}" if subtitles else ""
    if brand.mode == "horizontal":
        x, y, frame_w, frame_h = brand.horizontal_frame(framing.src_w / framing.src_h)
        radius = min(round(w * brand.border_radius), (frame_w - 1) // 2, (frame_h - 1) // 2)
        return (f"[0:v]split[a][b];[a]{blurred_source(w, h)},eq=brightness=-0.12[bg];"
                f"{rounded_foreground(frame_w, frame_h, radius)};"
                f"[bg][fg]overlay={x}:{y},setsar=1[base];"
                f"[base][1:v]overlay=eof_action=repeat{subs}[v]")
    x, y, area_w, area_h = brand.video_area
    if brand.mode == "existing" or (brand.mode == "fill" and framing.mode == "track"):
        layout = video_filter(framing, None, width=w, height=h, output="layout")
        zoom = brand.zoom if brand.mode == "existing" else max(1, brand.zoom)
        scaled_w = max(2, round(area_w * zoom / 2) * 2)
        scaled_h = max(2, round(area_h * zoom / 2) * 2)
        if zoom == 1 and not (x or y):
            geometry = "[layout]null"
        else:
            aspect = "increase" if brand.mode == "fill" else "decrease"
            geometry = (f"[layout]scale={scaled_w}:{scaled_h}:force_original_aspect_ratio={aspect}"
                        ":force_divisible_by=2")
    else:
        layout = ""
        zoom = max(1, brand.zoom)
        scaled_w = max(2, round(area_w * zoom / 2) * 2)
        scaled_h = max(2, round(area_h * zoom / 2) * 2)
        geometry = f"[0:v]scale={scaled_w}:{scaled_h}:force_original_aspect_ratio=increase:force_divisible_by=2"
    color = "black"
    if brand.background_enabled:
        rgb = brand.background_color[1:]
        color = "0x" + "".join(f"{round(int(rgb[i:i + 2], 16) * brand.background_opacity):02x}"
                                for i in (0, 2, 4))
    px, py = (1 + brand.offset_x) / 2, (1 + brand.offset_y) / 2
    geometry += (f",crop='min(iw,{area_w})':'min(ih,{area_h})':'(iw-ow)*{px}':'(ih-oh)*{py}',"
                 f"pad={w}:{h}:'{x}+({area_w}-iw)*{px}':'{y}+({area_h}-ih)*{py}':color={color},"
                 "setsar=1[base]")
    return ";".join(part for part in (
        layout, geometry, f"[base][1:v]overlay=eof_action=repeat{subs}[v]",
    ) if part)


def _audio_filter_args(duration: float, normalize_audio: bool) -> list[str]:
    if not normalize_audio:
        return []
    filters = ["loudnorm=I=-14:TP=-1.5:LRA=11", "afade=t=in:st=0:d=0.04"]
    if duration > 0.2:
        filters.append(f"afade=t=out:st={max(0.0, duration - 0.04):.3f}:d=0.04")
    return ["-af", ",".join(filters)]


def _draw_safe_zones(image_path: Path) -> None:
    import cv2
    import numpy as np

    data = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return
    h, w = img.shape[:2]
    overlay = img.copy()
    # Top UI bar (~10%), bottom caption/description bar (~20%), right social action bar (~15%)
    top_h = round(h * 0.10)
    bottom_y = round(h * 0.80)
    right_x = round(w * 0.85)
    cv2.rectangle(overlay, (0, 0), (w, top_h), (0, 0, 220), -1)
    cv2.rectangle(overlay, (0, bottom_y), (w, h), (0, 0, 220), -1)
    cv2.rectangle(overlay, (right_x, top_h), (w, bottom_y), (0, 140, 255), -1)
    blended = cv2.addWeighted(overlay, 0.28, img, 0.72, 0)
    cv2.rectangle(blended, (0, top_h), (right_x, bottom_y), (0, 255, 0), 2)
    cv2.putText(blended, "SAFE ZONE", (18, top_h + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", blended)
    if ok:
        buf.tofile(str(image_path))


def _render_args(video: Path, start: float, end: float, framing: Framing,
                 subtitles: Path | None, directory: Path, brand: Branding | None,
                 overlay: Path | None, width: int, height: int,
                 background_mode: str, frame_color: str) -> list[str]:
    sub_name = None
    if subtitles is not None:
        if subtitles.parent.resolve() != directory.resolve():
            raise ValueError("subtitles file must live next to the output clip")
        sub_name = subtitles.name
    args = ["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(video.resolve())]
    graph = video_filter(framing, sub_name, width, height, background_mode, frame_color)
    if brand is not None and brand.enabled:
        if overlay is None:
            raise ValueError("ViraCortes artwork is required")
        args += ["-i", str(overlay.resolve())]
        graph = branded_filter(framing, sub_name, brand)
    return [*args, "-filter_complex", graph, "-map", "[v]"]


def render_clip(video: Path, start: float, end: float, framing: Framing,
                subtitles: Path | None, output: Path, width: int = OUT_W, height: int = OUT_H,
                background_mode: str = "blur", frame_color: str = "0x00E5FF@0.95", *,
                brand: Branding | None = None,
                overlay: Path | None = None,
                normalize_audio: bool = False) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    args = [
        *_render_args(video, start, end, framing, subtitles, output.parent, brand, overlay,
                     width, height, background_mode, frame_color),
        "-map", "0:a?",
        *_audio_filter_args(end - start, normalize_audio),
    ]
    tail = ["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", output.name]
    encoder = preferred_encoder()
    if encoder[0] in _FAILED_ENCODERS:
        encoder = CPU_ENCODER
    print(f"[render] encoder {encoder[0]}", flush=True)
    try:
        ffmpeg.run([*args, "-c:v", *encoder, *tail], cwd=output.parent)
    except ffmpeg.FFmpegError:
        if encoder == CPU_ENCODER:
            raise
        _FAILED_ENCODERS.add(encoder[0])
        print(f"[render] {encoder[0]} failed; retrying with {CPU_ENCODER[0]}", flush=True)
        ffmpeg.run([*args, "-c:v", *CPU_ENCODER, *tail], cwd=output.parent)
    return output


def render_preview(video: Path, start: float, end: float, framing: Framing,
                   subtitles: Path | None, output: Path, *, brand: Branding | None = None,
                   overlay: Path | None = None, at: float = 0,
                   width: int = OUT_W, height: int = OUT_H, background_mode: str = "blur",
                   frame_color: str = "0x00E5FF@0.95", safe_zone: bool = False) -> Path:
    """Render one frame at a clip-relative time, keeping tracking and ASS timestamps."""
    if not 0 <= at < end - start:
        raise ValueError("preview time must be within the clip")
    output.parent.mkdir(parents=True, exist_ok=True)
    args = _render_args(video, start, end, framing, subtitles, output.parent, brand, overlay,
                     width, height, background_mode, frame_color)
    ffmpeg.run([*args, "-ss", f"{at:.3f}", "-frames:v", "1", "-update", "1", output.name],
               cwd=output.parent)
    if safe_zone:
        _draw_safe_zones(output)
    return output
