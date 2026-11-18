from pathlib import Path

from clipper import ffmpeg
from clipper.reframe import Framing, x_expression

OUT_W, OUT_H = 1080, 1920


def video_filter(framing: Framing, subtitles: str | None) -> str:
    subs = f",ass={subtitles}" if subtitles else ""
    if framing.mode == "track":
        x = x_expression(framing.points)
        return (f"[0:v]crop=w={framing.crop_w}:h={framing.src_h}:x='{x}':y=0,"
                f"scale={OUT_W}:{OUT_H},setsar=1{subs}[v]")
    if framing.mode == "fit":
        return (f"[0:v]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
                f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2,setsar=1{subs}[v]")
    if framing.mode == "gameplay":
        # Full source frame on a blurred vertical canvas: all of a 16:9 game capture stays visible.
        return (f"[0:v]split[a][b];"
                f"[a]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,crop={OUT_W}:{OUT_H},boxblur=24:4[bg];"
                f"[b]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1{subs}[v]")
    return (f"[0:v]split[a][b];"
            f"[a]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,crop={OUT_W}:{OUT_H},boxblur=24:4[bg];"
            f"[b]scale={OUT_W}:-2[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1{subs}[v]")


def render_clip(video: Path, start: float, end: float, framing: Framing,
                subtitles: Path | None, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    sub_name = None
    if subtitles is not None:
        if subtitles.parent.resolve() != output.parent.resolve():
            raise ValueError("subtitles file must live next to the output clip")
        sub_name = subtitles.name
    args = [
        "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(video.resolve()),
        "-filter_complex", video_filter(framing, sub_name),
        "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
        output.name,
    ]
    ffmpeg.run(args, cwd=output.parent)
    return output
