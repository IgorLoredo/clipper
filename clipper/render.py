from pathlib import Path

from clipper import ffmpeg
from clipper.reframe import Framing, x_expression

OUT_W, OUT_H = 1080, 1920


def video_filter(framing: Framing, subtitles: str | None, width: int = OUT_W, height: int = OUT_H,
                 background_mode: str = "blur", frame_color: str = "0x00E5FF@0.95") -> str:
    subs = f",ass={subtitles}" if subtitles else ""
    if framing.mode == "track":
        x = x_expression(framing.points)
        return (f"[0:v]crop=w={framing.crop_w}:h={framing.src_h}:x='{x}':y=0,"
                f"scale={width}:{height},setsar=1{subs}[v]")
    if framing.mode == "fit":
        return (f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1{subs}[v]")
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
                    f"setsar=1{subs}[v]")
        if background_mode == "cover":
            return (f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
                    f"crop={width}:{height},setsar=1{subs}[v]")
        # Full source frame on a blurred vertical canvas: all of a 16:9 game capture stays visible.
        return (f"[0:v]split[a][b];"
                f"[a]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},boxblur=24:4[bg];"
                f"[b]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1{subs}[v]")
    return (f"[0:v]split[a][b];"
            f"[a]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},boxblur=24:4[bg];"
            f"[b]scale={width}:-2[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1{subs}[v]")


def render_clip(video: Path, start: float, end: float, framing: Framing,
                subtitles: Path | None, output: Path, width: int = OUT_W, height: int = OUT_H,
                background_mode: str = "blur", frame_color: str = "0x00E5FF@0.95") -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    sub_name = None
    if subtitles is not None:
        if subtitles.parent.resolve() != output.parent.resolve():
            raise ValueError("subtitles file must live next to the output clip")
        sub_name = subtitles.name
    args = [
        "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(video.resolve()),
        "-filter_complex", video_filter(framing, sub_name, width, height, background_mode, frame_color),
        "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
        output.name,
    ]
    ffmpeg.run(args, cwd=output.parent)
    return output
