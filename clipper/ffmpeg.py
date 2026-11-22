import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


class FFmpegError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise FFmpegError("ffmpeg not found: install it or `pip install imageio-ffmpeg`") from exc


def run(args: Sequence[str], cwd: Path | None = None) -> None:
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise FFmpegError(f"ffmpeg failed ({proc.returncode}): {proc.stderr.strip()[-2000:]}")


def probe_duration(path: Path) -> float:
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
    if not match:
        raise FFmpegError(f"could not read duration of {path}")
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def iter_audio_seconds(path: Path, sample_rate: int = 8000, *, check: bool = False,
                       align: bool = False) -> Iterator[np.ndarray]:
    cmd = [
        ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        *(["-af", "aresample=async=1:first_pts=0"] if align else []), "-f", "s16le", "-",
    ]
    bytes_per_second = sample_rate * 2
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(bytes_per_second)
            if not chunk:
                break
            usable = len(chunk) - (len(chunk) % 2)
            yield np.frombuffer(chunk[:usable], dtype=np.int16)
        if check and proc.wait():
            raise FFmpegError("could not decode an audio stream")
