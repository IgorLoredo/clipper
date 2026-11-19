import urllib.request
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TARGET_RATIO = 9 / 16
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
DEFAULT_FACE_MODEL = Path.home() / ".cache" / "clipper" / "face_detection_yunet_2023mar.onnx"


@dataclass
class Framing:
    mode: str
    src_w: int
    src_h: int
    crop_w: int = 0
    points: list[tuple[float, int]] = field(default_factory=list)
    webcam: tuple[int, int, int, int] | None = None


def plan_gameplay_split(video: Path, start: float, face_model: str,
                        webcam: tuple[int, int, int, int] | None = None) -> Framing:
    from clipper.ffmpeg import ffmpeg_exe

    result = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-ss", f"{start:.3f}", "-i", str(video),
         "-frames:v", "1", "-f", "image2pipe", "-c:v", "ppm", "-"],
        capture_output=True,
    )
    if result.returncode or not result.stdout:
        raise ValueError(f"could not extract the opening frame from {video}")
    _, dimensions, _, pixels = result.stdout.split(b"\n", 3)
    src_w, src_h = map(int, dimensions.split())
    if webcam is None:
        try:
            detect = _detector(face_model)
        except ImportError as exc:
            raise ValueError("webcam detection requires OpenCV; use --webcam-crop X,Y,W,H") from exc
        rgb = np.frombuffer(pixels, dtype=np.uint8).reshape(src_h, src_w, 3)
        faces = [face for face in detect(rgb[:, :, ::-1].copy())
                 if face[0] + face[2] / 2 < 0.35 * src_w
                 or face[0] + face[2] / 2 > 0.65 * src_w
                 or face[1] + face[3] / 2 < 0.28 * src_h
                 or face[1] + face[3] / 2 > 0.72 * src_h]
        if not faces:
            raise ValueError("no webcam face found in the opening frame; use --webcam-crop X,Y,W,H")
        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        crop_w = min(even(src_w), max(2, even(w * 2.5)))
        crop_h = min(even(src_h), max(2, even(h * 1.9)))
        left = even(np.clip(x + w / 2 - crop_w / 2, 0, src_w - crop_w))
        top = even(np.clip(y + h * 0.68 - crop_h / 2, 0, src_h - crop_h))
        webcam = left, top, crop_w, crop_h
    x, y, w, h = webcam
    if min(x, y) < 0 or min(w, h) < 2 or x + w > src_w or y + h > src_h:
        raise ValueError(f"webcam crop must fit inside the {src_w}x{src_h} source frame")
    print(f"[layout] webcam crop {x},{y},{w},{h}", flush=True)
    return Framing("gameplay-split", src_w, src_h, webcam=webcam)


GAMEPLAY_PATTERN = re.compile(
    r"\b(gameplay|jogando|walkthrough|playthrough|speedrun|game\s*play|gta\s*(?:v|vi|5|6)?|"
    r"grand\s+theft\s+auto|fortnite|minecraft|roblox|valorant|call\s+of\s+duty|cs2|counter\s*strike|"
    r"league\s+of\s+legends|lol\b|fifa|ea\s*sports|free\s*fire)\b",
    re.IGNORECASE,
)
ANIME_PATTERN = re.compile(
    r"\b(anime|anim[eé]|mang[aá]|otaku|dublado|legendado|epis[oó]dio|ep\.?)\b",
    re.IGNORECASE,
)
MASS_PATTERN = re.compile(
    r"\b(santa\s+missa|missa|homilia|evangelho|liturgia|celebra[cç][aã]o|padre|par[oó]quia)\b",
    re.IGNORECASE,
)


def resolve_content_type(title: str, requested: str = "auto") -> str:
    """Choose a safe default layout; manual selection always takes precedence."""
    if requested not in {"auto", "podcast", "gameplay", "anime", "mass"}:
        raise ValueError("content type must be auto, podcast, gameplay, anime or mass")
    if requested != "auto":
        return requested
    if MASS_PATTERN.search(title or ""):
        return "mass"
    if ANIME_PATTERN.search(title or ""):
        return "anime"
    return "gameplay" if GAMEPLAY_PATTERN.search(title or "") else "podcast"


def even(value: float) -> int:
    v = int(value)
    return v - (v % 2)


def fill_gaps(samples: list[float | None]) -> list[float] | None:
    known = [i for i, v in enumerate(samples) if v is not None]
    if not known:
        return None
    filled = []
    for i, value in enumerate(samples):
        if value is not None:
            filled.append(value)
        else:
            nearest = min(known, key=lambda k: abs(k - i))
            filled.append(samples[nearest])
    return filled


def median_filter(values: list[float], window: int = 5) -> list[float]:
    half = window // 2
    return [float(np.median(values[max(0, i - half): i + half + 1])) for i in range(len(values))]


def camera_path(times: list[float], centers: list[float], deadzone: float = 0.12,
                min_hold: float = 1.5) -> list[tuple[float, float]]:
    smooth = median_filter(centers)
    position, last_move = smooth[0], times[0]
    path = [(times[0], position)]
    for t, c in zip(times[1:], smooth[1:]):
        if abs(c - position) > deadzone and t - last_move >= min_hold:
            position, last_move = c, t
            path.append((t, position))
    return path


def to_points(path: list[tuple[float, float]], src_w: int, crop_w: int,
              transition: float = 0.35) -> list[tuple[float, int]]:
    def x_for(center: float) -> int:
        return int(np.clip(center * src_w - crop_w / 2, 0, src_w - crop_w))

    points = [(0.0, x_for(path[0][1]))]
    for t, center in path[1:]:
        x = x_for(center)
        if x == points[-1][1]:
            continue
        points.append((round(t, 2), points[-1][1]))
        points.append((round(t + transition, 2), x))
    return points


def x_expression(points: list[tuple[float, int]]) -> str:
    if len(points) == 1:
        return str(points[0][1])
    expr = str(points[-1][1])
    for (t0, x0), (t1, x1) in reversed(list(zip(points, points[1:]))):
        if x0 == x1:
            segment = str(x0)
        else:
            segment = f"{x0}+({x1 - x0})*(t-{t0})/{round(t1 - t0, 3)}"
        expr = f"if(lt(t,{t1}),{segment},{expr})"
    return expr


def ensure_face_model(face_model: str) -> Path | None:
    path = Path(face_model) if face_model else DEFAULT_FACE_MODEL
    if path.is_file():
        return path
    if face_model:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        print("[reframe] downloading YuNet face model", flush=True)
        urllib.request.urlretrieve(YUNET_URL, path)
        return path
    except Exception as exc:
        print(f"[reframe] could not download YuNet ({exc}); falling back", flush=True)
        path.unlink(missing_ok=True)
        return None


def _detector(face_model: str):
    import cv2

    model = ensure_face_model(face_model)
    if model is not None and hasattr(cv2, "FaceDetectorYN"):
        yunet = cv2.FaceDetectorYN.create(str(model), "", (320, 320), 0.7)

        def detect(frame):
            yunet.setInputSize((frame.shape[1], frame.shape[0]))
            _, faces = yunet.detect(frame)
            return [] if faces is None else [tuple(f[:4]) for f in faces]

        return detect

    if not hasattr(cv2, "CascadeClassifier"):
        return lambda frame: []

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return [tuple(f) for f in cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(36, 36))]

    return detect


def sample_faces(video: Path, start: float, end: float, face_model: str,
                 step: float = 0.5, analysis_width: int = 640) -> tuple[int, int, list[float], list[float | None]]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detect = _detector(face_model)
    scale = analysis_width / src_w if src_w > analysis_width else 1.0
    times, centers = [], []
    t = start
    while t < end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        faces = detect(frame)
        if faces:
            x, _, w, _ = max(faces, key=lambda f: f[2] * f[3])
            centers.append((x + w / 2) / frame.shape[1])
        else:
            centers.append(None)
        times.append(t - start)
        t += step
    cap.release()
    return src_w, src_h, times, centers


def plan(video: Path, start: float, end: float, face_model: str = "", min_face_ratio: float = 0.4,
         content_type: str = "podcast") -> Framing:
    if content_type in {"gameplay", "anime", "mass"}:
        import cv2
        cap = cv2.VideoCapture(str(video))
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if src_w <= 0 or src_h <= 0:
            raise ValueError(f"could not read video dimensions: {video}")
        # Keep contextual visual information: HUD and action, animated art/subtitles,
        # or the altar and congregation in a church wide shot.
        return Framing(mode=content_type, src_w=src_w, src_h=src_h)
    src_w, src_h, times, centers = sample_faces(video, start, end, face_model)
    crop_w = even(src_h * TARGET_RATIO)
    if crop_w >= src_w or src_w / max(src_h, 1) <= TARGET_RATIO + 0.01:
        return Framing(mode="fit", src_w=src_w, src_h=src_h)
    seen = sum(c is not None for c in centers)
    filled = fill_gaps(centers)
    if not times or filled is None or seen / len(centers) < min_face_ratio:
        return Framing(mode="blur", src_w=src_w, src_h=src_h)
    points = to_points(camera_path(times, filled), src_w, crop_w)
    return Framing(mode="track", src_w=src_w, src_h=src_h, crop_w=crop_w, points=points)
