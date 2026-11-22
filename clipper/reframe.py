import json
import re
import subprocess
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile

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
    boxes: tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None = None


def _extract_ppm_frame(video: Path, timestamp: float) -> tuple[int, int, np.ndarray] | None:
    from clipper.ffmpeg import ffmpeg_exe

    result = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-ss", f"{max(0.0, timestamp):.3f}", "-i", str(video),
         "-frames:v", "1", "-f", "image2pipe", "-c:v", "ppm", "-"],
        capture_output=True,
    )
    if result.returncode or not result.stdout:
        return None
    parts = result.stdout.split(b"\n", 3)
    if len(parts) < 4:
        return None
    src_w, src_h = map(int, parts[1].split())
    rgb = np.frombuffer(parts[3], dtype=np.uint8).reshape(src_h, src_w, 3)
    return src_w, src_h, rgb


def _is_peripheral(face: tuple[float, float, float, float], src_w: int, src_h: int) -> bool:
    cx = face[0] + face[2] / 2
    cy = face[1] + face[3] / 2
    return cx < 0.35 * src_w or cx > 0.65 * src_w or cy < 0.28 * src_h or cy > 0.72 * src_h


def _webcam_box_from_face(face: tuple[float, float, float, float], src_w: int, src_h: int) -> tuple[int, int, int, int]:
    x, y, w, h = face
    crop_w = min(even(src_w), max(2, even(w * 2.5)))
    crop_h = min(even(src_h), max(2, even(h * 1.9)))
    left = even(np.clip(x + w / 2 - crop_w / 2, 0, src_w - crop_w))
    top = even(np.clip(y + h * 0.68 - crop_h / 2, 0, src_h - crop_h))
    return left, top, crop_w, crop_h


def _load_cached_webcam(cache_path: Path | None, video: Path, src_w: int, src_h: int) -> tuple[int, int, int, int] | None:
    if cache_path is None or not cache_path.is_file():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        crop = tuple(int(v) for v in data["webcam_crop"])
        if len(crop) != 4:
            return None
        if data.get("manual"):
            x, y, w, h = crop
            if min(x, y) >= 0 and min(w, h) >= 2 and x + w <= src_w and y + h <= src_h:
                return crop
        stat = video.stat()
        key = [str(video.resolve()), stat.st_size, stat.st_mtime_ns]
        if data.get("key") == key:
            x, y, w, h = crop
            if min(x, y) >= 0 and min(w, h) >= 2 and x + w <= src_w and y + h <= src_h:
                return crop
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return None


def _save_cached_webcam(cache_path: Path | None, video: Path, webcam: tuple[int, int, int, int],
                        *, manual: bool = False) -> None:
    if cache_path is None:
        return
    try:
        stat = video.stat()
        payload = {
            "key": [str(video.resolve()), stat.st_size, stat.st_mtime_ns],
            "webcam_crop": list(webcam),
            "manual": manual,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache_path.parent, delete=False) as tmp:
            temp_path = Path(tmp.name)
            json.dump(payload, tmp, ensure_ascii=False, indent=2)
        try:
            temp_path.replace(cache_path)
        finally:
            temp_path.unlink(missing_ok=True)
    except OSError:
        pass


def _pick_stationary_webcam(detections: list[tuple[float, float, float, float]],
                            src_w: int, src_h: int) -> tuple[int, int, int, int] | None:
    """Cluster peripheral face detections across frames; pick the most stationary one."""
    if not detections:
        return None
    tol_x = 0.12 * src_w
    tol_y = 0.12 * src_h
    clusters: list[list[tuple[float, float, float, float]]] = []
    for face in detections:
        cx, cy = face[0] + face[2] / 2, face[1] + face[3] / 2
        placed = False
        for group in clusters:
            gx = float(np.median([f[0] + f[2] / 2 for f in group]))
            gy = float(np.median([f[1] + f[3] / 2 for f in group]))
            if abs(cx - gx) <= tol_x and abs(cy - gy) <= tol_y:
                group.append(face)
                placed = True
                break
        if not placed:
            clusters.append([face])

    def rank(group: list[tuple[float, float, float, float]]) -> tuple[int, float, float]:
        xs = [f[0] + f[2] / 2 for f in group]
        ys = [f[1] + f[3] / 2 for f in group]
        spread = float(np.std(xs) + np.std(ys))
        area = float(np.median([f[2] * f[3] for f in group]))
        return (len(group), -spread, area)

    best = max(clusters, key=rank)
    med_face = tuple(float(np.median([f[i] for f in best])) for i in range(4))
    return _webcam_box_from_face(med_face, src_w, src_h)


def plan_gameplay_split(video: Path, start: float, face_model: str,
                        webcam: tuple[int, int, int, int] | None = None, *,
                        end: float | None = None, cache_path: Path | None = None) -> Framing:
    extracted = _extract_ppm_frame(video, start)
    if extracted is None:
        raise ValueError(f"could not extract the opening frame from {video}")
    src_w, src_h, rgb = extracted
    if webcam is None:
        webcam = _load_cached_webcam(cache_path, video, src_w, src_h)
    if webcam is None:
        try:
            detect = _detector(face_model)
        except ImportError as exc:
            raise ValueError("webcam detection requires OpenCV; use --webcam-crop X,Y,W,H") from exc
        detections: list[tuple[float, float, float, float]] = [
            face for face in detect(rgb[:, :, ::-1].copy()) if _is_peripheral(face, src_w, src_h)
        ]
        # Sample additional frames across the clip so a head turn, cutscene, or moving NPC
        # on frame 0 does not break webcam localization.
        span = max(0.0, (end - start) if end is not None else 20.0)
        extra_times = (
            [start + span * frac for frac in (0.15, 0.35, 0.55, 0.75, 0.9)]
            if span >= 2.0 else []
        )
        if not detections or end is not None:
            for sample_t in extra_times:
                sample = _extract_ppm_frame(video, sample_t)
                if sample is None:
                    continue
                _, _, sample_rgb = sample
                detections.extend(
                    face for face in detect(sample_rgb[:, :, ::-1].copy()) if _is_peripheral(face, src_w, src_h)
                )
        if not detections:
            raise ValueError("no webcam face found in the opening frame; use --webcam-crop X,Y,W,H")
        webcam = _pick_stationary_webcam(detections, src_w, src_h)
        if webcam is None:
            raise ValueError("no webcam face found in the opening frame; use --webcam-crop X,Y,W,H")
        _save_cached_webcam(cache_path, video, webcam, manual=False)
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


def _mouth_activity(gray: np.ndarray, prev_gray: np.ndarray | None, face: tuple[float, float, float, float]) -> float:
    if prev_gray is None or gray.shape != prev_gray.shape:
        return 0.0
    x, y, w, h = (int(round(v)) for v in face)
    x0 = max(0, x + w // 5)
    x1 = min(gray.shape[1], x + w - w // 5)
    y0 = max(0, y + int(h * 0.55))
    y1 = min(gray.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    diff = np.abs(gray[y0:y1, x0:x1].astype(np.float32) - prev_gray[y0:y1, x0:x1].astype(np.float32))
    return float(np.mean(diff))


def sample_faces(video: Path, start: float, end: float, face_model: str,
                 step: float = 0.5, analysis_width: int = 640, *,
                 active_speaker: bool = False) -> tuple[int, int, list[float], list[float | None]]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detect = _detector(face_model)
    scale = analysis_width / src_w if src_w > analysis_width else 1.0
    times, centers = [], []
    prev_gray = None
    t = start
    while t < end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if active_speaker else None
        faces = detect(frame)
        if faces:
            if active_speaker and len(faces) >= 2 and prev_gray is not None and gray is not None:
                chosen = max(
                    faces,
                    key=lambda f: (_mouth_activity(gray, prev_gray, f), f[2] * f[3]),
                )
            else:
                chosen = max(faces, key=lambda f: f[2] * f[3])
            x, _, w, _ = chosen
            centers.append((x + w / 2) / frame.shape[1])
        else:
            centers.append(None)
        prev_gray = gray
        times.append(t - start)
        t += step
    cap.release()
    return src_w, src_h, times, centers


def _plan_podcast_split(video: Path, start: float, end: float, face_model: str) -> Framing | None:
    import cv2

    cap = cv2.VideoCapture(str(video))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if src_w <= 0 or src_h <= 0:
        cap.release()
        return None
    detect = _detector(face_model)
    left_faces: list[tuple[float, float, float, float]] = []
    right_faces: list[tuple[float, float, float, float]] = []
    span = max(1.0, end - start)
    for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
        cap.set(cv2.CAP_PROP_POS_MSEC, (start + span * frac) * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        faces = detect(frame)
        if len(faces) < 2:
            continue
        top_two = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:2]
        top_two.sort(key=lambda f: f[0] + f[2] / 2)
        left_faces.append(top_two[0])
        right_faces.append(top_two[1])
    cap.release()
    if len(left_faces) < 2:
        return None

    def make_box(group: list[tuple[float, float, float, float]]) -> tuple[int, int, int, int]:
        x, y, w, h = (float(np.median([f[i] for f in group])) for i in range(4))
        crop_w = min(even(src_w), max(2, even(max(w * 2.8, src_w * 0.42))))
        crop_h = min(even(src_h), max(2, even(crop_w * 8 / 9)))
        left = even(np.clip(x + w / 2 - crop_w / 2, 0, src_w - crop_w))
        top = even(np.clip(y + h * 0.55 - crop_h / 2, 0, src_h - crop_h))
        return left, top, crop_w, crop_h

    return Framing("podcast-split", src_w, src_h, boxes=(make_box(left_faces), make_box(right_faces)))


def plan(video: Path, start: float, end: float, face_model: str = "", min_face_ratio: float = 0.4,
         content_type: str = "podcast", *, active_speaker: bool = False,
         podcast_split: bool = False) -> Framing:
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
    if podcast_split:
        split = _plan_podcast_split(video, start, end, face_model)
        if split is not None:
            return split
    src_w, src_h, times, centers = sample_faces(video, start, end, face_model, active_speaker=active_speaker)
    crop_w = even(src_h * TARGET_RATIO)
    if crop_w >= src_w or src_w / max(src_h, 1) <= TARGET_RATIO + 0.01:
        return Framing(mode="fit", src_w=src_w, src_h=src_h)
    seen = sum(c is not None for c in centers)
    filled = fill_gaps(centers)
    if not times or filled is None or seen / len(centers) < min_face_ratio:
        return Framing(mode="blur", src_w=src_w, src_h=src_h)
    points = to_points(camera_path(times, filled), src_w, crop_w)
    return Framing(mode="track", src_w=src_w, src_h=src_h, crop_w=crop_w, points=points)
