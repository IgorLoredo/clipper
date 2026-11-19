"""Configuration and geometry for the optional ViraCortes identity."""
import json
from dataclasses import dataclass, fields
from pathlib import Path
import re


@dataclass(frozen=True)
class Branding:
    enabled: bool = True
    width: int = 1080
    height: int = 1920
    scale: float = 1.0
    theme: str = "neon"
    logo_text: str = "Vira"
    logo_label: str = "CORTES"
    logo_font: str = "duplex"
    logo_color: str = "#FAF7FF"
    logo_size: float = 0.075
    logo_x: float = 0.5
    logo_y: float = 0.05
    logo_icon: bool = True
    border_margin: float = 0.011
    border_top: float = 0.107
    border_bottom: float = 0.918
    border_width: float = 0.004
    border_radius: float = 0.038
    border_color: str = "#A748F5"
    border_opacity: float = 0.95
    neon_color: str = "#7B26EA"
    neon_intensity: float = 1.0
    neon_blur: float = 0.009
    neon_opacity: float = 0.70
    footer_visible: bool = True
    footer_text: str = "MELHORES MOMENTOS"
    footer_label: str = "VIRACORTES / SHORTS"
    footer_y: float = 0.951
    footer_width: float = 0.36
    footer_size: float = 0.018
    mode: str = "horizontal"
    zoom: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    video_margin: float = 0.0
    background_enabled: bool = False
    background_color: str = "#100C1B"
    background_opacity: float = 0.94
    caption_bottom: float | None = None
    caption_margin: float = 0.056
    title_visible: bool = True
    title_text: str = ""
    title_color: str = "#49137D"
    title_size: float = 0.064
    title_width: float = 0.88
    title_gap: float = 0.078

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def window(self) -> tuple[int, int, int, int]:
        x = round(self.width * self.border_margin)
        return x, round(self.height * self.border_top), self.width - x - 1, round(self.height * self.border_bottom)

    @property
    def video_area(self) -> tuple[int, int, int, int]:
        if self.mode != "horizontal":
            inset = round(self.width * self.video_margin / 2) * 2
            return inset, inset, self.width - 2 * inset, self.height - 2 * inset
        left, top, right, bottom = self.window
        inset = round(self.width * (self.video_margin + self.border_width))
        left = (left + inset + 1) // 2 * 2
        if self.title_visible:
            top = max(top, round(self.height * 0.245))
        bottom = min(bottom, round(self.height * 0.82))
        top = (top + inset + 1) // 2 * 2
        return left, top, self.width - 2 * left, (bottom - inset - top) // 2 * 2

    def horizontal_frame(self, source_aspect: float) -> tuple[int, int, int, int]:
        x, y, area_w, area_h = self.video_area
        zoom = min(1, self.zoom)
        frame_w = max(2, int(min(area_w, area_h * source_aspect) * zoom / 2) * 2)
        frame_h = max(2, int(min(area_h, area_w / source_aspect) * zoom / 2) * 2)
        return (
            x + round((area_w - frame_w) * (1 + self.offset_x) / 4) * 2,
            y + round((area_h - frame_h) * (1 + self.offset_y) / 4) * 2,
            frame_w, frame_h,
        )

    def caption_margins(self, source_aspect: float = 16 / 9) -> tuple[float, float]:
        default = 560 / 1920
        if self.mode == "horizontal":
            _, top, _, video_h = self.horizontal_frame(source_aspect)
            video_bottom = top + video_h
            default = max(1 - self.border_bottom + 0.02, 1 - video_bottom / self.height - 0.09)
        bottom = default if self.caption_bottom is None else self.caption_bottom
        minimum = 1 - self.border_bottom + 0.02
        maximum = 1 - self.border_top - 0.14
        return max(minimum, min(maximum, bottom)), max(self.caption_margin, self.border_margin + 0.01)


def load_branding(path: Path | None = None, mode: str | None = None) -> Branding:
    default_path = Path(__file__).resolve().parent.parent / "assets/templates/viracortes.json"
    source = path or default_path
    try:
        values = json.loads(source.read_text(encoding="utf-8-sig")) if source.is_file() else {}
        if not isinstance(values, dict):
            raise ValueError("expected a JSON object")
        defaults = Branding()
        for field in fields(defaults):
            if field.name not in values:
                continue
            value, default = values[field.name], getattr(defaults, field.name)
            if default is None or type(default) is float:
                bounds = {
                    "scale": (0.25, 2), "zoom": (0.25, 3),
                    "offset_x": (-1, 1), "offset_y": (-1, 1), "neon_intensity": (0, 3),
                }
                low, high = bounds.get(field.name, (0, 1))
                if default is None and value is None:
                    continue
                if type(value) not in (float, int) or not low <= value <= high:
                    raise ValueError(f"{field.name} must be between {low} and {high}")
            elif type(value) is not type(default):
                raise ValueError(f"{field.name} must be {type(default).__name__}")
            if field.name.endswith("_color") and not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
                raise ValueError(f"{field.name} must use #RRGGBB")
        if mode is not None:
            values["mode"] = mode
        brand = Branding(**values)
        choices = {
            "mode": {"existing", "horizontal", "fill"},
            "theme": {"neon", "minimal"},
            "logo_font": {"duplex", "simplex", "complex", "triplex", "script"},
        }
        for name, allowed in choices.items():
            if getattr(brand, name) not in allowed:
                raise ValueError(f"{name} must be one of {', '.join(sorted(allowed))}")
        if any(d < 64 or d > 4096 or d % 2 for d in brand.size):
            raise ValueError("width and height must be even integers between 64 and 4096")
        if brand.width * 16 != brand.height * 9:
            raise ValueError("width and height must form a 9:16 canvas")
        if brand.border_bottom - brand.border_top < 0.2 or min(brand.video_area[2:]) < 2:
            raise ValueError("the video window is too small")
        if brand.border_margin + brand.border_width >= 0.49:
            raise ValueError("the side borders leave no room for the video")
        if max(brand.caption_margin, brand.border_margin + 0.01) >= 0.49:
            raise ValueError("caption margins leave no room for text")
        return brand
    except (TypeError, ValueError, OSError) as exc:
        raise ValueError(f"invalid ViraCortes template {source}: {exc}") from exc
