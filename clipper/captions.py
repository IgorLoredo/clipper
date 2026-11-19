import json
import re
from dataclasses import dataclass, fields
from pathlib import Path

from clipper.transcribe import Word

PUNCTUATION = (".", "!", "?", ",", "…", ":", ";")


@dataclass(frozen=True)
class CaptionStyle:
    font: str = "Arial Black"
    size: int = 92
    primary: str = "&H00FFFFFF"
    highlight: str = "&H00E5FF&"
    base: str = "&HFFFFFF&"
    outline: str = "&H00000000"
    outline_width: int = 7
    shadow: int = 3
    margin_v: int = 560
    max_words: int = 3
    max_gap: float = 0.6


@dataclass(frozen=True)
class CaptionTemplate:
    name: str = "default"
    width: int = 1080
    height: int = 1920
    caption: CaptionStyle = CaptionStyle()
    title_enabled: bool = False
    title_size: int = 54
    title_color: str = "&HFFFFFF&"
    title_margin_v: int = 150
    credit_enabled: bool = False
    credit_size: int = 28
    credit_color: str = "&HCCCCCC&"
    credit_margin_v: int = 70
    credit: str = ""
    background_mode: str = "blur"
    title_box: bool = True
    credit_box: bool = True
    brand: str = ""
    brand_enabled: bool = False
    frame_color: str = "0x00E5FF@0.95"


def _integer(data: dict, key: str, default: int, minimum: int) -> int:
    try:
        value = int(data.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"template {key} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"template {key} must be at least {minimum}")
    return value


def _number(data: dict, key: str, default: float, minimum: float) -> float:
    try:
        value = float(data.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"template {key} must be a number") from exc
    if value < minimum:
        raise ValueError(f"template {key} must be at least {minimum}")
    return value


def _ass_color(value: object, key: str) -> str:
    color = str(value)
    if not re.fullmatch(r"&H[0-9A-Fa-f]{6,8}&?", color):
        raise ValueError(f"template {key} must be an ASS color such as &H00FFFFFF")
    return color


def _caption_style(data: dict) -> CaptionStyle:
    font = str(data.get("font", CaptionStyle.font))
    if not font.strip() or any(character in font for character in ",\r\n"):
        raise ValueError("template caption font contains invalid characters")
    return CaptionStyle(
        font=font, size=_integer(data, "size", CaptionStyle.size, 8),
        primary=_ass_color(data.get("primary", CaptionStyle.primary), "caption.primary"),
        highlight=_ass_color(data.get("highlight", CaptionStyle.highlight), "caption.highlight"),
        base=_ass_color(data.get("base", CaptionStyle.base), "caption.base"),
        outline=_ass_color(data.get("outline", CaptionStyle.outline), "caption.outline"),
        outline_width=_integer(data, "outline_width", CaptionStyle.outline_width, 0),
        shadow=_integer(data, "shadow", CaptionStyle.shadow, 0),
        margin_v=_integer(data, "margin_v", CaptionStyle.margin_v, 0),
        max_words=_integer(data, "max_words", CaptionStyle.max_words, 1),
        max_gap=_number(data, "max_gap", CaptionStyle.max_gap, 0),
    )


def load_template(path: Path | None) -> CaptionTemplate:
    if path is None:
        return CaptionTemplate()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"template is not valid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError("template must contain a JSON object")
    allowed = {item.name for item in fields(CaptionTemplate)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError("unknown template fields: " + ", ".join(unknown))
    caption_data = data.get("caption", {})
    if not isinstance(caption_data, dict):
        raise ValueError("template caption must be a JSON object")
    caption_fields = {item.name for item in fields(CaptionStyle)}
    caption_unknown = sorted(set(caption_data) - caption_fields)
    if caption_unknown:
        raise ValueError("unknown caption fields: " + ", ".join(caption_unknown))
    style = _caption_style(caption_data)
    mode = str(data.get("background_mode", "blur"))
    if mode not in {"blur", "cover", "framed"}:
        raise ValueError("template background_mode must be blur, cover or framed")
    frame_color = str(data.get("frame_color", "0x00E5FF@0.95"))
    if not re.fullmatch(r"0x[0-9A-Fa-f]{6}(?:@[01](?:\.\d+)?)?", frame_color):
        raise ValueError("template frame_color must look like 0xRRGGBB or 0xRRGGBB@opacity")
    return CaptionTemplate(
        name=str(data.get("name", path.stem)), width=_integer(data, "width", 1080, 240),
        height=_integer(data, "height", 1920, 240), caption=style,
        title_enabled=bool(data.get("title_enabled", False)),
        title_size=_integer(data, "title_size", 54, 8),
        title_color=_ass_color(data.get("title_color", "&HFFFFFF&"), "title_color"),
        title_margin_v=_integer(data, "title_margin_v", 150, 0),
        credit_enabled=bool(data.get("credit_enabled", False)),
        credit_size=_integer(data, "credit_size", 28, 8),
        credit_color=_ass_color(data.get("credit_color", "&HCCCCCC&"), "credit_color"),
        credit_margin_v=_integer(data, "credit_margin_v", 70, 0), credit=str(data.get("credit", "")),
        background_mode=mode, title_box=bool(data.get("title_box", True)),
        credit_box=bool(data.get("credit_box", True)), brand=str(data.get("brand", "")),
        brand_enabled=bool(data.get("brand_enabled", False)), frame_color=frame_color,
    )


def ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    cs = int(round(seconds * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def clean(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")").upper()


def ass_text(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", "\\N")


def group_words(words: list[Word], max_words: int, max_gap: float) -> list[list[Word]]:
    groups: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        if current and (len(current) >= max_words or word.start - current[-1].end > max_gap):
            groups.append(current)
            current = []
        current.append(word)
        if word.text.endswith(PUNCTUATION):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def build_ass(words: list[Word], clip_start: float, clip_end: float,
              style: CaptionStyle = CaptionStyle(), width: int = 1080, height: int = 1920,
              template: CaptionTemplate | None = None, title: str = "", credit: str = "",
              include_captions: bool = True) -> str:
    template = template or CaptionTemplate(width=width, height=height, caption=style)
    style, width, height = template.caption, template.width, template.height
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Main,{style.font},{style.size},{style.primary},{style.primary},{style.outline},&H80000000,-1,0,0,0,100,100,0,0,1,{style.outline_width},{style.shadow},2,60,60,{style.margin_v},1
Style: TemplateTitle,{style.font},{template.title_size},{template.title_color},{template.title_color},{style.outline},{'&H80000000' if template.title_box else '&HFF000000'},-1,0,0,0,100,100,0,0,1,3,1,8,60,60,{template.title_margin_v},1
Style: TemplateBrand,{style.font},34,&H00E5FF&,&H00E5FF&,&H00000000,&HFF000000,-1,0,0,0,100,100,0,0,1,2,1,8,60,60,85,1
Style: TemplateCredit,Arial,{template.credit_size},{template.credit_color},{template.credit_color},{style.outline},{'&H80000000' if template.credit_box else '&HFF000000'},0,0,0,0,100,100,0,0,1,2,1,2,60,60,{template.credit_margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    inside = [w for w in words if w.end > clip_start and w.start < clip_end]
    clip_len = clip_end - clip_start
    lines = []
    if template.title_enabled and title.strip():
        lines.append(f"Dialogue: 0,0:00:00.00,{ass_time(clip_len)},TemplateTitle,,0,0,0,,{ass_text(title)}")
    if template.brand_enabled and template.brand.strip():
        lines.append(f"Dialogue: 0,0:00:00.00,{ass_time(clip_len)},TemplateBrand,,0,0,0,,{ass_text(template.brand)}")
    credit_text = credit.strip() or template.credit.strip()
    if template.credit_enabled and credit_text:
        lines.append(f"Dialogue: 0,0:00:00.00,{ass_time(clip_len)},TemplateCredit,,0,0,0,,{ass_text(credit_text)}")
    if not include_captions:
        return header + "\n".join(lines) + "\n"
    for group in group_words(inside, style.max_words, style.max_gap):
        for index, word in enumerate(group):
            start = word.start - clip_start
            end = group[index + 1].start - clip_start if index + 1 < len(group) else word.end - clip_start
            start, end = max(0.0, start), min(clip_len, max(end, start + 0.05))
            if end <= start:
                continue
            text = " ".join(
                f"{{\\c{style.highlight}}}{clean(w.text)}{{\\c{style.base}}}" if i == index else clean(w.text)
                for i, w in enumerate(group)
            )
            pop = "{\\fscx108\\fscy108\\t(0,90,\\fscx100\\fscy100)}" if index == 0 else ""
            lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Main,,0,0,0,,{pop}{text}")
    return header + "\n".join(lines) + "\n"
