from dataclasses import dataclass

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


def ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    cs = int(round(seconds * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def clean(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")").upper()


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
              style: CaptionStyle = CaptionStyle(), width: int = 1080, height: int = 1920) -> str:
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Main,{style.font},{style.size},{style.primary},{style.primary},{style.outline},&H80000000,-1,0,0,0,100,100,0,0,1,{style.outline_width},{style.shadow},2,60,60,{style.margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    inside = [w for w in words if w.end > clip_start and w.start < clip_end]
    clip_len = clip_end - clip_start
    lines = []
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
