"""Lineup stills and original roster portraits for the two teams in a film."""

import base64
import binascii
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from film_analysis.agents.errors import VideoAnalysisError
from film_analysis.agents.play_boundaries import DEFAULT_MODEL
from film_analysis.storage.catalog import PlayRecord

_FALLBACK = ("#1c3f73", "#f2efe6")
_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
_KLEIN = "@cf/black-forest-labs/flux-2-klein-4b"
_SCHNELL = "@cf/black-forest-labs/flux-1-schnell"
_COLORS = frozenset(
    {
        "white",
        "black",
        "gray",
        "gold",
        "silver",
        "blue",
        "navy",
        "red",
        "green",
        "purple",
        "orange",
        "yellow",
        "maroon",
        "cardinal",
        "crimson",
        "pink",
        "brown",
        "teal",
    }
)
_MODS = frozenset({"dark", "light"})
_PLAYER_PROMPT = (
    "Image 0 is the reference photo of one football player from this team. "
    "Generate one representative Madden-style video-game player wearing only this uniform. "
    "Match the colors and the style of the uniform in image 0. "
    "Do not brighten, darken, or shift any cloth color. "
    "Copy the exact clothes from image 0: the same jersey, pants, helmet, "
    "stripes, sleeves, socks, and shoe colors. "
    "Stylized 3D football video-game player, helmet on, not a photograph, "
    "standing full length, facing the camera, both feet on the ground, arms relaxed, "
    "{jersey} jersey {jersey_hex}, {pants} pants {pants_hex}, {pants_style}, "
    "{helmet} helmet {helmet_hex}, {helmet_style}, jersey numbers {number} {number_hex}, "
    "only the number 00 on the jersey, no other text, do not write color codes, "
    "cleats, plain dark studio background, entire body from helmet to cleats in frame, "
    "no field, no crowd, no other team, no badges."
)
_DETECT_PROMPT = (
    'These frames are from the football film "{title}". '
    "Two teams are in the plays. "
    "One team wears a white jersey, usually the away team. "
    "The other wears a solid color jersey, usually the home team. "
    "Identify those two uniforms. "
    "The left team is lined up on the left side of the frame. "
    "The right team is lined up on the right side of the frame. "
    "Name each team the way a high school roster would: the school, then the mascot. "
    "Use that shape: Johns Creek Gladiators, HF Wildcats, Walton Raiders. "
    "Read jerseys, helmets, the scoreboard, and the field. "
    "A short school name is right when that is what is written. "
    "For each side give jersey, pants, and helmet as color words. "
    "Read each hex from the cloth in these photos, not from a remembered school color. "
    "Also give jersey_hex, pants_hex, helmet_hex, and number_hex as #rrggbb. "
    "number_hex is the color of the digits on the jersey. "
    "helmet_style is the shell, such as solid or striped. "
    "pants_style is the cut and stripe, such as solid or striped. "
    "Reply with JSON only, no markdown: "
    '{{"left":{{"name":"","jersey":"","jersey_hex":"#000000",'
    '"pants":"","pants_hex":"#000000","pants_style":"",'
    '"helmet":"","helmet_hex":"#000000","helmet_style":"","number_hex":"#000000"}},'
    '"right":{{"name":"","jersey":"","jersey_hex":"#000000",'
    '"pants":"","pants_hex":"#000000","pants_style":"",'
    '"helmet":"","helmet_hex":"#000000","helmet_style":"","number_hex":"#000000"}}}}'
)


@dataclass(frozen=True)
class Uniform:
    """The uniform the model saw for one side of the plays."""

    name: str
    jersey: str
    pants: str
    helmet: str
    number: str = ""
    helmet_style: str = ""
    pants_style: str = ""
    jersey_hex: str = ""
    pants_hex: str = ""
    helmet_hex: str = ""
    number_hex: str = ""


def lineup_moment(plays: Sequence[PlayRecord]) -> float:
    """A second inside the first scrimmage, where both teams are usually lined up."""

    chosen = next((play for play in plays if play.family == "scrimmage"), None)
    if chosen is None:
        chosen = plays[0]
    length = max(0.0, chosen.end_seconds - chosen.start_seconds)
    return chosen.start_seconds + min(0.6, length / 2)


def split_lineup(video: Path, at: float) -> tuple[bytes, bytes]:
    """Return JPEG stills of the left and right halves of one frame."""

    if shutil.which("ffmpeg") is None:
        raise VideoAnalysisError("Could not read a lineup from that film.")
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        frame = root / "frame.jpg"
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{max(0.0, at):.3f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-vf",
                "scale=960:-2",
                "-q:v",
                "4",
                str(frame),
            ]
        )
        left = root / "left.jpg"
        right = root / "right.jpg"
        _crop(frame, "crop=iw/2:ih:0:0", left)
        _crop(frame, "crop=iw/2:ih:iw/2:0", right)
        return left.read_bytes(), right.read_bytes()


def focus_player(still: bytes, side: str) -> bytes:
    """Crop a lineup half toward the line of scrimmage, where that side stands."""

    if shutil.which("ffmpeg") is None or not still.startswith(b"\xff\xd8\xff"):
        return still
    crop = "crop=iw*0.62:ih:iw*0.38:0" if side == "left" else "crop=iw*0.62:ih:0:0"
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "still.jpg"
        dest = Path(folder) / "players.jpg"
        source.write_bytes(still)
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-vf",
                crop,
                "-q:v",
                "4",
                str(dest),
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0 or not dest.is_file():
            return still
        focused = dest.read_bytes()
    return focused if focused.startswith(b"\xff\xd8\xff") else still


def dominant_colors(still: bytes) -> tuple[str, str]:
    """Jersey and pants colors from a lineup still, skipping grass and sky."""

    try:
        pixels = _sample_pixels(still)
    except (OSError, VideoAnalysisError):
        return _FALLBACK
    return colors_from_pixels(pixels)


def colors_from_pixels(pixels: Sequence[tuple[int, int, int]]) -> tuple[str, str]:
    """Pick the two strongest uniform colors, averaged from the photo itself."""

    palette = _palette(pixels)
    if not palette:
        return _FALLBACK
    jersey = palette[0]
    pants = next((color for color in palette[1:] if _apart(jersey, color)), None)
    return _hex(jersey), _FALLBACK[1] if pants is None else _hex(pants)


def play_frames(video: Path, plays: Sequence[PlayRecord], limit: int = 4) -> list[bytes]:
    """A few lineup frames spread through the film, not only the opening plays."""

    frames: list[bytes] = []
    for play in sample_plays(plays, limit):
        try:
            frames.append(_frame(video, lineup_moment((play,))))
        except (VideoAnalysisError, OSError):
            continue
    return frames


def sample_plays(plays: Sequence[PlayRecord], limit: int = 4) -> list[PlayRecord]:
    """Scrimmage plays spaced across the film, so a handful of frames covers both uniforms."""

    chosen = [play for play in plays if play.family == "scrimmage"] or list(plays)
    if limit < 1 or not chosen:
        return []
    if len(chosen) <= limit:
        return chosen
    if limit == 1:
        return [chosen[0]]
    step = (len(chosen) - 1) / (limit - 1)
    indexes = sorted({round(index * step) for index in range(limit)})
    return [chosen[index] for index in indexes]


def detect_uniforms(
    frames: Sequence[bytes],
    title: str,
    left_still: bytes | None = None,
    right_still: bytes | None = None,
) -> tuple[Uniform, Uniform]:
    """Ask the model which two teams are in the plays, then keep a color fallback."""

    found = _ask_uniforms(frames, title, left_still, right_still)
    if found is not None:
        return (
            _snap_uniform(found[0], left_still),
            _snap_uniform(found[1], right_still),
        )
    return _color_uniforms(title, left_still, right_still)


def parse_uniforms(reply: str, title: str) -> tuple[Uniform, Uniform] | None:
    """Read the model's JSON into two uniforms, or return None when it is unusable."""

    payload = _json_object(reply)
    if payload is None:
        return None
    fallback = suggest_team_names(title)
    left = _uniform_from_model(payload.get("left"), fallback[0])
    right = _uniform_from_model(payload.get("right"), fallback[1])
    if left is None or right is None:
        return None
    return left, right


def encode_uniforms(left: Uniform, right: Uniform) -> str:
    """Store both uniforms as one JSON string."""

    return json.dumps(
        {"left": _uniform_dict(left), "right": _uniform_dict(right)},
        separators=(",", ":"),
    )


def decode_uniforms(value: str) -> tuple[Uniform, Uniform] | None:
    """Restore uniforms saved with `encode_uniforms`."""

    if not value:
        return None
    return parse_uniforms(value, "")


def uniform_line(uniform: Uniform) -> str:
    """One short line for the uniform the standing player will wear."""

    jersey = _labeled(uniform.jersey, uniform.jersey_hex)
    pants = _labeled(uniform.pants, uniform.pants_hex, uniform.pants_style)
    helmet = _labeled(uniform.helmet, uniform.helmet_hex, uniform.helmet_style)
    return f"{jersey} jersey, {pants} pants, {helmet} helmet"


def _labeled(color: str, hex_code: str = "", style: str = "") -> str:
    parts = [part for part in (color, style, hex_code) if part]
    return " ".join(parts)


def portrait_uniform(uniform: Uniform, name: str) -> Uniform:
    """Keep the detected colors and style when the displayed team name changes."""

    return replace(uniform, name=name or uniform.name)


def player_prompt(uniform: Uniform) -> str:
    """The one standing-player prompt. It requires the clothes in image 0."""

    jersey = color_phrase(uniform.jersey) or "white"
    pants = color_phrase(uniform.pants) or "white"
    helmet = color_phrase(uniform.helmet) or jersey
    number = color_phrase(uniform.number) or "the number color in the photo"
    return _PLAYER_PROMPT.format(
        jersey=jersey,
        pants=pants,
        helmet=helmet,
        number=number,
        jersey_hex=_hex_color(uniform.jersey_hex) or "from the photo",
        pants_hex=_hex_color(uniform.pants_hex) or "from the photo",
        helmet_hex=_hex_color(uniform.helmet_hex) or "from the photo",
        number_hex=_hex_color(uniform.number_hex) or "from the photo",
        helmet_style=style_phrase(uniform.helmet_style) or "as in the photo",
        pants_style=style_phrase(uniform.pants_style) or "as in the photo",
    )


def make_portrait(uniform: Uniform, still: bytes | None = None) -> tuple[bytes, str]:
    """A Madden player in number 00, matching this side's uniform and the play photo."""

    image = _render(player_prompt(uniform), still)
    if image is not None:
        return image, ".jpg"
    shirt = _swatch(uniform.jersey_hex or uniform.jersey)
    legs = _swatch(uniform.pants_hex or uniform.pants)
    return roster_svg(shirt, legs), ".svg"


def color_phrase(value: str) -> str:
    """Keep only color words, so a team name or hex code cannot enter the prompt."""

    words = re.findall(r"[a-z]+", value.lower())
    kept: list[str] = []
    for word in words:
        if word == "grey":
            word = "gray"
        if word in _COLORS or word in _MODS:
            kept.append(word)
        if len(kept) == 3:
            break
    if not any(word in _COLORS for word in kept):
        return ""
    return " ".join(kept)


def _hex_note(value: str) -> str:
    color = _hex_color(value)
    return f" {color}" if color else ""


def _extra(style: str, hex_color: str) -> str:
    parts = [part for part in (style_phrase(style), _hex_color(hex_color)) if part]
    return f" {' '.join(parts)}" if parts else ""


def style_phrase(value: str) -> str:
    """A few style words, so a logo or a school name cannot enter the prompt."""

    words = re.findall(r"[a-z]+", value.lower())
    kept = [word for word in words if word in _STYLES][:4]
    return " ".join(kept)


_STYLES = frozenset(
    {
        "solid",
        "stripe",
        "stripes",
        "striped",
        "glossy",
        "matte",
        "shell",
        "center",
        "side",
        "two",
        "tone",
        "dark",
        "light",
        "white",
        "black",
        "gold",
        "silver",
    }
)


_TEAM_PROMPT = """
The first photo is the left side of one high school football lineup.
The second photo is the right side.
Name each team the way a roster would: the school, then the mascot.
Use that shape. Johns Creek Gladiators. HF Wildcats. Walton Raiders.
Read the jerseys, helmets, scoreboard, and field.
A short school name is right when that is what is written.
Keep each name under 40 characters. The photos decide which team is on which side.
The film title may help: {title}
Reply with JSON only: {{"left": "school mascot", "right": "school mascot"}}
""".strip()


def suggest_team_names(
    title: str,
    left_still: bytes | None = None,
    right_still: bytes | None = None,
) -> tuple[str, str]:
    """Name both sides as a school and mascot, then fall back to the film title."""

    if left_still and right_still:
        named = _read_team_names(title, left_still, right_still)
        if named is not None:
            return named
    matchup = _matchup_names(title)
    if matchup is not None:
        return matchup
    return "Home", "Away"


def names_from_reply(reply: str) -> tuple[str, str] | None:
    """Read two school names from the model's JSON."""

    text = reply.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        found = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if found is None:
            return None
        try:
            payload = json.loads(found.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    left = _school_name(payload.get("left"))
    right = _school_name(payload.get("right"))
    if not left or not right or left.casefold() == right.casefold():
        return None
    return left, right


def color_words(value: str) -> str:
    """A short color name, so the image model does not paint a hex code on the jersey."""

    if not _COLOR.fullmatch(value):
        return "the color in the photo"
    red, green, blue = (int(value[index : index + 2], 16) for index in (1, 3, 5))
    peak = max(red, green, blue)
    floor = min(red, green, blue)
    if peak - floor < 28:
        if peak > 200:
            return "white"
        if peak < 50:
            return "black"
        return "gray"
    if blue >= red and blue >= green:
        name = "blue"
    elif green >= red and green > blue + 15:
        name = "green"
    elif red > 150 and green > 110 and blue < 90:
        name = "gold"
    else:
        name = "red"
    if peak < 130:
        return f"dark {name}"
    if floor > 150:
        return f"light {name}"
    return name


def roster_svg(jersey: str, pants: str) -> bytes:
    """An original roster drawing: helmet, jersey number 00, and pants."""

    shirt = _safe_color(jersey, _FALLBACK[0])
    legs = _safe_color(pants, _FALLBACK[1])
    lines = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 320">',
        '<rect width="240" height="320" fill="#14161a"/>',
        f'<path d="M86 36c0-22 68-22 68 0v22H86z" fill="{shirt}"/>',
        '<ellipse cx="120" cy="72" rx="22" ry="18" fill="#c4a574"/>',
        f'<path d="M70 108h100l18 36v70H52v-70z" fill="{shirt}"/>',
        '<text x="120" y="176" text-anchor="middle" font-size="46"',
        ' font-family="sans-serif" font-weight="700" fill="#f7f4ee">00</text>',
        f'<path d="M68 214h104l10 86H58z" fill="{legs}"/>',
        "</svg>",
    )
    return "\n".join(lines).encode()


def _crop(frame: Path, crop: str, dest: Path) -> None:
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(frame),
            "-vf",
            crop,
            "-q:v",
            "4",
            str(dest),
        ]
    )


def _run(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True, check=False)
    if result.returncode != 0:
        raise VideoAnalysisError("Could not read a lineup from that film.")


def _sample_pixels(still: bytes) -> list[tuple[int, int, int]]:
    if shutil.which("ffmpeg") is None:
        raise VideoAnalysisError("Could not read colors from that still.")
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "still.jpg"
        source.write_bytes(still)
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-vf",
                "scale=48:48",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            capture_output=True,
            check=False,
        )
    raw = result.stdout
    if result.returncode != 0 or len(raw) < 3:
        raise VideoAnalysisError("Could not read colors from that still.")
    usable = len(raw) - (len(raw) % 3)
    return [(raw[index], raw[index + 1], raw[index + 2]) for index in range(0, usable, 3)]


def _skip_color(red: int, green: int, blue: int) -> bool:
    peak = max(red, green, blue)
    if peak < 35:
        return True
    if green > red + 12 and green > blue + 8 and green > 70:
        return True
    return peak - min(red, green, blue) < 18 and peak < 170


def _palette(pixels: Sequence[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    groups: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    for red, green, blue in pixels:
        if _skip_color(red, green, blue):
            continue
        key = (red // 16 * 16, green // 16 * 16, blue // 16 * 16)
        groups.setdefault(key, []).append((red, green, blue))
    ranked = sorted(groups, key=lambda color: len(groups[color]), reverse=True)
    return [_average(groups[color]) for color in ranked[:4]]


def _average(samples: Sequence[tuple[int, int, int]]) -> tuple[int, int, int]:
    count = len(samples)
    return (
        round(sum(red for red, _green, _blue in samples) / count),
        round(sum(green for _red, green, _blue in samples) / count),
        round(sum(blue for _red, _green, blue in samples) / count),
    )


def _snap_uniform(uniform: Uniform, still: bytes | None) -> Uniform:
    if not still:
        return uniform
    try:
        measured = [_hex(color) for color in _palette(_sample_pixels(still))]
    except (OSError, VideoAnalysisError):
        return uniform
    if not measured:
        return uniform
    jersey = _nearest(uniform.jersey_hex, measured, required=True)
    pants = _nearest(uniform.pants_hex, measured, required=True)
    helmet = _nearest(uniform.helmet_hex or jersey, measured, required=False)
    number = _nearest(uniform.number_hex, measured, required=False) if uniform.number_hex else ""
    return replace(
        uniform,
        jersey=color_words(jersey),
        pants=color_words(pants),
        helmet=color_words(helmet),
        jersey_hex=jersey,
        pants_hex=pants,
        helmet_hex=helmet,
        number_hex=number or uniform.number_hex,
    )


def _nearest(guess: str, measured: list[str], *, required: bool) -> str:
    if not measured:
        return guess
    if not _hex_color(guess):
        return measured[0]
    best = min(measured, key=lambda color: _rgb_distance(guess, color))
    if required or _rgb_distance(guess, best) <= 120 * 120 * 3:
        return best
    return guess


def _rgb_distance(left: str, right: str) -> int:
    first = tuple(int(left[index : index + 2], 16) for index in (1, 3, 5))
    second = tuple(int(right[index : index + 2], 16) for index in (1, 3, 5))
    return sum((one - two) ** 2 for one, two in zip(first, second, strict=True))


def _apart(left: tuple[int, int, int], right: tuple[int, int, int]) -> bool:
    return sum(abs(a - b) for a, b in zip(left, right, strict=True)) >= 48


def _hex(color: tuple[int, int, int]) -> str:
    return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"


_MATCHUP = re.compile(r"\s+(?:vs\.?|versus|v\.|at|@)\s+", re.IGNORECASE)


def _matchup_names(title: str) -> tuple[str, str] | None:
    parts = _MATCHUP.split(title.strip(), maxsplit=1)
    if len(parts) != 2:
        return None
    left = _clean_name(parts[0])
    right = _clean_name(parts[1])
    if not left or not right or left.casefold() == right.casefold():
        return None
    return left, right


def _clean_name(value: str) -> str:
    text = " ".join(value.replace("|", " ").split()).strip(" -–—:|")
    return text[:40]


def _school_name(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.replace("|", " ").split()).strip(" -–—:|.,")
    if len(text) > 40:
        text = text[:40].rsplit(" ", 1)[0].strip()
    if len(text) < 2:
        return ""
    return text


def _read_team_names(
    title: str,
    left_still: bytes,
    right_still: bytes,
) -> tuple[str, str] | None:
    if not left_still.startswith(b"\xff\xd8\xff") or not right_still.startswith(b"\xff\xd8\xff"):
        return None
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return None
    hint = " ".join(title.split()).replace("{", "").replace("}", "")[:120] or "unknown"
    content: list[dict[str, object]] = [
        _image_block(left_still),
        _image_block(right_still),
        {"type": "text", "text": _TEAM_PROMPT.format(title=hint)},
    ]
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=os.environ.get("FILM_ANALYSIS_MODEL", "claude-sonnet-5-5"),
            max_tokens=200,
            messages=[{"role": "user", "content": content}],
        )
    except Exception:
        return None
    texts = [
        block.text
        for block in getattr(message, "content", [])
        if getattr(block, "type", None) == "text" and getattr(block, "text", "")
    ]
    return names_from_reply("\n".join(texts))


def _image_block(jpeg: bytes) -> dict[str, object]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.standard_b64encode(jpeg).decode("ascii"),
        },
    }


def _safe_color(value: str, fallback: str) -> str:
    return value if _COLOR.fullmatch(value) else fallback


def _frame(video: Path, at: float) -> bytes:
    if shutil.which("ffmpeg") is None:
        raise VideoAnalysisError("Could not read a lineup from that film.")
    with tempfile.TemporaryDirectory() as folder:
        dest = Path(folder) / "frame.jpg"
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{max(0.0, at):.3f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-vf",
                "scale=640:-2",
                "-q:v",
                "4",
                str(dest),
            ]
        )
        return dest.read_bytes()


def _ask_uniforms(
    frames: Sequence[bytes],
    title: str,
    left_still: bytes | None = None,
    right_still: bytes | None = None,
) -> tuple[Uniform, Uniform] | None:
    labeled: list[tuple[str, bytes]] = []
    for frame in frames:
        if len(labeled) >= 4:
            break
        if isinstance(frame, bytes) and frame.startswith(b"\xff\xd8\xff"):
            labeled.append((f"Play frame {len(labeled) + 1}", frame))
    if left_still and left_still.startswith(b"\xff\xd8\xff"):
        labeled.append(("Left team, closer on the players", focus_player(left_still, "left")))
    if right_still and right_still.startswith(b"\xff\xd8\xff"):
        labeled.append(("Right team, closer on the players", focus_player(right_still, "right")))
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not labeled or not api_key:
        return None
    hint = " ".join(title.split()).replace("{", "").replace("}", "")[:120] or "this film"
    content: list[dict[str, object]] = []
    for label, frame in labeled:
        content.append({"type": "text", "text": label})
        content.append(_image_block(frame))
    try:
        content.append({"type": "text", "text": _DETECT_PROMPT.format(title=hint)})
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=os.environ.get("FILM_ANALYSIS_MODEL", DEFAULT_MODEL),
            max_tokens=700,
            messages=[{"role": "user", "content": content}],
        )
    except Exception:
        return None
    texts = [
        block.text
        for block in getattr(message, "content", [])
        if getattr(block, "type", None) == "text" and getattr(block, "text", "")
    ]
    return parse_uniforms("\n".join(texts), title)


def _color_uniforms(
    title: str,
    left_still: bytes | None,
    right_still: bytes | None,
) -> tuple[Uniform, Uniform]:
    names = _matchup_names(title) or ("Home", "Away")
    return (
        _uniform_from_colors(names[0], left_still),
        _uniform_from_colors(names[1], right_still),
    )


def _uniform_from_colors(name: str, still: bytes | None) -> Uniform:
    jersey, pants = dominant_colors(still or b"")
    shirt = color_words(jersey)
    legs = color_words(pants)
    if shirt == "the color in the photo":
        shirt = "dark blue"
    if legs == "the color in the photo":
        legs = "white"
    return Uniform(
        name,
        shirt,
        legs,
        shirt,
        jersey_hex=jersey if _hex_color(jersey) else "",
        pants_hex=pants if _hex_color(pants) else "",
        helmet_hex=jersey if _hex_color(jersey) else "",
    )


def _hex_color(value: object) -> str:
    text = str(value or "").strip()
    return text.lower() if _COLOR.fullmatch(text) else ""


def _style(value: object) -> str:
    return style_phrase(str(value or ""))


def _uniform_from_model(value: object, fallback_name: str) -> Uniform | None:
    if not isinstance(value, dict):
        return None
    jersey_hex = _hex_color(value.get("jersey_hex"))
    pants_hex = _hex_color(value.get("pants_hex"))
    helmet_hex = _hex_color(value.get("helmet_hex"))
    number_hex = _hex_color(value.get("number_hex"))
    jersey = color_phrase(str(value.get("jersey") or "")) or (
        color_words(jersey_hex) if jersey_hex else ""
    )
    pants = color_phrase(str(value.get("pants") or "")) or (
        color_words(pants_hex) if pants_hex else ""
    )
    helmet = color_phrase(str(value.get("helmet") or "")) or (
        color_words(helmet_hex) if helmet_hex else ""
    ) or jersey
    if not jersey or not pants:
        return None
    name = _school_name(value.get("name")) or fallback_name
    return Uniform(
        name,
        jersey,
        pants,
        helmet,
        number=color_phrase(str(value.get("number") or "")) or (
            color_words(number_hex) if number_hex else ""
        ),
        helmet_style=_style(value.get("helmet_style")),
        pants_style=_style(value.get("pants_style")),
        jersey_hex=jersey_hex,
        pants_hex=pants_hex,
        helmet_hex=helmet_hex,
        number_hex=number_hex,
    )


def _uniform_dict(uniform: Uniform) -> dict[str, str]:
    return {
        "name": uniform.name,
        "jersey": uniform.jersey,
        "pants": uniform.pants,
        "helmet": uniform.helmet,
        "number": uniform.number,
        "jersey_hex": uniform.jersey_hex,
        "pants_hex": uniform.pants_hex,
        "helmet_hex": uniform.helmet_hex,
        "number_hex": uniform.number_hex,
        "helmet_style": uniform.helmet_style,
        "pants_style": uniform.pants_style,
    }


def _json_object(reply: str) -> dict[str, object] | None:
    text = reply.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        found = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if found is None:
            return None
        try:
            payload = json.loads(found.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    return payload


def _swatch(phrase: str) -> str:
    if _COLOR.fullmatch(phrase):
        return phrase
    text = color_phrase(phrase)
    if text in _SWATCH:
        return _SWATCH[text]
    last = text.split()[-1] if text else ""
    return _SWATCH.get(last, _FALLBACK[0])


_SWATCH = {
    "white": "#f4f1ea",
    "black": "#1a1a1a",
    "gray": "#8a8a8a",
    "gold": "#c0a020",
    "silver": "#c0c4c8",
    "blue": "#1c4fbf",
    "navy": "#1c3f73",
    "dark blue": "#1c3f73",
    "light blue": "#8eb4e8",
    "red": "#b42318",
    "green": "#1f7a3a",
    "purple": "#5b2a86",
    "orange": "#d86b1f",
    "yellow": "#e2c044",
    "maroon": "#6e1d2a",
}


def _render(prompt: str, still: bytes | None = None) -> bytes | None:
    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
    token = os.environ.get("CLOUDFLARE_API_TOKEN", "")
    if not account or not token:
        return None
    reference = _reference_jpeg(still) if still else None
    if reference is not None:
        body, content_type = _form(
            [
                ("prompt", prompt),
                ("width", "768"),
                ("height", "1280"),
                ("guidance", "5"),
            ],
            reference,
        )
        return _post_image(account, token, _KLEIN, body, content_type)
    body, content_type = _fields([("prompt", prompt), ("width", "768"), ("height", "1280")])
    image = _post_image(account, token, _KLEIN, body, content_type)
    if image is not None:
        return image
    return _run_prompt(account, token, prompt)


def _form(fields: list[tuple[str, str]], reference: bytes) -> tuple[bytes, str]:
    boundary = "film-player-boundary"
    chunks: list[bytes] = []
    for name, value in fields:
        header = f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
        chunks.append(f"{header}{value}\r\n".encode())
    chunks.append(
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="input_image_0"; '
            'filename="player.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'
        ).encode()
    )
    chunks.append(reference)
    chunks.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _reference_jpeg(still: bytes) -> bytes | None:
    """A close crop of the uniform, under 512 pixels on each side.

    Flux Klein ignores a reference that is 512 pixels or larger, and a wide
    lineup makes the jersey too small to copy.
    """

    if shutil.which("ffmpeg") is None or not still.startswith(b"\xff\xd8\xff"):
        return None
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "still.jpg"
        dest = Path(folder) / "reference.jpg"
        source.write_bytes(still)
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-vf",
                "scale=480:480:force_original_aspect_ratio=increase,crop=480:480",
                "-q:v",
                "5",
                str(dest),
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0 or not dest.is_file():
            return None
        data = dest.read_bytes()
    if not data.startswith(b"\xff\xd8\xff") or len(data) > 1_500_000:
        return None
    return data


def _fields(fields: list[tuple[str, str]]) -> tuple[bytes, str]:
    boundary = "film-player-boundary"
    chunks: list[bytes] = []
    for name, value in fields:
        header = f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
        chunks.append(f"{header}{value}\r\n".encode())
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _run_prompt(account: str, token: str, prompt: str) -> bytes | None:
    body = json.dumps({"prompt": prompt}).encode()
    return _post_image(account, token, _SCHNELL, body, "application/json")


def _post_image(
    account: str,
    token: str,
    model: str,
    body: bytes,
    content_type: str,
) -> bytes | None:
    url = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}"
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": content_type},
    )
    try:
        with urllib.request.urlopen(request, timeout=70) as response:
            payload = json.loads(response.read().decode())
    except (OSError, urllib.error.URLError, json.JSONDecodeError, UnicodeError, TimeoutError):
        return None
    return _image_bytes(payload)


def _image_bytes(payload: object) -> bytes | None:
    if not isinstance(payload, dict):
        return None
    result = payload.get("result", payload)
    image = result.get("image") if isinstance(result, dict) else None
    if not isinstance(image, str) or not image:
        return None
    if image.startswith("data:"):
        image = image.split(",", 1)[-1]
    try:
        decoded = base64.b64decode(image)
    except (binascii.Error, ValueError):
        return None
    if decoded.startswith(b"\xff\xd8\xff"):
        return decoded
    if decoded.startswith(b"\x89PNG\r\n\x1a\n"):
        return _jpeg_from_png(decoded)
    return None


def _jpeg_from_png(png: bytes) -> bytes | None:
    if shutil.which("ffmpeg") is None:
        return None
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "player.png"
        dest = Path(folder) / "player.jpg"
        source.write_bytes(png)
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-q:v",
                "3",
                str(dest),
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0 or not dest.is_file():
            return None
        data = dest.read_bytes()
    return data if data.startswith(b"\xff\xd8\xff") else None
