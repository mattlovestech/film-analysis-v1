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
from dataclasses import dataclass
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
    "The reference photo is a football player from this team's game film. "
    "Turn that player into one Madden-style video-game render. "
    "Follow the player's build, stance, helmet, and uniform from the photo. "
    "Stylized 3D football video-game player, helmet on, not a photograph, "
    "standing full length, facing the camera, both feet on the ground, arms relaxed, "
    "{jersey} jersey, {pants} pants, {helmet} helmet, shoulder pads, "
    "only the number 00 on the jersey, no other text, cleats, "
    "plain dark studio background, entire body from helmet to cleats in frame, "
    "no field, no crowd, no badges."
)
_DETECT_PROMPT = (
    'These frames are from the football film "{title}". '
    "Two teams are in the plays. "
    "The left team is lined up on the left side of the frame. "
    "The right team is lined up on the right side of the frame. "
    "Name each team the way a high school roster would: the school, then the mascot. "
    "Use that shape: Johns Creek Gladiators, HF Wildcats, Walton Raiders. "
    "Read jerseys, helmets, the scoreboard, and the field. "
    "A short school name is right when that is what is written. "
    "Describe jersey, pants, and helmet with color words only, "
    "such as white, dark blue, or gold. "
    "Do not put a logo or a jersey number in the color words. "
    "Reply with JSON only, no markdown: "
    '{"left":{"name":"","jersey":"","pants":"","helmet":""},'
    '"right":{"name":"","jersey":"","pants":"","helmet":""}}'
)


@dataclass(frozen=True)
class Uniform:
    """The uniform the model saw for one side of the plays."""

    name: str
    jersey: str
    pants: str
    helmet: str


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
    """Pick the two strongest uniform colors in already-sampled pixels."""

    counts: dict[tuple[int, int, int], int] = {}
    for red, green, blue in pixels:
        if _skip_color(red, green, blue):
            continue
        key = (red // 32 * 32, green // 32 * 32, blue // 32 * 32)
        counts[key] = counts.get(key, 0) + 1
    ranked = sorted(counts, key=lambda color: counts[color], reverse=True)
    if not ranked:
        return _FALLBACK
    jersey = ranked[0]
    pants = next((color for color in ranked[1:] if _apart(jersey, color)), None)
    return _hex(jersey), _FALLBACK[1] if pants is None else _hex(pants)


def play_frames(video: Path, plays: Sequence[PlayRecord], limit: int = 3) -> list[bytes]:
    """A lineup frame from each of the first scrimmage plays."""

    chosen = [play for play in plays if play.family == "scrimmage"] or list(plays)
    frames: list[bytes] = []
    for play in chosen[:limit]:
        try:
            frames.append(_frame(video, lineup_moment((play,))))
        except (VideoAnalysisError, OSError):
            continue
    return frames


def detect_uniforms(
    frames: Sequence[bytes],
    title: str,
    left_still: bytes | None = None,
    right_still: bytes | None = None,
) -> tuple[Uniform, Uniform]:
    """Ask the model which two teams are in the plays, then keep a color fallback."""

    found = _ask_uniforms(frames, title)
    if found is not None:
        return found
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

    return f"{uniform.jersey} jersey, {uniform.pants} pants, {uniform.helmet} helmet"


def make_portrait(uniform: Uniform, still: bytes | None = None) -> tuple[bytes, str]:
    """A Madden-style player based on this side of the film, wearing the uniform."""

    image = _render(player_prompt(uniform), still)
    if image is not None:
        return image, ".jpg"
    return roster_svg(_swatch(uniform.jersey), _swatch(uniform.pants)), ".svg"


def player_prompt(uniform: Uniform) -> str:
    """The one standing-player prompt. Only the uniform colors change."""

    jersey = color_phrase(uniform.jersey) or "white"
    pants = color_phrase(uniform.pants) or "white"
    helmet = color_phrase(uniform.helmet) or jersey
    return _PLAYER_PROMPT.format(jersey=jersey, pants=pants, helmet=helmet)


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
                "scale=16:16",
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
    if max(red, green, blue) < 40 or min(red, green, blue) > 210:
        return True
    if max(red, green, blue) - min(red, green, blue) < 18:
        return True
    return green > red + 12 and green > blue + 8 and green > 70


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


def _ask_uniforms(frames: Sequence[bytes], title: str) -> tuple[Uniform, Uniform] | None:
    jpegs = [frame for frame in frames if frame.startswith(b"\xff\xd8\xff")][:3]
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not jpegs or not api_key:
        return None
    hint = " ".join(title.split())[:120] or "this film"
    content: list[dict[str, object]] = []
    for index, frame in enumerate(jpegs, start=1):
        content.append({"type": "text", "text": f"Play frame {index}"})
        content.append(_image_block(frame))
    content.append({"type": "text", "text": _DETECT_PROMPT.format(title=hint)})
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=os.environ.get("FILM_ANALYSIS_MODEL", DEFAULT_MODEL),
            max_tokens=400,
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
    return Uniform(name, shirt, legs, shirt)


def _uniform_from_model(value: object, fallback_name: str) -> Uniform | None:
    if not isinstance(value, dict):
        return None
    jersey = color_phrase(str(value.get("jersey") or ""))
    pants = color_phrase(str(value.get("pants") or ""))
    helmet = color_phrase(str(value.get("helmet") or "")) or jersey
    if not jersey or not pants:
        return None
    name = _school_name(value.get("name")) or fallback_name
    return Uniform(name, jersey, pants, helmet)


def _uniform_dict(uniform: Uniform) -> dict[str, str]:
    return {
        "name": uniform.name,
        "jersey": uniform.jersey,
        "pants": uniform.pants,
        "helmet": uniform.helmet,
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
            [("prompt", prompt), ("width", "768"), ("height", "1280")],
            reference,
        )
        image = _post_image(account, token, _KLEIN, body, content_type)
        if image is not None:
            return image
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
    """A portrait-sized JPEG of the lineup player, small enough to send with the prompt."""

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
                "scale=768:1280:force_original_aspect_ratio=decrease",
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
