"""Lineup stills and the players found in the film."""

import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from film_analysis.ingest.errors import VideoAnalysisError
from film_analysis.plays.boundaries import DEFAULT_MODEL
from film_analysis.storage.catalog import PlayRecord
from film_analysis.teams.players import (
    PlayerColors,
    full_body_shot,
    lone_players,
    team_colors_from_players,
    uniform_crops,
)

_FALLBACK = ("#1c3f73", "#f2efe6")
_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
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
_DETECT_PROMPT = (
    'These frames are from the football film "{title}". '
    "Two teams are in the plays. "
    "One team wears a white jersey, the home team. "
    "The other wears a solid color jersey, the away team. "
    "Identify those two uniforms. "
    "A referee wears a black-and-white striped shirt and is not either team. "
    "The two jerseys are different colors. Never give both teams the same jersey color. "
    "The left team is lined up on the left side of the frame. "
    "The right team is lined up on the right side of the frame. "
    "Name the white-jersey team Home and the solid-color team Away. "
    "The name is only Home or Away. Do not guess a school, a mascot, or a city. "
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
    """Crop the part of a lineup half where this team stands, away from the other team."""

    if shutil.which("ffmpeg") is None or not still.startswith(b"\xff\xd8\xff"):
        return still
    crop = "crop=iw*0.7:ih:0:0" if side == "left" else "crop=iw*0.7:ih:iw*0.3:0"
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


# Team cards need a sharp close crop. Analysis frames stay small (768); these
# pulls stay near broadcast width so one player is more than a few dozen pixels.
_TEAM_FRAME_SCALE = "scale=1920:-2"
_TEAM_FRAME_QUALITY = "2"


def player_frames(video: Path, plays: Sequence[PlayRecord], limit: int = 48) -> list[bytes]:
    """Full-size frames spread through the film, so each uniform is large enough to crop."""

    if limit < 1 or not plays or not video.is_file() or video.stat().st_size < 4096:
        return []
    start = min(play.start_seconds for play in plays)
    end = max(play.end_seconds for play in plays)
    span = end - start
    if span <= 1:
        moments = [start]
    elif span <= 200:
        return _clip_frames(
            video, start, end, f"fps=1,{_TEAM_FRAME_SCALE}", quality=_TEAM_FRAME_QUALITY
        )
    else:
        count = min(limit, 40)
        moments = [start + span * (index + 0.5) / count for index in range(count)]
    frames: list[bytes] = []
    for at in moments:
        try:
            frames.append(_frame(video, at, _TEAM_FRAME_SCALE, quality=_TEAM_FRAME_QUALITY))
        except (VideoAnalysisError, OSError):
            continue
    return frames


def _clip_frames(
    video: Path,
    start: float,
    end: float,
    scale: str,
    *,
    quality: str = "4",
) -> list[bytes]:
    """One pass over a short film. A lone player can be in frame for only a second."""

    if shutil.which("ffmpeg") is None:
        return []
    with tempfile.TemporaryDirectory() as folder:
        dest = Path(folder) / "%03d.jpg"
        try:
            _run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{max(0.0, start):.3f}",
                    "-to",
                    f"{max(0.0, end):.3f}",
                    "-i",
                    str(video),
                    "-vf",
                    scale,
                    "-q:v",
                    quality,
                    str(dest),
                ]
            )
        except VideoAnalysisError:
            return []
        return [path.read_bytes() for path in sorted(Path(folder).glob("*.jpg"))]


def team_cards(
    frames: Sequence[bytes],
    left: Uniform,
    right: Uniform,
    avoid: Sequence[bytes] = (),
) -> tuple[bytes | None, bytes | None]:
    """A cropped player for each side, matched to that jersey.

    A live full-body shot of the jersey wins. The white-versus-color split fills
    a side that had no jersey match. Photos in ``avoid`` are skipped when the
    film has another player for that team.
    """

    left_shot = team_player_shot(frames, left, right, avoid=avoid)
    right_shot = team_player_shot(frames, right, left, avoid=avoid)
    if left_shot is not None and left_shot == right_shot:
        right_shot = None
    if left_shot is None or right_shot is None:
        white, _colored = lone_players(frames, avoid)
        if left_shot is None and _lighter(left):
            left_shot = _spare_for_uniform(white, left, right)
        if right_shot is None and _lighter(right):
            right_shot = _spare_for_uniform(white, right, left)
    if left_shot is not None and left_shot == right_shot:
        right_shot = None
    return left_shot, right_shot


def _spare_for_uniform(
    shot: bytes | None,
    uniform: Uniform,
    other: Uniform | None,
) -> bytes | None:
    """Keep a white/color spare only when its cloth is closer to this jersey."""

    if shot is None:
        return None
    target = _hex_color(uniform.jersey_hex) or _swatch(uniform.jersey)
    target_rgb = _rgb_tuple(target)
    if target_rgb is None:
        return shot
    sample = _shot_jersey(shot)
    if sample is None:
        return None
    own = _rgb_distance(_hex(sample), target)
    if other is not None:
        other_hex = _hex_color(other.jersey_hex) or _swatch(other.jersey)
        if _hex_color(other_hex) and _rgb_distance(_hex(sample), other_hex) <= own:
            return None
    limit = 70 * 70 * 3 if _lighter(uniform) else 62 * 62 * 3
    return shot if own <= limit else None


def _shot_jersey(shot: bytes) -> tuple[int, int, int] | None:
    from film_analysis.teams.players import _bgr, _jersey_sample_rgb

    image = _bgr(shot)
    if image is None:
        return None
    return _jersey_sample_rgb(image)


def _rgb_tuple(value: str) -> tuple[int, int, int] | None:
    color = _hex_color(value)
    if not color:
        return None
    return (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))


def _lighter(uniform: Uniform) -> bool:
    words = color_phrase(uniform.jersey)
    if "white" in words.split():
        return True
    hex_color = _hex_color(uniform.jersey_hex)
    if not hex_color:
        return False
    red = int(hex_color[1:3], 16)
    green = int(hex_color[3:5], 16)
    blue = int(hex_color[5:7], 16)
    return max(red, green, blue) > 185 and max(red, green, blue) - min(red, green, blue) < 55


def play_frames(video: Path, plays: Sequence[PlayRecord], limit: int = 4) -> list[bytes]:
    """A few lineup frames spread through the film, not only the opening plays."""

    frames: list[bytes] = []
    for play in sample_plays(plays, limit):
        try:
            frames.append(
                _frame(
                    video,
                    lineup_moment((play,)),
                    _TEAM_FRAME_SCALE,
                    quality=_TEAM_FRAME_QUALITY,
                )
            )
        except (VideoAnalysisError, OSError):
            continue
    return frames


def player_action_frames(
    video: Path,
    plays: Sequence[PlayRecord],
    limit: int = 32,
) -> list[bytes]:
    """Near-broadcast frames from live action, spread across longer plays.

    A lineup still is a small group at the snap. A player card needs a moment
    after the snap, when one uniform is large enough to crop — and the frame
    itself has to stay sharp enough that the crop is not a handful of pixels.
    """

    if limit < 1 or not plays or not video.is_file() or video.stat().st_size < 4096:
        return []
    chosen = [play for play in plays if play.family == "scrimmage"] or list(plays)
    long = [play for play in chosen if play.end_seconds - play.start_seconds >= 4] or chosen
    frames: list[bytes] = []
    for play in sample_plays(long, min(16, len(long))):
        length = max(0.0, play.end_seconds - play.start_seconds)
        for offset in _live_offsets(length):
            try:
                frames.append(
                    _frame(
                        video,
                        play.start_seconds + offset,
                        _TEAM_FRAME_SCALE,
                        quality=_TEAM_FRAME_QUALITY,
                    )
                )
            except (VideoAnalysisError, OSError):
                continue
            if len(frames) >= limit:
                return frames
    return frames


def action_moments(plays: Sequence[PlayRecord], limit: int = 8) -> list[tuple[int, float]]:
    """Seconds into sampled plays, after the snap and before the whistle.

    A lineup is the opening of a play. Live action is further in. Longer
    scrimmage plays are kept ahead of a short snap, and each of those plays
    contributes two moments so a sideline cut can be skipped.
    """

    chosen = [play for play in plays if play.family == "scrimmage"] or list(plays)
    long = [play for play in chosen if play.end_seconds - play.start_seconds >= 4]
    pool = long or chosen
    moments: list[tuple[int, float]] = []
    for play in sample_plays(pool, min(4, max(1, limit))):
        length = max(0.0, play.end_seconds - play.start_seconds)
        for offset in _live_offsets(length):
            moments.append((play.play_number, offset))
            if len(moments) >= limit:
                return moments
    return moments


def _live_offsets(length: float) -> list[float]:
    """Moments after the snap and before the whistle, inside one play."""

    if length <= 0:
        return []
    if length < 3:
        return [min(length * 0.5, max(0.3, length - 0.15))]
    start = min(max(1.5, length * 0.32), length - 0.8)
    later = min(max(start + 0.8, length * 0.58), length - 0.4)
    if later - start < 0.4:
        return [start]
    return [start, later]


def still_frame(video: Path, at: float) -> bytes:
    """One wide JPEG from a film or a play clip."""

    return _frame(video, at, _TEAM_FRAME_SCALE, quality=_TEAM_FRAME_QUALITY)


def choose_action_shot(frames: Sequence[bytes]) -> bytes | None:
    """The frame that looks most like live play, not a graphic or a turf closeup."""

    best: bytes | None = None
    best_score = -1.0
    for frame in frames:
        if not isinstance(frame, bytes) or not frame.startswith(b"\xff\xd8\xff"):
            continue
        score = score_field_shot(frame)
        if score > best_score:
            best = frame
            best_score = score
    return best


def score_field_shot(jpeg: bytes) -> float:
    """Higher when players are on the grass, not a title card or empty turf."""

    try:
        import cv2
        import numpy as np
    except ImportError:
        return 1.0
    array = np.frombuffer(jpeg, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return 0.0
    small = cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA)
    blue, green, red = cv2.split(small)
    grass_mask = (
        (green.astype(np.int16) > red.astype(np.int16) + 12)
        & (green.astype(np.int16) > blue.astype(np.int16) + 8)
        & (green > 40)
    )
    field = grass_mask[24:82, 8:152]
    grass = float(np.count_nonzero(field)) / field.size
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_32F).var()) / 180.0
    field_gray = gray[24:82, 8:152]
    players = (~field) & (field_gray > 28) & (field_gray < 245)
    player = float(np.count_nonzero(players)) / players.size
    columns = np.count_nonzero(players, axis=0)
    active = np.flatnonzero(columns > 0)
    spread = 0.0 if active.size == 0 else float(active[-1] - active[0]) / players.shape[1]
    if grass < 0.12:
        return min(sharp, 2.0) * 0.15
    if grass > 0.9 and player < 0.04:
        return 0.3
    bodies = max(0.0, 1.0 - abs(player - 0.16) / 0.16)
    return grass * 1.4 + bodies * 1.6 + min(sharp, 2.0) * 0.5 + spread


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
    photos = [item for item in (*frames, left_still, right_still) if isinstance(item, bytes)]
    if found is not None:
        pair = _keep_distinct(
            _snap_uniform(found[0], left_still),
            _snap_uniform(found[1], right_still),
            found[0],
            found[1],
        )
    else:
        pair = _color_uniforms(title, left_still, right_still)
    return label_home_away(*_colors_from_players(pair, photos))


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
    return label_home_away(left, right)


def encode_uniforms(left: Uniform, right: Uniform) -> str:
    """Store both uniforms as one JSON string."""

    return json.dumps(
        {"left": _uniform_dict(left), "right": _uniform_dict(right)},
        separators=(",", ":"),
    )


def uniforms_collapsed(value: str) -> bool:
    """True when the two saved uniforms do not describe different jerseys."""

    pair = decode_uniforms(value)
    if pair is None:
        return True
    return not _jerseys_differ(*pair)


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
    pieces = [f"{jersey} jersey", f"{pants} pants", f"{helmet} helmet"]
    number = _hex_color(uniform.number_hex)
    if number:
        pieces.append(f"number {number}")
    return ", ".join(pieces)


def uniform_colors(uniform: Uniform) -> list[dict[str, str]]:
    """Jersey, pants, helmet, and number colors read from the film."""

    chips: list[dict[str, str]] = []
    for label, value in (
        ("Jersey", uniform.jersey_hex),
        ("Pants", uniform.pants_hex),
        ("Helmet", uniform.helmet_hex),
        ("Number", uniform.number_hex),
    ):
        color = _hex_color(value)
        if color:
            chips.append({"label": label, "hex": color})
    return chips


def _labeled(color: str, hex_code: str = "", style: str = "") -> str:
    parts = [part for part in (color, style, hex_code) if part]
    return " ".join(parts)


def portrait_uniform(uniform: Uniform, name: str) -> Uniform:
    """Keep the detected colors and style when the displayed team name changes."""

    return replace(uniform, name=name or uniform.name)


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


def suggest_team_names(
    title: str,
    left_still: bytes | None = None,
    right_still: bytes | None = None,
) -> tuple[str, str]:
    """Home and away. The film title and the photos do not supply a name."""

    del title, left_still, right_still
    return "Home", "Away"


def label_home_away(left: Uniform, right: Uniform) -> tuple[Uniform, Uniform]:
    """Name the white jersey Home and the colored jersey Away."""

    left_white = _white_jersey(left)
    right_white = _white_jersey(right)
    if left_white and not right_white:
        return replace(left, name="Home"), replace(right, name="Away")
    if right_white and not left_white:
        return replace(left, name="Away"), replace(right, name="Home")
    return replace(left, name="Home"), replace(right, name="Away")


def _white_jersey(uniform: Uniform) -> bool:
    words = set(color_phrase(uniform.jersey).split())
    if "white" in words and "dark" not in words:
        return True
    return color_words(uniform.jersey_hex) == "white"


def color_words(value: str) -> str:
    """A short color name for this jersey."""

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
    jersey, jersey_hex = _adopt(uniform.jersey, uniform.jersey_hex, measured, required=True)
    pants, pants_hex = _adopt(uniform.pants, uniform.pants_hex, measured, required=True)
    helmet, helmet_hex = _adopt(
        uniform.helmet or jersey,
        uniform.helmet_hex or jersey_hex,
        measured,
        required=False,
    )
    number = _nearest(uniform.number_hex, measured, required=False) if uniform.number_hex else ""
    return replace(
        uniform,
        jersey=jersey,
        pants=pants,
        helmet=helmet,
        jersey_hex=jersey_hex,
        pants_hex=pants_hex,
        helmet_hex=helmet_hex,
        number_hex=number or uniform.number_hex,
    )


def _adopt(
    words: str,
    guess_hex: str,
    measured: list[str],
    *,
    required: bool,
) -> tuple[str, str]:
    """Use a measured color only when it is actually this cloth, not the other team."""

    seed = _hex_color(guess_hex) or _swatch(words)
    chosen = _nearest(seed, measured, required=required)
    if not _hex_color(guess_hex) and chosen == seed:
        return words, ""
    return color_words(chosen) or words, chosen


def _nearest(guess: str, measured: list[str], *, required: bool) -> str:
    if not measured:
        return guess
    if not _hex_color(guess):
        return measured[0] if required else guess
    best = min(measured, key=lambda color: _rgb_distance(guess, color))
    limit = 80 * 80 * 3 if required else 50 * 50 * 3
    if _rgb_distance(guess, best) <= limit:
        return best
    return guess


def _keep_distinct(
    left: Uniform,
    right: Uniform,
    original_left: Uniform,
    original_right: Uniform,
) -> tuple[Uniform, Uniform]:
    """Keep each team's own jersey when a shared photo would paint both the same."""

    if _jerseys_differ(left, right):
        return left, right
    if _jerseys_differ(original_left, original_right):
        return original_left, original_right
    return left, right


def _jerseys_differ(left: Uniform, right: Uniform) -> bool:
    left_color = _hex_color(left.jersey_hex) or _swatch(left.jersey)
    right_color = _hex_color(right.jersey_hex) or _swatch(right.jersey)
    if not _hex_color(left_color) or not _hex_color(right_color):
        return color_phrase(left.jersey) != color_phrase(right.jersey)
    return _rgb_distance(left_color, right_color) >= 45 * 45 * 3


def player_photo(
    frames: Sequence[bytes],
    uniform: Uniform,
    other: Uniform | None = None,
) -> bytes | None:
    """One photo of this team's players, chosen after the two uniforms are known."""

    photos = player_photos(frames, uniform, other)
    return photos[0] if photos else None


def team_player_shot(
    frames: Sequence[bytes],
    uniform: Uniform,
    other: Uniform | None = None,
    avoid: Sequence[bytes] = (),
) -> bytes | None:
    """One full-body photo of this team, with the other uniform left out."""

    target = _hex_color(uniform.jersey_hex) or _swatch(uniform.jersey)
    if not _hex_color(target):
        return None
    other_hex = ""
    if other is not None:
        other_hex = _hex_color(other.jersey_hex) or _swatch(other.jersey)
        if not _hex_color(other_hex):
            other_hex = ""
    return full_body_shot(frames, target, other_hex, avoid)


def player_picture(
    frames: Sequence[bytes],
    uniform: Uniform,
    other: Uniform | None = None,
) -> tuple[bytes, str] | None:
    """A player from this team in the film. Nothing is drawn when the film has no shot."""

    shot = team_player_shot(frames, uniform, other)
    if shot is not None:
        return shot, ".jpg"
    photos = player_photos(frames, uniform, other)[:1]
    if not photos:
        return None
    return photos[0], ".jpg"


def reference_for_model(
    frames: Sequence[bytes],
    uniform: Uniform,
    other: Uniform | None = None,
) -> bytes | None:
    """A crop of this team from the film."""

    photos = player_photos(frames, uniform, other)[:1]
    if not photos:
        photos = [
            frame
            for frame in frames
            if isinstance(frame, bytes) and frame.startswith(b"\xff\xd8\xff")
        ][:1]
    return sent_reference(photos)


def player_photos(
    frames: Sequence[bytes],
    uniform: Uniform,
    other: Uniform | None = None,
) -> list[bytes]:
    """The clearest photo of this team only, chosen after the two uniforms are known."""

    target = _hex_color(uniform.jersey_hex) or _swatch(uniform.jersey)
    if not _hex_color(target):
        return []
    other_hex = ""
    if other is not None:
        other_hex = _hex_color(other.jersey_hex) or _swatch(other.jersey)
        if not _hex_color(other_hex):
            other_hex = ""
    ranked: list[tuple[int, bytes]] = []
    seen: set[bytes] = set()
    for frame in frames:
        if not isinstance(frame, bytes) or not frame.startswith(b"\xff\xd8\xff"):
            continue
        for piece in uniform_crops(frame, target):
            if piece in seen:
                continue
            seen.add(piece)
            try:
                jersey, _pants = dominant_colors(piece)
            except (OSError, VideoAnalysisError):
                continue
            if not _hex_color(jersey):
                continue
            own = _rgb_distance(jersey, target)
            if other_hex and _rgb_distance(jersey, other_hex) <= own:
                continue
            ranked.append((own, piece))
    ranked.sort(key=lambda item: item[0])
    return [piece for _score, piece in ranked[:1]]


def _rgb_distance(left: str, right: str) -> int:
    first = tuple(int(left[index : index + 2], 16) for index in (1, 3, 5))
    second = tuple(int(right[index : index + 2], 16) for index in (1, 3, 5))
    return sum((one - two) ** 2 for one, two in zip(first, second, strict=True))


def _apart(left: tuple[int, int, int], right: tuple[int, int, int]) -> bool:
    return sum(abs(a - b) for a, b in zip(left, right, strict=True)) >= 48


def _hex(color: tuple[int, int, int]) -> str:
    return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"


def _image_block(jpeg: bytes) -> dict[str, object]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.standard_b64encode(jpeg).decode("ascii"),
        },
    }


def _frame(
    video: Path,
    at: float,
    scale: str = "scale=640:-2",
    *,
    quality: str = "4",
) -> bytes:
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
                scale,
                "-q:v",
                quality,
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
            thinking={"type": "between_tools"},
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


def _colors_from_players(
    pair: tuple[Uniform, Uniform],
    frames: Sequence[bytes],
) -> tuple[Uniform, Uniform]:
    """Replace a shared field color with the jersey and pants measured on each team."""

    measured = team_colors_from_players(frames)
    if measured is None:
        return pair
    light, solid = measured
    left, right = pair
    left_color = _hex_color(left.jersey_hex) or _swatch(left.jersey)
    if _hex_color(left_color) and _rgb_distance(left_color, solid.jersey) < _rgb_distance(
        left_color, light.jersey
    ):
        return _wear(left, solid), _wear(right, light)
    return _wear(left, light), _wear(right, solid)


def _wear(uniform: Uniform, colors: PlayerColors) -> Uniform:
    jersey = color_words(colors.jersey)
    pants = color_words(colors.pants)
    helmet = color_words(colors.helmet)
    if jersey == "the color in the photo":
        jersey = uniform.jersey
    if pants == "the color in the photo":
        pants = uniform.pants
    if helmet == "the color in the photo":
        helmet = uniform.helmet or jersey
    return replace(
        uniform,
        jersey=jersey,
        pants=pants,
        helmet=helmet,
        jersey_hex=colors.jersey,
        pants_hex=colors.pants,
        helmet_hex=colors.helmet,
    )


def _color_uniforms(
    title: str,
    left_still: bytes | None,
    right_still: bytes | None,
) -> tuple[Uniform, Uniform]:
    del title
    return label_home_away(
        _uniform_from_colors("Home", left_still),
        _uniform_from_colors("Away", right_still),
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
    name = fallback_name or "Home"
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


def sent_reference(still: bytes | Sequence[bytes] | None) -> bytes | None:
    """The first player crop from the film."""

    for photo in _photo_list(still):
        jpeg = _reference_jpeg(photo)
        if jpeg is not None:
            return jpeg
    return None


def _photo_list(still: bytes | Sequence[bytes] | None) -> list[bytes]:
    if still is None:
        return []
    if isinstance(still, bytes):
        return [still]
    return [photo for photo in still if isinstance(photo, bytes)]


def _reference_jpeg(still: bytes) -> bytes | None:
    """A player crop from the film, small enough to store beside the card."""

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
                "scale=-2:480,scale=min(480\\,iw):-2",
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


