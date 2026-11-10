"""Isolate the scouted uniform on one play.

The film already knows both jerseys. This read finds those colors in a pre-snap
frame, drops referees, and decides which color is lined up on offense.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from film_analysis.teams.players import PlayerBox, _striped, find_players
from film_analysis.teams.uniforms import Uniform

# OpenCV Lab. Two jerseys closer than this are the same dark cloth unless
# pants or helmets separate them. White against red is far past it.
_SEPARATE = 28.0
# A pixel or a torso must be this much closer to one uniform than the other.
_MARGIN = 18.0
# Farther than this from both uniforms is grass, a shadow, or the crowd.
_LIMIT = 42.0
_TEAM_SIZE = 11


@dataclass(frozen=True)
class IsolatedPlayer:
    """One player kept for a uniform, in fractions of the frame.

    ``x`` is the center of the body. ``y`` is the feet, so a team lined up
    deeper in the picture sits at a larger ``y``.
    """

    x: float
    y: float
    confidence: str = "high"


@dataclass(frozen=True)
class PlayIsolation:
    """The scouted uniform on one snap, and which side of the ball it is on."""

    mine: tuple[IsolatedPlayer, ...] = ()
    other: tuple[IsolatedPlayer, ...] = ()
    unit: str | None = None
    confidence: str = "low"
    frame: bytes = b""


@dataclass(frozen=True)
class _Swatch:
    jersey: tuple[float, float, float]
    pants: tuple[float, float, float] | None = None
    helmet: tuple[float, float, float] | None = None


def isolate_play(
    frames: Sequence[bytes],
    mine: Uniform,
    other: Uniform,
) -> PlayIsolation:
    """Find the scouted jersey in these frames and which side of the ball it is on.

    A clear row with teammates behind it, facing the other color, is the
    offense. Close colors and thin frames stay low confidence and name no side.
    """

    if not _can_separate(mine, other):
        return PlayIsolation()
    best: tuple[int, bytes, list[IsolatedPlayer], list[IsolatedPlayer]] | None = None
    for frame in list(frames)[:8]:
        if not isinstance(frame, bytes) or not frame.startswith(b"\xff\xd8\xff"):
            continue
        mine_players, other_players = split_players(frame, mine, other)
        score = len(mine_players) + len(other_players)
        if best is None or score > best[0]:
            best = (score, frame, mine_players, other_players)
    if best is None:
        return PlayIsolation()
    unit = _unit(best[2], best[3])
    return PlayIsolation(
        tuple(best[2]),
        tuple(best[3]),
        unit,
        "high" if unit in {"offense", "defense"} else "low",
        best[1],
    )


def split_players(
    jpeg: bytes,
    first: Uniform,
    second: Uniform,
) -> tuple[list[IsolatedPlayer], list[IsolatedPlayer]]:
    """Players wearing ``first``, then players wearing ``second``."""

    if not isinstance(jpeg, bytes) or not jpeg.startswith(b"\xff\xd8\xff"):
        return [], []
    if not _can_separate(first, second):
        return [], []
    image = _bgr(jpeg)
    swatches = (_swatch(first), _swatch(second))
    if image is None or swatches[0] is None or swatches[1] is None:
        return [], []
    height, width = image.shape[:2]
    frame_area = height * width
    first_players: list[IsolatedPlayer] = []
    second_players: list[IsolatedPlayer] = []
    for box in _candidates(jpeg, image, swatches[0], swatches[1], frame_area):
        crop = _crop(image, box)
        if crop is None or _striped(crop):
            continue
        side = _assign(crop, swatches[0], swatches[1], box, frame_area)
        if side is None:
            continue
        player = IsolatedPlayer(
            (box.x + box.width / 2) / width,
            (box.y + box.height) / height,
        )
        if side == "first":
            first_players.append(player)
        else:
            second_players.append(player)
    return first_players[:_TEAM_SIZE], second_players[:_TEAM_SIZE]


def _can_separate(first: Uniform, second: Uniform) -> bool:
    left = _swatch(first)
    right = _swatch(second)
    if left is None or right is None:
        return False
    if _dist(left.jersey, right.jersey) >= _SEPARATE:
        return True
    for pair in ((left.pants, right.pants), (left.helmet, right.helmet)):
        if pair[0] is not None and pair[1] is not None and _dist(pair[0], pair[1]) >= _SEPARATE:
            return True
    return False


def _unit(
    mine: Sequence[IsolatedPlayer],
    other: Sequence[IsolatedPlayer],
) -> str | None:
    """Offense is the color with a tight row and teammates behind that row."""

    if len(mine) < 3 or len(other) < 3:
        return None
    mine_pts = [(player.x, player.y) for player in mine]
    other_pts = [(player.x, player.y) for player in other]
    mine_center = _centroid(mine_pts)
    other_center = _centroid(other_pts)
    if math.hypot(mine_center[0] - other_center[0], mine_center[1] - other_center[1]) < 0.06:
        return None
    mine_behind = _backfield(mine_pts, other_center)
    other_behind = _backfield(other_pts, mine_center)
    if mine_behind >= 1 and mine_behind > other_behind:
        return "offense"
    if other_behind >= 1 and other_behind > mine_behind:
        return "defense"
    return None


def _backfield(
    points: Sequence[tuple[float, float]],
    opponent: tuple[float, float],
) -> int:
    """How many teammates stand behind a tight row, away from the other color."""

    if len(points) < 3:
        return 0
    center = _centroid(points)
    axis = (center[0] - opponent[0], center[1] - opponent[1])
    span = math.hypot(axis[0], axis[1])
    if span < 0.05:
        return 0
    unit_x, unit_y = axis[0] / span, axis[1] / span
    depths = [(x - opponent[0]) * unit_x + (y - opponent[1]) * unit_y for x, y in points]
    front = min(depths)
    band = 0.045
    front_depths = [depth for depth in depths if depth <= front + band]
    if len(front_depths) < 3 or max(front_depths) - min(front_depths) > band:
        return 0
    return sum(1 for depth in depths if depth > front + band + 0.04)


def _centroid(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    count = len(points)
    return (
        sum(x for x, _y in points) / count,
        sum(y for _x, y in points) / count,
    )


def _candidates(
    jpeg: bytes,
    image,
    first: _Swatch,
    second: _Swatch,
    frame_area: int,
) -> list[PlayerBox]:
    boxes = list(find_players(jpeg))
    boxes.extend(_mask_boxes(image, first, second, frame_area))
    return _dedupe(boxes)


def _mask_boxes(
    image,
    first: _Swatch,
    second: _Swatch,
    frame_area: int,
) -> list[PlayerBox]:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []
    boxes: list[PlayerBox] = []
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    for own, away in _mask_pairs(first, second):
        distance_own = np.linalg.norm(lab - np.array(own, dtype=np.float32), axis=2)
        distance_away = np.linalg.norm(lab - np.array(away, dtype=np.float32), axis=2)
        mask = ((distance_own + _MARGIN < distance_away) & (distance_own < _LIMIT)).astype(
            np.uint8
        ) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            box = PlayerBox(int(x), int(y), int(width), int(height))
            area = box.width * box.height
            if area < frame_area * 0.0008 or area > frame_area * 0.12:
                continue
            boxes.append(box)
    return boxes


def _mask_pairs(
    first: _Swatch,
    second: _Swatch,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """Colors that can paint a mask. Jersey first; pants or helmet when it is dark-on-dark."""

    if _dist(first.jersey, second.jersey) >= _SEPARATE:
        return [(first.jersey, second.jersey), (second.jersey, first.jersey)]
    pairs: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    for left, right in ((first.pants, second.pants), (first.helmet, second.helmet)):
        if left is None or right is None or _dist(left, right) < _SEPARATE:
            continue
        pairs.append((left, right))
        pairs.append((right, left))
    return pairs


def _assign(
    crop,
    first: _Swatch,
    second: _Swatch,
    box: PlayerBox,
    frame_area: int,
) -> str | None:
    if _standing(box, frame_area):
        jersey = _band_lab(crop, 0.22, 0.52)
        if jersey is not None:
            own = _dist(jersey, first.jersey)
            away = _dist(jersey, second.jersey)
            if _decisive(own, away):
                return "first" if own < away else "second"
        extra_first = _extra(crop, first)
        extra_second = _extra(crop, second)
        if extra_first is None or extra_second is None or not _decisive(extra_first, extra_second):
            return None
        return "first" if extra_first < extra_second else "second"
    return _pants_label(crop, first, second)


def _extra(crop, swatch: _Swatch) -> float | None:
    """Pants and helmet distance, used when the two jerseys are both dark."""

    parts: list[float] = []
    pants = _band_lab(crop, 0.55, 0.85)
    helmet = _band_lab(crop, 0.02, 0.20)
    if pants is not None and swatch.pants is not None:
        parts.append(_dist(pants, swatch.pants))
    if helmet is not None and swatch.helmet is not None:
        parts.append(_dist(helmet, swatch.helmet))
    if not parts:
        return None
    return sum(parts) / len(parts)


def _pants_label(crop, first: _Swatch, second: _Swatch) -> str | None:
    if first.pants is None or second.pants is None:
        return None
    sample = _mean_lab(crop)
    if sample is None:
        return None
    own = _dist(sample, first.pants)
    away = _dist(sample, second.pants)
    if not _decisive(own, away):
        return None
    return "first" if own < away else "second"


def _decisive(own: float, away: float) -> bool:
    return min(own, away) <= _LIMIT and abs(own - away) >= _MARGIN


def _standing(box: PlayerBox, frame_area: int) -> bool:
    area = box.width * box.height
    tall = box.height >= 28 and box.height >= box.width * 1.15 and box.width >= 10
    sized = frame_area * 0.0015 <= area <= frame_area * 0.12
    return tall and sized


def _band_lab(crop, top: float, bottom: float) -> tuple[float, float, float] | None:
    height = crop.shape[0]
    width = crop.shape[1]
    band = crop[int(height * top) : int(height * bottom), int(width * 0.25) : int(width * 0.75)]
    return _mean_lab(band)


def _mean_lab(crop) -> tuple[float, float, float] | None:
    if crop is None or crop.size == 0:
        return None
    kept: list[tuple[int, int, int]] = []
    for blue, green, red in crop.reshape(-1, 3)[::2]:
        red, green, blue = int(red), int(green), int(blue)
        if green > red + 12 and green > blue + 8 and green > 70:
            continue
        kept.append((red, green, blue))
    if len(kept) < 8:
        return None
    count = len(kept)
    red = round(sum(pixel[0] for pixel in kept) / count)
    green = round(sum(pixel[1] for pixel in kept) / count)
    blue = round(sum(pixel[2] for pixel in kept) / count)
    return _lab_color((red, green, blue))


def _crop(image, box: PlayerBox):
    height, width = image.shape[:2]
    x0 = max(0, box.x)
    y0 = max(0, box.y)
    x1 = min(width, box.x + box.width)
    y1 = min(height, box.y + box.height)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return image[y0:y1, x0:x1]


def _dedupe(boxes: Sequence[PlayerBox]) -> list[PlayerBox]:
    ordered = sorted(boxes, key=lambda box: box.width * box.height, reverse=True)
    kept: list[PlayerBox] = []
    for box in ordered:
        if any(_overlaps(box, other) for other in kept):
            continue
        kept.append(box)
    return kept


def _overlaps(left: PlayerBox, right: PlayerBox) -> bool:
    if _iou(left, right) >= 0.3:
        return True
    center_x = left.x + left.width / 2
    center_y = left.y + left.height / 2
    return (
        right.x <= center_x <= right.x + right.width
        and right.y <= center_y <= right.y + right.height
    )


def _iou(left: PlayerBox, right: PlayerBox) -> float:
    x0 = max(left.x, right.x)
    y0 = max(left.y, right.y)
    x1 = min(left.x + left.width, right.x + right.width)
    y1 = min(left.y + left.height, right.y + right.height)
    overlap = max(0, x1 - x0) * max(0, y1 - y0)
    if overlap == 0:
        return 0.0
    union = left.width * left.height + right.width * right.height - overlap
    if union <= 0:
        return 0.0
    return overlap / union


def _swatch(uniform: Uniform) -> _Swatch | None:
    jersey = _lab_color(_color(uniform.jersey_hex, uniform.jersey))
    if jersey is None:
        return None
    return _Swatch(
        jersey,
        _lab_color(_rgb(uniform.pants_hex)),
        _lab_color(_rgb(uniform.helmet_hex)),
    )


def _color(hex_code: str, name: str) -> tuple[int, int, int] | None:
    return _rgb(hex_code) or _rgb_from_name(name)


def _rgb(hex_code: str) -> tuple[int, int, int] | None:
    text = (hex_code or "").strip()
    if len(text) != 7 or not text.startswith("#"):
        return None
    try:
        return int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16)
    except ValueError:
        return None


def _rgb_from_name(name: str) -> tuple[int, int, int] | None:
    words = (name or "").lower()
    if "white" in words:
        return 244, 241, 234
    if "red" in words:
        return 180, 35, 24
    if "blue" in words or "navy" in words:
        return 28, 63, 115
    if "black" in words or "dark" in words:
        return 24, 24, 24
    if "green" in words:
        return 28, 110, 48
    if "yellow" in words or "gold" in words:
        return 220, 180, 40
    return None


def _lab_color(rgb: tuple[int, int, int] | None) -> tuple[float, float, float] | None:
    if rgb is None:
        return None
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    pixel = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0, 0]
    return float(lab[0]), float(lab[1]), float(lab[2])


def _dist(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _bgr(jpeg: bytes):
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    array = np.frombuffer(jpeg, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    return image
