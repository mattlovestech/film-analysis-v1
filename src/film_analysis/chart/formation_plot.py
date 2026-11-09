"""Spatial 11-on-11 plot of both teams before the snap.

OpenCV finds people in a film frame. Jersey color splits them into offense and
defense. Missing spots are filled with a best guess so each side always has 11
markers, and those guesses are marked low confidence on the plot.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from film_analysis.teams.players import PlayerBox, find_players
from film_analysis.teams.uniforms import Uniform

_TEAM_SIZE = 11


@dataclass(frozen=True)
class PlottedPlayer:
    """One player on the formation plot, in field coordinates 0..1."""

    x: float
    y: float
    side: str
    confidence: str


@dataclass(frozen=True)
class FormationLayout:
    """Both teams plotted, with a JPEG bird's-eye view of the snap."""

    offense: tuple[PlottedPlayer, ...]
    defense: tuple[PlottedPlayer, ...]
    plot: bytes
    frame: bytes
    guessed: int


def plot_formation(
    frames: Sequence[bytes],
    offense: Uniform,
    defense: Uniform,
    *,
    focus: str = "offense",
) -> FormationLayout | None:
    """Build an 11-on-11 plot from the clearest pre-snap frame.

    Coordinates are LOS-relative using the O-line as the stable anchor. ``focus``
    is ``offense`` or ``defense`` — that side's jersey color is drawn primary.
    """

    picked = _best_frame(frames, offense, defense)
    if picked is None:
        return None
    jpeg, offense_pts, defense_pts = picked
    offense_pts, defense_pts = normalize_to_los(offense_pts, defense_pts)
    offense_players = _fill_team(offense_pts, "offense", backfield=True)
    defense_players = _fill_team(defense_pts, "defense", backfield=False)
    guessed = sum(
        1 for player in (*offense_players, *defense_players) if player.confidence == "low"
    )
    drawing = render_formation_plot(
        offense_players,
        defense_players,
        offense,
        defense,
        focus=focus if focus in {"offense", "defense"} else "offense",
    )
    if not drawing:
        return None
    return FormationLayout(
        tuple(offense_players),
        tuple(defense_players),
        drawing,
        jpeg,
        guessed,
    )


def normalize_to_los(
    offense: Sequence[tuple[float, float]],
    defense: Sequence[tuple[float, float]],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Shift points so the O-line sits on y=0.5 and the center is x=0.5.

    Offense stays below the LOS (y > 0.5); defense above (y < 0.5). Ball
    direction is mirrored so the offense always faces up the diagram.
    """

    if not offense:
        return list(offense), list(defense)
    from film_analysis.chart.formation_features import _oline_anchor

    los_y, center_x = _oline_anchor(offense)
    # If most "offense" points are above the line, the detector flipped sides —
    # keep geometry but mirror so offense is in the bottom half for drawing.
    offense_below = sum(1 for _x, y in offense if y >= los_y)
    mirror = offense_below < len(offense) / 2

    def map_point(x: float, y: float, *, offense_side: bool) -> tuple[float, float]:
        dx = x - center_x
        dy = y - los_y
        if mirror:
            dy = -dy
        # Diagram: LOS at 0.5; offense deeper = larger y; defense = smaller y.
        out_y = 0.5 + abs(dy) * 0.9 if offense_side else 0.5 - abs(dy) * 0.9
        return _clamp(0.5 + dx), _clamp(out_y)

    return (
        [map_point(x, y, offense_side=True) for x, y in offense],
        [map_point(x, y, offense_side=False) for x, y in defense],
    )


def layout_payload(
    layout: FormationLayout,
    *,
    features: dict[str, object] | None = None,
    ai_guess: str = "",
    focus: str = "offense",
) -> dict[str, object]:
    """JSON saved beside the plot for the movable graph and grouping."""

    return {
        "offense": [_player_dict(player) for player in layout.offense],
        "defense": [_player_dict(player) for player in layout.defense],
        "guessed": layout.guessed,
        "ai_guess": ai_guess,
        "focus": focus,
        "features": features or {},
    }


def players_from_payload(
    payload: dict[str, object],
) -> tuple[list[PlottedPlayer], list[PlottedPlayer]]:
    """Rebuild plotted players from a saved layout JSON."""

    return (
        [_player_from_dict(item, "offense") for item in _list(payload.get("offense"))],
        [_player_from_dict(item, "defense") for item in _list(payload.get("defense"))],
    )


def render_formation_plot(
    offense: Sequence[PlottedPlayer],
    defense: Sequence[PlottedPlayer],
    offense_uniform: Uniform,
    defense_uniform: Uniform,
    *,
    focus: str = "offense",
) -> bytes:
    """Draw a bird's-eye view of the snap. The focused side is louder."""

    try:
        import cv2
        import numpy as np
    except ImportError:
        return b""
    width, height = 720, 480
    field = np.zeros((height, width, 3), dtype=np.uint8)
    field[:] = (34, 92, 48)
    for yard in range(1, 10):
        x = int(width * yard / 10)
        cv2.line(field, (x, 24), (x, height - 24), (48, 120, 64), 1, cv2.LINE_AA)
    mid = height // 2
    cv2.line(field, (24, mid), (width - 24, mid), (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(
        field,
        "LOS",
        (width - 70, mid - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    offense_color = _bgr_from_hex(offense_uniform.jersey_hex) or (240, 240, 240)
    defense_color = _bgr_from_hex(defense_uniform.jersey_hex) or (40, 40, 200)
    # Scouted focus: primary side is large and saturated; the other is muted.
    offense_primary = focus != "defense"
    for player in defense:
        _draw_marker(
            field,
            player,
            defense_color if not offense_primary else _mute(defense_color),
            width,
            height,
            mid,
            offense_side=False,
            primary=not offense_primary,
        )
    for player in offense:
        _draw_marker(
            field,
            player,
            offense_color if offense_primary else _mute(offense_color),
            width,
            height,
            mid,
            offense_side=True,
            primary=offense_primary,
        )
    cv2.putText(
        field,
        "Offense",
        (24, height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        offense_color if offense_primary else _mute(offense_color),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        field,
        "Defense",
        (24, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        defense_color if not offense_primary else _mute(defense_color),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        field,
        "Hollow = low confidence",
        (width - 250, height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    ok, encoded = cv2.imencode(".jpg", field, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        return b""
    return encoded.tobytes()


def _best_frame(
    frames: Sequence[bytes],
    offense: Uniform,
    defense: Uniform,
) -> tuple[bytes, list[tuple[float, float]], list[tuple[float, float]]] | None:
    best: tuple[int, bytes, list[tuple[float, float]], list[tuple[float, float]]] | None = None
    for jpeg in frames[:8]:
        if not isinstance(jpeg, bytes) or not jpeg.startswith(b"\xff\xd8\xff"):
            continue
        offense_pts, defense_pts = _team_points(jpeg, offense, defense)
        score = len(offense_pts) + len(defense_pts)
        if score < 4:
            continue
        if best is None or score > best[0]:
            best = (score, jpeg, offense_pts, defense_pts)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _team_points(
    jpeg: bytes,
    offense: Uniform,
    defense: Uniform,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return [], []
    array = np.frombuffer(jpeg, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return [], []
    height, width = image.shape[:2]
    if width < 8 or height < 8:
        return [], []
    offense_rgb = _rgb(offense.jersey_hex) or _rgb_from_name(offense.jersey)
    defense_rgb = _rgb(defense.jersey_hex) or _rgb_from_name(defense.jersey)
    offense_pts: list[tuple[float, float, int]] = []
    defense_pts: list[tuple[float, float, int]] = []
    for box in find_players(jpeg):
        sample = _jersey_sample(image, box)
        if sample is None:
            continue
        own_off = _distance(sample, offense_rgb) if offense_rgb else 10**9
        own_def = _distance(sample, defense_rgb) if defense_rgb else 10**9
        x = (box.x + box.width / 2) / width
        y = (box.y + box.height) / height
        if own_off <= own_def:
            offense_pts.append((x, y, box.width * box.height))
        else:
            defense_pts.append((x, y, box.width * box.height))
    offense_pts.sort(key=lambda item: item[2], reverse=True)
    defense_pts.sort(key=lambda item: item[2], reverse=True)
    return (
        [(x, y) for x, y, _area in offense_pts[:_TEAM_SIZE]],
        [(x, y) for x, y, _area in defense_pts[:_TEAM_SIZE]],
    )


def _fill_team(
    points: Sequence[tuple[float, float]],
    side: str,
    *,
    backfield: bool,
) -> list[PlottedPlayer]:
    """Keep detected players and invent the rest so the side always has 11."""

    kept = sorted(points, key=lambda point: point[0])[:_TEAM_SIZE]
    players = [
        PlottedPlayer(_clamp(x), _clamp(y), side, "high") for x, y in kept
    ]
    missing = _TEAM_SIZE - len(players)
    if missing <= 0:
        return players
    template = _template_slots(backfield)
    used = {(round(player.x, 2), round(player.y, 2)) for player in players}
    for x, y in template:
        if len(players) >= _TEAM_SIZE:
            break
        key = (round(x, 2), round(y, 2))
        if key in used:
            continue
        # Prefer gaps: skip slots that sit on top of a high-confidence player.
        if any(abs(player.x - x) < 0.05 and abs(player.y - y) < 0.05 for player in players):
            continue
        players.append(PlottedPlayer(x, y, side, "low"))
        used.add(key)
    while len(players) < _TEAM_SIZE:
        index = len(players)
        x = 0.08 + (0.84 * index / max(1, _TEAM_SIZE - 1))
        y = 0.72 if backfield else 0.28
        players.append(PlottedPlayer(_clamp(x), _clamp(y), side, "low"))
    return sorted(players, key=lambda player: (player.y, player.x))


def _template_slots(backfield: bool) -> list[tuple[float, float]]:
    """A generic 11-man shape used only to fill holes the detector missed."""

    if backfield:
        # Offense: OL across the LOS, backs/receivers behind.
        line = [(0.18 + 0.16 * index, 0.58) for index in range(5)]
        skill = [
            (0.08, 0.70),
            (0.22, 0.78),
            (0.50, 0.82),
            (0.78, 0.78),
            (0.92, 0.70),
            (0.50, 0.68),
        ]
        return line + skill
    line = [(0.18 + 0.16 * index, 0.42) for index in range(5)]
    secondary = [
        (0.12, 0.28),
        (0.28, 0.22),
        (0.50, 0.18),
        (0.72, 0.22),
        (0.88, 0.28),
        (0.50, 0.32),
    ]
    return line + secondary


def _draw_marker(
    field,
    player: PlottedPlayer,
    color: tuple[int, int, int],
    width: int,
    height: int,
    mid: int,
    *,
    offense_side: bool,
    primary: bool = True,
) -> None:
    import cv2

    x = int(24 + player.x * (width - 48))
    # Points are already LOS-normalized: y>0.5 offense, y<0.5 defense.
    y = int(24 + player.y * (height - 48))
    y = max(30, min(height - 30, y))
    radius = 12 if primary else 7
    if player.confidence == "low":
        cv2.circle(field, (x, y), radius, color, 2, cv2.LINE_AA)
        if primary:
            cv2.putText(
                field,
                "?",
                (x - 4, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
    else:
        cv2.circle(field, (x, y), radius, color, -1, cv2.LINE_AA)
        cv2.circle(field, (x, y), radius, (20, 20, 20), 1, cv2.LINE_AA)


def _mute(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(min(255, int(40 + channel * 0.45)) for channel in color)


def _player_dict(player: PlottedPlayer) -> dict[str, object]:
    return {
        "x": round(player.x, 4),
        "y": round(player.y, 4),
        "side": player.side,
        "confidence": player.confidence,
    }


def _player_from_dict(value: object, side: str) -> PlottedPlayer:
    if not isinstance(value, dict):
        return PlottedPlayer(0.5, 0.5, side, "low")
    return PlottedPlayer(
        _clamp(float(value.get("x") or 0.5)),
        _clamp(float(value.get("y") or 0.5)),
        str(value.get("side") or side),
        "high" if value.get("confidence") == "high" else "low",
    )


def _list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    return []


def _jersey_sample(image, box: PlayerBox) -> tuple[int, int, int] | None:
    height, width = image.shape[:2]
    x0 = max(0, box.x)
    y0 = max(0, box.y)
    x1 = min(width, box.x + box.width)
    y1 = min(height, box.y + max(1, int(box.height * 0.55)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    # OpenCV is BGR.
    mean = crop.reshape(-1, 3).mean(axis=0)
    return int(mean[2]), int(mean[1]), int(mean[0])


def _rgb(hex_color: str) -> tuple[int, int, int] | None:
    text = (hex_color or "").strip()
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
    if "blue" in words:
        return 28, 63, 115
    if "black" in words or "dark" in words:
        return 24, 24, 24
    if "green" in words:
        return 28, 110, 48
    if "yellow" in words or "gold" in words:
        return 220, 180, 40
    return None


def _bgr_from_hex(hex_color: str) -> tuple[int, int, int] | None:
    rgb = _rgb(hex_color)
    if rgb is None:
        return None
    return rgb[2], rgb[1], rgb[0]


def _distance(left: tuple[int, int, int], right: tuple[int, int, int] | None) -> int:
    if right is None:
        return 10**9
    return sum((a - b) ** 2 for a, b in zip(left, right, strict=True))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
