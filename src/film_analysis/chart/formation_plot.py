"""Spatial 11-on-11 plot of both teams before the snap.

The isolator splits a frame into the two jerseys. Missing spots are filled
with a best guess so each side always has 11 markers, and those guesses are
marked low confidence on the plot.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from film_analysis.chart.isolate import split_players
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
    """Draw a static XY chart of the lineup. X is width and Y is depth.

    The line of scrimmage sits at y=0.5. Offense is the larger-y side of that
    line. The focused side — the team being scouted — is drawn larger.
    """

    try:
        import cv2
        import numpy as np
    except ImportError:
        return b""
    width, height = 720, 480
    chart = np.zeros((height, width, 3), dtype=np.uint8)
    chart[:] = (22, 24, 22)
    left, top, right, bottom = 72, 40, width - 28, height - 52
    axis = (170, 174, 170)
    cv2.line(chart, (left, top), (left, bottom), axis, 1, cv2.LINE_AA)
    cv2.line(chart, (left, bottom), (right, bottom), axis, 1, cv2.LINE_AA)
    los = int(top + 0.5 * (bottom - top))
    cv2.line(chart, (left, los), (right, los), (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(
        chart,
        "LOS",
        (right - 52, los - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        chart,
        "Width",
        ((left + right) // 2 - 28, height - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        axis,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        chart,
        "Depth",
        (8, (top + bottom) // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        axis,
        1,
        cv2.LINE_AA,
    )
    offense_color = _bgr_from_hex(offense_uniform.jersey_hex) or (240, 240, 240)
    defense_color = _bgr_from_hex(defense_uniform.jersey_hex) or (40, 40, 200)
    offense_primary = focus != "defense"
    box = (left, top, right, bottom)
    for player in defense:
        _draw_marker(
            chart,
            player,
            defense_color if not offense_primary else _mute(defense_color),
            box,
            primary=not offense_primary,
        )
    for player in offense:
        _draw_marker(
            chart,
            player,
            offense_color if offense_primary else _mute(offense_color),
            box,
            primary=offense_primary,
        )
    cv2.putText(
        chart,
        "Offense",
        (left, height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        offense_color if offense_primary else _mute(offense_color),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        chart,
        "Defense",
        (left, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        defense_color if not offense_primary else _mute(defense_color),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        chart,
        "Hollow = guessed",
        (width - 210, height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    ok, encoded = cv2.imencode(".jpg", chart, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
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
    """The isolator's boxes, so the plot uses the same players as the side call."""

    offense_players, defense_players = split_players(jpeg, offense, defense)
    return (
        [(player.x, player.y) for player in offense_players[:_TEAM_SIZE]],
        [(player.x, player.y) for player in defense_players[:_TEAM_SIZE]],
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
    box: tuple[int, int, int, int],
    *,
    primary: bool = True,
) -> None:
    import cv2

    left, top, right, bottom = box
    x = int(left + player.x * (right - left))
    # Points are already LOS-normalized: y>0.5 offense, y<0.5 defense.
    y = int(top + player.y * (bottom - top))
    y = max(top + 8, min(bottom - 8, y))
    x = max(left + 8, min(right - 8, x))
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


def _rgb(hex_color: str) -> tuple[int, int, int] | None:
    text = (hex_color or "").strip()
    if len(text) != 7 or not text.startswith("#"):
        return None
    try:
        return int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16)
    except ValueError:
        return None


def _bgr_from_hex(hex_color: str) -> tuple[int, int, int] | None:
    rgb = _rgb(hex_color)
    if rgb is None:
        return None
    return rgb[2], rgb[1], rgb[0]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
