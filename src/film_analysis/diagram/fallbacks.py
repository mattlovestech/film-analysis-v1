"""Best guesses for when strict color matching cannot find both teams.

The strict reader in ``chart.isolate`` keeps a player only when the jersey is
clearly one team's color. On wide, blurry, or dark broadcast frames that leaves
too few players to draw. These fallbacks trade certainty for a diagram:

* ``loose_split`` — every person-shaped box, sent to the nearer jersey.
* ``model_points`` — Claude places both teams from the snap frames.
* ``fill_to_eleven`` — any spot still empty is filled from a standard
  formation and marked ``seen: False`` so the page can draw it as a guess.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from film_analysis.chart.isolate import (
    IsolatedPlayer,
    PlayIsolation,
    _band_lab,
    _bgr,
    _candidates,
    _crop,
    _dist,
    _mean_lab,
    _swatch,
    _unit,
)
from film_analysis.chart.play_diagram import _ask, parse_points
from film_analysis.teams.uniforms import Uniform, uniform_line

TEAM_SIZE = 11
# Farther than this from both jerseys is grass, the crowd, or a referee.
_LOOSE_LIMIT = 70.0

# Shotgun, 11 personnel: the usual look on Friday nights and Saturdays.
OFFENSE_TEMPLATE: tuple[tuple[float, float, str, str], ...] = (
    (0.0, -0.5, "center", "square"),
    (-1.3, -0.5, "line", "open"),
    (1.3, -0.5, "line", "open"),
    (-2.6, -0.6, "line", "open"),
    (2.6, -0.6, "line", "open"),
    (0.0, -5.0, "back", "filled"),
    (1.5, -5.5, "back", "filled"),
    (4.0, -0.8, "line", "open"),
    (-17.0, -0.5, "line", "open"),
    (15.0, -1.5, "back", "filled"),
    (9.0, -1.5, "back", "filled"),
)
# 4-2-5 over that look.
DEFENSE_TEMPLATE: tuple[tuple[float, float, str, str], ...] = (
    (-1.0, 1.0, "line", "open"),
    (1.0, 1.0, "line", "open"),
    (-3.2, 1.0, "line", "open"),
    (3.2, 1.0, "line", "open"),
    (-2.5, 5.0, "coverage", "filled"),
    (2.5, 5.0, "coverage", "filled"),
    (-17.0, 6.0, "coverage", "filled"),
    (15.0, 6.0, "coverage", "filled"),
    (9.0, 4.0, "coverage", "filled"),
    (-7.0, 12.0, "coverage", "filled"),
    (7.0, 12.0, "coverage", "filled"),
)

_MODEL_PROMPT = (
    "These frames are the start of one American football scrimmage play, in order. "
    "Team A wears {first}. Team B wears {second}. "
    "Use the frame with the widest view of the pre-snap lineup. "
    "First decide which team is on offense: the offense has the ball, the "
    "quarterback, and a row of linemen beside the center. "
    "Then place all 22 players as best you can, even ones partly hidden or out "
    "of frame; guess their spots from a normal lineup when you cannot see them, "
    'and set "seen" false for a guess. '
    "x is yards to the viewer's right of the ball; left is negative; the field "
    "is 53.3 yards wide. y is yards toward the goal the offense attacks; the "
    "line of scrimmage is y=0, the offense has negative y, the defense positive. "
    "Offense symbols: square for the center, open for the other linemen and "
    "ends on the line, filled for backs and receivers off the line. "
    "Reply with JSON only: "
    '{{"offense_team":"A","offense":[{{"x":0,"y":-0.5,"role":"center","symbol":"square","seen":true}}],'
    '"defense":[{{"x":0,"y":1,"role":"line","symbol":"open","seen":true}}]}}'
)


def loose_split(frames: Sequence[bytes], home: Uniform, away: Uniform) -> PlayIsolation:
    """Every person found, sent to the nearer jersey. Best frame wins."""

    first, second = _swatch(home), _swatch(away)
    if first is None or second is None:
        return PlayIsolation()
    best: tuple[int, bytes, list[IsolatedPlayer], list[IsolatedPlayer]] | None = None
    for frame in list(frames)[:8]:
        image = _bgr(frame)
        if image is None:
            continue
        height, width = image.shape[:2]
        area = height * width
        mine: list[IsolatedPlayer] = []
        other: list[IsolatedPlayer] = []
        for box in _candidates(frame, image, first, second, area):
            crop = _crop(image, box)
            if crop is None:
                continue
            color = _band_lab(crop, 0.2, 0.55) or _mean_lab(crop)
            if color is None:
                continue
            near_home, near_away = _dist(color, first.jersey), _dist(color, second.jersey)
            if min(near_home, near_away) > _LOOSE_LIMIT:
                continue
            player = IsolatedPlayer(
                (box.x + box.width / 2) / width, (box.y + box.height) / height, "low"
            )
            if not 0.12 < player.y < 0.95:
                continue
            (mine if near_home <= near_away else other).append(player)
        mine, other = mine[:TEAM_SIZE], other[:TEAM_SIZE]
        score = min(len(mine), len(other)) * 100 + len(mine) + len(other)
        if best is None or score > best[0]:
            best = (score, frame, mine, other)
    if best is None:
        return PlayIsolation()
    return PlayIsolation(tuple(best[2]), tuple(best[3]), _unit(best[2], best[3]), "low", best[1])


def model_points(
    frames: Sequence[bytes], home: Uniform, away: Uniform
) -> tuple[dict[str, object], str] | None:
    """Claude's placement of both teams, and which team it put on offense."""

    reply = _ask(
        frames,
        _MODEL_PROMPT.format(first=uniform_line(home), second=uniform_line(away)),
        limit=4,
        tokens=3000,
    )
    if reply is None:
        return None
    parsed = parse_points(reply)
    if parsed is None:
        return None
    side = "home"
    try:
        start = reply.index("{")
        payload = json.loads(reply[start : reply.rindex("}") + 1])
        if str(payload.get("offense_team") or "").strip().upper() == "B":
            side = "away"
    except (ValueError, AttributeError):
        pass
    return parsed, side


def fill_to_eleven(players: list[dict[str, object]], template: Sequence[tuple]) -> int:
    """Add template spots no seen player already covers. Returns how many were added."""

    if len(players) >= TEAM_SIZE:
        return 0
    spots = list(template)
    # Each seen player claims the nearest open spot, closest pairs first.
    pairs = sorted(
        (
            (float(p["x"]) - spot[0]) ** 2 + (float(p["y"]) - spot[1]) ** 2,
            index,
            spot_index,
        )
        for index, p in enumerate(players)
        for spot_index, spot in enumerate(spots)
    )
    taken_players: set[int] = set()
    taken_spots: set[int] = set()
    for _distance, index, spot_index in pairs:
        if index in taken_players or spot_index in taken_spots:
            continue
        taken_players.add(index)
        taken_spots.add(spot_index)
    added = 0
    for spot_index, (x, y, role, symbol) in enumerate(spots):
        if len(players) >= TEAM_SIZE:
            break
        if spot_index in taken_spots:
            continue
        players.append({"x": x, "y": y, "role": role, "symbol": symbol, "label": "", "seen": False})
        added += 1
    return added


def mark_players(jpeg: bytes, found: PlayIsolation, home: Uniform, away: Uniform) -> bytes | None:
    """Circle the players a fallback kept, home in white and away in red."""

    image = _bgr(jpeg)
    if image is None:
        return None
    try:
        import cv2
    except ImportError:
        return None
    height, width = image.shape[:2]
    radius = max(12, int(min(width, height) * 0.03))
    for players, color in ((found.mine, (245, 245, 245)), (found.other, (40, 40, 220))):
        for player in players:
            center = (int(player.x * width), int(min(height - 6, player.y * height)))
            cv2.circle(image, center, radius, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.circle(image, center, radius, color, 2, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return encoded.tobytes() if ok else None
