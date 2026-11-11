"""Read one scrimmage clip in order: the players on each jersey, then their x and y.

The diagram is JSON. A standard field is 120 yards long and 53.3 yards wide.
The ball is the origin on the 50. Positive y runs toward the goal the offense
is attacking. A player at the edge of the frame stays about 12 yards from the ball.
"""

from __future__ import annotations

import base64
import json
import os
import re
from collections.abc import Callable, Sequence

from film_analysis.chart.formation import FORMATIONS, parse_formation
from film_analysis.chart.formation_plot import normalize_to_los
from film_analysis.chart.isolate import isolate_play
from film_analysis.plays.boundaries import DEFAULT_MODEL
from film_analysis.teams.uniforms import Uniform, label_home_away, uniform_line

FIELD_LENGTH = 120.0
FIELD_WIDTH = 53.3
_HALF_WIDTH = FIELD_WIDTH / 2
_HALF_LENGTH = FIELD_LENGTH / 2
# Half the frame is the lineup we can see, kept to about 12 yards so a back
# at the bottom of the picture stays on the diagram.
_SIGHT_YARDS = 12.0
_FRAME_TO_YARDS = _SIGHT_YARDS / 0.5
_SYMBOLS = frozenset({"square", "open", "filled"})
_CALLS = frozenset({"run", "pass"})

_CALL_PROMPT = (
    "These frames are one scrimmage play, in order. "
    "The offense wears {offense}. The defense wears {defense}. "
    "The offense is the team that has the ball. "
    "Say whether the offense ran or passed. "
    "A run is a handoff, a pitch, or a quarterback keeper or scramble. "
    "A pass is a throw, including a sack or an incompletion, "
    "when the play was an attempted pass. "
    "Do not name a formation. Do not explain. "
    'Reply with JSON only: {{"call":"run"}}'
)
_PLACE_PROMPT = (
    "These frames are the pre-snap lineup. "
    "The offense wears {offense}. The defense wears {defense}. "
    "A standard American football field is {length} yards long and {width} yards wide. "
    "Place every player you can see. x is yards to the viewer's right of the "
    "center; left is negative, and x stays within half the field width. "
    "y is yards toward the goal the offense is attacking. The line of "
    "scrimmage is y=0, the 50 yard line. Players behind the ball have a negative y. "
    "Skip the referee, the sideline, and the crowd. "
    "Offense symbols: square for the center, open for the rest of the line, "
    "filled for a back. Label skill players X, Z, Y, or H when the spot is clear. "
    "Defense on the line is labeled with a technique number such as 1, 3, or 5. "
    "Off the line, label B, C, or S. Leave the label empty when you are unsure. "
    "Do not name the formation. "
    "Reply with JSON only: "
    '{{"offense":[{{"x":0,"y":-1,"role":"center","symbol":"square","label":"","seen":true}}],'
    '"defense":[{{"x":0,"y":1,"role":"line","symbol":"open","label":"1","seen":true}}]}}'
)
_FORMATION_PROMPT = (
    "This play is already a {call}. The offense and defense are already placed "
    "as x and y yards from the ball on the 50. Positive y attacks the goal. "
    "Name the formation from these points, not from a picture. "
    "shotgun when the quarterback is about five yards behind the center; "
    "pistol when he is about three or four yards back with a back behind him; "
    "i_form when he is under center with two backs stacked behind him; "
    "ace when he is under center with one back behind him; "
    "wing when a back or tight end is just outside the end of the line; "
    "goal_line for a heavy set near the goal line; "
    "empty only when no running back is beside or behind the quarterback; "
    "trips or bunch only when the receivers are clear and the backfield is not. "
    "If the points do not show a lineup, use null. "
    "Points: {points} "
    'Reply with JSON only: {{"formation":"shotgun"}}'
)


def blank_diagram(
    offense: Uniform | None = None,
    defense: Uniform | None = None,
) -> dict[str, object]:
    """The field every diagram shares, with no players yet."""

    return {
        "field": _field(offense, defense),
        "teams": _teams(offense, defense),
        "call": "",
        "formation": "",
        "offense": [],
        "defense": [],
        "title": "",
        "source": "film",
    }


def read_scrimmage(
    frames: Sequence[bytes],
    left: Uniform,
    right: Uniform,
    on_step: Callable[[str, int], None] | None = None,
) -> dict[str, object]:
    """The players on each jersey, then every one of them as x and y."""

    def mark(step: str, percent: int) -> None:
        if on_step is not None:
            on_step(step, percent)

    home, away = label_home_away(left, right)
    mark("type", 100)
    mark("players", 40)
    found = isolate_play(frames, home, away)
    offense, defense, offense_players, defense_players = _sides(found, home, away)
    mark("players", 100 if offense_players and defense_players else 0)

    mark("plot", 30)
    diagram = _plot_players(offense, defense, offense_players, defense_players)
    mark("plot", 100 if _has_points(diagram) else 0)
    return diagram


def diagram_from_frames(
    frames: Sequence[bytes],
    offense: Uniform,
    defense: Uniform,
) -> dict[str, object]:
    """Place the players the film already found, in yards from the 50."""

    best: tuple[int, list[tuple[float, float]], list[tuple[float, float]]] | None = None
    for frame in list(frames)[:4]:
        if not isinstance(frame, bytes) or not frame.startswith(b"\xff\xd8\xff"):
            continue
        from film_analysis.chart.isolate import split_players

        offense_players, defense_players = split_players(frame, offense, defense)
        score = len(offense_players) + len(defense_players)
        if score < 2:
            continue
        if best is None or score > best[0]:
            best = (
                score,
                [(player.x, player.y) for player in offense_players],
                [(player.x, player.y) for player in defense_players],
            )
    diagram = blank_diagram(offense, defense)
    if best is None:
        return diagram
    offense_xy, defense_xy = normalize_to_los(best[1], best[2])
    diagram["offense"] = _label_offense([_point(x, y, "offense") for x, y in offense_xy])
    diagram["defense"] = _label_defense([_point(x, y, "defense") for x, y in defense_xy])
    diagram["source"] = "film"
    return diagram


def parse_points(reply: str) -> dict[str, object] | None:
    """Read a model's point list. A reply without both sides is ignored."""

    payload = _json_object(reply)
    if payload is None:
        return None
    offense = _players(payload.get("offense"))
    defense = _players(payload.get("defense"))
    if offense is None or defense is None or not offense and not defense:
        return None
    return {"offense": offense, "defense": defense, "source": "model"}


def steps_done(layout: dict[str, object] | None) -> dict[str, int]:
    """How far a saved scrimmage play has gotten. 100 is finished, 0 has not started.

    Type is already known for a scrimmage clip. Players is full when each jersey
    has someone. Plot is full when any of those people has x and y.
    """

    offense = layout.get("offense") if isinstance(layout, dict) else None
    defense = layout.get("defense") if isinstance(layout, dict) else None
    both = (
        isinstance(offense, list)
        and isinstance(defense, list)
        and any(_is_point(item) for item in offense)
        and any(_is_point(item) for item in defense)
    )
    return {
        "type": 100,
        "players": 100 if both else 0,
        "plot": 100 if _has_points(layout if isinstance(layout, dict) else None) else 0,
    }


def _sides(
    found: object,
    home: Uniform,
    away: Uniform,
) -> tuple[Uniform, Uniform, tuple[object, ...], tuple[object, ...]]:
    """Offense, defense, and the people found on each jersey.

    A named side from the film wins. Otherwise the jersey sitting deeper in
    the picture is the offense. A short side is kept, and a side with nobody
    stays an empty list.
    """

    home_players = tuple(getattr(found, "mine", ()) or ())
    away_players = tuple(getattr(found, "other", ()) or ())
    unit = getattr(found, "unit", None)
    if unit == "defense" or (unit not in {"offense", "defense"} and _depth(away_players) > _depth(home_players)):
        return away, home, away_players, home_players
    return home, away, home_players, away_players


def _depth(players: Sequence[object]) -> float:
    if not players:
        return -1.0
    return sum(float(getattr(player, "y", 0.0)) for player in players) / len(players)


def _plot_players(
    offense: Uniform,
    defense: Uniform,
    offense_players: Sequence[object],
    defense_players: Sequence[object],
) -> dict[str, object]:
    offense_xy, defense_xy = normalize_to_los(
        [(float(player.x), float(player.y)) for player in offense_players],
        [(float(player.x), float(player.y)) for player in defense_players],
    )
    diagram = blank_diagram(offense, defense)
    diagram["offense"] = _label_offense([_point(x, y, "offense") for x, y in offense_xy])
    diagram["defense"] = _label_defense([_point(x, y, "defense") for x, y in defense_xy])
    diagram["call"] = ""
    diagram["formation"] = ""
    diagram["source"] = "film"
    return diagram


def _ask_call(frames: Sequence[bytes], offense: Uniform, defense: Uniform) -> str | None:
    reply = _ask(
        frames,
        _CALL_PROMPT.format(offense=uniform_line(offense), defense=uniform_line(defense)),
        limit=4,
        tokens=400,
    )
    if reply is None:
        return None
    payload = _json_object(reply)
    if payload is None:
        return None
    call = str(payload.get("call") or "").strip().lower()
    return call if call in _CALLS else None


def _ask_points(
    frames: Sequence[bytes],
    offense: Uniform,
    defense: Uniform,
) -> dict[str, object] | None:
    reply = _ask(
        frames,
        _PLACE_PROMPT.format(
            offense=uniform_line(offense),
            defense=uniform_line(defense),
            length=int(FIELD_LENGTH),
            width=FIELD_WIDTH,
        ),
        limit=4,
        tokens=2000,
    )
    if reply is None:
        return None
    parsed = parse_points(reply)
    if parsed is None:
        return None
    diagram = blank_diagram(offense, defense)
    diagram["offense"] = parsed["offense"]
    diagram["defense"] = parsed["defense"]
    diagram["source"] = "model"
    return diagram


def _ask_formation(diagram: dict[str, object], call: str) -> str | None:
    points = json.dumps(
        {"offense": diagram.get("offense"), "defense": diagram.get("defense")},
        separators=(",", ":"),
    )
    reply = _ask(
        (),
        _FORMATION_PROMPT.format(call=call, points=points),
        limit=0,
        tokens=400,
    )
    if reply is None:
        return None
    return parse_formation(reply)


def _ask(frames: Sequence[bytes], prompt: str, *, limit: int, tokens: int) -> str | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return None
    jpegs = [
        frame
        for frame in frames
        if isinstance(frame, bytes) and frame.startswith(b"\xff\xd8\xff")
    ][:limit]
    if limit and not jpegs:
        return None
    content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
    for index, frame in enumerate(jpegs, start=1):
        content.append({"type": "text", "text": f"Frame {index}"})
        content.append(_image_block(frame))
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=os.environ.get("FILM_ANALYSIS_MODEL", DEFAULT_MODEL),
            max_tokens=tokens,
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
    return "\n".join(texts) or None


def _field(offense: Uniform | None, defense: Uniform | None) -> dict[str, object]:
    return {
        "ball": 50,
        "attack": "goal",
        "length": FIELD_LENGTH,
        "width": FIELD_WIDTH,
        "offense_color": (offense.jersey_hex if offense else "") or "#f4f4f4",
        "defense_color": (defense.jersey_hex if defense else "") or "#c4382c",
    }


def _teams(offense: Uniform | None, defense: Uniform | None) -> dict[str, object]:
    if offense is None or defense is None:
        return {"offense": "", "defense": ""}
    return {
        "offense": offense.name or "Home",
        "defense": defense.name or "Away",
        "offense_hex": offense.jersey_hex,
        "defense_hex": defense.jersey_hex,
    }


def _point(x: float, y: float, side: str) -> dict[str, object]:
    """One seen player. In the normalized frame, a larger y is the offensive backfield."""

    yards_x = _clamp((x - 0.5) * FIELD_WIDTH, _HALF_WIDTH)
    # The goal the offense attacks is the small-y side of that frame, so both
    # sides flip: offense behind the ball is negative, defense is positive.
    # The edge of the frame is about 12 yards, so a back stays on the drawing.
    yards_y = _clamp(-(y - 0.5) * _FRAME_TO_YARDS, _SIGHT_YARDS)
    on_line = abs(yards_y) <= 1.5
    if side == "offense" and on_line and abs(yards_x) <= 1.2:
        symbol, role = "square", "center"
    elif on_line:
        symbol, role = "open", "line"
    else:
        symbol, role = "filled", "back" if side == "offense" else "coverage"
    return {
        "x": round(yards_x, 1),
        "y": round(yards_y, 1),
        "role": role,
        "symbol": symbol,
        "label": "",
        "seen": True,
    }


def _label_offense(players: list[dict[str, object]]) -> list[dict[str, object]]:
    """X is the wide split, Y the tight end, Z the wide flanker, H the slot."""

    line = [player for player in players if player["role"] in {"line", "center"}]
    line_ids = {id(player) for player in line}
    backs = [player for player in players if id(player) not in line_ids]
    on_line = [player for player in line if player["role"] != "center"]
    if on_line:
        widest = max(on_line, key=lambda player: abs(float(player["x"])))
        if abs(float(widest["x"])) >= 8:
            widest["label"] = "X"
        tight = [
            player
            for player in on_line
            if player is not widest and 2.5 <= abs(float(player["x"])) <= 8
        ]
        if tight:
            min(tight, key=lambda player: abs(float(player["x"])))["label"] = "Y"
        elif widest["label"] != "X":
            widest["label"] = "Y"
    if backs:
        widest = max(backs, key=lambda player: abs(float(player["x"])))
        if abs(float(widest["x"])) >= 6:
            widest["label"] = "Z"
        slots = [
            player
            for player in backs
            if player is not widest and abs(float(player["x"])) >= 3
        ]
        if slots:
            max(slots, key=lambda player: abs(float(player["x"])))["label"] = "H"
    return players


def _label_defense(players: list[dict[str, object]]) -> list[dict[str, object]]:
    """A lineman gets a technique. Off the ball is B, C, or S."""

    for player in players:
        yards_x = float(player["x"])
        yards_y = float(player["y"])
        if player["role"] == "line":
            player["label"] = _technique(abs(yards_x))
            continue
        if abs(yards_x) >= 10 and yards_y <= 12:
            player["label"] = "C"
        elif yards_y >= 8:
            player["label"] = "S"
        else:
            player["label"] = "B"
    return players


def _technique(yards_from_center: float) -> str:
    cuts = (0.8, 1.6, 2.5, 3.5, 5.0, 7.0, 10.0)
    names = ("0", "1", "2", "3", "4", "5", "7", "9")
    for limit, name in zip(cuts, names, strict=False):
        if yards_from_center < limit:
            return name
    return names[-1]


def _players(value: object) -> list[dict[str, object]] | None:
    if not isinstance(value, list):
        return None
    players: list[dict[str, object]] = []
    for item in value[:11]:
        if not isinstance(item, dict):
            continue
        x = _number(item.get("x"))
        y = _number(item.get("y"))
        if x is None or y is None:
            continue
        symbol = str(item.get("symbol") or "open").strip().lower()
        if symbol not in _SYMBOLS:
            symbol = "open"
        players.append(
            {
                "x": round(_clamp(x, _HALF_WIDTH), 1),
                "y": round(_clamp(y, _HALF_LENGTH), 1),
                "role": str(item.get("role") or "player").strip()[:24] or "player",
                "symbol": symbol,
                "label": str(item.get("label") or "").strip()[:4],
                "seen": True if item.get("seen") is None else bool(item.get("seen")),
            }
        )
    return players


def _has_points(layout: dict[str, object] | None) -> bool:
    if not isinstance(layout, dict):
        return False
    offense = layout.get("offense")
    defense = layout.get("defense")
    if not isinstance(offense, list) or not isinstance(defense, list):
        return False
    return any(_is_point(item) for item in (*offense, *defense))


def _is_point(item: object) -> bool:
    return isinstance(item, dict) and _number(item.get("x")) is not None and _number(item.get("y")) is not None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


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


def _image_block(jpeg: bytes) -> dict[str, object]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.standard_b64encode(jpeg).decode("ascii"),
        },
    }
