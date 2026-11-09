"""Second read of a play: run or pass, or the special-teams type for your team."""

import base64
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass

from film_analysis.learning import apply_coach_notes
from film_analysis.plays.boundaries import DEFAULT_MODEL
from film_analysis.teams.uniforms import Uniform, uniform_line

SCRIMMAGE = "scrimmage"
SPECIAL_TEAMS = "special_teams"
RUN_OR_PASS = frozenset({"run", "pass"})
SPECIAL_CALLS = frozenset(
    {"kickoff", "kick_return", "punt", "punt_return", "field_goal", "extra_point"}
)

_SCRIMMAGE_PROMPT = (
    "These frames are one scrimmage play, in order, from the snap to the whistle. "
    "Team A is {mine_name}. Team A wears {mine}. "
    "Team B is {other_name}. Team B wears {other}.{contrast} "
    "First decide which team has the ball. "
    "If a broadcast scorebug is visible, it usually marks the team with the ball "
    "with a small bar, dot, or football beside that team's name or score; trust it. "
    "Otherwise the offense is the team with the quarterback taking the snap and the "
    "backs lined up behind its line, and the defense faces it across the ball. "
    "After the snap the ball carrier or the receiver belongs to the offense. "
    "Then say whether the offense ran or passed. "
    "A run is a handoff, a pitch, or a quarterback keeper or scramble. "
    "A pass is a throw, including a sack or an incompletion on a dropback. "
    "Do not explain. The whole reply is one JSON object. "
    'Reply with JSON only: {{"offense":"A","call":"run"}}'
)


@dataclass(frozen=True)
class ChartedPlay:
    """The call, and which side your team is on when the frames show it."""

    call: str
    side: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


_SPECIAL_PROMPT = (
    "These frames are one special-teams play, in order. "
    "Your team wears {mine}. The other team wears {other}. "
    "Name the play from your team's side. "
    "kickoff when your team kicks off. kick_return when your team receives the kickoff. "
    "punt when your team punts. punt_return when your team receives the punt. "
    "field_goal for a field goal. extra_point for an extra point. "
    'Reply with JSON only: {{"call":"kick_return"}}'
)


def chart_prompt(mine: Uniform, other: Uniform, kind: str = SCRIMMAGE) -> str:
    """Ask for a run or a pass, or for the special-teams play your team is on."""

    if kind == SPECIAL_TEAMS:
        prompt = _SPECIAL_PROMPT.format(mine=uniform_line(mine), other=uniform_line(other))
    else:
        prompt = _SCRIMMAGE_PROMPT.format(
            mine=uniform_line(mine),
            other=uniform_line(other),
            mine_name=_team_name(mine, "your team"),
            other_name=_team_name(other, "the opponent"),
            contrast=_contrast(mine, other),
        )
    return apply_coach_notes(prompt)


def _team_name(uniform: Uniform, fallback: str) -> str:
    name = (uniform.name or "").strip()
    if not name or name.lower() in {"home", "away", "left", "right", "team a", "team b"}:
        return fallback
    return f"{name}, {fallback}"


def _contrast(mine: Uniform, other: Uniform) -> str:
    """Name the darker jersey when the color words alone may not tell them apart."""

    mine_light = _lightness(mine.jersey_hex)
    other_light = _lightness(other.jersey_hex)
    if mine_light is None or other_light is None or abs(mine_light - other_light) < 0.15:
        return ""
    if mine_light < other_light:
        return " Team A's jerseys are darker than Team B's."
    return " Team B's jerseys are darker than Team A's."


def _lightness(hex_code: str) -> float | None:
    text = (hex_code or "").strip().lstrip("#")
    if len(text) != 6:
        return None
    try:
        red, green, blue = (int(text[index : index + 2], 16) / 255 for index in (0, 2, 4))
    except ValueError:
        return None
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def parse_play_call(reply: str, kind: str = SCRIMMAGE) -> str | None:
    """Read the call. Offense and defense plays are only run or pass."""

    payload = _json_object(reply)
    if payload is None:
        return None
    call = str(payload.get("call") or "").strip().lower().replace(" ", "_")
    allowed = SPECIAL_CALLS if kind == SPECIAL_TEAMS else RUN_OR_PASS
    if call not in allowed:
        return None
    return call


def parse_play_side(reply: str, kind: str = SCRIMMAGE) -> str | None:
    """Which side your team is on.

    The model names the team with the ball: A is your team, B the opponent.
    Older replies named the opponent's side instead, and those still parse.
    """

    if kind != SCRIMMAGE:
        return None
    payload = _json_object(reply)
    if payload is not None:
        offense = str(payload.get("offense") or "").strip().lower()
        if offense in {"a", "team a"}:
            return "offense"
        if offense in {"b", "team b"}:
            return "defense"
        opponent = str(payload.get("opponent") or "").strip().lower()
        if opponent == "offense":
            return "defense"
        if opponent == "defense":
            return "offense"
        side = str(payload.get("side") or "").strip().lower()
        if side in {"offense", "defense"}:
            return side
    text = reply.lower()
    if "opponent is on offense" in text or "opponent on offense" in text:
        return "defense"
    if "opponent is on defense" in text or "opponent on defense" in text:
        return "offense"
    return None


def chart_play(
    frames: Sequence[bytes],
    mine: Uniform,
    other: Uniform,
    kind: str = SCRIMMAGE,
) -> ChartedPlay | None:
    """Look at a few frames and name the call for this kind of play."""

    jpegs = [frame for frame in frames if frame.startswith(b"\xff\xd8\xff")][:6]
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not jpegs or not api_key:
        return None
    content: list[dict[str, object]] = []
    for index, frame in enumerate(jpegs, start=1):
        content.append({"type": "text", "text": f"Frame {index}"})
        content.append(_image_block(frame))
    content.append({"type": "text", "text": chart_prompt(mine, other, kind)})
    try:
        import anthropic

        model = os.environ.get("FILM_ANALYSIS_MODEL", DEFAULT_MODEL)
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=400,
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
    reply = "\n".join(texts)
    call = parse_play_call(reply, kind)
    side = parse_play_side(reply, kind)
    if call is None and side is None:
        return None
    usage = getattr(message, "usage", None)
    return ChartedPlay(
        call or "",
        side,
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
        model,
    )


def _image_block(jpeg: bytes) -> dict[str, object]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.standard_b64encode(jpeg).decode("ascii"),
        },
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
