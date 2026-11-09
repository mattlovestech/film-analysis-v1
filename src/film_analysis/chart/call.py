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
    "These frames are one play, in order. "
    "Your team wears {mine}. The opponent wears {other}. "
    "Decide whether this is an offensive or defensive play for the opponent. "
    "The opponent is on offense when the opponent has the ball. "
    "The opponent is on defense when your team has the ball. "
    "Say whether the offense ran or passed. "
    "A run is a handoff or a keeper. A pass is a throw. "
    "Do not explain. The whole reply is one JSON object. "
    'Reply with JSON only: {{"call":"run","opponent":"offense"}}'
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

    template = _SPECIAL_PROMPT if kind == SPECIAL_TEAMS else _SCRIMMAGE_PROMPT
    return apply_coach_notes(
        template.format(mine=uniform_line(mine), other=uniform_line(other))
    )


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

    The model names the opponent's side. Opponent offense means your team is
    on defense, and opponent defense means your team has the ball.
    """

    if kind != SCRIMMAGE:
        return None
    payload = _json_object(reply)
    if payload is not None:
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

    jpegs = [frame for frame in frames if frame.startswith(b"\xff\xd8\xff")][:4]
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
