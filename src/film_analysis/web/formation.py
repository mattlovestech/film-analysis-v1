"""Last read of an offensive play: the formation the offense is lined up in.

This pass is experimental. It runs only after the team is known and the
opponent has the ball, and it leaves the play alone when it is not sure.
"""

import base64
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass

from film_analysis.agents.play_boundaries import DEFAULT_MODEL
from film_analysis.web.teams import Uniform, uniform_line


@dataclass(frozen=True)
class FormationRead:
    """The formation, and the tokens the model used to name it."""

    formation: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""

FORMATIONS = {
    "ace": "Ace",
    "i_form": "I-formation",
    "shotgun": "Shotgun",
    "pistol": "Pistol",
    "trips": "Trips",
    "bunch": "Bunch",
    "empty": "Empty",
    "wing": "Wing",
    "goal_line": "Goal line",
}
_ALIASES = {
    "i": "i_form",
    "iform": "i_form",
    "i_formation": "i_form",
    "goalline": "goal_line",
    "goal-line": "goal_line",
    "wing_t": "wing",
    "singleback": "ace",
}
_PROMPT = (
    "These frames are one offensive play, lined up before the snap. "
    "The offense wears {mine}. The defense wears {other}. "
    "Name the formation the offense is in. "
    "Choose one of: ace, i_form, shotgun, pistol, trips, bunch, empty, wing, goal_line. "
    "This read is experimental. If the formation is not clear, use null. "
    'Reply with JSON only: {{"formation":"shotgun"}}'
)


def formation_prompt(mine: Uniform, other: Uniform) -> str:
    """Ask for the offense's formation, and allow an unsure answer."""

    return _PROMPT.format(mine=uniform_line(mine), other=uniform_line(other))


def formation_label(key: str | None) -> str:
    """The name to show for a stored formation."""

    if not key:
        return ""
    return FORMATIONS.get(key, "")


def parse_formation(reply: str) -> str | None:
    """Read one known formation. Anything unsure or unknown is left blank."""

    payload = _json_object(reply)
    if payload is None or payload.get("formation") is None:
        return None
    key = str(payload["formation"]).strip().lower().replace(" ", "_").replace("-", "_")
    key = _ALIASES.get(key, key)
    if key not in FORMATIONS:
        return None
    return key


def read_formation(
    frames: Sequence[bytes], mine: Uniform, other: Uniform
) -> FormationRead | str | None:
    """Name the formation. No frames or no key means this play is left alone."""

    jpegs = [frame for frame in frames if frame.startswith(b"\xff\xd8\xff")][:4]
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not jpegs or not api_key:
        return None
    content: list[dict[str, object]] = []
    for index, frame in enumerate(jpegs, start=1):
        content.append({"type": "text", "text": f"Frame {index}"})
        content.append(_image_block(frame))
    content.append({"type": "text", "text": formation_prompt(mine, other)})
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
    named = parse_formation("\n".join(texts))
    if named is None:
        return None
    usage = getattr(message, "usage", None)
    return FormationRead(
        named,
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
