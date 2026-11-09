"""Read one scrimmage play as a run or a pass, from the user's side of the ball."""

import base64
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass

from film_analysis.agents.play_boundaries import DEFAULT_MODEL
from film_analysis.web.teams import Uniform, uniform_line

_PROMPT = (
    "These frames are one football play, in order. "
    "Your team wears {mine}. "
    "The other team wears {other}. "
    "The offense is the team that has the ball at the snap. "
    "If your team has the ball, unit is offense. "
    "If the other team has the ball, unit is defense. "
    "call is run when the ball is handed off or kept, and pass when it is thrown. "
    'Reply with JSON only: {{"unit":"offense","call":"run"}}'
)


@dataclass(frozen=True)
class PlayCall:
    """A scrimmage play from the user's point of view."""

    call: str
    unit: str


def chart_prompt(mine: Uniform, other: Uniform) -> str:
    """Tell the model which uniform is the user's team, then ask run or pass."""

    return _PROMPT.format(mine=uniform_line(mine), other=uniform_line(other))


def parse_play_call(reply: str) -> PlayCall | None:
    """Read a run or pass, and whether the user's team is on offense."""

    payload = _json_object(reply)
    if payload is None:
        return None
    call = str(payload.get("call") or "").strip().lower()
    unit = str(payload.get("unit") or "").strip().lower()
    if call not in {"run", "pass"} or unit not in {"offense", "defense"}:
        return None
    return PlayCall(call, unit)


def chart_play(
    frames: Sequence[bytes],
    mine: Uniform,
    other: Uniform,
) -> PlayCall | None:
    """Look at a few frames and decide run or pass for the user's offense or defense."""

    jpegs = [frame for frame in frames if frame.startswith(b"\xff\xd8\xff")][:4]
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not jpegs or not api_key:
        return None
    content: list[dict[str, object]] = []
    for index, frame in enumerate(jpegs, start=1):
        content.append({"type": "text", "text": f"Frame {index}"})
        content.append(_image_block(frame))
    content.append({"type": "text", "text": chart_prompt(mine, other)})
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=os.environ.get("FILM_ANALYSIS_MODEL", DEFAULT_MODEL),
            max_tokens=120,
            messages=[{"role": "user", "content": content}],
        )
    except Exception:
        return None
    texts = [
        block.text
        for block in getattr(message, "content", [])
        if getattr(block, "type", None) == "text" and getattr(block, "text", "")
    ]
    return parse_play_call("\n".join(texts))


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
