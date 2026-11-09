"""Formation read after the play call is known.

Builds an 11-on-11 spatial plot of both teams, then asks Claude to name the
offensive formation with that plot plus the already-charted call (run/pass/etc).
"""

from __future__ import annotations

import base64
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass

from film_analysis.chart.formation_plot import FormationLayout, plot_formation
from film_analysis.plays.boundaries import DEFAULT_MODEL
from film_analysis.teams.uniforms import Uniform, uniform_line

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
    "This snap is already charted as a {call} play. "
    "These frames are the offense lined up before the snap. "
    "The offense wears {offense}. The defense wears {defense}. "
    "The last image is a bird's-eye spatial plot of both teams with exactly "
    "11 players on each side. Hollow circles with a ? are low-confidence "
    "guesses where a player was not clearly seen. "
    "Given that this is a {call}, name the formation the offense is in. "
    "Choose one of: ace, i_form, shotgun, pistol, trips, bunch, empty, wing, goal_line. "
    "This read is experimental. If the formation is not clear, use null. "
    'Reply with JSON only: {{"formation":"shotgun"}}'
)


@dataclass(frozen=True)
class FormationRead:
    """Named formation plus the spatial plot used to decide it."""

    formation: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    plot: bytes = b""
    frame: bytes = b""
    guessed: int = 0


def formation_prompt(offense: Uniform, defense: Uniform, call: str = "") -> str:
    """Ask for the offense's formation after the call is known."""

    play_call = (call or "").strip().lower() or "scrimmage"
    return _PROMPT.format(
        call=play_call,
        offense=uniform_line(offense),
        defense=uniform_line(defense),
    )


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
    frames: Sequence[bytes],
    offense: Uniform,
    defense: Uniform,
    call: str = "",
) -> FormationRead | None:
    """Name the formation after the call is known, using an 11-on-11 plot.

    Always keeps the spatial plot when one can be built, even if the model is
    unsure. That plot is what a person uses to override the guess.
    """

    play_call = (call or "").strip().lower()
    if play_call not in {"run", "pass"}:
        return None
    jpegs = [
        frame
        for frame in frames
        if isinstance(frame, bytes) and frame.startswith(b"\xff\xd8\xff")
    ]
    if not jpegs:
        return None
    layout = plot_formation(jpegs, offense, defense)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        if layout is None:
            return None
        return FormationRead(
            "",
            plot=layout.plot,
            frame=layout.frame,
            guessed=layout.guessed,
        )
    content: list[dict[str, object]] = [
        {"type": "text", "text": formation_prompt(offense, defense, play_call)},
    ]
    if layout is not None:
        content.append({"type": "text", "text": "Film frame"})
        content.append(_image_block(layout.frame))
        content.append({"type": "text", "text": "Spatial plot (11 on each side)"})
        content.append(_image_block(layout.plot))
    else:
        for index, frame in enumerate(jpegs[:4], start=1):
            content.append({"type": "text", "text": f"Frame {index}"})
            content.append(_image_block(frame))
    try:
        import anthropic

        model = os.environ.get("FILM_ANALYSIS_MODEL", DEFAULT_MODEL)
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=400,
            messages=[{"role": "user", "content": content}],
        )
    except Exception:
        if layout is None:
            return None
        return FormationRead(
            "",
            plot=layout.plot,
            frame=layout.frame,
            guessed=layout.guessed,
        )
    texts = [
        block.text
        for block in getattr(message, "content", [])
        if getattr(block, "type", None) == "text" and getattr(block, "text", "")
    ]
    named = parse_formation("\n".join(texts)) or ""
    usage = getattr(message, "usage", None)
    return FormationRead(
        named,
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
        model,
        layout.plot if layout is not None else b"",
        layout.frame if layout is not None else b"",
        layout.guessed if layout is not None else 0,
    )


def layout_from_frames(
    frames: Sequence[bytes],
    offense: Uniform,
    defense: Uniform,
) -> FormationLayout | None:
    """Expose the spatial plot helper for tests and tooling."""

    return plot_formation(frames, offense, defense)


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
