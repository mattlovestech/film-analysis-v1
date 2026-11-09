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
    "The frames are in order, starting at the snap, so the first frames show "
    "how the offense lined up. "
    "The offense wears {offense}. The defense wears {defense}. "
    "{plot}"
    "Given that this is a {call}, name the formation the offense is in. "
    "Name the backfield first: "
    "shotgun when the quarterback stands about five yards behind the center; "
    "pistol when he is about three or four yards back with a running back directly "
    "behind him; "
    "i_form when he is under center with two backs stacked behind him; "
    "ace when he is under center with one back behind him; "
    "wing when a back or tight end lines up just outside the end of the line; "
    "goal_line for a heavy set with extra linemen or tight ends near the goal line; "
    "empty only when you can see that no running back stands beside or behind "
    "the quarterback; a back next to a shotgun quarterback is still shotgun. "
    "Use trips or bunch only when the receivers are clear and the backfield is not. "
    "If you cannot see the offense lined up, use null. "
    'Reply with JSON only: {{"formation":"shotgun"}}'
)
_PLOT_NOTE = (
    "The last image is a bird's-eye spatial plot of both teams with 11 players on "
    "each side. Hollow circles with a ? are guesses where a player was not seen. "
)
# A plot that guessed more than this many of the 22 players is not shown to the model.
_PLOT_GUESS_LIMIT = 6


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


def formation_prompt(
    offense: Uniform, defense: Uniform, call: str = "", *, plot: bool = True
) -> str:
    """Ask for the offense's formation after the call is known."""

    play_call = (call or "").strip().lower() or "scrimmage"
    return _PROMPT.format(
        call=play_call,
        offense=uniform_line(offense),
        defense=uniform_line(defense),
        plot=_PLOT_NOTE if plot else "",
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
        frame for frame in frames if isinstance(frame, bytes) and frame.startswith(b"\xff\xd8\xff")
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
    # The plot is kept for a person to check, but a plot made mostly of guesses
    # only misleads the read, so the model then sees the film frames alone.
    show_plot = layout is not None and layout.guessed <= _PLOT_GUESS_LIMIT
    content: list[dict[str, object]] = [
        {
            "type": "text",
            "text": formation_prompt(offense, defense, play_call, plot=show_plot),
        },
    ]
    for index, frame in enumerate(jpegs[:4], start=1):
        content.append({"type": "text", "text": f"Frame {index}"})
        content.append(_image_block(frame))
    if show_plot and layout is not None:
        content.append({"type": "text", "text": "Spatial plot (11 on each side)"})
        content.append(_image_block(layout.plot))
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
