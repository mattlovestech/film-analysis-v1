"""Coach corrections, and the notes those corrections teach the next read."""

import json
import os
import threading
from collections.abc import Sequence

from film_analysis.storage.catalog import CoachEdit

_NOTES = ""
_LOCK = threading.Lock()
_LIMIT = 1200

_LEARN_PROMPT = """
You improve how American football film is cut and tagged.
Coaches corrected the reads below. Rewrite the notes so the next read follows those corrections.

Current notes:
{current}

Edits:
{edits}

Rules:
- Keep a note only when the edits support it. Drop a note the edits contradict.
- Say how a play should be cut when an edit moved the start or the end.
- Say how a play should be tagged when an edit changed offense, defense, run, pass, special teams, or not a play.
- Stay under 900 characters.
- Do not explain. The whole reply is one JSON object: {{"guidance":"..."}}
""".strip()  # noqa: E501


def publish_guidance(text: str) -> None:
    """Make these notes visible to the next cut and tag on this process."""

    global _NOTES
    with _LOCK:
        _NOTES = text.strip()[:_LIMIT]


def published_guidance() -> str:
    with _LOCK:
        return _NOTES


def apply_coach_notes(prompt: str) -> str:
    """Append the learned coach notes when any have been published."""

    notes = published_guidance()
    if not notes:
        return prompt
    return (
        f"{prompt}\n\n"
        "Coaches corrected earlier films. Follow these notes when they apply:\n"
        f"{notes}"
    )


def learn_prompt(guidance: str, edits: Sequence[CoachEdit]) -> str:
    """The request that turns coach corrections into notes for the next read."""

    lines = [
        f"- {edit.film}, play {edit.play_number or '?'}: {edit.action}. "
        f"Was: {edit.before} Now: {edit.after}."
        for edit in edits
    ]
    current = guidance.strip() or "No notes yet."
    return _LEARN_PROMPT.format(current=current, edits="\n".join(lines))


def guidance_from_reply(reply: str) -> str:
    """Read the rewritten notes. Empty when the reply is not usable."""

    start = reply.find("{")
    end = reply.rfind("}")
    if start < 0 or end <= start:
        return ""
    try:
        payload = json.loads(reply[start : end + 1])
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict):
        return ""
    text = str(payload.get("guidance") or "").strip()
    return text[:_LIMIT]


def propose_guidance(guidance: str, edits: Sequence[CoachEdit]) -> str:
    """Ask the model to rewrite the coach notes from these corrections."""

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key or not edits:
        raise ValueError("No coach edits to learn from.")
    import anthropic

    from film_analysis.plays.boundaries import DEFAULT_MODEL

    model = os.environ.get("FILM_ANALYSIS_MODEL", DEFAULT_MODEL)
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=800,
        thinking={"type": "between_tools"},
        messages=[{"role": "user", "content": learn_prompt(guidance, edits)}],
    )
    texts = [
        block.text
        for block in getattr(message, "content", [])
        if getattr(block, "type", None) == "text" and getattr(block, "text", "")
    ]
    notes = guidance_from_reply("\n".join(texts))
    if not notes:
        raise ValueError("The model did not return coach notes.")
    return notes
