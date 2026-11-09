"""Play windows on a film file, printed as clock ranges."""

import json
import re

from pydantic import BaseModel, ConfigDict, Field, ValidationError

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class PlaySegment(BaseModel):
    """One play, measured in seconds from the start of the uploaded file."""

    model_config = ConfigDict(frozen=True)

    number: int = Field(ge=1)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)

    @property
    def label(self) -> str:
        return format_play(self.number, self.start_seconds, self.end_seconds)


class FilmSpan(BaseModel):
    """One labeled stretch of film, measured in seconds from the start of the file."""

    model_config = ConfigDict(frozen=True)

    kind: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)


_KINDS = {
    "play": "play",
    "huddle": "huddle",
    "dead": "dead",
    "dead ball": "dead",
    "replay": "replay",
    "crowd": "crowd",
    "crowd shot": "crowd",
    "graphic": "graphic",
    "graphics": "graphic",
}


class _RawPlay(BaseModel):
    start_seconds: float
    end_seconds: float


class _RawPlays(BaseModel):
    plays: list[_RawPlay]


def format_clock(seconds: float) -> str:
    """Format seconds as `M:SS`, matching lines like `0:01`."""

    minutes, secs = divmod(_whole_seconds(seconds), 60)
    return f"{minutes}:{secs:02d}"


def format_play(number: int, start_seconds: float, end_seconds: float) -> str:
    """Format one play as `Play 1: 0:01 - 0:02`."""

    if number < 1:
        raise ValueError(f"play number must be >= 1, got {number}")
    if end_seconds <= start_seconds:
        raise ValueError("end must be after start")
    return f"Play {number}: {format_span(start_seconds, end_seconds)}"


def format_span(start_seconds: float, end_seconds: float) -> str:
    """Format a range. Sub-second plays keep a decimal so the range stays visible."""

    start = format_clock(start_seconds)
    end = format_clock(end_seconds)
    if start != end:
        return f"{start} - {end}"
    precise_start = _format_precise(start_seconds, 1)
    precise_end = _format_precise(end_seconds, 1)
    if precise_start != precise_end:
        return f"{precise_start} - {precise_end}"
    return f"{_format_precise(start_seconds, 2)} - {_format_precise(end_seconds, 2)}"


def parse_film_spans(raw: str) -> tuple[FilmSpan, ...]:
    """Turn a model JSON payload into play and non-play spans."""

    try:
        payload = json.loads(_strip_fences(raw))
    except json.JSONDecodeError as error:
        raise ValueError("film spans were not valid JSON") from error
    if isinstance(payload, list):
        payload = {"spans": payload}
    if not isinstance(payload, dict):
        raise ValueError("film spans must be a JSON object")

    raw_spans = payload.get("spans")
    if raw_spans is None and isinstance(payload.get("plays"), list):
        raw_spans = [
            {**play, "kind": "play"} if isinstance(play, dict) else play
            for play in payload["plays"]
        ]
    if not isinstance(raw_spans, list):
        raise ValueError("spans must be a list")

    spans: list[FilmSpan] = []
    for item in raw_spans:
        if not isinstance(item, dict):
            raise ValueError("each span must be an object")
        start = item.get("start_seconds", item.get("start"))
        end = item.get("end_seconds", item.get("end"))
        if start is None or end is None:
            raise ValueError("each span needs a start and end")
        try:
            start_seconds = float(start)
            end_seconds = float(end)
        except (TypeError, ValueError) as error:
            raise ValueError("each span needs a start and end") from error
        if end_seconds <= start_seconds or start_seconds < 0:
            continue
        kind = _KINDS.get(str(item.get("kind", "dead")).strip().lower(), "dead")
        spans.append(FilmSpan(kind=kind, start_seconds=start_seconds, end_seconds=end_seconds))
    return tuple(spans)


def parse_play_segments(raw: str) -> tuple[PlaySegment, ...]:
    """Turn a model JSON payload into numbered, non-overlapping plays."""

    try:
        payload = json.loads(_strip_fences(raw))
    except json.JSONDecodeError as error:
        raise ValueError("play timestamps were not valid JSON") from error

    if isinstance(payload, list):
        payload = {"plays": payload}
    if not isinstance(payload, dict):
        raise ValueError("play timestamps must be a JSON object")

    plays = payload.get("plays", [])
    if not isinstance(plays, list):
        raise ValueError("plays must be a list")

    try:
        parsed = _RawPlays.model_validate({"plays": [_coerce_play(play) for play in plays]})
    except (ValidationError, ValueError) as error:
        raise ValueError("play timestamps were missing start or end") from error

    ordered = sorted(parsed.plays, key=lambda play: (play.start_seconds, play.end_seconds))
    segments: list[PlaySegment] = []
    previous_end: float | None = None
    for play in ordered:
        if play.end_seconds <= play.start_seconds:
            continue
        number = len(segments) + 1
        if previous_end is not None and play.start_seconds < previous_end:
            raise ValueError(f"Play {number} overlaps the previous play")
        segments.append(
            PlaySegment(
                number=number,
                start_seconds=play.start_seconds,
                end_seconds=play.end_seconds,
            )
        )
        previous_end = play.end_seconds
    return tuple(segments)


def _coerce_play(play: object) -> dict[str, float]:
    if not isinstance(play, dict):
        raise ValueError("each play must be an object")
    start = play.get("start_seconds", play.get("start"))
    end = play.get("end_seconds", play.get("end"))
    if start is None or end is None:
        raise ValueError("each play needs a start and end")
    return {"start_seconds": start, "end_seconds": end}


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("```"):
        return text
    return _FENCE.sub("", text).strip()


def _whole_seconds(seconds: float) -> int:
    if seconds < 0:
        raise ValueError(f"timestamp must be >= 0, got {seconds}")
    return int(round(seconds))


def _format_precise(seconds: float, places: int) -> str:
    factor = 10**places
    units = _whole_seconds(seconds * factor)
    minutes, remainder = divmod(units, 60 * factor)
    whole, fraction = divmod(remainder, factor)
    return f"{minutes}:{whole:02d}.{fraction:0{places}d}"
