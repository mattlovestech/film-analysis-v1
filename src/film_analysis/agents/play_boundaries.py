"""Agent that watches film frames and marks each play."""

import base64
import contextvars
import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

from film_analysis.agents.errors import MissingApiKeyError, VideoAnalysisError
from film_analysis.agents.frames import extract_frames
from film_analysis.agents.segments import (
    UNCERTAIN,
    FilmSpan,
    PlaySegment,
    parse_film_spans,
    parse_play_segments,
)
from film_analysis.agents.youtube import canonical_youtube_url, download_youtube

PROMPT = """
You chart American football film.
Each image is labeled with its time in seconds from the start of the video.

A play starts at the snap, kickoff, or punt and ends when the ball is dead.

Return JSON with this shape and nothing else:
{"plays": [{"play": 1, "start_seconds": 0.0, "end_seconds": 0.0}]}

Use the frame times. List plays in order. Do not overlap them. Include every snap and kick.
""".strip()

SPAN_PROMPT = """
You sort American football film before anyone charts it.
Each image is labeled with its time in seconds from the start of the video.
This window runs from {start:.1f}s to {end:.1f}s. Label every moment in that range.

Decide in this order, and do not skip ahead:
1. Offensive or defensive play. Both teams are lined up across the line of scrimmage,
the ball is snapped, and the play runs until the ball is dead. Use kind scrimmage.
A huddle, a walk to the line, or a celebration is not a play.
2. If it is not that, special teams. A kickoff, punt, field goal, extra point,
or the return that follows. Use kind special_teams.
3. If it is not that, it is junk. Dead ball, replays, crowd, graphics, warmups,
sideline, commercials, and anything that is not a live play. Use kind junk.
4. If you are not sure it is a live play, use kind uncertain. Set it aside.
A wrong play is worse than leaving the stretch for later.

Return JSON with this shape and nothing else:
{{"spans": [{{"kind": "scrimmage", "start_seconds": 0.0, "end_seconds": 0.0}}]}}

Use the frame times. Cover the window from its first time to its last. Do not overlap spans.
""".strip()

REVIEW_PROMPT = """
You set this stretch of American football film aside because it was not clearly a play.
Each image is labeled with its time in seconds from the start of the video.
The stretch runs from {start:.1f}s to {end:.1f}s. Look once more, in this order:
1. Offensive or defensive play, from the snap until the ball is dead. Use kind scrimmage.
2. Special teams: kickoff, punt, field goal, extra point, or return. Use kind special_teams.
3. Junk: not a live play. Use kind junk.
4. Still not sure. Use kind uncertain.

Only mark a play when the frames show one. Leaving it uncertain is fine.

Return JSON with this shape and nothing else:
{{"spans": [{{"kind": "uncertain", "start_seconds": 0.0, "end_seconds": 0.0}}]}}

Use the frame times. Cover the stretch. Do not overlap spans.
""".strip()

DEFAULT_MODEL = "claude-sonnet-5-5"
_LOG = logging.getLogger("film_analysis")
_WINDOW = 40

FrameSampler = Callable[[Path], list[tuple[float, bytes]]]
Downloader = Callable[[str, Path], Path]
ReadNote = Callable[[float, float, int, int, str, str], None]
_READS: contextvars.ContextVar[ReadNote | None] = contextvars.ContextVar(
    "film_read_note",
    default=None,
)


def bind_read_note(note: ReadNote | None) -> None:
    """Record failed reads for the label happening on this thread."""

    _READS.set(note)


@runtime_checkable
class PlayBoundaryAgent(Protocol):
    """Watch one film and return its plays."""

    name: str

    def label_window(self, samples: list[tuple[float, bytes]]) -> tuple[FilmSpan, ...]:
        """Label one stretch as a scrimmage play, special teams, junk, or unsure."""

    def review_window(self, samples: list[tuple[float, bytes]]) -> tuple[FilmSpan, ...]:
        """Look again at a stretch that was set aside."""

    def segment(self, video_path: Path) -> tuple[PlaySegment, ...]:
        """Return plays in an uploaded file, in kickoff-to-whistle order."""

    def segment_youtube(self, url: str) -> tuple[PlaySegment, ...]:
        """Return plays in a public YouTube video, in kickoff-to-whistle order."""


class AnthropicPlayAgent:
    """Show Claude timestamped frames and read play boundaries back."""

    name = "anthropic-play-boundaries"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        client: object | None = None,
        frames: FrameSampler | None = None,
        downloader: Downloader | None = None,
        note_unreadable: ReadNote | None = None,
    ) -> None:
        if not api_key.strip():
            raise MissingApiKeyError("Set ANTHROPIC_API_KEY to analyze film.")
        self.model = model
        self._api_key = api_key
        self._client = client
        self._frames = frames or extract_frames
        self._downloader = downloader or download_youtube
        self.note_unreadable = note_unreadable

    def segment(self, video_path: Path) -> tuple[PlaySegment, ...]:
        if not video_path.is_file():
            raise VideoAnalysisError(f"Video not found: {video_path.name}")
        samples = self._frames(video_path)
        if not samples:
            raise VideoAnalysisError("Could not read frames from that video.")

        try:
            groups = []
            start = 0
            while start < len(samples):
                window = samples[start : start + _WINDOW]
                groups.append(parse_play_segments(self._ask(window)))
                if len(window) < _WINDOW:
                    break
                start += _WINDOW - 1
        except ValueError as error:
            raise VideoAnalysisError(
                f"The model returned play timestamps that could not be read. {error}"
            ) from error
        return _merge(groups)

    def label_window(self, samples: list[tuple[float, bytes]]) -> tuple[FilmSpan, ...]:
        """Label one short stretch as a scrimmage play, special teams, junk, or unsure."""

        return self._label(samples, SPAN_PROMPT)

    def review_window(self, samples: list[tuple[float, bytes]]) -> tuple[FilmSpan, ...]:
        """Look again at a stretch that was set aside on the first pass."""

        return self._label(samples, REVIEW_PROMPT)

    def _label(self, samples: list[tuple[float, bytes]], template: str) -> tuple[FilmSpan, ...]:
        if not samples:
            return ()
        start = samples[0][0]
        end = samples[-1][0]
        reply = ""
        try:
            reply = self._ask(samples, template.format(start=start, end=end))
            return parse_film_spans(reply)
        except ValueError as error:
            self._miss(start, end, samples, str(error), reply)
            if end <= start:
                return ()
            return (FilmSpan(kind=UNCERTAIN, start_seconds=start, end_seconds=end),)
        except VideoAnalysisError as error:
            self._miss(start, end, samples, str(error), reply)
            if _stops_the_film(error):
                raise
            if end <= start:
                return ()
            return (FilmSpan(kind=UNCERTAIN, start_seconds=start, end_seconds=end),)

    def _miss(
        self,
        start: float,
        end: float,
        samples: list[tuple[float, bytes]],
        reason: str,
        reply: str,
    ) -> None:
        frames = len(samples)
        nbytes = sum(len(jpeg) for _, jpeg in samples)
        _LOG.warning(
            "could not read %.1f–%.1f (%s frames, %s bytes): %s reply=%s",
            start,
            end,
            frames,
            nbytes,
            reason,
            _excerpt(reply),
        )
        note = _READS.get() or self.note_unreadable
        if note is None:
            return
        try:
            note(start, end, frames, nbytes, reason, reply)
        except Exception:
            _LOG.warning("could not record the failed read at %.1f–%.1f", start, end, exc_info=True)

    def segment_youtube(self, url: str) -> tuple[PlaySegment, ...]:
        try:
            canonical = canonical_youtube_url(url)
        except ValueError as error:
            raise VideoAnalysisError(str(error)) from error

        with tempfile.TemporaryDirectory() as tmp:
            video = self._downloader(canonical, Path(tmp))
            return self.segment(video)

    def _ask(self, samples: list[tuple[float, bytes]], prompt: str = PROMPT) -> str:
        client = self._client or _anthropic_client(self._api_key)
        content: list[dict[str, object]] = []
        for seconds, jpeg in samples:
            content.append({"type": "text", "text": f"t={seconds:.1f}s"})
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.standard_b64encode(jpeg).decode("ascii"),
                    },
                }
            )
        content.append({"type": "text", "text": prompt})
        try:
            message = client.messages.create(
                model=self.model,
                max_tokens=1536,
                messages=[{"role": "user", "content": content}],
            )
        except VideoAnalysisError:
            raise
        except Exception as error:
            raise VideoAnalysisError(_model_failure(error)) from error

        texts = [
            block.text
            for block in getattr(message, "content", [])
            if getattr(block, "type", None) == "text" and getattr(block, "text", "")
        ]
        text = "\n".join(texts).strip()
        if not text:
            raise VideoAnalysisError("The model returned no play timestamps.")
        return text


def agent_from_env() -> AnthropicPlayAgent:
    """Build the play agent from `ANTHROPIC_API_KEY`."""

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise MissingApiKeyError("Set ANTHROPIC_API_KEY to analyze film.")
    model = os.environ.get("FILM_ANALYSIS_MODEL", DEFAULT_MODEL)
    return AnthropicPlayAgent(api_key=api_key, model=model)


def _merge(groups: list[tuple[PlaySegment, ...]]) -> tuple[PlaySegment, ...]:
    ordered = sorted(
        (play for group in groups for play in group),
        key=lambda play: (play.start_seconds, play.end_seconds),
    )
    kept: list[PlaySegment] = []
    for play in ordered:
        if kept and play.start_seconds < kept[-1].end_seconds:
            if play.end_seconds > kept[-1].end_seconds:
                previous = kept[-1]
                kept[-1] = PlaySegment(
                    number=previous.number,
                    start_seconds=previous.start_seconds,
                    end_seconds=play.end_seconds,
                )
            continue
        kept.append(play)
    return tuple(
        PlaySegment(
            number=index,
            start_seconds=play.start_seconds,
            end_seconds=play.end_seconds,
        )
        for index, play in enumerate(kept, start=1)
    )


def _stops_the_film(error: BaseException) -> bool:
    """True when the next window would fail for the same reason."""

    text = str(error).lower()
    if "out of credits" in text or "credit balance" in text or "billing" in text:
        return True
    if any(code in text for code in ("http 401", "http 402", "http 413")):
        return True
    return "request_too_large" in text or "request too large" in text


def _excerpt(reply: str) -> str:
    text = " ".join(reply.split())
    if len(text) > 240:
        return text[:237] + "..."
    return text or "(empty)"


_CREDIT = "This film couldn't be read. The analysis account is out of credits."


def _model_failure(error: Exception) -> str:
    detail = _api_message(error)
    code = getattr(error, "error_code", None)
    status = getattr(error, "status_code", None)
    tags = [type(error).__name__]
    if isinstance(code, str) and code:
        tags.append(code)
    if isinstance(status, int):
        tags.append(f"HTTP {status}")
    where = ", ".join(tags)
    if code == "billing_frozen" or "credit balance" in detail.lower():
        return f"{_CREDIT} ({where})"
    if not detail:
        return f"The model could not read that video. ({where})"
    if len(detail) > 800:
        detail = detail[:797] + "..."
    return f"The model could not read that video. ({where}) {detail}"


def _api_message(error: Exception) -> str:
    api_message = getattr(error, "api_message", None)
    if isinstance(api_message, str) and api_message.strip():
        return " ".join(api_message.split())
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        nested = body.get("error")
        if isinstance(nested, dict):
            text = nested.get("message")
            if isinstance(text, str) and text.strip():
                return " ".join(text.split())
    return " ".join(str(error).split())


def _anthropic_client(api_key: str) -> object:
    import anthropic

    return anthropic.Anthropic(api_key=api_key)
