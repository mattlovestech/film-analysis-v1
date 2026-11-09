import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from film_analysis.ingest.errors import MissingApiKeyError, VideoAnalysisError
from film_analysis.ingest.frames import extract_frames
from film_analysis.plays.boundaries import AnthropicPlayAgent, agent_from_env


class _Messages:
    def __init__(self, text: str, usage: object = None) -> None:
        self.text = text
        self.usage = usage
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.text)],
            usage=self.usage,
        )


class _Client:
    def __init__(self, messages: _Messages) -> None:
        self.messages = messages


def _frames(path: Path) -> list[tuple[float, bytes]]:
    assert path.is_file()
    return [(1.0, b"jpeg-a"), (2.0, b"jpeg-b")]


def test_agent_sends_timestamped_frames(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    messages = _Messages('{"plays": [{"start_seconds": 1, "end_seconds": 2}]}')
    agent = AnthropicPlayAgent(api_key="test-key", client=_Client(messages), frames=_frames)

    segments = agent.segment(video)

    assert [segment.label for segment in segments] == ["Play 1: 0:01 - 0:02"]
    call = messages.calls[0]
    assert call["model"] == "claude-sonnet-5-5"
    content = call["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "t=1.0s"}
    assert content[1]["type"] == "image"
    assert content[1]["source"]["media_type"] == "image/jpeg"
    assert "snap" in content[-1]["text"]


def test_label_window_sorts_scrimmage_before_junk(tmp_path: Path) -> None:
    del tmp_path
    messages = _Messages(
        '{"spans": ['
        '{"kind": "scrimmage", "start_seconds": 1, "end_seconds": 3},'
        '{"kind": "crowd", "start_seconds": 3, "end_seconds": 5}'
        "]}"
    )
    agent = AnthropicPlayAgent(api_key="test-key", client=_Client(messages), frames=_frames)

    spans = agent.label_window([(1.0, b"jpeg-a"), (5.0, b"jpeg-b")])

    assert [span.kind for span in spans] == ["scrimmage", "junk"]
    prompt = messages.calls[0]["messages"][0]["content"][-1]["text"]
    assert "Offensive or defensive" in prompt
    assert "special teams" in prompt
    assert "uncertain" in prompt


def test_a_successful_read_records_its_tokens() -> None:
    notes: list[tuple[object, ...]] = []
    messages = _Messages(
        '{"spans": [{"kind": "scrimmage", "start_seconds": 1, "end_seconds": 3}]}',
        usage=SimpleNamespace(input_tokens=1000, output_tokens=100),
    )
    agent = AnthropicPlayAgent(
        api_key="test-key",
        client=_Client(messages),
        frames=_frames,
        note_unreadable=lambda *args: notes.append(args),
    )

    agent.label_window([(1.0, b"jpeg-a"), (5.0, b"jpeg-b")])

    assert notes == [
        (1.0, 5.0, 2, len(b"jpeg-a") + len(b"jpeg-b"), "", "", (1000, 100, "claude-sonnet-5-5"))
    ]


def test_review_window_looks_again_at_a_held_stretch() -> None:
    messages = _Messages('{"spans": [{"kind": "punt", "start_seconds": 4, "end_seconds": 9}]}')
    agent = AnthropicPlayAgent(api_key="test-key", client=_Client(messages), frames=_frames)

    spans = agent.review_window([(4.0, b"jpeg-a"), (9.0, b"jpeg-b")])

    assert [span.kind for span in spans] == ["special_teams"]
    prompt = messages.calls[0]["messages"][0]["content"][-1]["text"]
    assert "set this stretch" in prompt
    assert "Still not sure" in prompt


def test_a_reply_that_is_not_json_records_the_frames_and_the_reply(tmp_path: Path) -> None:
    del tmp_path
    notes: list[tuple[object, ...]] = []
    messages = _Messages("I could not decide. The frames were blurry.")
    agent = AnthropicPlayAgent(
        api_key="test-key",
        client=_Client(messages),
        frames=_frames,
        note_unreadable=lambda *args: notes.append(args),
    )

    spans = agent.label_window([(30.0, b"ab"), (70.0, b"cdef")])

    assert [(span.kind, span.start_seconds, span.end_seconds) for span in spans] == [
        ("uncertain", 30.0, 70.0)
    ]
    assert notes == [
        (
            30.0,
            70.0,
            2,
            6,
            "film spans were not valid JSON",
            "I could not decide. The frames were blurry.",
        )
    ]


def test_one_overloaded_window_is_set_aside_and_tracked() -> None:
    class _Overload(Exception):
        def __init__(self) -> None:
            super().__init__("Overloaded")
            self.status_code = 529
            self.api_message = "Overloaded"

    class _Broken:
        def create(self, **kwargs: object) -> SimpleNamespace:
            raise _Overload()

    notes: list[tuple[object, ...]] = []
    agent = AnthropicPlayAgent(
        api_key="test-key",
        client=SimpleNamespace(messages=_Broken()),
        frames=_frames,
        note_unreadable=lambda *args: notes.append(args),
    )

    spans = agent.label_window([(1.0, b"aa"), (4.0, b"bbb")])

    assert [span.kind for span in spans] == ["uncertain"]
    assert notes[0][2:4] == (2, 5)
    assert "HTTP 529" in str(notes[0][4])


def test_a_reply_that_is_not_json_is_set_aside(tmp_path: Path) -> None:
    del tmp_path
    messages = _Messages("I could not decide. The frames were blurry.")
    agent = AnthropicPlayAgent(api_key="test-key", client=_Client(messages), frames=_frames)

    spans = agent.label_window([(30.0, b"jpeg-a"), (70.0, b"jpeg-b")])

    assert [(span.kind, span.start_seconds, span.end_seconds) for span in spans] == [
        ("uncertain", 30.0, 70.0)
    ]


def test_youtube_link_is_downloaded_then_charted(tmp_path: Path) -> None:
    seen: list[str] = []

    def download(url: str, directory: Path) -> Path:
        seen.append(url)
        video = directory / "film.mp4"
        video.write_bytes(b"video")
        return video

    messages = _Messages('{"plays": [{"start_seconds": 1, "end_seconds": 2}]}')
    agent = AnthropicPlayAgent(
        api_key="test-key",
        client=_Client(messages),
        frames=_frames,
        downloader=download,
    )

    segments = agent.segment_youtube("https://www.youtube.com/watch?v=5mHoasOclJs")

    assert [segment.label for segment in segments] == ["Play 1: 0:01 - 0:02"]
    assert seen == ["https://www.youtube.com/watch?v=5mHoasOclJs"]


def test_a_low_credit_balance_still_stops_the_window() -> None:
    class _Billing(Exception):
        def __init__(self) -> None:
            super().__init__("credit balance is too low")
            self.api_message = "Your credit balance is too low to access the Anthropic API."
            self.error_code = None
            self.body = {"error": {"message": self.api_message}}

    class _Broken:
        def create(self, **kwargs: object) -> SimpleNamespace:
            raise _Billing()

    agent = AnthropicPlayAgent(
        api_key="test-key",
        client=SimpleNamespace(messages=_Broken()),
        frames=_frames,
    )

    with pytest.raises(VideoAnalysisError, match="out of credits"):
        agent.label_window([(1.0, b"jpeg")])


def test_a_low_credit_balance_is_explained(tmp_path: Path) -> None:
    class _Billing(Exception):
        def __init__(self) -> None:
            super().__init__(
                "Error code: 400 - {'type': 'error', 'error': {"
                "'type': 'invalid_request_error', "
                "'message': 'Your credit balance is too low to access the Anthropic API.'"
                "}}"
            )
            self.api_message = "Your credit balance is too low to access the Anthropic API."
            self.error_code = None
            self.body = {"error": {"type": "invalid_request_error", "message": self.api_message}}

    class _Broken:
        def create(self, **kwargs: object) -> SimpleNamespace:
            raise _Billing()

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    agent = AnthropicPlayAgent(
        api_key="test-key",
        client=SimpleNamespace(messages=_Broken()),
        frames=_frames,
    )

    with pytest.raises(
        VideoAnalysisError,
        match="The analysis account is out of credits.",
    ) as caught:
        agent.segment(video)

    assert "Error code" not in str(caught.value)


def test_non_youtube_link_is_rejected_before_download() -> None:
    def download(url: str, directory: Path) -> Path:
        raise AssertionError("should not download")

    agent = AnthropicPlayAgent(
        api_key="test-key",
        client=_Client(_Messages("{}")),
        downloader=download,
    )

    with pytest.raises(VideoAnalysisError, match="YouTube"):
        agent.segment_youtube("https://vimeo.com/123456789")


def test_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(MissingApiKeyError, match="ANTHROPIC_API_KEY"):
        agent_from_env()


def test_extract_frames_from_a_short_clip(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=160x120:d=4",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )

    frames = extract_frames(video)

    assert [timestamp for timestamp, _jpeg in frames[:2]] == [0.0, 2.0]
    assert frames[0][1].startswith(b"\xff\xd8")


def test_sample_frames_spreads_over_a_play_and_handles_a_short_clip(tmp_path: Path) -> None:
    from film_analysis.ingest.frames import sample_frames

    def clip(name: str, seconds: float) -> Path:
        video = tmp_path / name
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c=black:s=320x180:d={seconds}",
                "-pix_fmt",
                "yuv420p",
                str(video),
            ],
            check=True,
        )
        return video

    frames = sample_frames(clip("play.mp4", 6), 6, 6)
    assert len(frames) == 6
    assert all(frame.startswith(b"\xff\xd8\xff") for frame in frames)
    # One frame every two seconds gave nothing for a clip under two seconds.
    assert len(sample_frames(clip("short.mp4", 0.9), 0.9, 6)) >= 1
