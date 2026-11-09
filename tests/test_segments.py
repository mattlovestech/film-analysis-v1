import pytest

from film_analysis.agents.segments import format_play, parse_play_segments


def test_play_line_matches_the_clock_format() -> None:
    assert format_play(1, 1, 2) == "Play 1: 0:01 - 0:02"


def test_clock_rolls_past_a_minute() -> None:
    assert format_play(2, 61, 75.2) == "Play 2: 1:01 - 1:15"


def test_sub_second_play_keeps_a_decimal() -> None:
    assert format_play(1, 1.2, 1.4) == "Play 1: 0:01.2 - 0:01.4"


def test_parse_numbers_plays_in_time_order() -> None:
    raw = '{"plays": [{"play": 9, "start_seconds": 4, "end_seconds": 9}, {"start": 1, "end": 2}]}'

    segments = parse_play_segments(raw)

    assert [segment.label for segment in segments] == [
        "Play 1: 0:01 - 0:02",
        "Play 2: 0:04 - 0:09",
    ]


def test_parse_strips_json_fences() -> None:
    raw = '```json\n{"plays": [{"start_seconds": 1, "end_seconds": 2}]}\n```'

    assert parse_play_segments(raw)[0].label == "Play 1: 0:01 - 0:02"


def test_parse_skips_a_play_that_ends_where_it_starts() -> None:
    raw = (
        '{"plays": [{"start_seconds": 1, "end_seconds": 2}, '
        '{"start_seconds": 39, "end_seconds": 39}]}'
    )

    assert [segment.label for segment in parse_play_segments(raw)] == ["Play 1: 0:01 - 0:02"]


def test_parse_rejects_an_overlap() -> None:
    raw = (
        '{"plays": [{"start_seconds": 1, "end_seconds": 5}, '
        '{"start_seconds": 4, "end_seconds": 8}]}'
    )

    with pytest.raises(ValueError, match="overlaps"):
        parse_play_segments(raw)
