"""The scouted uniform is isolated on each play by jersey color."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from film_analysis.chart.formation_plot import plot_formation
from film_analysis.chart.isolate import isolate_play
from film_analysis.teams.players import _striped
from film_analysis.teams.uniforms import Uniform

_WHITE = Uniform(
    "Home",
    "white",
    "white",
    "white",
    jersey_hex="#f4f1ea",
    pants_hex="#f4f1ea",
    helmet_hex="#f4f1ea",
)
_RED = Uniform(
    "Away",
    "red",
    "white",
    "red",
    jersey_hex="#b42318",
    pants_hex="#f2efe6",
    helmet_hex="#b42318",
)
_NAVY = Uniform(
    "Home",
    "navy",
    "white",
    "navy",
    jersey_hex="#0a1628",
    pants_hex="#f2efe6",
    helmet_hex="#0a1628",
)
_BLACK = Uniform(
    "Away",
    "black",
    "white",
    "black",
    jersey_hex="#141414",
    pants_hex="#f2efe6",
    helmet_hex="#161616",
)
_WIDTH = 640
_HEIGHT = 360
_BODY = (28, 64)


def test_white_bodies_stay_on_the_scouted_team() -> None:
    frame = _jpeg(_bottom_lineup(_bgr("#f4f1ea"), _bgr("#b42318")))

    isolation = isolate_play([frame], _WHITE, _RED)

    assert len(isolation.mine) == 8
    assert len(isolation.other) == 5
    assert min(player.y for player in isolation.mine) > max(player.y for player in isolation.other)
    assert isolation.unit == "offense"
    assert isolation.confidence == "high"


def test_a_striped_official_is_left_out() -> None:
    bodies = _bottom_lineup(_bgr("#f4f1ea"), _bgr("#b42318"))
    referee = (18, 16, 42, 96)
    frame = _jpeg(bodies, referee=referee)
    image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
    x, y, width, height = referee

    assert _striped(image[y : y + height, x : x + width])
    isolation = isolate_play([frame], _WHITE, _RED)
    assert len(isolation.mine) == 8
    assert len(isolation.other) == 5
    assert all(player.x > 0.12 for player in (*isolation.mine, *isolation.other))


def test_navy_against_black_does_not_invent_a_side() -> None:
    frame = _jpeg(_bottom_lineup(_bgr("#0a1628"), _bgr("#141414")))

    isolation = isolate_play([frame], _NAVY, _BLACK)

    assert isolation.confidence == "low"
    assert isolation.unit is None
    assert isolation.mine == ()
    assert isolation.other == ()


def test_a_thin_frame_does_not_invent_a_side() -> None:
    white = _bgr("#f4f1ea")
    red = _bgr("#b42318")
    frame = _jpeg(
        [
            (200, 80, *_BODY, red),
            (280, 180, *_BODY, white),
            (360, 180, *_BODY, white),
        ]
    )

    isolation = isolate_play([frame], _WHITE, _RED)

    assert isolation.confidence == "low"
    assert isolation.unit is None


def test_the_row_with_backs_behind_it_is_the_offense_from_either_direction() -> None:
    frame = _jpeg(_right_lineup(_bgr("#f4f1ea"), _bgr("#b42318")))

    offense = isolate_play([frame], _WHITE, _RED)
    defense = isolate_play([frame], _RED, _WHITE)

    assert offense.unit == "offense"
    assert offense.confidence == "high"
    assert min(player.x for player in offense.mine) > max(player.x for player in offense.other)
    assert defense.unit == "defense"
    assert defense.confidence == "high"
    assert max(player.x for player in defense.mine) < min(player.x for player in defense.other)


def test_pants_color_splits_two_dark_jerseys() -> None:
    navy = Uniform(
        "Home",
        "navy",
        "white",
        "navy",
        jersey_hex="#0a1628",
        pants_hex="#f4f1ea",
        helmet_hex="#0a1628",
    )
    black = Uniform(
        "Away",
        "black",
        "red",
        "black",
        jersey_hex="#141414",
        pants_hex="#b42318",
        helmet_hex="#141414",
    )
    frame = _jpeg(_dark_uniforms())

    isolation = isolate_play([frame], navy, black)

    assert len(isolation.mine) == 3
    assert len(isolation.other) == 3
    assert max(player.x for player in isolation.mine) < min(player.x for player in isolation.other)


def test_the_formation_plot_uses_the_isolated_players() -> None:
    frame = _jpeg(_bottom_lineup(_bgr("#f4f1ea"), _bgr("#b42318")))

    layout = plot_formation([frame], _WHITE, _RED)

    assert layout is not None
    assert sum(1 for player in layout.offense if player.confidence == "high") == 8
    assert sum(1 for player in layout.defense if player.confidence == "high") == 5


def test_a_clear_lineup_sets_the_side_and_keeps_the_model_call(tmp_path: Path, monkeypatch) -> None:
    from film_analysis.chart.call import ChartedPlay
    from film_analysis.storage.catalog import MemoryCatalog
    from film_analysis.web.app import _chart_play

    store = MemoryCatalog(tmp_path)
    project_id = store.create_project("Friday", None)
    clip = tmp_path / "play.mp4"
    clip.write_bytes(b"clip")
    store.add_play(project_id, 1, 0.0, 6.0, clip, "scrimmage")
    project = store.get_project(project_id)
    frame = _jpeg(_bottom_lineup(_bgr("#f4f1ea"), _bgr("#b42318")))
    monkeypatch.setattr(
        "film_analysis.web.app._scrimmage_frames",
        lambda *_args, **_kwargs: [frame],
    )

    def reader(*_args: object) -> ChartedPlay:
        return ChartedPlay("pass", "defense")

    _chart_play(store, project, project.plays[0], _WHITE, _RED, reader, frozenset({"scrimmage"}))

    play = store.get_project(project_id).plays[0]
    assert play.unit == "offense"
    assert play.play_type == "pass"


def test_a_known_call_does_not_ask_the_model_when_the_colors_are_clear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from film_analysis.storage.catalog import MemoryCatalog
    from film_analysis.web.app import _chart_play

    store = MemoryCatalog(tmp_path)
    project_id = store.create_project("Friday", None)
    clip = tmp_path / "play.mp4"
    clip.write_bytes(b"clip")
    store.add_play(project_id, 1, 0.0, 6.0, clip, "scrimmage")
    store.set_play_type(project_id, 1, "run", "scrimmage")
    project = store.get_project(project_id)
    frame = _jpeg(_bottom_lineup(_bgr("#f4f1ea"), _bgr("#b42318")))
    monkeypatch.setattr(
        "film_analysis.web.app._scrimmage_frames",
        lambda *_args, **_kwargs: [frame],
    )
    called: list[str] = []

    def reader(*_args: object) -> str:
        called.append("model")
        return "pass"

    _chart_play(store, project, project.plays[0], _WHITE, _RED, reader, frozenset({"scrimmage"}))

    play = store.get_project(project_id).plays[0]
    assert called == []
    assert play.unit == "offense"
    assert play.play_type == "run"


def test_close_colors_leave_the_side_to_the_model(tmp_path: Path, monkeypatch) -> None:
    from film_analysis.chart.call import ChartedPlay
    from film_analysis.storage.catalog import MemoryCatalog
    from film_analysis.web.app import _chart_play

    store = MemoryCatalog(tmp_path)
    project_id = store.create_project("Friday", None)
    clip = tmp_path / "play.mp4"
    clip.write_bytes(b"clip")
    store.add_play(project_id, 1, 0.0, 6.0, clip, "scrimmage")
    project = store.get_project(project_id)
    frame = _jpeg(_bottom_lineup(_bgr("#0a1628"), _bgr("#141414")))
    monkeypatch.setattr(
        "film_analysis.web.app._scrimmage_frames",
        lambda *_args, **_kwargs: [frame],
    )

    def reader(*_args: object) -> ChartedPlay:
        return ChartedPlay("run", "defense")

    _chart_play(store, project, project.plays[0], _NAVY, _BLACK, reader, frozenset({"scrimmage"}))

    play = store.get_project(project_id).plays[0]
    assert play.unit == "defense"
    assert play.play_type == "run"


def _bottom_lineup(
    offense: tuple[int, int, int],
    defense: tuple[int, int, int],
) -> list[tuple[int, int, int, int, tuple[int, int, int]]]:
    width, height = _BODY
    spots: list[tuple[int, int, int, int, tuple[int, int, int]]] = []
    for x in (120, 200, 280, 360, 440):
        spots.append((x, 40, width, height, defense))
        spots.append((x, 150, width, height, offense))
    for x in (200, 280, 360):
        spots.append((x, 250, width, height, offense))
    return spots


def _right_lineup(
    offense: tuple[int, int, int],
    defense: tuple[int, int, int],
) -> list[tuple[int, int, int, int, tuple[int, int, int]]]:
    width, height = _BODY
    spots: list[tuple[int, int, int, int, tuple[int, int, int]]] = []
    for y in (30, 110, 190, 270):
        spots.append((70, y, width, height, defense))
        spots.append((280, y, width, height, offense))
    for y in (70, 150, 230):
        spots.append((460, y, width, height, offense))
    return spots


def _dark_uniforms() -> list[tuple[int, int, int, int, tuple[int, int, int]]]:
    """Three navy bodies with white pants, three black bodies with red pants."""

    navy = _bgr("#0a1628")
    black = _bgr("#141414")
    white = _bgr("#f4f1ea")
    red = _bgr("#b42318")
    spots: list[tuple[int, int, int, int, tuple[int, int, int]]] = []
    for index, x in enumerate((80, 180, 280)):
        spots.append((x, 120, 28, 38, navy))
        spots.append((x, 158, 28, 26, white))
        other = 420 + index * 70
        spots.append((other, 120, 28, 38, black))
        spots.append((other, 158, 28, 26, red))
    return spots


def _jpeg(
    bodies: list[tuple[int, int, int, int, tuple[int, int, int]]],
    referee: tuple[int, int, int, int] | None = None,
) -> bytes:
    image = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
    image[:] = (40, 140, 50)
    for x, y, width, height, color in bodies:
        image[y : y + height, x : x + width] = color
    if referee is not None:
        x, y, width, height = referee
        for column in range(width):
            light = (column // 6) % 2 == 0
            image[y : y + height, x + column] = (245, 245, 245) if light else (8, 8, 8)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok
    return encoded.tobytes()


def _bgr(hex_code: str) -> tuple[int, int, int]:
    red = int(hex_code[1:3], 16)
    green = int(hex_code[3:5], 16)
    blue = int(hex_code[5:7], 16)
    return blue, green, red
