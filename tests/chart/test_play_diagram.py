"""A scrimmage clip is read as the players on each jersey, then their x and y."""

from film_analysis.chart.isolate import IsolatedPlayer, PlayIsolation
from film_analysis.chart.play_diagram import (
    parse_points,
    read_scrimmage,
    steps_done,
)
from film_analysis.teams.uniforms import Uniform

_HOME = Uniform("Home", "white", "white", "white", jersey_hex="#f7f7f7")
_AWAY = Uniform("Away", "red", "white", "red", jersey_hex="#aa2222")


def test_points_stay_on_the_field() -> None:
    parsed = parse_points(
        '{"offense":[{"x":80,"y":-90,"role":"back","symbol":"filled","label":"H","seen":true}],'
        '"defense":[{"x":0,"y":4,"role":"line","symbol":"open","label":"1","seen":true}]}'
    )

    assert parsed is not None
    assert parsed["offense"][0]["x"] == 26.6
    assert parsed["offense"][0]["y"] == -60.0
    assert parsed["defense"][0]["y"] == 4.0


def test_the_offense_is_the_team_with_the_ball(monkeypatch) -> None:
    monkeypatch.setattr(
        "film_analysis.chart.play_diagram.isolate_play",
        lambda frames, left, right: PlayIsolation(unit="defense", confidence="high"),
    )

    diagram = read_scrimmage([b"\xff\xd8\xff"], _HOME, _AWAY)

    assert diagram["teams"] == {
        "offense": "Away",
        "defense": "Home",
        "offense_hex": "#aa2222",
        "defense_hex": "#f7f7f7",
    }
    assert diagram["call"] == ""
    assert diagram["formation"] == ""
    assert diagram["offense"] == []
    assert diagram["defense"] == []


def test_a_short_lineup_still_plots(monkeypatch) -> None:
    found = PlayIsolation(
        mine=(IsolatedPlayer(0.3, 0.85), IsolatedPlayer(0.7, 0.9)),
        other=(IsolatedPlayer(0.5, 0.2),),
        unit=None,
    )
    monkeypatch.setattr(
        "film_analysis.chart.play_diagram.isolate_play",
        lambda frames, left, right: found,
    )

    diagram = read_scrimmage([b"\xff\xd8\xff"], _HOME, _AWAY)

    assert diagram["teams"]["offense"] == "Home"
    assert diagram["teams"]["defense"] == "Away"
    assert len(diagram["offense"]) == 2
    assert len(diagram["defense"]) == 1
    saved = diagram["offense"] + diagram["defense"]
    assert all("x" in player and "y" in player for player in saved)
    assert all(player["y"] <= 0 for player in diagram["offense"])
    assert all(player["y"] >= 0 for player in diagram["defense"])
    assert all(abs(player["y"]) <= 12 for player in saved)
    assert steps_done(diagram) == {"type": 100, "players": 100, "plot": 100}


def test_a_player_at_the_bottom_of_the_frame_stays_on_the_diagram() -> None:
    from film_analysis.chart.play_diagram import _point

    point = _point(0.5, 1.0, "offense")

    assert point["y"] < 0
    assert abs(point["y"]) <= 12


def test_a_missing_lineup_saves_no_points(monkeypatch) -> None:
    monkeypatch.setattr(
        "film_analysis.chart.play_diagram.isolate_play",
        lambda frames, left, right: PlayIsolation(unit=None),
    )

    diagram = read_scrimmage([], _HOME, _AWAY)

    assert diagram["formation"] == ""
    assert diagram["offense"] == []
    assert diagram["defense"] == []
    assert steps_done(diagram)["players"] == 0
    assert steps_done(diagram)["plot"] == 0
    assert steps_done(diagram)["type"] == 100


def test_the_defense_is_placed_toward_the_goal() -> None:
    from film_analysis.chart.play_diagram import _point

    offense = _point(0.5, 0.7, "offense")
    defense = _point(0.5, 0.3, "defense")

    assert offense["y"] < 0
    assert defense["y"] > 0
    assert defense["label"] == ""
