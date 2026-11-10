"""Formation reads use a spatial 11-on-11 plot after the call is known."""

import cv2
import numpy as np

from film_analysis.chart.formation import formation_label, formation_prompt, parse_formation
from film_analysis.chart.formation_features import (
    extract_features,
    lineup_label,
    same_lineup,
)
from film_analysis.chart.formation_plot import PlottedPlayer, _fill_team, render_formation_plot
from film_analysis.teams.uniforms import Uniform

_OFFENSE = Uniform("Home", "white", "white", "white", jersey_hex="#f4f1ea")
_DEFENSE = Uniform("Away", "red", "white", "red", jersey_hex="#b42318")


def test_a_clear_formation_is_kept() -> None:
    prompt = formation_prompt(_OFFENSE, _DEFENSE, call="run")

    assert "already charted as a run play" in prompt
    assert "The offense wears" in prompt
    assert "The defense wears" in prompt
    assert "spatial plot" in prompt
    assert "Name the backfield first" in prompt
    assert "spatial plot" not in formation_prompt(_OFFENSE, _DEFENSE, call="run", plot=False)
    assert parse_formation('{"formation":"shotgun"}') == "shotgun"
    assert parse_formation('{"formation":"I formation"}') == "i_form"
    assert formation_label("shotgun") == "Shotgun"
    assert formation_label("trips") == "Trips"


def test_an_unsure_or_unknown_formation_is_left_blank() -> None:
    assert parse_formation('{"formation":null}') is None
    assert parse_formation('{"formation":"wildcat"}') is None
    assert parse_formation("not json") is None
    assert formation_label(None) == ""
    assert formation_label("wildcat") == ""


def test_each_side_always_has_eleven_players() -> None:
    offense = _fill_team([(0.2, 0.6), (0.5, 0.7)], "offense", backfield=True)
    defense = _fill_team([(0.4, 0.3)], "defense", backfield=False)

    assert len(offense) == 11
    assert len(defense) == 11
    assert sum(1 for player in offense if player.confidence == "low") == 9
    assert sum(1 for player in defense if player.confidence == "low") == 10


def test_formation_plot_draws_both_teams() -> None:
    offense = _fill_team([(0.3, 0.65), (0.5, 0.8), (0.7, 0.65)], "offense", backfield=True)
    defense = _fill_team([(0.25, 0.35), (0.5, 0.2), (0.75, 0.35)], "defense", backfield=False)

    plot = render_formation_plot(offense, defense, _OFFENSE, _DEFENSE)

    assert plot.startswith(b"\xff\xd8\xff")
    assert len(plot) > 1000


def test_the_xy_chart_puts_offense_and_defense_on_opposite_sides_of_the_los() -> None:
    offense = [PlottedPlayer(0.5, 0.78, "offense", "high")]
    defense = [PlottedPlayer(0.5, 0.22, "defense", "high")]
    hollow = [PlottedPlayer(0.2, 0.78, "offense", "low")]

    filled = _image(render_formation_plot(offense, defense, _OFFENSE, _DEFENSE, focus="offense"))
    guessed = _image(render_formation_plot(hollow, defense, _OFFENSE, _DEFENSE, focus="offense"))
    offense_rows = _rows_near(filled, (234, 241, 244))
    defense_rows = _rows_near(filled, _muted((24, 35, 180)))

    assert offense_rows.size and defense_rows.size
    assert float(offense_rows.mean()) > float(defense_rows.mean())
    assert _count_near(filled, (234, 241, 244)) > _count_near(guessed, (234, 241, 244))


def test_similar_lineups_group_without_matching_coordinates() -> None:
    first = extract_features(_offense_look(0.0), _defense_look(0.0))
    shifted = extract_features(_offense_look(0.03), _defense_look(0.02))
    other = extract_features(_offense_look(0.0, heavy=True), _defense_look(0.0, nickel=True))

    assert same_lineup(first, shifted, "offense")
    assert same_lineup(first, shifted, "defense")
    assert not same_lineup(first, other, "offense")
    assert not same_lineup(first, other, "defense")
    assert lineup_label(first, "offense") == "Shotgun, 3 in the backfield, 2 wide"
    assert lineup_label(first, "defense") == "4 linemen, 3 linebackers, 4 deep"
    assert lineup_label(other, "offense") == "Shotgun, heavy"
    assert lineup_label(None, "defense", "I-formation") == "Needs a look"


def _offense_look(shift: float, *, heavy: bool = False) -> list[PlottedPlayer]:
    line = [
        PlottedPlayer(x + shift, 0.40, "offense", "high")
        for x in (0.30, 0.40, 0.50, 0.60, 0.70)
    ]
    backs = [
        PlottedPlayer(0.50 + shift, 0.62, "offense", "high"),
        PlottedPlayer(0.40 + shift, 0.70, "offense", "high"),
        PlottedPlayer(0.60 + shift, 0.70, "offense", "high"),
    ]
    if heavy:
        wings = [
            PlottedPlayer(0.18 + shift, 0.49, "offense", "high"),
            PlottedPlayer(0.82 + shift, 0.49, "offense", "high"),
        ]
    else:
        wings = [
            PlottedPlayer(0.05 + shift, 0.42, "offense", "high"),
            PlottedPlayer(0.95 + shift, 0.42, "offense", "high"),
        ]
    return line + backs + wings


def _defense_look(shift: float, *, nickel: bool = False) -> list[PlottedPlayer]:
    if nickel:
        spots = [
            (0.32, 0.34),
            (0.50, 0.34),
            (0.68, 0.34),
            (0.28, 0.22),
            (0.72, 0.22),
            (0.08, 0.06),
            (0.30, 0.05),
            (0.50, 0.04),
            (0.70, 0.05),
            (0.92, 0.06),
            (0.50, 0.16),
        ]
    else:
        spots = [(0.32, 0.35), (0.44, 0.35), (0.56, 0.35), (0.68, 0.35)]
        spots += [(0.36, 0.22), (0.50, 0.22), (0.64, 0.22)]
        spots += [(0.10, 0.05), (0.36, 0.05), (0.64, 0.05), (0.90, 0.05)]
    return [PlottedPlayer(x + shift, y, "defense", "high") for x, y in spots]


def _image(plot: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(plot, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    return image


def _muted(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(min(255, int(40 + channel * 0.45)) for channel in color)


def _rows_near(image: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    return np.where(_near(image, color))[0]


def _count_near(image: np.ndarray, color: tuple[int, int, int]) -> int:
    return int(_near(image, color).sum())


def _near(image: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    diff = np.abs(image.astype(np.int16) - np.array(color, dtype=np.int16))
    return diff.max(axis=2) <= 28
