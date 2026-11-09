"""Formation reads use a spatial 11-on-11 plot after the call is known."""

from film_analysis.chart.formation import formation_label, formation_prompt, parse_formation
from film_analysis.chart.formation_plot import _fill_team, render_formation_plot
from film_analysis.teams.uniforms import Uniform

_OFFENSE = Uniform("Home", "white", "white", "white", jersey_hex="#f4f1ea")
_DEFENSE = Uniform("Away", "red", "white", "red", jersey_hex="#b42318")


def test_a_clear_formation_is_kept() -> None:
    prompt = formation_prompt(_OFFENSE, _DEFENSE, call="run")

    assert "already charted as a run play" in prompt
    assert "The offense wears" in prompt
    assert "The defense wears" in prompt
    assert "spatial plot" in prompt
    assert "experimental" in prompt
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
