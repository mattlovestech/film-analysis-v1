"""The experimental formation read names only a known offensive look."""

from film_analysis.web.formation import formation_label, formation_prompt, parse_formation
from film_analysis.web.teams import Uniform

_MINE = Uniform("Home", "white", "white", "white", jersey_hex="#f4f1ea")
_OTHER = Uniform("Away", "red", "white", "red", jersey_hex="#b42318")


def test_a_clear_formation_is_kept() -> None:
    prompt = formation_prompt(_MINE, _OTHER)

    assert "The offense wears" in prompt
    assert "The defense wears" in prompt
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
