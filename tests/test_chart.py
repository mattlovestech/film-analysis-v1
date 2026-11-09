"""Run or pass on offense and defense, and the return when your team receives."""

from film_analysis.web.chart import chart_prompt, parse_play_call, parse_play_side
from film_analysis.web.teams import Uniform

_MINE = Uniform("Home", "white", "white", "white", jersey_hex="#f4f1ea")
_OTHER = Uniform("Away", "red", "white", "red", jersey_hex="#b42318")


def test_an_offense_play_is_only_run_or_pass() -> None:
    prompt = chart_prompt(_MINE, _OTHER, "scrimmage")

    assert "Your team wears" in prompt
    assert "white" in prompt
    assert "The opponent wears" in prompt
    assert "red" in prompt
    assert "run" in prompt
    assert "pass" in prompt
    assert "offensive or defensive play for the opponent" in prompt
    assert "The opponent is on offense when the opponent has the ball" in prompt
    assert "kick_return" not in prompt
    assert parse_play_call('{"call":"run"}', "scrimmage") == "run"
    assert parse_play_side('{"call":"run","opponent":"offense"}', "scrimmage") == "defense"
    assert parse_play_side('{"call":"run","opponent":"defense"}', "scrimmage") == "offense"
    assert parse_play_side('{"call":"run","side":"defense"}', "scrimmage") == "defense"
    assert parse_play_side("The opponent is on offense.", "scrimmage") == "defense"
    assert parse_play_side("The opponent is on defense.", "scrimmage") == "offense"
    assert parse_play_side('{"call":"kick_return"}', "special_teams") is None
    assert parse_play_call('{"call":"pass","unit":"defense"}', "scrimmage") == "pass"
    assert parse_play_call('{"call":"kick_return"}', "scrimmage") is None


def test_special_teams_is_named_from_your_team() -> None:
    prompt = chart_prompt(_MINE, _OTHER, "special_teams")

    assert "kick_return when your team receives the kickoff" in prompt
    assert "punt_return when your team receives the punt" in prompt
    assert parse_play_call('{"call":"kick_return"}', "special_teams") == "kick_return"
    assert parse_play_call('{"call":"punt return"}', "special_teams") == "punt_return"
    assert parse_play_call('{"call":"run"}', "special_teams") is None
    assert parse_play_call("not json", "special_teams") is None
