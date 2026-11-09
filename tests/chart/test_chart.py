"""Run or pass on offense and defense, and the return when your team receives."""

from film_analysis.chart.call import chart_prompt, parse_play_call, parse_play_side
from film_analysis.teams.uniforms import Uniform

_MINE = Uniform("Home", "white", "white", "white", jersey_hex="#f4f1ea")
_OTHER = Uniform("Away", "red", "white", "red", jersey_hex="#b42318")


def test_an_offense_play_is_only_run_or_pass() -> None:
    prompt = chart_prompt(_MINE, _OTHER, "scrimmage")

    assert "Team A is your team. Team A wears white" in prompt
    assert "Team B is the opponent. Team B wears red" in prompt
    assert "Team B's jerseys are darker than Team A's." in prompt
    assert "which team has the ball" in prompt
    assert "scorebug" in prompt
    assert "run" in prompt
    assert "pass" in prompt
    assert "kick_return" not in prompt
    assert parse_play_side('{"offense":"A","call":"run"}', "scrimmage") == "offense"
    assert parse_play_side('{"offense":"B","call":"pass"}', "scrimmage") == "defense"
    assert parse_play_call('{"call":"run"}', "scrimmage") == "run"
    assert parse_play_side('{"call":"run","opponent":"offense"}', "scrimmage") == "defense"
    assert parse_play_side('{"call":"run","opponent":"defense"}', "scrimmage") == "offense"
    assert parse_play_side('{"call":"run","side":"defense"}', "scrimmage") == "defense"
    assert parse_play_side("The opponent is on offense.", "scrimmage") == "defense"
    assert parse_play_side("The opponent is on defense.", "scrimmage") == "offense"
    assert parse_play_side('{"call":"kick_return"}', "special_teams") is None
    assert parse_play_call('{"call":"pass","unit":"defense"}', "scrimmage") == "pass"
    assert parse_play_call('{"call":"kick_return"}', "scrimmage") is None


def test_the_prompt_names_both_teams() -> None:
    mine = Uniform("Pike County", "red", "red", "red", jersey_hex="#c8102e")
    other = Uniform("Lamar County", "white", "white", "navy", jersey_hex="#f2f2f2")

    prompt = chart_prompt(mine, other, "scrimmage")

    assert "Team A is Pike County, your team." in prompt
    assert "Team B is Lamar County, the opponent." in prompt
    assert "Team A's jerseys are darker than Team B's." in prompt


def test_close_jersey_shades_are_not_called_darker() -> None:
    mine = Uniform("Home", "gray", "gray", "gray", jersey_hex="#808080")
    other = Uniform("Away", "gray", "gray", "gray", jersey_hex="#8a8a8a")

    assert "darker" not in chart_prompt(mine, other, "scrimmage")


def test_special_teams_is_named_from_your_team() -> None:
    prompt = chart_prompt(_MINE, _OTHER, "special_teams")

    assert "kick_return when your team receives the kickoff" in prompt
    assert "punt_return when your team receives the punt" in prompt
    assert parse_play_call('{"call":"kick_return"}', "special_teams") == "kick_return"
    assert parse_play_call('{"call":"punt return"}', "special_teams") == "punt_return"
    assert parse_play_call('{"call":"run"}', "special_teams") is None
    assert parse_play_call("not json", "special_teams") is None
