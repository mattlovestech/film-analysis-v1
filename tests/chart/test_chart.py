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


def test_the_play_read_asks_for_a_team_card_player() -> None:
    for kind in ("scrimmage", "special_teams"):
        prompt = chart_prompt(_MINE, _OTHER, kind)

        assert "best photo of one player from each team" in prompt
        assert "Never pick a referee" in prompt
        assert '"players":{"A":{"frame":2,"box":[0.29,0.38,0.33,0.62],"score":7}}' in prompt


def test_player_picks_keep_only_real_frames_and_boxes() -> None:
    from film_analysis.chart.call import PlayerPick, parse_player_picks

    reply = (
        '{"offense":"A","call":"run","players":{'
        '"A":{"frame":2,"box":[0.29,0.38,0.33,0.62],"score":12},'
        '"Team B":{"frame":4,"box":[-0.1,0.2,0.1,0.5],"score":6}}}'
    )

    assert parse_player_picks(reply, 6) == {
        "A": PlayerPick(2, (0.29, 0.38, 0.33, 0.62), 10),
        "B": PlayerPick(4, (0.0, 0.2, 0.1, 0.5), 6),
    }
    assert parse_player_picks(reply, 3) == {"A": PlayerPick(2, (0.29, 0.38, 0.33, 0.62), 10)}
    empty_box = '{"players":{"A":{"frame":1,"box":[0.5,0.5,0.5,0.5],"score":9}}}'
    unknown_team = '{"players":{"C":{"frame":1,"box":[0.1,0.1,0.2,0.4],"score":9}}}'
    assert parse_player_picks(empty_box, 6) == {}
    assert parse_player_picks(unknown_team, 6) == {}
    assert parse_player_picks('{"call":"run"}', 6) == {}
