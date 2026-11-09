import pytest
from pydantic import ValidationError

from film_analysis.domain.models import Game, Hash, Play, PlayType, clock_to_seconds
from film_analysis.domain.personnel import personnel_from_label
from film_analysis.domain.situations import DownDistanceBucket
from film_analysis.plays.protocol import AgentFinding, FilmAgent, FilmContext
from film_analysis.scout.tendencies import count_by_personnel, success_rate


def _game() -> Game:
    return Game(game_id="2024-01-kc-buf", season=2024, week=1, home="BUF", away="KC")


def _play(**overrides: object) -> Play:
    payload: dict[str, object] = {
        "play_id": "1",
        "game_id": "2024-01-kc-buf",
        "offense": "KC",
        "defense": "BUF",
        "quarter": 1,
        "clock_seconds": clock_to_seconds("12:00"),
        "down": 1,
        "distance": 10,
        "yards_to_goal": 75,
        "play_type": PlayType.PASS,
        "personnel": personnel_from_label("11"),
        "formation": "Gun Trips",
        "hash_mark": Hash.LEFT,
        "yards_gained": 6,
    }
    payload.update(overrides)
    return Play.model_validate(payload)


def test_clock_parses_minutes_and_seconds() -> None:
    assert clock_to_seconds("7:22") == 7 * 60 + 22


def test_play_derives_situation_flags() -> None:
    play = _play(down=3, distance=5, yards_to_goal=5, yards_gained=5)

    assert play.down_distance_bucket is DownDistanceBucket.THIRD_AND_MEDIUM
    assert play.red_zone is True
    assert play.goal_to_go is True


def test_distance_cannot_extend_past_the_goal_line() -> None:
    with pytest.raises(ValidationError):
        _play(distance=11, yards_to_goal=10)


def test_kickoff_can_omit_down_and_distance() -> None:
    play = _play(
        play_type=PlayType.KICKOFF,
        down=None,
        distance=None,
        yards_to_goal=None,
        personnel=None,
        yards_gained=None,
    )

    assert play.down_distance_bucket is None
    assert play.red_zone is None


def test_personnel_counts_and_success_rate() -> None:
    plays = (
        _play(play_id="1", yards_gained=4),
        _play(play_id="2", yards_gained=3, personnel=personnel_from_label("12")),
        _play(play_id="3", play_type=PlayType.KNEEL, yards_gained=-1),
    )

    assert count_by_personnel(plays) == {"11": 2, "12": 1}
    assert success_rate(plays) == 0.5


def test_success_rate_is_none_without_scrimmage_plays() -> None:
    assert success_rate((_play(play_type=PlayType.PUNT, yards_gained=40),)) is None


def test_film_agent_contract() -> None:
    class EchoAgent:
        name = "echo"

        def run(self, context: FilmContext) -> AgentFinding:
            summary = context.focus or "no focus"
            return AgentFinding(
                agent=self.name,
                summary=summary,
                evidence_play_ids=context.play_ids,
            )

    context = FilmContext(game=_game(), plays=(_play(),), focus="11 personnel")
    agent: FilmAgent = EchoAgent()

    finding = agent.run(context)

    assert isinstance(agent, FilmAgent)
    assert finding.summary == "11 personnel"
    assert finding.evidence_play_ids == ("1",)
