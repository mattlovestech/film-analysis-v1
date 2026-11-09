"""Handoff types between the chart and an analysis agent."""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from film_analysis.domain.models import Game, Play


class FilmContext(BaseModel):
    """The slice of film an agent is asked to read."""

    model_config = ConfigDict(frozen=True)

    game: Game
    plays: tuple[Play, ...]
    focus: str = ""

    @property
    def play_ids(self) -> tuple[str, ...]:
        return tuple(play.play_id for play in self.plays)


class AgentFinding(BaseModel):
    """A citeable read. Evidence points back at charted play ids."""

    model_config = ConfigDict(frozen=True)

    agent: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    evidence_play_ids: tuple[str, ...] = ()


@runtime_checkable
class FilmAgent(Protocol):
    """Something that can read a film context and return one finding."""

    name: str

    def run(self, context: FilmContext) -> AgentFinding:
        """Inspect the context and return a finding grounded in its plays."""
