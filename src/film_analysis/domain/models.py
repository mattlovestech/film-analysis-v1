"""Games and charted plays."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from film_analysis.domain.personnel import Personnel
from film_analysis.domain.situations import (
    DownDistanceBucket,
    bucket_down_and_distance,
    is_goal_to_go,
    is_red_zone,
)


class PlayType(StrEnum):
    """High-level play family. Sacks and scrambles can be tagged later as details."""

    RUN = "run"
    PASS = "pass"
    KNEEL = "kneel"
    SPIKE = "spike"
    PUNT = "punt"
    KICKOFF = "kickoff"
    KICK_RETURN = "kick_return"
    PUNT_RETURN = "punt_return"
    FIELD_GOAL = "field_goal"
    EXTRA_POINT = "extra_point"
    TWO_POINT = "two_point"


class Hash(StrEnum):
    """Ball placement between the hashes, from the offense's perspective."""

    LEFT = "left"
    MIDDLE = "middle"
    RIGHT = "right"


def clock_to_seconds(clock: str) -> int:
    """Convert a game clock like '7:22' into seconds remaining in the period."""

    parts = clock.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"clock must look like M:SS, got {clock!r}")
    try:
        minutes = int(parts[0])
        seconds = int(parts[1])
    except ValueError as error:
        raise ValueError(f"clock must look like M:SS, got {clock!r}") from error
    if minutes < 0 or not 0 <= seconds <= 59:
        raise ValueError(f"clock must look like M:SS, got {clock!r}")
    return minutes * 60 + seconds


class Game(BaseModel):
    """One contest. Team names stay free-form so this is not NFL-only."""

    model_config = ConfigDict(frozen=True)

    game_id: str = Field(min_length=1)
    season: int = Field(ge=1900, le=2100)
    week: int = Field(ge=0, le=30)
    home: str = Field(min_length=1)
    away: str = Field(min_length=1)

    @field_validator("game_id", "home", "away")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned


class Play(BaseModel):
    """One charted snap.

    `yards_to_goal` is the distance to the opponent's end zone, from 1 to 99.
    Down and distance are omitted for kicks and other non-scrimmage snaps.
    """

    model_config = ConfigDict(frozen=True)

    play_id: str = Field(min_length=1)
    game_id: str = Field(min_length=1)
    offense: str = Field(min_length=1)
    defense: str = Field(min_length=1)
    quarter: int = Field(ge=1, le=8)
    clock_seconds: int = Field(ge=0, le=20 * 60)
    down: int | None = Field(default=None, ge=1, le=4)
    distance: int | None = Field(default=None, ge=0, le=99)
    yards_to_goal: int | None = Field(default=None, ge=1, le=99)
    play_type: PlayType | None = None
    personnel: Personnel | None = None
    formation: str | None = None
    hash_mark: Hash | None = None
    yards_gained: int | None = None
    description: str = ""

    @field_validator("play_id", "game_id", "offense", "defense")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned

    @field_validator("formation")
    @classmethod
    def _strip_formation(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @model_validator(mode="after")
    def _distance_fits_on_field(self) -> Self:
        if (
            self.distance is not None
            and self.yards_to_goal is not None
            and self.distance > self.yards_to_goal
        ):
            raise ValueError("distance cannot exceed yards_to_goal")
        return self

    @property
    def down_distance_bucket(self) -> DownDistanceBucket | None:
        if self.down is None or self.distance is None:
            return None
        return bucket_down_and_distance(self.down, self.distance)

    @property
    def red_zone(self) -> bool | None:
        if self.yards_to_goal is None:
            return None
        return is_red_zone(self.yards_to_goal)

    @property
    def goal_to_go(self) -> bool | None:
        if self.distance is None or self.yards_to_goal is None:
            return None
        return is_goal_to_go(self.distance, self.yards_to_goal)
