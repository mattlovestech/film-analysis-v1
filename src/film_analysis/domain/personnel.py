"""Offensive personnel groupings.

NFL-style labels use two digits: running backs, then tight ends.
The remaining skill players, out of five, are wide receivers.
"11" is 1 RB, 1 TE, and 3 WR. The quarterback is not counted.
"""

from pydantic import BaseModel, ConfigDict, Field


class Personnel(BaseModel):
    """A personnel grouping such as 11 or 12 personnel."""

    model_config = ConfigDict(frozen=True)

    label: str = Field(min_length=2, max_length=2)
    running_backs: int = Field(ge=0, le=5)
    tight_ends: int = Field(ge=0, le=5)
    wide_receivers: int = Field(ge=0, le=5)


def personnel_from_label(label: str) -> Personnel:
    """Parse a two-digit personnel label into skill-position counts."""

    cleaned = label.strip()
    if len(cleaned) != 2 or not cleaned.isdigit():
        raise ValueError(f"personnel label must be two digits, got {label!r}")

    running_backs = int(cleaned[0])
    tight_ends = int(cleaned[1])
    wide_receivers = 5 - running_backs - tight_ends
    if wide_receivers < 0:
        raise ValueError(
            f"personnel {cleaned} needs more than five skill players "
            f"({running_backs} RB, {tight_ends} TE)"
        )

    return Personnel(
        label=cleaned,
        running_backs=running_backs,
        tight_ends=tight_ends,
        wide_receivers=wide_receivers,
    )
