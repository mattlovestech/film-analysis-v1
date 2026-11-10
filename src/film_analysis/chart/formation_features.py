"""Football structure read from a formation plot.

O-line anchors the LOS. From that, the offense's starting look is shotgun or
under center, how many men are in the backfield, and the WR/TE shape. Defense
points are binned into line, linebackers, and secondary for context. Plays that
match within a similarity range are the same starting formation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from film_analysis.chart.formation_plot import PlottedPlayer

# Depth behind the LOS (offense y increases into the backfield after normalize).
_UNDER_CENTER_MAX = 0.08
_PISTOL_MAX = 0.16
_SHOTGUN_MIN = 0.14
_BACKFIELD_MIN = 0.10
_TE_WIDTH = 0.28
_WR_WIDTH = 0.38
_DL_DEPTH = 0.12
_LB_DEPTH = 0.28
NEEDS_LOOK = "Needs a look"
_BACKFIELD_NAMES = {
    "shotgun": "Shotgun",
    "pistol": "Pistol",
    "under_center": "Under center",
}


@dataclass(frozen=True)
class FormationFeatures:
    """Structural starting formation for one snap."""

    backfield: str
    backs: int
    receivers: str
    wideouts: int
    tight_ends: int
    defense_front: str
    sketch: str
    quality: str
    defense_quality: str = "low"

    def bucket(self) -> str:
        """Offensive similarity key. Matching shapes are not identical coordinates."""

        return f"{self.backfield}|{self.backs}|{self.receivers}"

    def defense_bucket(self) -> str:
        """Defensive similarity key from linemen, linebackers, and deep players."""

        return self.defense_front if self.defense_front not in {"", "unknown"} else ""


@dataclass(frozen=True)
class FormationGroupView:
    """Plays that came out in the same starting formation."""

    id: str
    bucket: str
    sketch: str
    count: int
    play_numbers: tuple[int, ...]
    ai_guess: str
    formation: str
    representative: int


def extract_features(
    offense: Sequence[PlottedPlayer],
    defense: Sequence[PlottedPlayer] = (),
) -> FormationFeatures:
    """Read the starting formation from plotted points."""

    offense_pts = [_xy(player) for player in offense]
    defense_pts = [_xy(player) for player in defense]
    if len(offense_pts) < 3:
        return FormationFeatures(
            "unknown",
            0,
            "unknown",
            0,
            0,
            "unknown",
            "unknown",
            "low",
        )
    los_y, center_x = _oline_anchor(offense_pts)
    relative = [((x - center_x), (y - los_y)) for x, y in offense_pts]
    # After normalize: y > 0 is into the offensive backfield.
    line = [point for point in relative if abs(point[1]) <= 0.07 and abs(point[0]) <= 0.42]
    skill = [point for point in relative if point not in line]
    qb = min(skill, key=lambda point: abs(point[0])) if skill else None
    backfield_kind, backs = _backfield(skill, qb)
    receivers, wideouts, tight_ends = _receivers(skill, qb)
    defense_front = _defense_front(defense_pts, los_y, center_x)
    high = sum(1 for player in offense if player.confidence == "high")
    def_high = sum(1 for player in defense if player.confidence == "high")
    quality = "high" if high >= 6 else "medium" if high >= 3 else "low"
    defense_quality = "high" if def_high >= 6 else "medium" if def_high >= 3 else "low"
    sketch = offense_title_parts(backfield_kind, backs, receivers, wideouts) or "unknown"
    return FormationFeatures(
        backfield_kind,
        backs,
        receivers,
        wideouts,
        tight_ends,
        defense_front,
        sketch,
        quality,
        defense_quality,
    )


def features_match(left: FormationFeatures, right: FormationFeatures) -> bool:
    """True when both snaps look like the same offensive formation."""

    return same_lineup(left, right, "offense")


def same_lineup(left: FormationFeatures, right: FormationFeatures, unit: str) -> bool:
    """True when the scouted side's shape matches, even if the XY points differ."""

    key = lineup_bucket(left, unit)
    return bool(key) and key == lineup_bucket(right, unit)


def lineup_bucket(features: FormationFeatures, unit: str) -> str:
    """The shape key for the scouted team. Empty when the plot is too thin to group."""

    if unit == "defense":
        if features.defense_quality == "low":
            return ""
        return features.defense_bucket()
    if features.quality == "low" or features.bucket() == "unknown|0|unknown":
        return ""
    return features.bucket()


def lineup_label(features: FormationFeatures | None, unit: str, fallback: str = "") -> str:
    """Plain name of how the scouted team lined up.

    Offense falls back to a stored formation name when no plot was kept.
    Defense never uses that name — it is the other team's offensive formation.
    """

    if features is not None:
        if unit == "defense":
            title = defense_title(features) if features.defense_quality != "low" else ""
        else:
            title = offense_title(features) if features.quality != "low" else ""
        if title:
            return title
    if unit != "defense" and fallback:
        return fallback
    return NEEDS_LOOK


def offense_title(features: FormationFeatures) -> str:
    """Shotgun, 3 in the backfield, 2 wide — or a heavy set."""

    return offense_title_parts(
        features.backfield, features.backs, features.receivers, features.wideouts
    )


def offense_title_parts(backfield: str, backs: int, receivers: str, wideouts: int) -> str:
    kind = _BACKFIELD_NAMES.get(backfield, "")
    if receivers == "heavy":
        return f"{kind}, heavy" if kind else "Heavy"
    if not kind:
        return ""
    men = backs if backfield == "under_center" else backs + 1
    parts = [kind, f"{men} in the backfield"]
    if wideouts:
        parts.append(f"{wideouts} wide")
    return ", ".join(parts)


def defense_title(features: FormationFeatures) -> str:
    """4 linemen, 3 linebackers, 4 deep."""

    parts = (features.defense_front or "").split("-")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        return ""
    linemen, backers, deep = (int(part) for part in parts)
    return (
        f"{_people(linemen, 'lineman', 'linemen')}, "
        f"{_people(backers, 'linebacker', 'linebackers')}, "
        f"{deep} deep"
    )


def _people(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


def group_by_features(
    rows: Sequence[tuple[int, FormationFeatures, str, str]],
) -> list[FormationGroupView]:
    """Group plays by football feature bucket, most used first.

    Each row is (play_number, features, ai_guess, confirmed_formation).
    """

    buckets: dict[str, list[tuple[int, FormationFeatures, str, str]]] = {}
    for row in rows:
        features = row[1]
        key = lineup_bucket(features, "offense") or f"low:{row[0]}"
        buckets.setdefault(key, []).append(row)
    groups: list[FormationGroupView] = []
    for key, members in buckets.items():
        members = sorted(members, key=lambda item: item[0])
        guesses = [guess for _number, _features, guess, _named in members if guess]
        confirmed = next(
            (named for _number, _features, _guess, named in members if named),
            "",
        )
        ai = _majority(guesses) if guesses else ""
        sketch = members[0][1].sketch
        numbers = tuple(number for number, _features, _guess, _named in members)
        groups.append(
            FormationGroupView(
                id=key.replace("|", "-").replace(":", "-"),
                bucket=key,
                sketch=sketch,
                count=len(members),
                play_numbers=numbers,
                ai_guess=ai,
                formation=confirmed,
                representative=numbers[0],
            )
        )
    return sorted(groups, key=lambda group: (-group.count, group.representative))


def features_from_dict(value: dict[str, object] | None) -> FormationFeatures | None:
    if not value:
        return None
    front = str(value.get("defense_front") or "unknown")
    stored_defense = value.get("defense_quality")
    if stored_defense is None and len(front.split("-")) == 3 and all(
        part.isdigit() for part in front.split("-")
    ):
        defense_quality = "medium"
    else:
        defense_quality = str(stored_defense or "low")
    try:
        return FormationFeatures(
            str(value.get("backfield") or "unknown"),
            int(value.get("backs") or 0),
            str(value.get("receivers") or "unknown"),
            int(value.get("wideouts") or 0),
            int(value.get("tight_ends") or 0),
            front,
            str(value.get("sketch") or "unknown"),
            str(value.get("quality") or "low"),
            defense_quality,
        )
    except (TypeError, ValueError):
        return None


def features_to_dict(features: FormationFeatures) -> dict[str, object]:
    return asdict(features)


def _xy(player: PlottedPlayer) -> tuple[float, float]:
    return float(player.x), float(player.y)


def _oline_anchor(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """LOS y and center x from the tightest horizontal cluster (the O-line)."""

    ordered = sorted(points, key=lambda point: point[1])
    # Prefer the densest band of ~5 players across width — the line of scrimmage.
    best: tuple[float, list[tuple[float, float]]] | None = None
    for _x, y in ordered:
        band = [point for point in ordered if abs(point[1] - y) <= 0.06]
        if len(band) < 3:
            continue
        spread = max(point[0] for point in band) - min(point[0] for point in band)
        score = len(band) * 2.0 + spread
        if best is None or score > best[0]:
            best = (score, band)
    if best is None:
        ys = [point[1] for point in points]
        xs = [point[0] for point in points]
        return sum(ys) / len(ys), sum(xs) / len(xs)
    band = best[1]
    # The five most central across the band are the O-line.
    across = sorted(band, key=lambda point: point[0])
    if len(across) > 5:
        start = max(0, (len(across) - 5) // 2)
        across = across[start : start + 5]
    los_y = sum(point[1] for point in across) / len(across)
    center_x = sum(point[0] for point in across) / len(across)
    return los_y, center_x


def _backfield(
    skill: Sequence[tuple[float, float]],
    qb: tuple[float, float] | None,
) -> tuple[str, int]:
    if qb is None:
        return "unknown", 0
    depth = qb[1]
    if depth <= _UNDER_CENTER_MAX:
        kind = "under_center"
    elif depth <= _PISTOL_MAX:
        kind = "pistol"
    else:
        kind = "shotgun"
    backs = 0
    for x, y in skill:
        if (x, y) == qb:
            continue
        if y < _BACKFIELD_MIN:
            continue
        # Backs stay near the QB hash; receivers flare wider.
        if abs(x - qb[0]) <= 0.22 and y >= _BACKFIELD_MIN:
            backs += 1
    backs = min(3, backs)
    if kind == "shotgun" and backs == 0 and depth >= _SHOTGUN_MIN:
        # Empty look still shotgun.
        pass
    return kind, backs


def _receivers(
    skill: Sequence[tuple[float, float]],
    qb: tuple[float, float] | None,
) -> tuple[str, int, int]:
    wideouts = 0
    tight_ends = 0
    for x, y in skill:
        if qb is not None and (x, y) == qb:
            continue
        if qb is not None and abs(x - qb[0]) <= 0.22 and y >= _BACKFIELD_MIN:
            continue
        width = abs(x)
        if width >= _WR_WIDTH and y <= 0.22:
            wideouts += 1
        elif _TE_WIDTH <= width < _WR_WIDTH and y <= 0.14:
            tight_ends += 1
        elif width >= _WR_WIDTH:
            wideouts += 1
        elif width >= 0.18:
            tight_ends += 1
    if wideouts == 0 and tight_ends == 0 and qb is not None:
        # Fallback from remaining skill count.
        others = [
            point
            for point in skill
            if point != qb and not (abs(point[0] - qb[0]) <= 0.22 and point[1] >= _BACKFIELD_MIN)
        ]
        wideouts = min(4, len(others))
    shape = _receiver_shape(wideouts, tight_ends)
    return shape, wideouts, tight_ends


def _receiver_shape(wideouts: int, tight_ends: int) -> str:
    if wideouts >= 4 and tight_ends == 0:
        return "empty"
    if wideouts >= 3 and tight_ends == 0:
        return "trips"
    if wideouts >= 2 and tight_ends >= 1:
        return "2x1"
    if wideouts >= 2 and tight_ends == 0:
        return "2x2"
    if wideouts == 1 and tight_ends >= 1:
        return "ace_wing"
    if tight_ends >= 2:
        return "heavy"
    if wideouts or tight_ends:
        return f"{wideouts}wr_{tight_ends}te"
    return "unknown"


def _defense_front(
    points: Sequence[tuple[float, float]],
    los_y: float,
    center_x: float,
) -> str:
    if not points:
        return "unknown"
    relative = [((x - center_x), (los_y - y)) for x, y in points]
    # Positive depth is into the defense.
    dl = sum(1 for _x, depth in relative if 0 <= depth <= _DL_DEPTH)
    lb = sum(1 for _x, depth in relative if _DL_DEPTH < depth <= _LB_DEPTH)
    db = sum(1 for _x, depth in relative if depth > _LB_DEPTH)
    return f"{dl}-{lb}-{db}"


def _majority(values: Sequence[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]
