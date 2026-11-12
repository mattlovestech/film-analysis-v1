"""One play through steps 2a to 2c, each step traced.

The pipeline does not know about the web app or the database. It reads the
play through a ``PlaySource`` and writes the finished diagram back through it,
so tests can hand it a fake film and the app hands it the real catalog.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from film_analysis.chart.call import SPECIAL_CALLS
from film_analysis.chart.isolate import PlayIsolation, isolate_play, mark_both_teams
from film_analysis.chart.play_diagram import _label_defense, _plot_players, _sides, blank_diagram
from film_analysis.diagram.fallbacks import (
    DEFENSE_TEMPLATE,
    OFFENSE_TEMPLATE,
    fill_to_eleven,
    loose_split,
    mark_players,
    model_points,
)
from film_analysis.diagram.trace import Run, RunStore, StepFailed, StepSkipped
from film_analysis.teams.uniforms import Uniform, label_home_away, uniform_line

TEAM_SIZE = 11
# Fewer than this many players on either jersey sends the play to the next method.
GOOD_SIDE = 7
_JPEG = b"\xff\xd8\xff"


@dataclass(frozen=True)
class PlayInfo:
    project: int
    play: int
    film: str
    family: str
    play_type: str | None
    start: float
    end: float


class PlaySource(Protocol):
    def play(self, project: int, play: int) -> PlayInfo: ...

    def uniforms(self, project: int) -> tuple[Uniform, Uniform] | None: ...

    def snap_frames(self, project: int, play: int) -> list[bytes]: ...

    def save(self, project: int, play: int, diagram: dict[str, object]) -> None: ...


Isolate = Callable[[Sequence[bytes], Uniform, Uniform], PlayIsolation]
Mark = Callable[[bytes, Uniform, Uniform], bytes | None]
AskModel = Callable[[Sequence[bytes], Uniform, Uniform], tuple[dict[str, object], str] | None]


@dataclass
class Placement:
    """Players found for the plot, and which method found them."""

    method: str
    found: PlayIsolation
    model: dict[str, object] | None = None
    model_offense: str = "home"


class DiagramPipeline:
    """Load → 2a classify → 2b frames, players → 2c plot → save."""

    def __init__(
        self,
        source: PlaySource,
        store: RunStore,
        *,
        isolate: Isolate | None = None,
        mark: Mark | None = None,
        loose: Isolate | None = None,
        ask_model: AskModel | None = None,
    ) -> None:
        self.source = source
        self.store = store
        self._isolate = isolate
        self._mark = mark
        self._loose = loose
        self._ask_model = ask_model

    def isolate(self, frames: Sequence[bytes], home: Uniform, away: Uniform) -> PlayIsolation:
        return (self._isolate or isolate_play)(frames, home, away)

    def mark(self, jpeg: bytes, home: Uniform, away: Uniform) -> bytes | None:
        return (self._mark or mark_both_teams)(jpeg, home, away)

    def loose(self, frames: Sequence[bytes], home: Uniform, away: Uniform) -> PlayIsolation:
        return (self._loose or loose_split)(frames, home, away)

    def ask_model(
        self, frames: Sequence[bytes], home: Uniform, away: Uniform
    ) -> tuple[dict[str, object], str] | None:
        return (self._ask_model or model_points)(frames, home, away)

    def run(self, run: Run, worker: str = "") -> Run:
        store = self.store
        store.set_state(run, "running", worker=worker, started_at=_now())
        store.event(run, "info", f"picked up by {worker or 'worker'}")
        try:
            info, home, away = self._load(run)
            self._classify(run, info)
            frames = self._frames(run, info)
            placement = self._players(run, frames, home, away)
            placement = self._model(run, frames, home, away, placement)
            diagram = self._plot(run, placement, home, away)
            self._save(run, info, diagram)
        except StepSkipped as reason:
            store.set_state(run, "skipped", detail=str(reason), ended_at=_now())
        except StepFailed as reason:
            store.set_state(run, "failed", detail=str(reason), error=str(reason), ended_at=_now())
        except Exception as error:
            store.set_state(
                run,
                "failed",
                detail=f"{type(error).__name__}: {error}",
                error=traceback.format_exc(),
                ended_at=_now(),
            )
        else:
            store.set_state(run, "done", detail="Diagram saved.", ended_at=_now())
        store.event(run, "info", f"run {run.status}")
        return run

    # Steps --------------------------------------------------------------

    def _load(self, run: Run) -> tuple[PlayInfo, Uniform, Uniform]:
        with self.store.step(run, "load", "1") as step:
            info = self.source.play(run.project, run.play)
            step.data.update(
                {
                    "film": info.film,
                    "clip_seconds": round(info.end - info.start, 2),
                    "family": info.family,
                    "play_type": info.play_type,
                }
            )
            pair = self.source.uniforms(run.project)
            if pair is None:
                # Guess light against dark so the play still gets a diagram.
                pair = (
                    Uniform("Home", "white", "white", "white", jersey_hex="#f2f2f2"),
                    Uniform("Away", "dark", "dark", "dark", jersey_hex="#2b2f4a"),
                )
                step.warnings.append("No uniforms picked for this film; guessed light vs dark.")
            first, second = label_home_away(*pair)
            # label_home_away renames in place; it does not put Home first.
            home, away = (first, second) if first.name == "Home" else (second, first)
            step.data["home"] = {
                "name": home.name,
                "hex": home.jersey_hex,
                "line": uniform_line(home),
            }
            step.data["away"] = {
                "name": away.name,
                "hex": away.jersey_hex,
                "line": uniform_line(away),
            }
            return info, home, away

    def _classify(self, run: Run, info: PlayInfo) -> None:
        with self.store.step(run, "classify", "2a") as step:
            special = info.family == "special_teams" or (info.play_type or "") in SPECIAL_CALLS
            kind = "special_teams" if special else "scrimmage"
            step.data.update({"kind": kind, "decided_by": "cut labeler"})
            if special:
                raise StepSkipped("Special teams play. Left out of diagrams for now.")
            step.detail = "Offense / defense snap."

    def _frames(self, run: Run, info: PlayInfo) -> list[bytes]:
        with self.store.step(run, "frames", "2b") as step:
            frames = [
                frame
                for frame in self.source.snap_frames(info.project, info.play)
                if isinstance(frame, bytes) and frame.startswith(_JPEG)
            ]
            step.data.update({"frames": len(frames), "bytes": sum(len(f) for f in frames)})
            for index, frame in enumerate(frames, start=1):
                self.store.put_image(run, step, f"snap-{index}", frame)
            if not frames:
                raise StepFailed("No frames came out of the clip.")
            if len(frames) < 4:
                step.warnings.append(f"Only {len(frames)} snap frames; expected 4.")
            return frames

    def _players(self, run: Run, frames: list[bytes], home: Uniform, away: Uniform) -> Placement:
        with self.store.step(run, "players", "2b") as step:
            strict = self.isolate(frames, home, away)
            placement = Placement("color", strict)
            step.data["color"] = _counts(strict, frames)
            if _side_score(strict) < GOOD_SIDE:
                loose = self.loose(frames, home, away)
                step.data["loose"] = _counts(loose, frames)
                if _score(loose) > _score(strict):
                    placement = Placement("loose", loose)
            found = placement.found
            mine, other = len(found.mine), len(found.other)
            step.data.update(
                {
                    "method": placement.method,
                    "home_players": mine,
                    "away_players": other,
                    "unit": found.unit,
                    "confidence": found.confidence,
                }
            )
            if found.frame:
                marked = (
                    self.mark(found.frame, home, away)
                    if placement.method == "color"
                    else mark_players(found.frame, found, home, away)
                )
                if marked:
                    self.store.put_image(run, step, f"players-{placement.method}", marked)
            if placement.method != "color":
                step.warnings.append(
                    "Strict jersey match found too few players; using every person "
                    "sent to the nearer jersey."
                )
            for count, name in ((mine, home.name or "home"), (other, away.name or "away")):
                if count < TEAM_SIZE:
                    step.warnings.append(f"{count} of {TEAM_SIZE} {name} players found.")
            step.detail = f"{mine} + {other} players by {placement.method}."
            return placement

    def _model(
        self,
        run: Run,
        frames: list[bytes],
        home: Uniform,
        away: Uniform,
        placement: Placement,
    ) -> Placement:
        with self.store.step(run, "model", "2b") as step:
            if _side_score(placement.found) >= GOOD_SIDE:
                step.skip(f"Not needed: {placement.method} found enough players.")
                return placement
            answer = self.ask_model(frames, home, away)
            if answer is None:
                step.warnings.append("Claude did not place the players (no key or no reply).")
                return placement
            diagram, offense_side = answer
            offense = [p for p in diagram.get("offense") or [] if isinstance(p, dict)]
            defense = [p for p in diagram.get("defense") or [] if isinstance(p, dict)]
            seen = sum(1 for p in (*offense, *defense) if p.get("seen", True))
            step.data.update(
                {
                    "offense_team": offense_side,
                    "offense": len(offense),
                    "defense": len(defense),
                    "seen": seen,
                }
            )
            if min(len(offense), len(defense)) <= _side_score(placement.found):
                step.warnings.append("Claude placed fewer players than the film; kept the film.")
                return placement
            step.detail = f"Claude placed {len(offense)} + {len(defense)} ({seen} seen)."
            return Placement("model", placement.found, diagram, offense_side)

    def _plot(
        self, run: Run, placement: Placement, home: Uniform, away: Uniform
    ) -> dict[str, object]:
        with self.store.step(run, "plot", "2c") as step:
            if placement.model is not None:
                offense, defense = (
                    (home, away) if placement.model_offense == "home" else (away, home)
                )
                diagram = blank_diagram(offense, defense)
                diagram["offense"] = [dict(p) for p in placement.model.get("offense") or []]
                diagram["defense"] = [dict(p) for p in placement.model.get("defense") or []]
                diagram["source"] = "model"
            else:
                offense, defense, offense_players, defense_players = _sides(
                    placement.found, home, away
                )
                diagram = _plot_players(offense, defense, offense_players, defense_players)
            off = diagram["offense"]
            dfn = diagram["defense"]
            seen_off, seen_def = len(off), len(dfn)
            added_off = fill_to_eleven(off, OFFENSE_TEMPLATE)
            added_def = fill_to_eleven(dfn, DEFENSE_TEMPLATE)
            position_labels(diagram)
            _label_defense(dfn)
            method = placement.method if seen_off or seen_def else "template"
            diagram["method"] = method
            diagram["guessed"] = {"offense": added_off, "defense": added_def}
            line = [p for p in off if p.get("role") in {"line", "center"}]
            step.data.update(
                {
                    "method": method,
                    "offense_team": offense.name,
                    "defense_team": defense.name,
                    "offense_seen": seen_off,
                    "defense_seen": seen_def,
                    "offense_guessed": added_off,
                    "defense_guessed": added_def,
                    "on_line": len(line),
                    "center": any(p.get("role") == "center" for p in off),
                    "qb": any(p.get("label") == "QB" for p in off),
                }
            )
            if added_off or added_def:
                step.warnings.append(
                    f"Filled {added_off} offense and {added_def} defense spots from a "
                    "standard shotgun / 4-2-5 look."
                )
            if len(line) < 7:
                step.warnings.append(
                    f"{len(line)} offensive players on the line; a legal set has 7."
                )
            run.diagram = diagram
            step.detail = (
                f"{seen_off}+{added_off} offense, {seen_def}+{added_def} defense ({method})."
            )
            return diagram

    def _save(self, run: Run, info: PlayInfo, diagram: dict[str, object]) -> None:
        with self.store.step(run, "save", "-") as step:
            self.source.save(info.project, info.play, diagram)
            step.detail = "Saved as the play layout."


def position_labels(diagram: dict[str, object]) -> None:
    """Name every offensive spot the way a play card does.

    The line out from the center is LT LG C RG RT. The closest back behind
    the center is QB. Wide players are X on the left and Z on the right, then
    H; a tight player on the line is Y; a back behind the quarterback is RB,
    and any other back is F.
    """

    offense = [p for p in diagram.get("offense") or [] if isinstance(p, dict)]
    if not offense:
        return
    for player in offense:
        player["label"] = ""
    centers = [p for p in offense if p.get("role") == "center"]
    pool = centers or [p for p in offense if p.get("role") == "line"]
    if not pool:
        return
    center = min(pool, key=lambda p: (abs(float(p["y"])), abs(float(p["x"]))))
    for extra in centers:
        if extra is center:
            continue
        # The plot calls anyone near the ball a center. Only one is.
        if float(extra["y"]) < -1.0:
            extra["role"], extra["symbol"] = "back", "filled"
        else:
            extra["role"], extra["symbol"] = "line", "open"
    center.update({"role": "center", "symbol": "square", "label": "C"})
    cx = float(center["x"])
    line = [p for p in offense if p.get("role") == "line"]
    for names, sign in ((("LG", "LT"), -1), (("RG", "RT"), 1)):
        side = sorted(
            (p for p in line if (float(p["x"]) - cx) * sign > 0),
            key=lambda p: abs(float(p["x"]) - cx),
        )
        last = cx
        for name, player in zip(names, side, strict=False):
            # Linemen stand shoulder to shoulder; a gap means a receiver.
            if abs(float(player["x"]) - last) > 2.5:
                break
            player["label"] = name
            last = float(player["x"])
    open_spots = [p for p in offense if not p["label"]]
    under = [
        p
        for p in open_spots
        if p.get("role") not in {"line", "center"}
        and abs(float(p["x"]) - cx) <= 2.5
        and -8.0 <= float(p["y"]) < 0
    ]
    if under:
        qb = max(under, key=lambda p: float(p["y"]))
        qb.update({"label": "QB", "role": "quarterback", "symbol": "filled"})
    used = {str(p["label"]) for p in offense if p["label"]}

    def take(*names: str) -> str:
        for name in names:
            if name not in used:
                used.add(name)
                return name
        return ""

    rest = sorted(
        (p for p in offense if not p["label"]),
        key=lambda p: abs(float(p["x"]) - cx),
        reverse=True,
    )
    for player in rest:
        x = float(player["x"]) - cx
        if abs(x) >= 8:
            player["label"] = take("X", "Z", "H", "W") if x < 0 else take("Z", "X", "H", "W")
        elif player.get("role") == "line":
            player["label"] = take("Y", "TE", "H")
        elif abs(x) <= 2.5:
            player["label"] = take("RB", "FB", "F")
        else:
            player["label"] = take("H", "F", "W")


def _score(found: PlayIsolation) -> int:
    return _side_score(found) * 100 + len(found.mine) + len(found.other)


def _side_score(found: PlayIsolation) -> int:
    return min(len(found.mine), len(found.other))


def _counts(found: PlayIsolation, frames: list[bytes]) -> dict[str, object]:
    return {
        "home": len(found.mine),
        "away": len(found.other),
        "frame": frames.index(found.frame) + 1 if found.frame in frames else None,
    }


def _now() -> float:
    return time.time()
