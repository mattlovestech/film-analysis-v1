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
from film_analysis.chart.play_diagram import _plot_players, _sides
from film_analysis.diagram.trace import Run, RunStore, StepFailed, StepSkipped
from film_analysis.teams.uniforms import Uniform, label_home_away, uniform_line

TEAM_SIZE = 11
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


class DiagramPipeline:
    """Load → 2a classify → 2b frames, players → 2c plot → save."""

    def __init__(
        self,
        source: PlaySource,
        store: RunStore,
        *,
        isolate: Isolate | None = None,
        mark: Mark | None = None,
    ) -> None:
        self.source = source
        self.store = store
        self._isolate = isolate
        self._mark = mark

    def isolate(self, frames: Sequence[bytes], home: Uniform, away: Uniform) -> PlayIsolation:
        return (self._isolate or isolate_play)(frames, home, away)

    def mark(self, jpeg: bytes, home: Uniform, away: Uniform) -> bytes | None:
        return (self._mark or mark_both_teams)(jpeg, home, away)

    def run(self, run: Run, worker: str = "") -> Run:
        store = self.store
        store.set_state(run, "running", worker=worker, started_at=_now())
        store.event(run, "info", f"picked up by {worker or 'worker'}")
        try:
            info, home, away = self._load(run)
            self._classify(run, info)
            frames = self._frames(run, info)
            found = self._players(run, frames, home, away)
            diagram = self._plot(run, found, home, away)
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
                raise StepFailed("This film has no uniforms yet. Pick the two teams first.")
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

    def _players(
        self, run: Run, frames: list[bytes], home: Uniform, away: Uniform
    ) -> PlayIsolation:
        with self.store.step(run, "players", "2b") as step:
            found = self.isolate(frames, home, away)
            mine, other = len(found.mine), len(found.other)
            chosen = frames.index(found.frame) + 1 if found.frame in frames else None
            step.data.update(
                {
                    "home_players": mine,
                    "away_players": other,
                    "unit": found.unit,
                    "confidence": found.confidence,
                    "best_frame": chosen,
                }
            )
            if found.frame:
                marked = self.mark(found.frame, home, away)
                if marked:
                    self.store.put_image(run, step, "players-marked", marked)
            if not mine or not other:
                raise StepFailed(
                    f"Found {mine} {home.name or 'home'} and {other} {away.name or 'away'} "
                    "players. Both jerseys are needed."
                )
            for count, name in ((mine, home.name or "home"), (other, away.name or "away")):
                if count < TEAM_SIZE:
                    step.warnings.append(f"{count} of {TEAM_SIZE} {name} players found.")
            if found.unit not in {"offense", "defense"}:
                step.warnings.append("Side of the ball guessed from depth, not lineup shape.")
            step.detail = f"{mine} + {other} players on frame {chosen}."
            return found

    def _plot(
        self, run: Run, found: PlayIsolation, home: Uniform, away: Uniform
    ) -> dict[str, object]:
        with self.store.step(run, "plot", "2c") as step:
            offense, defense, offense_players, defense_players = _sides(found, home, away)
            diagram = _plot_players(offense, defense, offense_players, defense_players)
            position_labels(diagram)
            off = [p for p in diagram["offense"] if isinstance(p, dict)]
            dfn = [p for p in diagram["defense"] if isinstance(p, dict)]
            line = [p for p in off if p.get("role") in {"line", "center"}]
            step.data.update(
                {
                    "offense_team": offense.name,
                    "defense_team": defense.name,
                    "offense": len(off),
                    "defense": len(dfn),
                    "on_line": len(line),
                    "center": any(p.get("role") == "center" for p in off),
                    "qb": any(p.get("label") == "QB" for p in off),
                }
            )
            if not step.data["center"]:
                step.warnings.append("No center found at the ball.")
            if len(line) < 7:
                step.warnings.append(
                    f"{len(line)} offensive players on the line; a legal set has 7."
                )
            if not off and not dfn:
                raise StepFailed("No player could be placed.")
            run.diagram = diagram
            step.detail = f"{len(off)} offense, {len(dfn)} defense placed."
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


def _now() -> float:
    return time.time()
