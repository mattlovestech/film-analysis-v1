"""One play through the traced diagram pipeline, with a fake film and fake vision."""

from pathlib import Path

from film_analysis.chart.isolate import IsolatedPlayer, PlayIsolation
from film_analysis.diagram.pipeline import DiagramPipeline, PlayInfo, position_labels
from film_analysis.diagram.trace import RunStore
from film_analysis.diagram.workers import DiagramWorkers
from film_analysis.teams.uniforms import Uniform

JPEG = b"\xff\xd8\xff" + b"0" * 32
HOME = Uniform("Home", "white", "white", "white", jersey_hex="#ffffff")
AWAY = Uniform("Away", "red", "white", "red", jersey_hex="#aa2222")

OFFENSE = (
    # Line: tackle, guard, center, guard, tackle, tight end, split end.
    (0.42, 0.50),
    (0.46, 0.50),
    (0.50, 0.50),
    (0.54, 0.50),
    (0.58, 0.50),
    (0.62, 0.50),
    (0.08, 0.50),
    # Quarterback, back, flanker, slot.
    (0.50, 0.58),
    (0.50, 0.66),
    (0.92, 0.53),
    (0.78, 0.53),
)
DEFENSE = tuple((x, 0.46) for x in (0.44, 0.48, 0.53, 0.57)) + (
    (0.40, 0.38),
    (0.50, 0.37),
    (0.60, 0.38),
    (0.08, 0.42),
    (0.92, 0.42),
    (0.40, 0.25),
    (0.62, 0.25),
)


class FakeSource:
    def __init__(self, family: str = "scrimmage", frames: int = 4, uniforms: bool = True) -> None:
        self.family = family
        self.frames = frames
        self.has_uniforms = uniforms
        self.saved: dict[tuple[int, int], dict[str, object]] = {}

    def play(self, project: int, play: int) -> PlayInfo:
        return PlayInfo(project, play, "Friday night", self.family, None, 2.0, 8.0)

    def uniforms(self, project: int):
        return (HOME, AWAY) if self.has_uniforms else None

    def snap_frames(self, project: int, play: int) -> list[bytes]:
        return [JPEG] * self.frames

    def save(self, project: int, play: int, diagram: dict[str, object]) -> None:
        self.saved[(project, play)] = diagram


def _isolate(frames, mine, other) -> PlayIsolation:
    return PlayIsolation(
        tuple(IsolatedPlayer(x, y) for x, y in OFFENSE),
        tuple(IsolatedPlayer(x, y) for x, y in DEFENSE),
        "offense",
        "high",
        frames[0],
    )


def _pipeline(source: FakeSource, root: Path | None = None) -> DiagramPipeline:
    return DiagramPipeline(source, RunStore(root), isolate=_isolate, mark=lambda jpeg, a, b: jpeg)


def test_a_scrimmage_play_is_traced_and_saved(tmp_path: Path) -> None:
    source = FakeSource()
    pipeline = _pipeline(source, tmp_path)
    run = pipeline.store.create("1-1", 1, 1)

    pipeline.run(run, worker="test")

    assert run.status == "done", run.error
    assert [step.name for step in run.steps] == [
        "load",
        "classify",
        "frames",
        "players",
        "plot",
        "save",
    ]
    assert all(step.ms is not None for step in run.steps)
    assert source.saved[(1, 1)] is run.diagram
    assert len(run.diagram["offense"]) == 11
    assert len(run.diagram["defense"]) == 11
    labels = sorted(p["label"] for p in run.diagram["offense"])
    assert labels == sorted(["LT", "LG", "C", "RG", "RT", "Y", "X", "Z", "H", "QB", "RB"])
    players = next(step for step in run.steps if step.name == "players")
    assert players.data["home_players"] == 11
    assert any(name.endswith("players-marked.jpg") for name in players.artifacts)
    # The trace and its images survive a restart.
    again = RunStore(tmp_path)
    loaded = again.get(run.run_id)
    assert loaded is not None and loaded.status == "done"
    assert again.image(run.run_id, players.artifacts[0]) == JPEG


def test_special_teams_is_skipped_at_classify() -> None:
    source = FakeSource(family="special_teams")
    pipeline = _pipeline(source)
    run = pipeline.store.create("1-2", 1, 2)

    pipeline.run(run)

    assert run.status == "skipped"
    assert run.steps[-1].name == "classify"
    assert run.steps[-1].status == "skipped"
    assert source.saved == {}


def test_missing_uniforms_fail_at_load_with_a_reason() -> None:
    pipeline = _pipeline(FakeSource(uniforms=False))
    run = pipeline.store.create("1-1", 1, 1)

    pipeline.run(run)

    assert run.status == "failed"
    assert run.steps[0].status == "failed"
    assert "uniforms" in run.detail


def test_short_frames_warn_but_finish() -> None:
    pipeline = _pipeline(FakeSource(frames=2))
    run = pipeline.store.create("1-1", 1, 1)

    pipeline.run(run)

    frames = next(step for step in run.steps if step.name == "frames")
    assert run.status == "done"
    assert frames.status == "warn"
    assert pipeline.store.stats()["runs"]["done"] == 1


def test_a_crash_inside_a_step_is_kept_with_its_traceback() -> None:
    def broken(frames, mine, other):
        raise RuntimeError("detector exploded")

    pipeline = DiagramPipeline(FakeSource(), RunStore(None), isolate=broken)
    run = pipeline.store.create("1-1", 1, 1)

    pipeline.run(run)

    assert run.status == "failed"
    assert "detector exploded" in run.detail
    assert "Traceback" in run.error
    assert run.steps[-1].name == "players"


def test_workers_do_not_queue_the_same_play_twice() -> None:
    workers = DiagramWorkers(_pipeline(FakeSource()), workers=1)
    first = workers.submit([(1, 1), (1, 1), (1, 2)])
    again = workers.submit([(1, 1), (1, 2)])
    assert [run.uid for run in first] == ["1-1", "1-2"]
    assert {run.run_id for run in again} <= {run.run_id for run in first} | {
        run.run_id for run in workers.store.recent()
    }
    workers.shutdown()


def test_line_names_run_out_from_the_center() -> None:
    diagram = {
        "offense": [
            {"x": 0.0, "y": 0.0, "role": "center", "symbol": "square", "label": ""},
            {"x": -1.2, "y": 0.0, "role": "line", "symbol": "open", "label": ""},
            {"x": -2.4, "y": 0.0, "role": "line", "symbol": "open", "label": ""},
            {"x": 1.2, "y": 0.0, "role": "line", "symbol": "open", "label": ""},
            {"x": 2.4, "y": 0.0, "role": "line", "symbol": "open", "label": ""},
            {"x": 0.0, "y": -5.0, "role": "back", "symbol": "filled", "label": ""},
            {"x": 0.0, "y": -7.5, "role": "back", "symbol": "filled", "label": ""},
        ]
    }

    position_labels(diagram)

    assert [p["label"] for p in diagram["offense"]] == ["C", "LG", "LT", "RG", "RT", "QB", "RB"]


def test_home_is_the_white_jersey_whichever_side_it_came_from() -> None:
    source = FakeSource()
    source.uniforms = lambda project: (AWAY, HOME)
    pipeline = _pipeline(source)
    run = pipeline.store.create("1-1", 1, 1)

    pipeline.run(run)

    load = run.steps[0]
    assert load.data["home"]["hex"] == "#ffffff"
    assert load.data["away"]["hex"] == "#aa2222"
