"""The /generate-diagram-test page queues plays to the diagram workers and shows each trace."""

import time
from pathlib import Path

from fastapi.testclient import TestClient

from film_analysis.chart.isolate import IsolatedPlayer, PlayIsolation
from film_analysis.storage.catalog import FilmTeams, MemoryCatalog
from film_analysis.web.app import create_app


def _film(tmp_path: Path) -> MemoryCatalog:
    store = MemoryCatalog(tmp_path)
    project_id = store.create_project("Friday night", None)
    store.set_progress(
        project_id, status="ready", duration_seconds=30, scanned_seconds=30, error=None
    )
    clip = tmp_path / "play.mp4"
    clip.write_bytes(b"clip")
    store.add_play(project_id, 1, 2.0, 8.0, clip, "scrimmage")
    store.add_play(project_id, 2, 10.0, 16.0, clip, "special_teams")
    store.set_film_teams(project_id, FilmTeams("Home", "Away", None, ""))
    return store


def _isolate(frames, mine, other) -> PlayIsolation:
    return PlayIsolation(
        tuple(IsolatedPlayer(x, 0.5) for x in (0.42, 0.46, 0.5, 0.54, 0.58))
        + (IsolatedPlayer(0.5, 0.6),),
        tuple(IsolatedPlayer(x, 0.45) for x in (0.45, 0.5, 0.55)),
        "offense",
        "high",
        frames[0],
    )


def _wait(client: TestClient, run_id: str) -> dict:
    for _ in range(200):
        run = client.get(f"/api/diagram-test/runs/{run_id}").json()
        if run["status"] not in {"queued", "running"}:
            return run
        time.sleep(0.02)
    raise AssertionError("run never finished")


def test_page_lists_scrimmage_plays_with_worker_stats(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FILM_DIAGRAM_RUNS", str(tmp_path / "runs"))
    client = TestClient(create_app(store=_film(tmp_path)))

    page = client.get("/generate-diagram-test")
    plays = client.get("/api/diagram-test/plays").json()

    assert page.status_code == 200
    assert "/api/diagram-test/runs" in page.text
    assert [play["uid"] for play in plays["plays"]] == ["1-1"]
    assert plays["plays"][0]["run"] is None
    assert plays["workers"]["workers"] >= 1


def test_a_queued_play_runs_and_exposes_trace_images_and_diagram(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FILM_DIAGRAM_RUNS", str(tmp_path / "runs"))
    monkeypatch.setattr(
        "film_analysis.web.app._scrimmage_frames", lambda *a, **k: [b"\xff\xd8\xff" + b"0" * 16] * 4
    )
    monkeypatch.setattr("film_analysis.diagram.pipeline.isolate_play", _isolate)
    monkeypatch.setattr("film_analysis.diagram.pipeline.mark_both_teams", lambda jpeg, a, b: None)
    monkeypatch.setattr("film_analysis.diagram.pipeline.loose_split", lambda *a: PlayIsolation())
    monkeypatch.setattr("film_analysis.diagram.pipeline.model_points", lambda *a: None)
    store = _film(tmp_path)
    client = TestClient(create_app(store=store))

    started = client.post("/api/diagram-test/runs", json={"uids": ["1-1", "1-1", "nope"]})
    assert started.status_code == 200
    runs = started.json()["runs"]
    assert len(runs) == 1

    run = _wait(client, runs[0]["run_id"])

    assert run["status"] == "done", run["error"]
    assert [step["name"] for step in run["steps"]] == [
        "load",
        "classify",
        "frames",
        "players",
        "model",
        "plot",
        "save",
    ]
    assert run["diagram"]["offense"]
    assert store.play_layout(1, 1) == run["diagram"]
    frames = next(step for step in run["steps"] if step["name"] == "frames")
    image = client.get(f"/api/diagram-test/runs/{run['run_id']}/images/{frames['artifacts'][0]}")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"
    listed = client.get("/api/diagram-test/plays").json()
    assert listed["plays"][0]["run"]["status"] == "done"
    assert listed["workers"]["runs"]["done"] == 1


def test_special_teams_play_is_skipped(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FILM_DIAGRAM_RUNS", str(tmp_path / "runs"))
    client = TestClient(create_app(store=_film(tmp_path)))

    run_id = client.post("/api/diagram-test/runs", json={"uids": ["1-2"]}).json()["runs"][0][
        "run_id"
    ]
    run = _wait(client, run_id)

    assert run["status"] == "skipped"
    assert run["steps"][-1]["name"] == "classify"


def test_an_unknown_run_is_404(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FILM_DIAGRAM_RUNS", str(tmp_path / "runs"))
    client = TestClient(create_app(store=_film(tmp_path)))

    assert client.get("/api/diagram-test/runs/missing").status_code == 404
    assert client.get("/api/diagram-test/runs/missing/images/..%2Frun.json").status_code == 404
    assert client.post("/api/diagram-test/runs", json={"uids": []}).status_code == 400
