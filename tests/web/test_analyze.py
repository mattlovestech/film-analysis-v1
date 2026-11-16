"""The analyze table lists scrimmage plays and shows type, players, then the plot."""

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
    source = tmp_path / "film.mp4"
    source.write_bytes(b"video")
    store.save_source(project_id, source)
    clip = tmp_path / "play.mp4"
    clip.write_bytes(b"clip")
    store.add_play(project_id, 1, 2.0, 8.0, clip, "scrimmage")
    store.add_play(project_id, 2, 10.0, 16.0, clip, "special_teams")
    store.set_film_teams(project_id, FilmTeams("Home", "Away", None, ""))
    return store


def _isolate(frames, mine, other) -> PlayIsolation:
    return PlayIsolation(
        tuple(IsolatedPlayer(0.42 + i * 0.02, 0.56) for i in range(8)),
        tuple(IsolatedPlayer(0.42 + i * 0.02, 0.42) for i in range(7)),
        "offense",
        "high",
        frames[0],
    )


def test_analyze_lists_only_scrimmage_plays(tmp_path: Path) -> None:
    client = TestClient(create_app(store=_film(tmp_path)))

    page = client.get("/analyze")
    rows = client.get("/api/analyze")

    assert page.status_code == 200
    assert ">Type<" in page.text
    assert ">Players<" in page.text
    assert ">Plot<" in page.text
    assert "plays/${parts[1]}" in page.text
    assert 'createElement("video")' in page.text
    assert 'createElement("canvas")' in page.text
    assert '["Step", "Status", "AI", "In", "Out", "Time"]' in page.text
    assert "Only the Model step calls an AI model." in page.text
    assert "This step does not call a model." in page.text
    assert "The model was not called." in page.text
    assert 'location.assign(`/analyze/play/${encodeURIComponent(uid)}`)' in page.text
    assert "setInterval(load" not in page.text
    assert "location.href" not in page.text
    assert rows.status_code == 200
    assert [row["play"] for row in rows.json()] == [1]
    assert rows.json()[0]["type"] == 100
    assert rows.json()[0]["players"] == 0
    assert rows.json()[0]["plot"] == 0


def test_running_a_play_saves_the_plotted_players(tmp_path: Path, monkeypatch) -> None:
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

    quiet = client.get("/api/plays/1-1/diagram")
    assert quiet.status_code == 200
    assert quiet.json()["run"] is None
    assert quiet.json()["running"] is False

    started = client.post("/api/plays/1-1/diagram")
    assert started.status_code == 200

    deadline = time.monotonic() + 2
    view = client.get("/api/plays/1-1/diagram").json()
    while time.monotonic() < deadline and view.get("running"):
        time.sleep(0.02)
        view = client.get("/api/plays/1-1/diagram").json()

    diagram = view["diagram"]
    assert diagram["offense"]
    assert diagram["defense"]
    assert diagram["field"]["ball"] == 50
    assert view["steps"]["type"] == 100
    assert view["steps"]["players"] == 100
    assert view["steps"]["plot"] == 100
    assert view["run"]["status"] == "done"
    assert view["workers"]["workers"] >= 1
    incoming = {step["name"]: step["incoming"] for step in view["run"]["steps"]}
    assert incoming["load"] == {"project": 1, "play": 1}
    assert incoming["frames"]["start"] == 2.0
    assert incoming["players"]["frames"] == 4
    players = next(step for step in view["run"]["steps"] if step["name"] == "players")
    assert "home_players" in players["data"]
    assert incoming["save"]["offense"] >= 1
    model = next(step for step in view["run"]["steps"] if step["name"] == "model")
    assert model["call"]["sent"] is False
    saved = store.get_project(1)
    play = next(item for item in saved.plays if item.play_number == 1)
    assert play.play_type is None
    assert play.formation is None
    assert play.unit is None
    assert client.post("/api/plays/1-2/diagram").status_code == 400
    offense = [dict(player) for player in diagram["offense"]]
    offense[0]["x"] = 4.2
    offense[0]["y"] = -3.1
    moved = client.put(
        "/api/plays/1-1/diagram",
        json={"offense": offense, "defense": diagram["defense"]},
    )
    assert moved.status_code == 200
    assert moved.json()["diagram"]["offense"][0]["x"] == 4.2
    assert moved.json()["diagram"]["offense"][0]["y"] == -3.1
    assert moved.json()["diagram"]["field"]["ball"] == 50
    page = client.get("/analyze/play/1-1")
    assert page.status_code == 200
    assert "<canvas" in page.text
    assert "<details" in page.text
    assert ">Points<" in page.text
    again = client.get("/api/plays/1-1/diagram").json()
    assert again["diagram"]["offense"][0]["x"] == 4.2
    saved = store.get_project(1)
    play = next(item for item in saved.plays if item.play_number == 1)
    assert play.play_type is None
    assert play.formation is None
    assert play.unit is None
