"""The analyze table lists scrimmage plays and shows type, players, then the plot."""

import time
from pathlib import Path

from fastapi.testclient import TestClient

from film_analysis.storage.catalog import FilmTeams, MemoryCatalog
from film_analysis.web.app import create_app


def _film(tmp_path: Path) -> MemoryCatalog:
    store = MemoryCatalog(tmp_path)
    project_id = store.create_project("Friday night", None)
    store.set_progress(project_id, status="ready", duration_seconds=30, scanned_seconds=30, error=None)
    source = tmp_path / "film.mp4"
    source.write_bytes(b"video")
    store.save_source(project_id, source)
    clip = tmp_path / "play.mp4"
    clip.write_bytes(b"clip")
    store.add_play(project_id, 1, 2.0, 8.0, clip, "scrimmage")
    store.add_play(project_id, 2, 10.0, 16.0, clip, "special_teams")
    store.set_film_teams(project_id, FilmTeams("Home", "Away", None, ""))
    return store


def _diagram(frames, left, right, on_step=None):
    if on_step is not None:
        for step in ("type", "players", "plot"):
            on_step(step, 100)
    return {
        "field": {
            "ball": 50,
            "attack": "goal",
            "length": 120,
            "width": 53.3,
            "offense_color": "#f4f4f4",
            "defense_color": "#aa2222",
        },
        "teams": {"offense": "Home", "defense": "Away"},
        "call": "",
        "formation": "",
        "offense": [
            {"x": 0, "y": -5, "role": "quarterback", "symbol": "filled", "label": "", "seen": True}
        ],
        "defense": [
            {"x": 1, "y": 2, "role": "line", "symbol": "open", "label": "1", "seen": True}
        ],
        "title": "",
        "source": "film",
    }


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
    assert "location.href" not in page.text
    assert rows.status_code == 200
    assert [row["play"] for row in rows.json()] == [1]
    assert rows.json()[0]["type"] == 100
    assert rows.json()[0]["players"] == 0
    assert rows.json()[0]["plot"] == 0


def test_running_a_play_saves_the_plotted_players(tmp_path: Path, monkeypatch) -> None:
    store = _film(tmp_path)
    monkeypatch.setattr("film_analysis.web.app.read_scrimmage", _diagram)
    client = TestClient(create_app(store=store))

    started = client.post("/api/plays/1-1/diagram")
    assert started.status_code == 200

    deadline = time.monotonic() + 2
    view = client.get("/api/plays/1-1/diagram").json()
    while time.monotonic() < deadline and view.get("running"):
        time.sleep(0.02)
        view = client.get("/api/plays/1-1/diagram").json()

    diagram = view["diagram"]
    assert diagram["offense"][0]["y"] == -5
    assert diagram["defense"][0]["x"] == 1
    assert diagram["field"]["ball"] == 50
    assert view["steps"]["type"] == 100
    assert view["steps"]["players"] == 100
    assert view["steps"]["plot"] == 100
    saved = store.get_project(1)
    play = next(item for item in saved.plays if item.play_number == 1)
    assert play.play_type is None
    assert play.formation is None
    assert play.unit is None
    assert client.post("/api/plays/1-2/diagram").status_code == 400
