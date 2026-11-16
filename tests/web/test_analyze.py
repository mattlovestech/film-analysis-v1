"""The analyze table lists scrimmage plays and shows type, players, then the plot."""

import base64
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
    assert 'id="drop"' in page.text
    assert ">Place<" in page.text
    assert ">Save<" in page.text
    assert ">Points<" in page.text
    assert ">Prompt<" in page.text
    assert page.text.count("<details") == 2
    frame = client.get("/api/plays/1-1/diagram/frame")
    assert frame.status_code == 200
    assert frame.content.startswith(b"\xff\xd8\xff")
    locked = client.post(
        "/api/plays/1-1/diagram/example",
        json={"offense": offense, "defense": diagram["defense"]},
    )
    assert locked.status_code == 200
    assert locked.json()["saved"] is True
    example = store.diagram_example(1, 1)
    assert example is not None
    assert example.points["offense"][0]["x"] == 4.2
    image = store.play_image(1, 1, "diagram")
    assert image is not None
    assert image.read_bytes().startswith(b"\xff\xd8\xff")
    again = client.get("/api/plays/1-1/diagram").json()
    assert again["diagram"]["offense"][0]["x"] == 4.2
    saved = store.get_project(1)
    play = next(item for item in saved.plays if item.play_number == 1)
    assert play.play_type is None
    assert play.formation is None
    assert play.unit is None


def test_a_dropped_frame_is_placed_with_the_standard_prompt(tmp_path: Path, monkeypatch) -> None:
    from film_analysis.chart.play_diagram import _Exchange

    def exchange(frames, prompt, *, limit, tokens, plain_retry=False):
        assert frames[0].startswith(b"\xff\xd8\xff")
        assert "Team A wears" in prompt
        reply = (
            '{"offense_team":"B","offense":[{"x":0,"y":-0.5,"role":"center",'
            '"symbol":"square","seen":true}],"defense":[{"x":1,"y":2,"role":"line",'
            '"symbol":"open","seen":true}]}'
        )
        return _Exchange(
            "claude-sonnet-5-5",
            prompt,
            1,
            reply,
            True,
            stop="end_turn",
            served_model="claude-sonnet-5-5-20261001",
            input_tokens=1_000_000,
            output_tokens=100_000,
            ms=4200,
            max_tokens=8000,
            thinking=True,
        )

    monkeypatch.setattr("film_analysis.web.app._exchange", exchange)
    store = _film(tmp_path)
    client = TestClient(create_app(store=store))

    prompt = client.get("/api/plays/1-1/diagram/prompt")
    assert prompt.status_code == 200
    assert "Team A wears" in prompt.json()["prompt"]
    assert "JSON only" in prompt.json()["prompt"]

    placed = client.post(
        "/api/plays/1-1/diagram/pass",
        data={"prompt": prompt.json()["prompt"]},
        files={"image": ("frame.jpg", b"\xff\xd8\xff" + b"0" * 16, "image/jpeg")},
    )
    assert placed.status_code == 200
    body = placed.json()
    assert body["error"] == ""
    assert body["call"] == {
        "sent": True,
        "frames": 1,
        "model": "claude-sonnet-5-5-20261001",
        "requested_model": "claude-sonnet-5-5",
        "input_tokens": 1_000_000,
        "output_tokens": 100_000,
        "cost_usd": 3.0,
        "ms": 4200,
        "max_tokens": 8000,
        "thinking": True,
        "retried": False,
        "attempt_ms": [],
        "stop": "end_turn",
        "image_bytes": len(b"\xff\xd8\xff" + b"0" * 16),
        "prompt_chars": len(prompt.json()["prompt"]),
    }
    assert body["diagram"]["offense"][0]["symbol"] == "square"
    assert body["diagram"]["defense"][0]["y"] == 2.0
    saved = client.post(
        "/api/plays/1-1/diagram/example",
        json={
            "offense": body["diagram"]["offense"],
            "defense": body["diagram"]["defense"],
            "prompt": prompt.json()["prompt"],
            "image": base64.b64encode(b"\xff\xd8\xff" + b"0" * 16).decode(),
        },
    )
    assert saved.status_code == 200
    row = store.diagram_example(1, 1)
    assert row is not None
    assert row.prompt.startswith("These frames")
    assert row.points["offense"][0]["symbol"] == "square"
    page = client.get("/analyze/play/1-1")
    assert "Drop the formation frame here." in page.text
    assert 'id="prompt"' in page.text
    assert 'id="points"' in page.text
    assert 'id="reply"' in page.text
    assert "Waiting on the model" in page.text
    assert "tokens/s" in page.text


def test_the_diagram_page_places_any_frame_without_a_film(tmp_path: Path, monkeypatch) -> None:
    from film_analysis.chart.play_diagram import _Exchange

    def exchange(frames, prompt, *, limit, tokens, plain_retry=False):
        assert prompt == "Place everyone."
        reply = (
            '{"offense_team":"A","offense":[{"x":0,"y":-0.5,"role":"center",'
            '"symbol":"square","seen":true}],"defense":[{"x":1,"y":2,"role":"line",'
            '"symbol":"open","seen":true}]}'
        )
        return _Exchange("claude-sonnet-5-5", prompt, 1, reply, True, input_tokens=10)

    monkeypatch.setattr("film_analysis.web.app._exchange", exchange)
    store = MemoryCatalog(tmp_path)
    client = TestClient(create_app(store=store))

    page = client.get("/diagram")
    assert page.status_code == 200
    assert 'id="prompt"' in page.text
    prompt = client.get("/api/diagram/prompt")
    assert "JSON only" in prompt.json()["prompt"]

    placed = client.post(
        "/api/diagram/pass",
        data={"prompt": "Place everyone."},
        files={"image": ("frame.jpg", b"\xff\xd8\xff" + b"0" * 16, "image/jpeg")},
    )
    assert placed.status_code == 200
    body = placed.json()
    assert body["diagram"]["offense"][0]["symbol"] == "square"
    assert body["call"]["input_tokens"] == 10
    assert store.list_projects() == []

    refused = client.post(
        "/api/diagram/pass", files={"image": ("note.txt", b"not a picture", "text/plain")}
    )
    assert refused.status_code == 400
