"""Scouting report PDFs from offense and defense tags."""

from pathlib import Path

from fastapi.testclient import TestClient

from film_analysis.storage.catalog import FilmTeams, MemoryCatalog, PlayRecord
from film_analysis.web.app import create_app
from film_analysis.web.report import scouting_blocks, untagged_plays


def test_the_report_counts_offense_and_defense() -> None:
    plays = (
        PlayRecord(1, 10, 16, "scrimmage", "run", unit="offense"),
        PlayRecord(2, 20, 28, "scrimmage", "pass", unit="offense"),
        PlayRecord(3, 40, 47, "scrimmage", "run", unit="defense"),
        PlayRecord(4, 60, 66, "special_teams", "kickoff"),
    )
    teams = FilmTeams("North", "South", "left")

    text = "\n".join(line for _, line in scouting_blocks("Friday night", teams, plays))

    assert untagged_plays(plays) == 0
    assert "North against South." in text
    assert "Read from 4 plays: 2 South defense, 1 South offense, 1 special teams." in text
    assert "2 run, 1 pass, 1 kickoff." in text
    assert "Play tendency" in text
    assert "South on offense." in text
    assert "Run\t1\t100%" in text
    assert "Pass\t0\t0%" in text
    assert "Offense\t1\t" in text
    assert "South ran 1 offensive snap." in text
    assert "1 run (100%) and 0 passes (0%)" in text
    assert "South defended 2 snaps." in text
    assert "what your offense ran against them" in text
    assert "1 run (50%) and 1 pass (50%)" in text
    assert "1 kickoff" in text
    assert "Play 1  0:10-0:16  South defense  Run  6s" in text
    assert "Play 3  0:40-0:47  South offense  Run  7s" in text
    assert "Play 4  1:00-1:06  Special teams  Kickoff  6s" in text


def test_the_report_names_both_teams_when_yours_is_not_marked() -> None:
    plays = (PlayRecord(1, 10, 16, "scrimmage", "run", unit="offense"),)
    text = "\n".join(
        line for _, line in scouting_blocks("Friday night", FilmTeams("North", "South"), plays)
    )

    assert "North and South." in text
    assert "North against South." not in text
    assert "Offense ran 1 offensive snap." in text


def test_a_scrimmage_play_without_a_side_is_still_open() -> None:
    plays = (
        PlayRecord(1, 0, 5, "scrimmage", "run", unit="offense"),
        PlayRecord(2, 8, 14, "scrimmage", "pass"),
        PlayRecord(3, 20, 25, "special_teams", "punt"),
    )

    assert untagged_plays(plays) == 1


def test_generating_the_report_reads_each_play(tmp_path: Path) -> None:
    store = _ready_film(tmp_path)
    client = TestClient(create_app(store=store))

    waiting = client.post("/api/projects/1/report")
    assert waiting.status_code == 400
    assert "Label your team" in waiting.json()["detail"]

    store.set_film_teams(1, FilmTeams("North", "South", "left"))
    created = client.post("/api/projects/1/report")

    assert created.status_code == 200
    body = created.json()
    assert body["project"] == 1
    assert body["plays"] == 2
    assert body["ready"] is True
    assert body["tendencies"] == ""
    assert body["sections"] == ["calls", "plays", "special", "tendencies"]
    assert body["blocks"][0]["text"] == "Scouting report: Friday night"


def test_generating_the_report_charts_run_and_pass(tmp_path: Path) -> None:
    from film_analysis.web.teams import Uniform

    store = _ready_film(tmp_path)
    clip = tmp_path / "kick.mp4"
    clip.write_bytes(b"clip")
    store.add_play(1, 3, 30.0, 36.0, clip, "special_teams")
    store.set_play_unit(1, 1, "offense")
    store.set_play_unit(1, 2, "offense")
    store.set_film_teams(1, FilmTeams("Home", "Away", "left"))
    seen: list[str] = []

    def chart(frames: list[bytes], mine: Uniform, other: Uniform, kind: str) -> str:
        seen.append(kind)
        assert frames == []
        assert mine.jersey == "white"
        assert other.jersey == "dark blue"
        return "pass" if kind == "scrimmage" else "kick_return"

    client = TestClient(create_app(store=store, chart=chart))
    created = client.post("/api/projects/1/report")

    assert created.status_code == 200
    assert set(seen) == {"scrimmage", "special_teams"}
    assert "Away defense: 1 run (50%) and 1 pass (50%)." in created.json()["tendencies"]
    assert "Home special teams: 1 kick return." in created.json()["tendencies"]
    plays = store.get_project(1).plays
    assert plays[0].play_type == "run"
    assert plays[0].unit == "offense"
    assert plays[1].play_type == "pass"
    assert plays[1].unit == "offense"
    assert plays[2].play_type == "kick_return"
    assert plays[2].family == "special_teams"


def test_the_pdf_is_ready_after_the_team_is_labeled(tmp_path: Path) -> None:
    store = _ready_film(tmp_path)
    client = TestClient(create_app(store=store))

    waiting = client.get("/api/projects/1/report.pdf")
    assert waiting.status_code == 400
    assert "Label your team" in waiting.json()["detail"]

    store.set_film_teams(1, FilmTeams("North", "South", "left"))
    report = client.get("/api/projects/1/report.pdf")

    assert report.status_code == 200
    assert report.headers["content-type"] == "application/pdf"
    assert "scouting-report.pdf" in report.headers["content-disposition"]
    assert report.content.startswith(b"%PDF")
    assert b"North" in report.content
    assert b"Run" in report.content
    assert b"Not read" in report.content
    assert b"Scrimmage" not in report.content


def test_the_csv_lists_each_play(tmp_path: Path) -> None:
    store = _ready_film(tmp_path)
    store.set_film_teams(1, FilmTeams("North", "South", "right"))
    store.set_play_unit(1, 1, "offense")
    client = TestClient(create_app(store=store))

    waiting = client.get("/api/projects/1/report.csv")
    assert waiting.status_code == 200
    rows = waiting.content.decode().splitlines()

    assert waiting.headers["content-type"].startswith("text/csv")
    assert "scouting-report.csv" in waiting.headers["content-disposition"]
    assert rows[0].split(",") == [
        "film",
        "your_team",
        "opponent",
        "play",
        "start_seconds",
        "end_seconds",
        "seconds",
        "call",
        "family",
        "side",
    ]
    assert rows[1] == "Friday night,South,North,1,2,8,6,Run,scrimmage,North defense"
    assert rows[2].endswith(",scrimmage,")


def test_the_live_report_fills_a_row_as_each_play_is_read(tmp_path: Path) -> None:
    import json

    from film_analysis.web.chart import ChartedPlay

    store = _ready_film(tmp_path)
    store.set_film_teams(1, FilmTeams("North", "South", "left"))

    def chart(frames: list[bytes], mine: object, other: object, kind: str) -> ChartedPlay:
        del frames, mine, other, kind
        return ChartedPlay(
            call="pass",
            side="defense",
            input_tokens=1000,
            output_tokens=100,
            model="claude-sonnet-5-5",
        )

    client = TestClient(create_app(store=store, chart=chart, workers=1))
    response = client.post("/api/projects/1/report/live", json={"sections": ["tendencies"]})

    assert response.status_code == 200
    lines = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    progress = [line["done"] for line in lines if "done" in line]
    assert progress[0] == 0
    assert progress[-1] == 2
    assert 1 in progress
    assert lines[0]["workers"] == 1
    assert lines[0]["rows"][0]["reading"] == ["side"]
    assert lines[0]["rows"][1]["reading"] == ["side", "call"]
    assert len(lines[0]["analysis"]) == 2
    assert {row["status"] for row in lines[0]["analysis"]} == {"pending"}
    assert lines[-1]["ready"] is True
    finished = "\n".join(block["text"] for block in lines[-1]["blocks"])
    assert "South on offense." in finished
    assert "Run\t1\t50%" in finished
    assert "Pass\t1\t50%" in finished
    rows = store.list_report_analysis(1)
    assert len(rows) == 2
    assert {row.status for row in rows} == {"ready"}
    assert all(row.cost_usd == 0.003 for row in rows)
    assert all(row.model == "claude-sonnet-5-5" for row in rows)
    analysis = client.get("/api/projects/1/report/analysis?sections=tendencies")
    assert analysis.status_code == 200
    body = analysis.json()
    assert body["running"] is False
    assert body["done"] == 2
    assert body["total"] == 2
    assert body["workers"] == 1
    assert len(body["analysis"]) == 2


def test_the_report_is_on_the_page_before_a_pdf(tmp_path: Path) -> None:
    store = _ready_film(tmp_path)
    store.set_play_unit(1, 2, "defense")
    store.set_play_type(1, 2, "pass", "scrimmage")
    store.set_film_teams(1, FilmTeams("North", "South", "left"))
    seen: list[str] = []

    def chart(frames: list[bytes], mine: object, other: object, kind: str) -> str:
        del frames, mine, other
        seen.append(kind)
        return "run"

    client = TestClient(create_app(store=store, chart=chart))
    preview = client.get("/api/projects/1/report?sections=tendencies")

    assert preview.status_code == 200
    assert seen == []
    text = "\n".join(block["text"] for block in preview.json()["blocks"])
    assert "Play tendency" in text
    assert "South on offense." in text
    assert "Pass\t1\t100%" in text
    pdf = client.get("/api/projects/1/report.pdf?sections=tendencies")
    assert pdf.status_code == 200
    assert pdf.content.startswith(b"%PDF")


def test_special_teams_can_be_the_only_section(tmp_path: Path) -> None:
    store = _ready_film(tmp_path)
    clip = tmp_path / "kick.mp4"
    clip.write_bytes(b"clip")
    store.add_play(1, 3, 30.0, 36.0, clip, "special_teams")
    store.set_play_unit(1, 1, "offense")
    store.set_play_unit(1, 2, "defense")
    store.set_film_teams(1, FilmTeams("North", "South", "left"))
    seen: list[str] = []

    def chart(frames: list[bytes], mine: object, other: object, kind: str) -> str:
        del frames, mine, other
        seen.append(kind)
        return "kickoff"

    client = TestClient(create_app(store=store, chart=chart))
    created = client.post("/api/projects/1/report", json={"sections": ["special"]})

    assert created.status_code == 200
    assert seen == ["special_teams"]
    text = "\n".join(block["text"] for block in created.json()["blocks"])
    assert "Special teams" in text
    assert "Kickoff" in text
    assert "Offense" not in text
    assert "Tendencies" not in text
    sheet = client.get("/api/projects/1/report.csv?sections=special")
    rows = sheet.content.decode().splitlines()
    assert rows[1].endswith(",Kickoff,special_teams,Special teams")
    assert len(rows) == 2


def test_formations_are_read_for_the_opponents_offense(tmp_path: Path) -> None:
    from film_analysis.web.chart import ChartedPlay

    store = _ready_film(tmp_path)
    store.set_play_unit(1, 2, "defense")
    store.set_film_teams(1, FilmTeams("Home", "Away", "left"))
    seen: list[str] = []

    def chart(frames: list[bytes], mine: object, other: object, kind: str) -> ChartedPlay:
        del frames, mine, other, kind
        return ChartedPlay("pass", "offense")

    def name(frames: list[bytes], mine: object, other: object) -> str:
        del frames, other
        seen.append(getattr(mine, "jersey", ""))
        return "shotgun"

    client = TestClient(create_app(store=store, chart=chart, formations=name))
    created = client.post("/api/projects/1/report", json={"sections": ["formations"]})

    assert created.status_code == 200
    assert seen == ["dark blue"]
    text = "\n".join(block["text"] for block in created.json()["blocks"])
    assert "Formations" in text
    assert "Away offense: 1 Shotgun." in text
    assert "This read is experimental." in text
    plays = store.get_project(1).plays
    assert plays[0].formation is None
    assert plays[0].unit == "offense"
    assert plays[1].formation == "shotgun"
    assert plays[1].unit == "defense"
    sheet = client.get("/api/projects/1/report.csv?sections=formations")
    rows = sheet.content.decode().splitlines()
    assert rows[0].split(",")[-1] == "formation"
    assert any("Shotgun" in row for row in rows)


def test_an_untagged_play_learns_your_side(tmp_path: Path) -> None:
    from film_analysis.web.chart import ChartedPlay

    store = _ready_film(tmp_path)
    store.set_reviewed(1, 1, False)
    store.set_film_teams(1, FilmTeams("Home", "Away", "left"))

    def chart(frames: list[bytes], mine: object, other: object, kind: str) -> ChartedPlay:
        del frames, mine, other
        return ChartedPlay("pass" if kind == "scrimmage" else "punt", "defense")

    client = TestClient(create_app(store=store, chart=chart))
    created = client.post("/api/projects/1/report", json={"sections": ["calls"]})

    assert created.status_code == 200
    plays = store.get_project(1).plays
    assert plays[0].play_type == "run"
    assert plays[0].unit == "defense"
    assert plays[0].reviewed is False
    assert plays[1].play_type == "pass"
    assert plays[1].unit == "defense"


def test_special_teams_do_not_block_the_report(tmp_path: Path) -> None:
    store = _ready_film(tmp_path)
    clip = tmp_path / "kick.mp4"
    clip.write_bytes(b"clip")
    store.add_play(1, 3, 30.0, 36.0, clip, "special_teams")
    store.set_play_type(1, 3, "kickoff", "special_teams")
    store.set_film_teams(1, FilmTeams("North", "South", "left"))
    store.set_play_unit(1, 1, "offense")
    store.set_play_unit(1, 2, "defense")
    client = TestClient(create_app(store=store))

    report = client.get("/api/projects/1/report.pdf")

    assert report.status_code == 200
    assert b"Kickoff" in report.content


def test_report_progress_survives_leaving_the_stream(tmp_path: Path) -> None:
    import json
    import threading
    import time

    from film_analysis.web.chart import ChartedPlay

    store = _ready_film(tmp_path)
    store.set_film_teams(1, FilmTeams("North", "South", "left"))
    gate = threading.Event()
    started = threading.Event()
    lines: list[str] = []

    def chart(frames: list[bytes], mine: object, other: object, kind: str) -> ChartedPlay:
        del frames, mine, other, kind
        started.set()
        gate.wait(timeout=2)
        return ChartedPlay(
            call="pass",
            side="defense",
            input_tokens=1000,
            output_tokens=100,
            model="claude-sonnet-5-5",
        )

    client = TestClient(create_app(store=store, chart=chart, workers=1))

    def follow() -> None:
        with client.stream(
            "POST",
            "/api/projects/1/report/live",
            json={"sections": ["tendencies"]},
        ) as response:
            assert response.status_code == 200
            lines.extend(line for line in response.iter_lines() if line)

    worker = threading.Thread(target=follow)
    worker.start()
    assert started.wait(timeout=2)
    mid = client.get("/api/projects/1/report/analysis?sections=tendencies")
    assert mid.status_code == 200
    body = mid.json()
    assert body["running"] is True
    assert body["total"] == 2
    assert body["done"] < 2
    gate.set()
    worker.join(timeout=5)
    assert any(json.loads(line).get("ready") is True for line in lines)
    for _ in range(50):
        done = client.get("/api/projects/1/report/analysis?sections=tendencies")
        if done.json()["running"] is False and done.json()["done"] == 2:
            break
        time.sleep(0.05)
    finished = client.get("/api/projects/1/report/analysis?sections=tendencies")
    assert finished.json()["running"] is False
    assert finished.json()["done"] == 2
    assert finished.json()["can_resume"] is False


def test_unfinished_report_rows_can_be_retried(tmp_path: Path) -> None:
    from film_analysis.storage.catalog import ReportAnalysis
    from film_analysis.web.chart import ChartedPlay

    store = _ready_film(tmp_path)
    store.set_film_teams(1, FilmTeams("North", "South", "left"))
    store.set_report_analysis(1, ReportAnalysis(1, "ready", call="run", side="offense"))
    store.set_report_analysis(1, ReportAnalysis(2, "pending"))
    seen: list[int] = []

    def chart(frames: list[bytes], mine: object, other: object, kind: str) -> ChartedPlay:
        del frames, mine, other, kind
        seen.append(1)
        return ChartedPlay(
            call="pass",
            side="defense",
            input_tokens=500,
            output_tokens=50,
            model="claude-sonnet-5-5",
        )

    client = TestClient(create_app(store=store, chart=chart, workers=1))
    waiting = client.get("/api/projects/1/report/analysis?sections=tendencies")
    assert waiting.json()["can_resume"] is True
    response = client.post(
        "/api/projects/1/report/live",
        json={"sections": ["tendencies"], "restart": False},
    )

    assert response.status_code == 200
    assert len(seen) == 1
    rows = {row.play_number: row for row in store.list_report_analysis(1)}
    assert rows[1].status == "ready"
    assert rows[1].call == "run"
    assert rows[2].status == "ready"
    assert rows[2].call == "pass"
    assert client.get("/api/projects/1/report/analysis").json()["can_resume"] is False


def _ready_film(tmp_path: Path) -> MemoryCatalog:
    store = MemoryCatalog(tmp_path)
    project_id = store.create_project("Friday night", None)
    store.set_progress(
        project_id,
        status="ready",
        duration_seconds=40,
        scanned_seconds=40,
        error=None,
    )
    clip = tmp_path / "play.mp4"
    clip.write_bytes(b"clip")
    store.add_play(project_id, 1, 2.0, 8.0, clip, "scrimmage")
    store.add_play(project_id, 2, 12.0, 19.0, clip, "scrimmage")
    store.set_play_type(project_id, 1, "run", "scrimmage")
    return store
