"""Scouting report PDFs from offense and defense tags."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from film_analysis.scout.report import scouting_blocks, untagged_plays
from film_analysis.storage.catalog import FilmTeams, MemoryCatalog, PlayRecord
from film_analysis.web.app import create_app


def test_a_discarded_play_is_left_out_of_the_report() -> None:
    plays = (
        PlayRecord(1, 10, 16, "scrimmage", "run", unit="offense"),
        PlayRecord(2, 20, 28, "scrimmage", "pass", unit="defense", discarded=True),
        PlayRecord(3, 40, 47, "scrimmage", "run", unit="defense"),
    )
    teams = FilmTeams("North", "South", "left")

    text = "\n".join(line for _, line in scouting_blocks("Friday night", teams, plays))

    assert "Read from 2 plays: 1 South defense, 1 South offense." in text
    assert "Play 2" not in text
    assert "1 pass" not in text
    assert "Play 1  0:10-0:16  South defense  Run  6s" in text
    assert "Play 3  0:40-0:47  South offense  Run  7s" in text


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
    assert "Run\t1" in text
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
    assert body["sections"] == ["calls", "formations", "plays", "special", "tendencies"]
    assert body["blocks"][0]["text"] == "Scouting report: Friday night"


def test_generating_the_report_charts_run_and_pass(tmp_path: Path) -> None:
    from film_analysis.teams.uniforms import Uniform

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
        "formation",
    ]
    assert rows[1] == "Friday night,South,North,1,2,8,6,Run,scrimmage,North defense,"
    assert rows[2].endswith(",scrimmage,,")


def test_the_live_report_fills_a_row_as_each_play_is_read(tmp_path: Path) -> None:
    import json

    from film_analysis.chart.call import ChartedPlay

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
    assert lines[0]["analysis"][0]["play"] == 1
    assert lines[0]["analysis"][0]["start"] == 2.0
    assert lines[0]["analysis"][0]["end"] == 8.0
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


def test_formation_waits_until_the_call_is_known(tmp_path: Path) -> None:
    store = _ready_film(tmp_path)
    store.set_play_unit(1, 2, "defense")
    store.set_film_teams(1, FilmTeams("Home", "Away", "left"))
    called: list[str] = []

    def chart(frames: list[bytes], mine: object, other: object, kind: str) -> None:
        del frames, mine, other, kind
        return None

    def name(frames: list[bytes], mine: object, other: object, call: str = "") -> str:
        del frames, mine, other
        called.append(call)
        return "shotgun"

    client = TestClient(create_app(store=store, chart=chart, formations=name))
    created = client.post("/api/projects/1/report", json={"sections": ["formations"]})

    assert created.status_code == 200
    assert called == []
    assert store.get_project(1).plays[1].formation is None


def test_charted_tags_are_stored_on_the_play(tmp_path: Path) -> None:
    from film_analysis.chart.call import ChartedPlay

    store = _ready_film(tmp_path)
    store.set_film_teams(1, FilmTeams("Home", "Away", "left"))

    def chart(frames: list[bytes], mine: object, other: object, kind: str) -> ChartedPlay:
        del frames, mine, other, kind
        return ChartedPlay("pass", "defense")

    client = TestClient(create_app(store=store, chart=chart))
    created = client.post("/api/projects/1/report", json={"sections": ["calls"]})

    assert created.status_code == 200
    play = store.get_project(1).plays[1]
    assert play.play_type == "pass"
    assert play.unit == "defense"
    detail = client.get("/api/plays/1-2")
    assert detail.status_code == 200
    body = detail.json()
    assert body["play_type"] == "pass"
    assert body["unit"] == "defense"
    analysis = client.get("/api/projects/1/report/analysis?sections=calls")
    assert analysis.status_code == 200
    rows = analysis.json()["analysis"]
    tagged = next(row for row in rows if row["play"] == 2)
    assert tagged["uid"] == "1-2"
    assert tagged["call"] == "pass"
    assert tagged["side"] == "defense"


def test_formation_setting_runs_during_call_charting(tmp_path: Path) -> None:
    from film_analysis.chart.call import ChartedPlay
    from film_analysis.chart.formation import FormationRead

    store = _ready_film(tmp_path)
    store.set_play_unit(1, 2, "defense")
    store.set_film_teams(1, FilmTeams("Home", "Away", "left"))
    plot = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"

    def chart(frames: list[bytes], mine: object, other: object, kind: str) -> ChartedPlay:
        del frames, mine, other, kind
        return ChartedPlay("pass", "offense")

    def name(frames: list[bytes], mine: object, other: object, call: str = "") -> FormationRead:
        del frames, mine, other
        return FormationRead("trips", plot=plot)

    client = TestClient(create_app(store=store, chart=chart, formations=name))
    created = client.post("/api/projects/1/report", json={"sections": ["calls"]})

    assert created.status_code == 200
    assert store.formation_reads() is True
    assert store.get_project(1).plays[1].formation == "trips"
    image = client.get("/projects/1/plays/2/formation.jpg")
    assert image.status_code == 200
    detail = client.get("/api/plays/1-2")
    assert detail.status_code == 200
    assert detail.json()["formation_plot"] == "/projects/1/plays/2/formation.jpg"
    assert detail.json()["steps"] == {
        "play": "detected",
        "side": "detected",
        "formation": "detected",
        "reviewed": "reviewed",
    }


def test_report_backfills_a_missing_formation_plot(tmp_path: Path) -> None:
    from film_analysis.chart.call import ChartedPlay
    from film_analysis.chart.formation import FormationRead

    store = _ready_film(tmp_path)
    store.set_play_type(1, 2, "pass", "scrimmage")
    store.set_play_unit(1, 2, "defense")
    store.set_play_formation(1, 2, "shotgun")
    store.set_film_teams(1, FilmTeams("Home", "Away", "left"))
    plot = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
    called: list[str] = []

    def chart(frames: list[bytes], mine: object, other: object, kind: str) -> ChartedPlay:
        del frames, mine, other, kind
        return ChartedPlay("pass", "defense")

    def name(frames: list[bytes], mine: object, other: object, call: str = "") -> FormationRead:
        del frames, mine, other
        called.append(call)
        return FormationRead("shotgun", plot=plot)

    client = TestClient(create_app(store=store, chart=chart, formations=name))
    created = client.post("/api/projects/1/report", json={"sections": ["calls", "formations"]})

    assert created.status_code == 200
    assert "pass" in called
    detail = client.get("/api/plays/1-2")
    assert detail.status_code == 200
    assert detail.json()["formation"] == "Shotgun"
    assert detail.json()["formation_plot"] == "/projects/1/plays/2/formation.jpg"
    assert client.get("/projects/1/plays/2/formation.jpg").status_code == 200


def test_a_person_can_override_the_formation(tmp_path: Path) -> None:
    from film_analysis.chart.formation import FormationRead

    store = _ready_film(tmp_path)
    store.set_play_type(1, 2, "pass", "scrimmage")
    store.set_play_unit(1, 2, "defense")
    store.set_film_teams(1, FilmTeams("Home", "Away", "left"))
    plot = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"

    def name(frames: list[bytes], mine: object, other: object, call: str = "") -> FormationRead:
        del frames, mine, other, call
        return FormationRead("shotgun", plot=plot)

    client = TestClient(create_app(store=store, formations=name))
    detected = client.post("/api/plays/1-2/formation")
    assert detected.status_code == 200
    assert detected.json()["formation"] == "Shotgun"
    assert detected.json()["formation_key"] == "shotgun"
    assert detected.json()["formation_plot"]

    changed = client.post("/api/plays/1-2", json={"formation": "trips"})
    assert changed.status_code == 200
    assert changed.json()["formation"] == "Trips"
    assert changed.json()["formation_key"] == "trips"
    assert store.get_project(1).plays[1].formation == "trips"

    cleared = client.post("/api/plays/1-2", json={"clear_formation": True})
    assert cleared.status_code == 200
    assert cleared.json()["formation"] is None
    assert store.get_project(1).plays[1].formation is None
    still = client.get("/projects/1/plays/2/formation.jpg")
    assert still.status_code == 200


def test_formations_are_read_for_both_offenses(tmp_path: Path) -> None:
    from film_analysis.chart.call import ChartedPlay
    from film_analysis.chart.formation import FormationRead

    store = _ready_film(tmp_path)
    store.set_play_unit(1, 2, "defense")
    store.set_film_teams(1, FilmTeams("Home", "Away", "left"))
    seen: list[tuple[str, str]] = []
    plot = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"

    def chart(frames: list[bytes], mine: object, other: object, kind: str) -> ChartedPlay:
        del frames, mine, other, kind
        return ChartedPlay("pass", "offense")

    def name(frames: list[bytes], mine: object, other: object, call: str = "") -> FormationRead:
        del frames, other
        seen.append((getattr(mine, "jersey", ""), call))
        return FormationRead("shotgun", plot=plot)

    client = TestClient(create_app(store=store, chart=chart, formations=name))
    created = client.post("/api/projects/1/report", json={"sections": ["formations"]})

    assert created.status_code == 200
    assert sorted(seen) == [("dark blue", "pass"), ("white", "run")]
    blocks = created.json()["blocks"]
    text = "\n".join(block["text"] for block in blocks)
    assert "experimental" not in text.lower()
    headings = [block["text"] for block in blocks if block["kind"] == "heading"]
    assert headings == ["Home offense", "Home defense"]
    assert "Home had the ball on 1 snap: 1 run (100%) and 0 passes (0%)." in text
    assert "Home defended 1 snap and faced 0 runs (0%) and 1 pass (100%) from Away." in text
    grids = [block["text"] for block in blocks if block["kind"] == "clips"]
    assert "Home offense · Shotgun\t1" in grids
    assert "Home defense · Needs a look\t2" in grids
    assert "Away offense · Shotgun\t2" not in grids
    assert "Home defense · Shotgun\t2" not in grids
    plays = store.get_project(1).plays
    assert plays[0].formation == "shotgun"
    assert plays[0].unit == "offense"
    assert plays[1].formation == "shotgun"
    assert plays[1].unit == "defense"
    image = client.get("/projects/1/plays/2/formation.jpg")
    assert image.status_code == 200
    assert image.content.startswith(b"\xff\xd8\xff")
    detail = client.get("/api/plays/1-2")
    assert detail.status_code == 200
    assert detail.json()["formation"] == "Shotgun"
    assert detail.json()["formation_plot"] == "/projects/1/plays/2/formation.jpg"
    sheet = client.get("/api/projects/1/report.csv?sections=formations")
    rows = sheet.content.decode().splitlines()
    assert rows[0].split(",")[-1] == "formation"
    assert any("Shotgun" in row for row in rows)


def test_an_untagged_play_learns_your_side(tmp_path: Path) -> None:
    from film_analysis.chart.call import ChartedPlay

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
    import threading
    import time

    from film_analysis.chart.call import ChartedPlay

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
            for line in response.iter_lines():
                if line:
                    lines.append(line)
                    return

    worker = threading.Thread(target=follow)
    worker.start()
    assert started.wait(timeout=2)
    worker.join(timeout=3)
    assert lines
    assert not worker.is_alive()
    mid = client.get("/api/projects/1/report/analysis?sections=tendencies")
    assert mid.status_code == 200
    body = mid.json()
    assert body["running"] is True
    assert body["total"] == 2
    assert body["done"] < 2
    listed = client.get("/api/projects").json()
    assert listed[0]["phase"] == "charting"
    assert listed[0]["running"] is True
    assert listed[0]["report_total"] == 2
    detail = client.get("/api/projects/1").json()
    assert detail["phase"] == "charting"
    assert detail["report_total"] == 2
    gate.set()
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
    from film_analysis.chart.call import ChartedPlay
    from film_analysis.storage.catalog import ReportAnalysis

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


def test_a_missed_play_can_be_tried_again_without_losing_the_rest(tmp_path: Path) -> None:
    from film_analysis.chart.call import ChartedPlay

    store = _ready_film(tmp_path)
    store.set_film_teams(1, FilmTeams("North", "South", "left"))
    calls = {"n": 0}

    def chart(frames: list[bytes], mine: object, other: object, kind: str) -> ChartedPlay:
        del frames, mine, other, kind
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("read failed")
        return ChartedPlay(
            call="pass",
            side="defense",
            input_tokens=500,
            output_tokens=50,
            model="claude-sonnet-5-5",
        )

    client = TestClient(create_app(store=store, chart=chart, workers=1))
    first = client.post("/api/projects/1/report/live", json={"sections": ["tendencies"]})

    assert first.status_code == 200
    stopped = client.get("/api/projects/1/report/analysis?sections=tendencies").json()
    assert stopped["running"] is False
    assert stopped["kept"] == 1
    assert stopped["missed"] == 1
    assert stopped["can_resume"] is True
    assert {row["play"]: row["status"] for row in stopped["analysis"]} == {
        1: "missed",
        2: "ready",
    }

    again = client.post(
        "/api/projects/1/report/live",
        json={"sections": ["tendencies"], "plays": [1]},
    )

    assert again.status_code == 200
    finished = client.get("/api/projects/1/report/analysis?sections=tendencies").json()
    assert finished["kept"] == 2
    assert finished["missed"] == 0
    assert finished["can_resume"] is False
    assert calls["n"] == 3


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


def test_the_scouted_team_groups_its_own_lineups() -> None:
    from film_analysis.scout.report import scouting_csv, scouting_pdf, team_breakdowns

    def snap(number: int, unit: str, call: str, formation: str | None) -> PlayRecord:
        start = number * 10.0
        return PlayRecord(number, start, start + 6, "scrimmage", call, False, unit, formation)

    plays = [
        snap(1, "defense", "run", "i_form"),
        snap(2, "defense", "run", "i_form"),
        snap(3, "defense", "run", "i_form"),
        snap(4, "defense", "pass", "shotgun"),
        snap(5, "defense", "pass", None),
        snap(6, "offense", "pass", "shotgun"),
        snap(7, "offense", "run", None),
        PlayRecord(8, 80.0, 85.0, "special_teams", "punt", formation="shotgun"),
    ]
    teams = FilmTeams("Pike County", "Lamar County", "left")
    layouts = {
        1: _front("4-3-4"),
        2: _front("4-3-4"),
        3: _front("4-3-4"),
        4: _front("3-3-5"),
        6: _shotgun_two_wide(),
    }

    units = team_breakdowns(teams, plays, layouts)

    assert [(unit.team, unit.unit) for unit in units] == [
        ("Pike County", "offense"),
        ("Pike County", "defense"),
    ]
    assert [(group.formation, group.plays) for group in units[0].groups] == [
        ("Shotgun, 3 in the backfield, 2 wide", (6,)),
        ("Needs a look", (7,)),
    ]
    assert units[0].groups[0].plot == 6
    pike_defense = units[1]
    assert pike_defense.snaps == 5
    assert [(group.formation, group.plays) for group in pike_defense.groups] == [
        ("4 linemen, 3 linebackers, 4 deep", (1, 2, 3)),
        ("3 linemen, 3 linebackers, 5 deep", (4,)),
        ("Needs a look", (5,)),
    ]
    assert "I-formation" not in {group.formation for group in pike_defense.groups}
    blocks = scouting_blocks("Week 6", teams, plays, ["formations", "special"], layouts)
    text = "\n".join(text for _kind, text in blocks)
    assert [line for kind, line in blocks if kind == "heading"] == [
        "Special teams",
        "Pike County offense",
        "Pike County defense",
    ]
    assert "Most used: 4 linemen, 3 linebackers, 4 deep, 3 of 4 (75%)" in text
    assert "the offense ran 100% of the time" in text
    assert "Pike County defense · I-formation" not in text
    assert "1 punt" in text
    punt_rows = [line for kind, line in blocks if kind == "row" and line.startswith("Play 8")]
    assert punt_rows and "Shotgun" not in punt_rows[0]
    kinds = [kind for kind, _text in blocks]
    assert "plot" in kinds
    pdf = scouting_pdf("Week 6", teams, plays, ["formations"], layouts)
    assert pdf.startswith(b"%PDF")
    assert b"4 linemen, 3 linebackers, 4 deep: Play 1, Play 2, Play 3" in pdf
    assert b"I-formation: Play 1, Play 2, Play 3" not in pdf
    sheet = scouting_csv("Week 6", teams, plays, ["formations", "special"], layouts)
    rows = sheet.splitlines()
    assert "4 linemen, 3 linebackers, 4 deep" in rows[1]
    assert "I-formation" not in sheet
    assert rows[-1].endswith(",")


def _front(front: str) -> dict[str, object]:
    return {
        "features": {
            "backfield": "unknown",
            "backs": 0,
            "receivers": "unknown",
            "wideouts": 0,
            "tight_ends": 0,
            "defense_front": front,
            "sketch": "unknown",
            "quality": "low",
            "defense_quality": "high",
        }
    }


def _shotgun_two_wide() -> dict[str, object]:
    return {
        "features": {
            "backfield": "shotgun",
            "backs": 2,
            "receivers": "2x2",
            "wideouts": 2,
            "tight_ends": 0,
            "defense_front": "4-3-4",
            "sketch": "Shotgun, 3 in the backfield, 2 wide",
            "quality": "high",
            "defense_quality": "high",
        }
    }


def test_a_team_with_no_snaps_says_so() -> None:
    from film_analysis.scout.report import scouting_blocks

    plays = [PlayRecord(1, 0.0, 5.0, "scrimmage", "run", False, "offense", "shotgun")]
    blocks = scouting_blocks("Week 1", FilmTeams("Home", "Away", "left"), plays, ["formations"])
    text = "\n".join(text for _kind, text in blocks)

    assert "Home had the ball on 1 snap: 1 run (100%) and 0 passes (0%)." in text
    assert "Home has no defensive snaps charted in this film." in text
    assert "Away has no offensive snaps" not in text


def test_the_play_read_puts_its_best_player_on_each_team_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cv2
    import numpy as np

    from film_analysis.chart.call import ChartedPlay, PlayerPick
    from film_analysis.teams.players import crop_picked_player

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :] = (40, 140, 40)
    frame[300:450, 200:250] = (40, 30, 140)
    frame[280:430, 900:950] = (230, 230, 230)
    full = cv2.imencode(".jpg", frame)[1].tobytes()
    monkeypatch.setattr("film_analysis.web.app.full_frame", lambda video, moment: full)

    store = _ready_film(tmp_path)
    store.set_film_teams(1, FilmTeams("North", "South", "right"))
    best = (900 / 1280, 280 / 720, 950 / 1280, 430 / 720)
    weaker = (200 / 1280, 300 / 720, 250 / 1280, 450 / 720)
    answers = iter(
        [
            {"A": PlayerPick(2, best, 9), "B": PlayerPick(1, weaker, 4)},
            {"A": PlayerPick(3, weaker, 7), "B": PlayerPick(5, weaker, 8)},
        ]
    )

    def chart(frames: list[bytes], mine: object, other: object, kind: str) -> ChartedPlay:
        del frames, mine, other, kind
        return ChartedPlay(call="pass", side="offense", players=next(answers))

    client = TestClient(create_app(store=store, chart=chart, workers=1))
    response = client.post("/api/projects/1/report/live", json={"sections": ["tendencies"]})

    assert response.status_code == 200
    scouted = store.team_image(1, "right-player")
    opponent = store.team_image(1, "left-player")
    assert scouted is not None and opponent is not None
    assert scouted.read_bytes() == crop_picked_player(full, best)
    assert opponent.read_bytes() == crop_picked_player(full, weaker)
    assert store.team_image(1, "right-reference").read_bytes() == scouted.read_bytes()


def test_one_report_step_reads_only_the_calls_again(tmp_path: Path) -> None:
    from film_analysis.chart.call import ChartedPlay

    store = _ready_film(tmp_path)
    store.set_film_teams(1, FilmTeams("Home", "Away", "left"))
    charted: list[str] = []
    plotted: list[str] = []

    def chart(frames: list[bytes], mine: object, other: object, kind: str) -> ChartedPlay:
        del frames, mine, other
        charted.append(kind)
        return ChartedPlay("pass", "offense")

    def name(frames: list[bytes], mine: object, other: object, call: str = "") -> None:
        del frames, mine, other
        plotted.append(call)

    client = TestClient(create_app(store=store, chart=chart, formations=name))
    created = client.post(
        "/api/projects/1/report",
        json={"sections": ["calls"], "step": "call", "plays": [1]},
    )

    assert created.status_code == 200
    assert charted == ["scrimmage"]
    assert plotted == []
    play = store.get_project(1).plays[0]
    assert (play.play_type, play.unit) == ("pass", "offense")


def test_one_report_step_plots_a_formation_again(tmp_path: Path) -> None:
    from film_analysis.chart.call import ChartedPlay
    from film_analysis.chart.formation import FormationRead

    store = _ready_film(tmp_path)
    store.set_play_type(1, 2, "pass", "scrimmage")
    store.set_play_unit(1, 2, "defense")
    store.set_play_formation(1, 2, "shotgun")
    old = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
    store.save_play_image(1, 2, "formation", old)
    store.set_film_teams(1, FilmTeams("Home", "Away", "left"))
    new = old[:-2] + b"\x00\xff\xd9"
    charted: list[str] = []

    def chart(frames: list[bytes], mine: object, other: object, kind: str) -> ChartedPlay:
        del frames, mine, other
        charted.append(kind)
        return ChartedPlay("run", "offense")

    def name(frames: list[bytes], mine: object, other: object, call: str = "") -> FormationRead:
        del frames, mine, other, call
        return FormationRead("trips", plot=new)

    client = TestClient(create_app(store=store, chart=chart, formations=name))
    created = client.post(
        "/api/projects/1/report",
        json={"sections": ["formations"], "step": "formation", "plays": [2]},
    )

    assert created.status_code == 200
    assert charted == []
    assert store.get_project(1).plays[1].formation == "trips"
    assert client.get("/projects/1/plays/2/formation.jpg").content == new


def test_an_unknown_report_step_is_refused(tmp_path: Path) -> None:
    client = TestClient(create_app(store=_ready_film(tmp_path)))

    response = client.post("/api/projects/1/report/live", json={"step": "teams"})

    assert response.status_code == 422
