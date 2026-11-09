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
    assert "Read from 4 plays: 2 run, 1 pass, 1 kickoff." in text
    assert "North ran 2 offensive snaps." in text
    assert "1 run (50%) and 1 pass (50%)" in text
    assert "North defended 1 snap." in text
    assert "1 kickoff" in text
    assert "Play 1  0:10-0:16  Run  6s  Offense" in text
    assert "Play 3  0:40-0:47  Run  7s  Defense" in text
    assert "Play 4  1:00-1:06  Kickoff  6s  Special teams" in text


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
    assert created.json() == {"project": 1, "plays": 2, "ready": True}


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
    assert b"Scrimmage" in report.content


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
    assert rows[1] == "Friday night,South,North,1,2,8,6,Run,scrimmage,offense"
    assert rows[2].endswith(",Scrimmage,scrimmage,")


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
