"""Team names, lineup stills, and roster portraits."""

import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from film_analysis.storage.catalog import MemoryCatalog, PlayRecord
from film_analysis.web.app import create_app
from film_analysis.web.teams import colors_from_pixels, lineup_moment, make_portrait, split_lineup


def test_lineup_moment_uses_the_first_scrimmage() -> None:
    plays = (
        PlayRecord(1, 10.0, 20.0, "special"),
        PlayRecord(2, 30.0, 40.0, "scrimmage"),
    )

    assert lineup_moment(plays) == 30.6


def test_colors_skip_grass_and_keep_two_uniform_colors() -> None:
    grass = (40, 140, 50)
    jersey = (20, 40, 180)
    pants = (210, 180, 40)
    pixels = [grass] * 20 + [jersey] * 8 + [pants] * 5

    assert colors_from_pixels(pixels) == ("#0020a0", "#c0a020")


def test_colors_fall_back_when_the_frame_is_only_grass() -> None:
    assert colors_from_pixels([(40, 140, 50)] * 10) == ("#1c3f73", "#f2efe6")


def test_portrait_falls_back_to_a_drawing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "film_analysis.web.teams._flux",
        lambda team_name, jersey, pants: None,
    )
    monkeypatch.setattr(
        "film_analysis.web.teams.dominant_colors",
        lambda still: ("#aa0000", "#eeeeee"),
    )

    data, suffix = make_portrait(b"still", "Home")

    assert suffix == ".svg"
    assert b"#aa0000" in data
    assert b">00<" in data


def test_portrait_keeps_a_generated_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "film_analysis.web.teams._flux",
        lambda team_name, jersey, pants: b"\xff\xd8\xfffake",
    )
    monkeypatch.setattr(
        "film_analysis.web.teams.dominant_colors",
        lambda still: ("#003366", "#f4f1ea"),
    )

    data, suffix = make_portrait(b"still", "Away")

    assert suffix == ".jpg"
    assert data.startswith(b"\xff\xd8\xff")


def test_split_lineup_crops_each_side(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    video = tmp_path / "film.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x32:d=1",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=64x32:d=1",
            "-filter_complex",
            "hstack",
            str(video),
        ],
        check=True,
    )

    left, right = split_lineup(video, 0.2)

    assert colors_from_pixels(_pixels(left))[0].startswith("#0000")
    assert colors_from_pixels(_pixels(right))[0].startswith("#e000") or colors_from_pixels(
        _pixels(right)
    )[0].startswith("#c000")


def test_naming_teams_saves_stills_and_portraits(tmp_path: Path) -> None:
    store = _ready_film(tmp_path)
    client = TestClient(
        create_app(
            store=store,
            lineup=lambda video, at: (b"left-jpeg", b"right-jpeg"),
            portrait=lambda still, name: (f"<svg>{name}</svg>".encode(), ".svg"),
        )
    )

    unnamed = client.post("/api/projects/1/teams/stills")
    assert unnamed.status_code == 200
    teams = unnamed.json()["teams"]
    assert teams["left_name"] == ""
    assert teams["mine"] is None
    assert client.get(teams["left_still"]).content == b"left-jpeg"
    assert client.get(teams["right_still"]).content == b"right-jpeg"

    missing = client.post(
        "/api/projects/1/teams",
        json={"left_name": "North", "right_name": "South", "mine": "sideline"},
    )
    assert missing.status_code == 422

    saved = client.post(
        "/api/projects/1/teams",
        json={"left_name": "  North  ", "right_name": "South", "mine": "left"},
    )
    assert saved.status_code == 200
    named = saved.json()["teams"]
    assert named["left_name"] == "North"
    assert named["right_name"] == "South"
    assert named["mine"] == "left"
    assert b"North" in client.get(named["left_player"]).content
    assert b"South" in client.get(named["right_player"]).content
    assert client.get("/projects/1/teams/neither").status_code == 404


def test_teams_wait_until_plays_are_identified(tmp_path: Path) -> None:
    store = _ready_film(tmp_path)
    store.set_progress(1, status="scanning", duration_seconds=30, scanned_seconds=4, error=None)
    client = TestClient(create_app(store=store, lineup=lambda video, at: (b"a", b"b")))

    response = client.post("/api/projects/1/teams/stills")

    assert response.status_code == 400
    assert response.json()["detail"] == "Name the teams after the plays are identified."


def _ready_film(tmp_path: Path) -> MemoryCatalog:
    store = MemoryCatalog(tmp_path)
    project_id = store.create_project("Friday night", None)
    store.set_progress(
        project_id,
        status="ready",
        duration_seconds=30,
        scanned_seconds=30,
        error=None,
    )
    source = tmp_path / "film.mp4"
    source.write_bytes(b"video")
    store.save_source(project_id, source)
    clip = tmp_path / "play.mp4"
    clip.write_bytes(b"clip")
    store.add_play(project_id, 1, 2.0, 8.0, clip, "scrimmage")
    return store


def _pixels(still: bytes) -> list[tuple[int, int, int]]:
    from film_analysis.web.teams import _sample_pixels

    return _sample_pixels(still)
