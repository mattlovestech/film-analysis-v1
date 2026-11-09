"""Team names, lineup stills, and roster portraits."""

import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from film_analysis.storage.catalog import MemoryCatalog, PlayRecord
from film_analysis.web.app import create_app
from film_analysis.web.teams import (
    Uniform,
    colors_from_pixels,
    focus_player,
    lineup_moment,
    make_portrait,
    names_from_reply,
    player_prompt,
    split_lineup,
    suggest_team_names,
)


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


def test_player_prompt_asks_for_a_standing_player() -> None:
    prompt = player_prompt(Uniform("North", "dark blue", "white", "dark blue"))

    assert "standing full length" in prompt
    assert "exact clothes" in prompt
    assert "reference photo is a football player" in prompt
    assert "Madden-style" in prompt
    assert "00" in prompt
    assert "dark blue jersey" in prompt
    assert "North" not in prompt


def test_both_teams_share_one_prompt() -> None:
    left = player_prompt(Uniform("Kentucky", "white", "white", "white"))
    right = player_prompt(Uniform("LSU", "purple", "gold", "gold"))

    def shell(prompt: str) -> str:
        for color in ("white", "purple", "gold"):
            prompt = prompt.replace(color, "COLOR")
        return prompt

    assert shell(left) == shell(right)


def test_a_model_reply_becomes_two_uniforms() -> None:
    from film_analysis.web.teams import parse_uniforms

    reply = (
        '{"left":{"name":"Kentucky","jersey":"white","pants":"white","helmet":"white"},'
        '"right":{"name":"LSU","jersey":"purple with a logo","pants":"gold","helmet":"gold"}}'
    )

    left, right = parse_uniforms(reply, "Kentucky vs LSU")

    assert left == Uniform("Kentucky", "white", "white", "white")
    assert right.jersey == "purple"
    assert right.pants == "gold"
    assert "with a logo" not in player_prompt(right)
    assert "purple jersey" in player_prompt(right)


def test_a_uniform_keeps_hex_colors_and_style() -> None:
    from film_analysis.web.teams import parse_uniforms

    reply = (
        '{"left":{"name":"Pike County Pirates","jersey":"white","jersey_hex":"#f4f1ea",'
        '"pants":"black","pants_hex":"#1a1a1a","pants_style":"solid",'
        '"helmet":"white","helmet_hex":"#f7f4ee","helmet_style":"solid shell",'
        '"number":"black","number_hex":"#111111"},'
        '"right":{"name":"ACE Charter Gryphons","jersey":"red","jersey_hex":"#b42318",'
        '"pants":"white","pants_hex":"#f4f1ea","pants_style":"striped",'
        '"helmet":"red","helmet_hex":"#b42318","helmet_style":"center stripe",'
        '"number_hex":"#f4f1ea"}}'
    )

    left, right = parse_uniforms(reply, "Pike County vs ACE Charter")
    prompt = player_prompt(left)

    assert left.jersey_hex == "#f4f1ea"
    assert left.pants_style == "solid"
    assert left.helmet_style == "solid shell"
    assert left.number == "black"
    assert right.helmet_style == "center stripe"
    assert "#f4f1ea" in prompt
    assert "solid shell" in prompt
    assert "00" in prompt
    assert "Pike" not in prompt
    assert "do not write color codes" in prompt


def test_play_samples_are_spread_across_the_film() -> None:
    from film_analysis.web.teams import sample_plays

    plays = tuple(
        PlayRecord(number, float(number), float(number) + 1, "scrimmage") for number in range(1, 9)
    )

    picked = sample_plays(plays, 4)

    assert [play.play_number for play in picked] == [1, 3, 6, 8]


def test_portrait_falls_back_to_a_drawing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("film_analysis.web.teams._render", lambda prompt, still=None: None)

    data, suffix = make_portrait(Uniform("Home", "red", "white", "red"))

    assert suffix == ".svg"
    assert b"#b42318" in data
    assert b">00<" in data


def test_portrait_keeps_a_generated_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "film_analysis.web.teams._render",
        lambda prompt, still=None: b"\xff\xd8\xfffake",
    )

    data, suffix = make_portrait(Uniform("Away", "dark blue", "white", "dark blue"))

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


def test_the_uniform_prompt_keeps_the_json_example() -> None:
    from film_analysis.web.teams import _DETECT_PROMPT

    text = _DETECT_PROMPT.format(title="Pike County")

    assert "Pike County" in text
    assert '"left"' in text
    assert '"right"' in text
    assert "white jersey" in text
    assert "jersey_hex" in text


def test_a_still_without_a_photo_stays_whole() -> None:
    assert focus_player(b"not-a-photo", "left") == b"not-a-photo"


def test_a_matchup_title_names_both_sides() -> None:
    assert suggest_team_names("Pike County vs. Lamar") == ("Pike County", "Lamar")


def test_school_and_mascot_come_from_the_lineup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "film_analysis.web.teams._read_team_names",
        lambda title, left, right: ("Johns Creek Gladiators", "HF Wildcats"),
    )

    assert suggest_team_names("Friday night", b"left", b"right") == (
        "Johns Creek Gladiators",
        "HF Wildcats",
    )


def test_a_reply_keeps_each_schools_mascot() -> None:
    reply = '{"left": "Johns Creek Gladiators", "right": "HF Wildcats"}'

    assert names_from_reply(reply) == ("Johns Creek Gladiators", "HF Wildcats")


def test_a_lineup_without_a_readable_name_stays_home_and_away(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "film_analysis.web.teams._read_team_names",
        lambda title, left, right: None,
    )

    assert suggest_team_names("Friday night", b"left", b"right") == ("Home", "Away")


def test_naming_teams_saves_stills_and_portraits(tmp_path: Path) -> None:
    store = _ready_film(tmp_path)
    client = TestClient(
        create_app(
            store=store,
            lineup=lambda video, at: (b"left-jpeg", b"right-jpeg"),
            portrait=lambda uniform, still=b"": (f"<svg>{uniform.name}</svg>".encode(), ".svg"),
        )
    )

    unnamed = client.post("/api/projects/1/teams/stills")
    assert unnamed.status_code == 200
    teams = unnamed.json()["teams"]
    assert teams["left_name"] == "Home"
    assert teams["right_name"] == "Away"
    assert teams["mine"] is None
    assert client.get(teams["left_still"]).content == b"left-jpeg"
    assert client.get(teams["right_still"]).content == b"right-jpeg"

    missing = client.post(
        "/api/projects/1/teams",
        json={"left_name": "North", "right_name": "South", "mine": "sideline"},
    )
    assert missing.status_code == 422

    optional = client.post(
        "/api/projects/1/teams",
        json={"left_name": "", "right_name": ""},
    )
    assert optional.status_code == 200
    assert optional.json()["teams"]["left_name"] == "Home"
    assert optional.json()["teams"]["right_name"] == "Away"
    assert optional.json()["teams"]["mine"] is None

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
