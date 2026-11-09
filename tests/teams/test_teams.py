"""Team names and the players found in the film."""

import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from film_analysis.storage.catalog import FilmTeams, MemoryCatalog, PlayRecord
from film_analysis.teams.uniforms import (
    Uniform,
    colors_from_pixels,
    detect_uniforms,
    focus_player,
    lineup_moment,
    split_lineup,
    suggest_team_names,
    team_cards,
)
from film_analysis.web.app import create_app


def _usable_player_bytes(mark: int = 0) -> bytes:
    """A standing white player on grass that passes the card quality checks."""

    image = np.zeros((240, 100, 3), dtype=np.uint8)
    image[:, :] = (40, 140, 40)
    tone = 220 + (mark % 20)
    # Fill most of the frame so the crop is not mostly field.
    image[18:228, 18:82] = (tone, tone, tone)
    image[10:40, 32:68] = (max(0, tone - 20), max(0, tone - 20), max(0, tone - 20))
    image[8:14, 8:14] = (mark % 255, (mark * 3) % 255, (mark * 7) % 255)
    return cv2.imencode(".jpg", image)[1].tobytes()


def _stub_team_cards(
    monkeypatch: pytest.MonkeyPatch,
    left: bytes | None = None,
    right: bytes | None = None,
    factory=None,
) -> None:
    """Team cards come from film crops, not lineup stills."""

    def cards(*_args, **_kwargs):
        if factory is not None:
            return factory()
        return left, right

    monkeypatch.setattr("film_analysis.web.app.team_cards", cards)


def test_team_cards_prefer_a_player_in_each_jersey(monkeypatch: pytest.MonkeyPatch) -> None:
    def shot(_frames, uniform: Uniform, _other=None, avoid=()):
        return b"home" if uniform.jersey == "white" else b"away"

    monkeypatch.setattr("film_analysis.teams.uniforms.team_player_shot", shot)
    monkeypatch.setattr(
        "film_analysis.teams.uniforms.lone_players",
        lambda *_args, **_kwargs: (None, None),
    )
    left = Uniform("Home", "white", "white", "white", jersey_hex="#ffffff")
    right = Uniform("Away", "purple", "white", "purple", jersey_hex="#461d7c")

    assert team_cards([b"frame"], left, right) == (b"home", b"away")


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

    assert colors_from_pixels(pixels) == ("#1428b4", "#d2b428")


def test_a_white_jersey_is_kept() -> None:
    white = (245, 245, 245)
    navy = (20, 40, 180)

    assert colors_from_pixels([white] * 12 + [navy] * 6)[0] == "#f5f5f5"


def test_photo_colors_replace_a_guessed_hex(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "film_analysis.teams.uniforms._ask_uniforms",
        lambda frames, title, left_still=None, right_still=None: (
            Uniform(
                "Home",
                "blue",
                "gold",
                "blue",
                jersey_hex="#0000ff",
                pants_hex="#ffff00",
                helmet_hex="#0000ff",
            ),
            Uniform("Away", "white", "white", "white"),
        ),
    )
    pixels = [(20, 40, 180)] * 30 + [(210, 180, 40)] * 20
    monkeypatch.setattr("film_analysis.teams.uniforms._sample_pixels", lambda still: pixels)

    left, _right = detect_uniforms([], "Friday night", b"left", b"right")

    assert left.jersey_hex == "#1428b4"
    assert left.pants_hex == "#d2b428"


def test_colors_fall_back_when_the_frame_is_only_grass() -> None:
    assert colors_from_pixels([(40, 140, 50)] * 10) == ("#1c3f73", "#f2efe6")


def test_a_far_color_does_not_repaint_both_teams(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "film_analysis.teams.uniforms._ask_uniforms",
        lambda frames, title, left_still=None, right_still=None: (
            Uniform(
                "Home",
                "white",
                "white",
                "white",
                jersey_hex="#f4f1ea",
                pants_hex="#f4f1ea",
                helmet_hex="#f4f1ea",
            ),
            Uniform(
                "Away",
                "red",
                "white",
                "red",
                jersey_hex="#b42318",
                pants_hex="#f4f1ea",
                helmet_hex="#b42318",
            ),
        ),
    )
    monkeypatch.setattr(
        "film_analysis.teams.uniforms._sample_pixels",
        lambda still: [(64, 128, 128)] * 40,
    )

    left, right = detect_uniforms([], "Friday night", b"left", b"right")

    assert left.jersey_hex == "#f4f1ea"
    assert right.jersey_hex == "#b42318"


def test_player_photo_uses_the_crop_that_matches_this_team(monkeypatch: pytest.MonkeyPatch) -> None:
    from film_analysis.teams.uniforms import player_photo

    def crops(frame: bytes, jersey_hex: str, limit: int = 2) -> list[bytes]:
        del frame, limit
        if jersey_hex == "#f4f1ea":
            return [b"\xff\xd8\xffwhite", b"\xff\xd8\xffred"]
        return []

    def colors(still: bytes) -> tuple[str, str]:
        if still.endswith(b"white"):
            return "#f4f1ea", "#f4f1ea"
        return "#b42318", "#f4f1ea"

    monkeypatch.setattr("film_analysis.teams.uniforms.uniform_crops", crops)
    monkeypatch.setattr("film_analysis.teams.uniforms.dominant_colors", colors)

    photo = player_photo(
        [b"\xff\xd8\xffframe"],
        Uniform("Home", "white", "white", "white", jersey_hex="#f4f1ea"),
    )

    assert photo == b"\xff\xd8\xffwhite"


def test_player_photos_skip_the_other_team(monkeypatch: pytest.MonkeyPatch) -> None:
    from film_analysis.teams.uniforms import player_photos

    def crops(frame: bytes, jersey_hex: str, limit: int = 2) -> list[bytes]:
        del frame, limit, jersey_hex
        return [b"\xff\xd8\xffwhite", b"\xff\xd8\xffred"]

    def colors(still: bytes) -> tuple[str, str]:
        if still.endswith(b"white"):
            return "#f4f1ea", "#f4f1ea"
        return "#b42318", "#f4f1ea"

    monkeypatch.setattr("film_analysis.teams.uniforms.uniform_crops", crops)
    monkeypatch.setattr("film_analysis.teams.uniforms.dominant_colors", colors)

    photos = player_photos(
        [b"\xff\xd8\xffframe"],
        Uniform("Home", "white", "white", "white", jersey_hex="#f4f1ea"),
        Uniform("Away", "red", "white", "red", jersey_hex="#b42318"),
    )

    assert photos == [b"\xff\xd8\xffwhite"]


def test_a_model_reply_becomes_two_uniforms() -> None:
    from film_analysis.teams.uniforms import parse_uniforms

    reply = (
        '{"left":{"name":"Kentucky","jersey":"white","pants":"white","helmet":"white"},'
        '"right":{"name":"LSU","jersey":"purple with a logo","pants":"gold","helmet":"gold"}}'
    )

    left, right = parse_uniforms(reply, "Kentucky vs LSU")

    assert left == Uniform("Away", "white", "white", "white")
    assert right.name == "Home"
    assert right.jersey == "purple"
    assert right.pants == "gold"
    assert "logo" not in right.jersey


def test_a_uniform_keeps_hex_colors_and_style() -> None:
    from film_analysis.teams.uniforms import parse_uniforms

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

    assert left.jersey_hex == "#f4f1ea"
    assert left.pants_style == "solid"
    assert left.helmet_style == "solid shell"
    assert left.number == "black"
    assert right.helmet_style == "center stripe"
    assert "Pike" not in left.name
    assert left.jersey == "white"
    assert left.pants == "black"


def test_action_moments_are_after_the_snap_on_longer_plays() -> None:
    from film_analysis.teams.uniforms import action_moments

    plays = (
        PlayRecord(1, 0.0, 2.0, "scrimmage"),
        PlayRecord(2, 10.0, 18.0, "scrimmage"),
        PlayRecord(3, 30.0, 34.0, "special"),
        PlayRecord(4, 40.0, 48.0, "scrimmage"),
        PlayRecord(5, 70.0, 78.0, "scrimmage"),
    )

    moments = action_moments(plays)

    assert {number for number, _offset in moments} == {2, 4, 5}
    by_play: dict[int, list[float]] = {}
    for number, offset in moments:
        assert 1.2 < offset < 7.7
        by_play.setdefault(number, []).append(offset)
    assert all(len(offsets) == 2 for offsets in by_play.values())
    assert all(offsets[1] - offsets[0] >= 0.5 for offsets in by_play.values())


def test_an_action_frame_beats_a_graphic_and_empty_turf() -> None:
    import cv2
    import numpy as np

    from film_analysis.teams.uniforms import choose_action_shot, score_field_shot

    def jpeg(image: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        return encoded.tobytes()

    action = np.full((180, 320, 3), (50, 160, 40), np.uint8)
    for left, color in ((40, (30, 40, 160)), (140, (230, 230, 230)), (230, (40, 50, 20))):
        action[70:130, left : left + 36] = color
    graphic = np.full((180, 320, 3), (40, 30, 20), np.uint8)
    turf = np.full((180, 320, 3), (50, 160, 40), np.uint8)
    shots = (jpeg(graphic), jpeg(turf), jpeg(action))

    assert score_field_shot(shots[2]) > score_field_shot(shots[1])
    assert score_field_shot(shots[2]) > score_field_shot(shots[0])
    assert choose_action_shot(shots) == shots[2]


def test_play_samples_are_spread_across_the_film() -> None:
    from film_analysis.teams.uniforms import sample_plays

    plays = tuple(
        PlayRecord(number, float(number), float(number) + 1, "scrimmage") for number in range(1, 9)
    )

    picked = sample_plays(plays, 4)

    assert [play.play_number for play in picked] == [1, 3, 6, 8]


def test_a_renamed_team_keeps_the_uniform_from_the_photo() -> None:
    from film_analysis.teams.uniforms import portrait_uniform

    original = Uniform(
        "Home",
        "white",
        "black",
        "white",
        number="black",
        jersey_hex="#f4f1ea",
        pants_hex="#1a1a1a",
        helmet_hex="#f7f4ee",
        number_hex="#111111",
        helmet_style="solid shell",
        pants_style="solid",
    )

    renamed = portrait_uniform(original, "Pike County Pirates")

    assert renamed.name == "Pike County Pirates"
    assert renamed.jersey_hex == "#f4f1ea"
    assert renamed.helmet_style == "solid shell"
    assert renamed.number_hex == "#111111"
    assert renamed.pants_style == "solid"


def test_a_player_card_uses_the_player_found_in_the_film(monkeypatch: pytest.MonkeyPatch) -> None:
    from film_analysis.teams.uniforms import player_picture

    monkeypatch.setattr(
        "film_analysis.teams.uniforms.team_player_shot",
        lambda frames, uniform, other=None: b"\xff\xd8\xffplayer",
    )

    found = player_picture(
        [b"\xff\xd8\xffframe"],
        Uniform("Home", "white", "white", "white", jersey_hex="#f4f1ea"),
        Uniform("Away", "red", "white", "red", jersey_hex="#b42318"),
    )

    assert found == (b"\xff\xd8\xffplayer", ".jpg")


def test_a_missing_player_is_not_drawn(monkeypatch: pytest.MonkeyPatch) -> None:
    from film_analysis.teams.uniforms import player_picture

    monkeypatch.setattr(
        "film_analysis.teams.uniforms.team_player_shot",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "film_analysis.teams.uniforms.player_photos",
        lambda frames, uniform, other=None: [],
    )

    assert (
        player_picture([b"\xff\xd8\xffframe"], Uniform("Home", "white", "white", "white")) is None
    )


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

    left_color = colors_from_pixels(_pixels(left))[0]
    assert left_color[5:7] > left_color[1:3] and left_color[5:7] > left_color[3:5]
    right_color = colors_from_pixels(_pixels(right))[0]
    assert right_color[1:3] > right_color[3:5] and right_color[1:3] > right_color[5:7]


def test_the_uniform_prompt_keeps_the_json_example() -> None:
    from film_analysis.teams.uniforms import _DETECT_PROMPT

    text = _DETECT_PROMPT.format(title="Pike County")

    assert "Pike County" in text
    assert '"left"' in text
    assert '"right"' in text
    assert "white jersey" in text
    assert "jersey_hex" in text


def test_the_reference_crop_stays_under_512(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg is not installed")
    from film_analysis.teams.uniforms import _reference_jpeg

    frame = tmp_path / "wide.jpg"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=white:s=1280x720:d=0.1",
            "-frames:v",
            "1",
            str(frame),
        ],
        check=True,
    )

    data = _reference_jpeg(frame.read_bytes())
    assert data is not None and data.startswith(b"\xff\xd8\xff")
    close = tmp_path / "close.jpg"
    close.write_bytes(data)
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(close),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    width, height = (int(part) for part in probe.stdout.strip().split("x"))

    assert width < 512
    assert height < 512
    assert max(width, height) == 480


def test_a_still_without_a_photo_stays_whole() -> None:
    assert focus_player(b"not-a-photo", "left") == b"not-a-photo"


def test_a_title_does_not_guess_school_names() -> None:
    assert suggest_team_names("Pike County vs. Lamar", b"left", b"right") == ("Home", "Away")


def test_each_team_card_uses_the_film_photo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _ready_film(tmp_path)
    left = _usable_player_bytes(1)
    right = _usable_player_bytes(2)
    _stub_team_cards(monkeypatch, left, right)
    client = TestClient(
        create_app(
            store=store,
            lineup=lambda video, at: (b"left-jpeg", b"right-jpeg"),
            portrait=lambda uniform, still=b"", other=None: (b"drawn", ".jpg"),
        )
    )

    teams = client.post("/api/projects/1/teams/stills").json()["teams"]

    assert client.get(teams["left_player"]).content == left
    assert client.get(teams["right_player"]).content == right
    assert client.get(teams["left_reference"]).content == left
    assert client.get(teams["right_reference"]).content == right


def test_naming_teams_saves_stills_and_portraits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _ready_film(tmp_path)
    left = _usable_player_bytes(3)
    right = _usable_player_bytes(4)
    _stub_team_cards(monkeypatch, left, right)
    client = TestClient(
        create_app(
            store=store,
            lineup=lambda video, at: (b"left-jpeg", b"right-jpeg"),
            portrait=lambda uniform, still=b"", other=None: (
                f"<svg>{uniform.name}</svg>".encode(),
                ".svg",
            ),
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
    assert client.get(teams["left_player"]).content == left
    assert client.get(teams["right_player"]).content == right

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
    assert client.get(named["left_player"]).content == left
    assert client.get(named["right_player"]).content == right
    assert client.get("/projects/1/teams/neither").status_code == 404


def test_a_visit_keeps_the_saved_players_until_generate_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _ready_film(tmp_path)
    frames = {"n": 0}
    cards = {"n": 0}

    def lineup(video, at):
        del video, at
        frames["n"] += 1
        return f"left-{frames['n']}".encode(), f"right-{frames['n']}".encode()

    def factory():
        cards["n"] += 1
        return _usable_player_bytes(cards["n"]), _usable_player_bytes(cards["n"] + 40)

    _stub_team_cards(monkeypatch, factory=factory)
    client = TestClient(
        create_app(
            store=store,
            lineup=lineup,
            portrait=lambda uniform, still=b"", other=None: (b"drawn", ".jpg"),
        )
    )

    first = client.post("/api/projects/1/teams/stills")
    assert first.status_code == 200
    saved = client.get(first.json()["teams"]["left_player"]).content
    made = frames["n"]
    assert made > 0

    visit = client.get("/api/projects/1")
    assert visit.status_code == 200
    assert client.get(visit.json()["teams"]["left_player"]).content == saved
    assert frames["n"] == made

    again = client.post("/api/projects/1/teams/stills?again=true")
    assert again.status_code == 200
    assert frames["n"] > made
    assert client.get(again.json()["teams"]["left_player"]).content != saved


def test_one_team_photo_can_be_regenerated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _ready_film(tmp_path)
    frames = {"n": 0}
    cards = {"n": 0}

    def lineup(video, at):
        del video, at
        frames["n"] += 1
        return f"left-{frames['n']}".encode(), f"right-{frames['n']}".encode()

    def factory():
        cards["n"] += 1
        return _usable_player_bytes(cards["n"]), _usable_player_bytes(cards["n"] + 40)

    _stub_team_cards(monkeypatch, factory=factory)
    client = TestClient(
        create_app(
            store=store,
            lineup=lineup,
            portrait=lambda uniform, still=b"", other=None: (b"drawn", ".jpg"),
        )
    )
    first = client.post("/api/projects/1/teams/stills")
    assert first.status_code == 200
    left = client.get(first.json()["teams"]["left_player"]).content
    right = client.get(first.json()["teams"]["right_player"]).content
    made = frames["n"]

    again = client.post("/api/projects/1/teams/stills?again=true&side=right")
    assert again.status_code == 200
    assert frames["n"] == made + 1
    assert client.get(again.json()["teams"]["left_player"]).content == left
    assert client.get(again.json()["teams"]["right_player"]).content != right


def test_opening_a_film_replaces_the_old_generated_players(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _ready_film(tmp_path)
    store.save_team_image(1, "left", b"left-jpeg", ".jpg")
    store.save_team_image(1, "right", b"right-jpeg", ".jpg")
    old = _jpeg_claiming(768, 1280)
    left = _usable_player_bytes(11)
    right = _usable_player_bytes(12)
    store.save_team_image(1, "left-player", old, ".jpg")
    store.save_team_image(1, "right-player", old, ".jpg")
    store.set_film_teams(1, FilmTeams("North", "South", "left", ""))
    _stub_team_cards(monkeypatch, left, right)

    client = TestClient(
        create_app(
            store=store,
            lineup=lambda video, at: (b"left-jpeg", b"right-jpeg"),
            portrait=lambda uniform, still=b"", other=None: (b"drawn", ".jpg"),
        )
    )

    visit = _wait_for_player(client, left)

    assert visit.status_code == 200
    teams = visit.json()["teams"]
    assert client.get(teams["left_player"]).content == left
    assert client.get(teams["right_player"]).content == right

    second = client.get("/api/projects/1")

    assert second.status_code == 200
    assert client.get(second.json()["teams"]["left_player"]).content == left


def test_opening_a_film_replaces_a_shot_that_is_too_small(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _ready_film(tmp_path)
    tiny = _jpeg_claiming(40, 72)
    kept = _usable_player_bytes(20)
    left = _usable_player_bytes(21)
    store.save_team_image(1, "left", b"left-jpeg", ".jpg")
    store.save_team_image(1, "right", b"right-jpeg", ".jpg")
    store.save_team_image(1, "left-player", tiny, ".jpg")
    store.save_team_image(1, "right-player", kept, ".jpg")
    store.set_film_teams(1, FilmTeams("North", "South", None, ""))
    _stub_team_cards(monkeypatch, left, kept)

    client = TestClient(
        create_app(
            store=store,
            lineup=lambda video, at: (b"left-jpeg", b"right-jpeg"),
            portrait=lambda uniform, still=b"", other=None: (b"drawn", ".jpg"),
        )
    )

    visit = _wait_for_player(client, left)

    assert visit.status_code == 200
    teams = visit.json()["teams"]
    assert client.get(teams["left_player"]).content == left
    assert client.get(teams["right_player"]).content == kept


def test_opening_a_film_keeps_a_player_at_512_by_768(tmp_path: Path) -> None:
    store = _ready_film(tmp_path)
    current = _jpeg_claiming(512, 768)
    store.save_team_image(1, "left", b"left-jpeg", ".jpg")
    store.save_team_image(1, "right", b"right-jpeg", ".jpg")
    store.save_team_image(1, "left-player", current, ".jpg")
    store.save_team_image(1, "right-player", current, ".jpg")
    store.set_film_teams(1, FilmTeams("North", "South", None, ""))
    count = {"n": 0}

    def portrait(uniform, still=b"", other=None):
        del uniform, still, other
        count["n"] += 1
        return b"new-player", ".svg"

    client = TestClient(
        create_app(
            store=store,
            lineup=lambda video, at: (b"left-jpeg", b"right-jpeg"),
            portrait=portrait,
        )
    )

    visit = client.get("/api/projects/1")

    assert visit.status_code == 200
    assert count["n"] == 0
    assert client.get(visit.json()["teams"]["left_player"]).content == current


def test_teams_wait_until_plays_are_identified(tmp_path: Path) -> None:
    store = _ready_film(tmp_path)
    store.set_progress(1, status="scanning", duration_seconds=30, scanned_seconds=4, error=None)
    client = TestClient(create_app(store=store, lineup=lambda video, at: (b"a", b"b")))

    response = client.post("/api/projects/1/teams/stills")

    assert response.status_code == 400
    assert response.json()["detail"] == "Name the teams after the plays are identified."


def test_regenerate_clears_the_old_team_photos_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _ready_film(tmp_path)
    store.save_team_image(1, "left", b"old-left", ".jpg")
    store.save_team_image(1, "right", b"old-right", ".jpg")
    store.save_team_image(1, "left-player", b"old-left-player", ".jpg")
    store.save_team_image(1, "right-player", b"old-right-player", ".jpg")
    store.save_team_image(1, "left-reference", b"old-left-ref", ".jpg")
    store.save_team_image(1, "right-reference", b"old-right-ref", ".jpg")
    store.set_film_teams(1, FilmTeams("North", "South", None, ""))
    left = _usable_player_bytes(31)
    right = _usable_player_bytes(32)
    _stub_team_cards(monkeypatch, left, right)

    client = TestClient(
        create_app(
            store=store,
            lineup=lambda video, at: (b"fresh-left", b"fresh-right"),
            portrait=lambda uniform, still=b"", other=None: (b"<svg>unused</svg>", ".svg"),
        )
    )

    again = client.post("/api/projects/1/teams/stills?again=true")

    assert again.status_code == 200
    teams = again.json()["teams"]
    assert client.get(teams["left_still"]).content == b"fresh-left"
    assert client.get(teams["right_still"]).content == b"fresh-right"
    assert client.get(teams["left_player"]).content == left
    assert client.get(teams["right_player"]).content == right
    assert client.get(teams["left_player"]).content != b"old-left-player"
    assert teams["progress"]["percent"] == 100
    assert teams["progress"]["running"] is False


def test_reset_rebuilds_team_photos_for_every_ready_film(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _ready_film(tmp_path)
    second = store.create_project("Saturday", None)
    store.set_progress(
        second,
        status="ready",
        duration_seconds=20,
        scanned_seconds=20,
        error=None,
    )
    source = tmp_path / "film-2.mp4"
    source.write_bytes(b"video-2")
    store.save_source(second, source)
    clip = tmp_path / "play-2.mp4"
    clip.write_bytes(b"clip-2")
    store.add_play(second, 1, 1.0, 5.0, clip, "scrimmage")
    store.save_team_image(1, "left-player", b"stale-1", ".jpg")
    store.save_team_image(1, "right-player", b"stale-1b", ".jpg")
    store.save_team_image(second, "left-player", b"stale-2", ".jpg")
    store.save_team_image(second, "right-player", b"stale-2b", ".jpg")
    lineups = {"n": 0}
    saw_live = {"ok": False}
    cards = {"n": 0}

    def lineup(video, at):
        del video, at
        lineups["n"] += 1
        time.sleep(0.2)
        return b"left-jpeg", b"right-jpeg"

    def factory():
        cards["n"] += 1
        return _usable_player_bytes(cards["n"]), _usable_player_bytes(cards["n"] + 50)

    _stub_team_cards(monkeypatch, factory=factory)
    client = TestClient(
        create_app(
            store=store,
            lineup=lineup,
            portrait=lambda uniform, still=b"", other=None: (b"<svg>unused</svg>", ".svg"),
        )
    )

    started = client.post("/api/teams/reset")
    assert started.status_code == 200
    body = started.json()
    assert body["running"] is True or body["done"] == body["total"]
    assert body["total"] == 2
    assert "films" in body
    assert body["steps"][0]["id"] == "queue"

    deadline = time.monotonic() + 5
    progress = body
    while time.monotonic() < deadline:
        progress = client.get("/api/teams/reset").json()
        running_rows = [row for row in progress["films"] if row["status"] == "running"]
        if running_rows and (
            running_rows[0].get("percent", 0) > 0
            or running_rows[0].get("detail")
        ):
            saw_live["ok"] = True
        if not progress["running"]:
            break
        time.sleep(0.05)

    assert progress["running"] is False
    assert progress["total"] == 2
    assert progress["done"] == 2
    assert progress["failed"] == 0
    assert saw_live["ok"] is True
    assert lineups["n"] >= 2
    assert store.team_image(1, "left-player") is not None
    assert store.team_image(second, "right-player") is not None
    assert store.team_image(1, "left-player").read_bytes() != b"stale-1"
    assert store.team_image(second, "right-player").read_bytes() != b"stale-2b"
    assert {row["status"] for row in progress["films"]} == {"ready"}
    assert all(row.get("percent") == 100 for row in progress["films"])


def test_reset_can_retry_only_the_failed_films(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _ready_film(tmp_path)
    second = store.create_project("Saturday", None)
    store.set_progress(
        second,
        status="ready",
        duration_seconds=20,
        scanned_seconds=20,
        error=None,
    )
    source = tmp_path / "film-2.mp4"
    source.write_bytes(b"video-2")
    store.save_source(second, source)
    clip = tmp_path / "play-2.mp4"
    clip.write_bytes(b"clip-2")
    store.add_play(second, 1, 1.0, 5.0, clip, "scrimmage")
    calls = {"n": 0}
    cards = {"n": 0}

    def lineup(video, at):
        del video, at
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("first film fails once")
        return b"left-jpeg", b"right-jpeg"

    def factory():
        cards["n"] += 1
        return _usable_player_bytes(cards["n"]), _usable_player_bytes(cards["n"] + 60)

    _stub_team_cards(monkeypatch, factory=factory)
    client = TestClient(
        create_app(
            store=store,
            lineup=lineup,
            portrait=lambda uniform, still=b"", other=None: (b"<svg>unused</svg>", ".svg"),
        )
    )

    first = client.post("/api/teams/reset")
    assert first.status_code == 200
    deadline = time.monotonic() + 5
    progress = first.json()
    while time.monotonic() < deadline:
        progress = client.get("/api/teams/reset").json()
        if not progress["running"]:
            break
        time.sleep(0.05)

    assert progress["failed"] >= 1
    assert any(row["status"] == "failed" for row in progress["films"])

    retry = client.post("/api/teams/reset?failed=true")
    assert retry.status_code == 200
    deadline = time.monotonic() + 5
    progress = retry.json()
    while time.monotonic() < deadline:
        progress = client.get("/api/teams/reset").json()
        if not progress["running"]:
            break
        time.sleep(0.05)

    assert progress["running"] is False
    assert progress["failed"] == 0
    assert progress["done"] >= 1
    assert {row["status"] for row in progress["films"]} == {"ready"}


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


def _wait_for_player(client: TestClient, expected: bytes) -> object:
    deadline = time.monotonic() + 3
    visit = client.get("/api/projects/1")
    while time.monotonic() < deadline:
        assert visit.status_code == 200
        teams = visit.json().get("teams") or {}
        url = teams.get("left_player")
        if url:
            try:
                got = client.get(url)
                if got.status_code == 200 and got.content == expected:
                    return visit
            except (OSError, RuntimeError):
                pass
        time.sleep(0.02)
        visit = client.get("/api/projects/1")
    return visit


def _jpeg_claiming(width: int, height: int) -> bytes:
    return (
        b"\xff\xd8"
        b"\xff\xc0\x00\x0b\x08"
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + b"\x01\x01\x11\x00"
        b"\xff\xd9"
    )


def _pixels(still: bytes) -> list[tuple[int, int, int]]:
    from film_analysis.teams.uniforms import _sample_pixels

    return _sample_pixels(still)
