"""Player boxes separate the two uniform colors."""

import cv2
import numpy as np
import pytest

from film_analysis.web.players import (
    PlayerBox,
    PlayerColors,
    _striped,
    full_body_shot,
    lone_players,
    team_colors_from_players,
    uniform_crops,
)
from film_analysis.web.teams import Uniform, detect_uniforms


def test_detected_players_split_a_white_team_from_a_colored_team(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    monkeypatch.setattr("film_analysis.web.players._bgr", lambda jpeg: image)

    def players(jpeg: bytes) -> list[PlayerBox]:
        del jpeg
        return [PlayerBox(0, 0, 30, 70), PlayerBox(40, 0, 30, 70)]

    def colors(frame, box: PlayerBox):
        del frame
        if box.x == 0:
            return (248, 246, 242), (28, 28, 28), (248, 246, 242)
        return (180, 35, 24), (248, 246, 242), (180, 35, 24)

    monkeypatch.setattr("film_analysis.web.players.find_players", players)
    monkeypatch.setattr("film_analysis.web.players._uniform_colors", colors)

    light, solid = team_colors_from_players([b"\xff\xd8\xffframe"])

    assert light.jersey == "#f8f6f2"
    assert light.pants == "#1c1c1c"
    assert solid.jersey == "#b42318"
    assert solid.helmet == "#b42318"


def test_player_colors_replace_one_shared_field_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "film_analysis.web.teams._ask_uniforms",
        lambda frames, title, left_still=None, right_still=None: (
            Uniform(
                "Home",
                "dark blue",
                "gray",
                "dark blue",
                jersey_hex="#408080",
                pants_hex="#808080",
                helmet_hex="#408080",
            ),
            Uniform(
                "Away",
                "dark blue",
                "gray",
                "dark blue",
                jersey_hex="#406060",
                pants_hex="#808080",
                helmet_hex="#406060",
            ),
        ),
    )
    monkeypatch.setattr(
        "film_analysis.web.teams.team_colors_from_players",
        lambda frames: (
            PlayerColors("#f4f1ea", "#1a1a1a", "#f7f4ee"),
            PlayerColors("#b42318", "#f4f1ea", "#b42318"),
        ),
    )

    left, right = detect_uniforms([b"\xff\xd8\xffframe"], "Friday night", b"left", b"right")

    jerseys = {left.jersey_hex, right.jersey_hex}
    assert jerseys == {"#f4f1ea", "#b42318"}
    assert left.jersey_hex != right.jersey_hex


def test_a_close_crop_keeps_the_matching_jersey() -> None:
    image = np.zeros((220, 360, 3), dtype=np.uint8)
    image[:, :] = (40, 140, 40)
    image[50:150, 30:90] = (242, 242, 242)
    image[50:150, 230:290] = (46, 16, 200)
    ok, encoded = cv2.imencode(".jpg", image)

    assert ok
    crops = uniform_crops(encoded.tobytes(), "#c8102e")

    assert len(crops) == 1
    found = cv2.imdecode(np.frombuffer(crops[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert found is not None
    red_cloth = ((found[:, :, 2] > 150) & (found[:, :, 1] < 80)).sum()
    assert red_cloth > 20


def test_full_body_shot_is_one_player_from_this_team(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.zeros((220, 240, 3), dtype=np.uint8)
    image[:, :] = (40, 120, 40)
    image[12:190, 12:52] = (242, 246, 248)
    image[12:190, 170:210] = (24, 35, 180)
    monkeypatch.setattr("film_analysis.web.players._bgr", lambda jpeg: image)

    def people(frame):
        del frame
        return [PlayerBox(12, 12, 40, 178), PlayerBox(170, 12, 40, 178)]

    monkeypatch.setattr("film_analysis.web.players._hog_people", people)

    shot = full_body_shot([b"\xff\xd8\xffframe"], "#f8f6f2", "#b42318")

    assert shot is not None
    assert shot.startswith(b"\xff\xd8\xff")
    found = cv2.imdecode(np.frombuffer(shot, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert found is not None
    assert max(found.shape[:2]) <= 512
    mean = found.reshape(-1, 3).mean(axis=0)
    assert mean[2] > 180
    assert abs(float(mean[2]) - float(mean[0])) < 40


def test_full_body_shot_skips_a_frame_shared_with_the_other_team(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((220, 240, 3), dtype=np.uint8)
    monkeypatch.setattr("film_analysis.web.players._bgr", lambda jpeg: image)
    monkeypatch.setattr(
        "film_analysis.web.players._hog_people",
        lambda frame: [PlayerBox(12, 12, 40, 178)],
    )
    monkeypatch.setattr(
        "film_analysis.web.players._uniform_colors",
        lambda frame, box: ((248, 246, 242), (28, 28, 28), (248, 246, 242)),
    )
    monkeypatch.setattr("film_analysis.web.players._other_share", lambda frame, box, other: 0.4)

    assert full_body_shot([b"\xff\xd8\xffframe"], "#f8f6f2", "#b42318") is None


def _blurred_official() -> np.ndarray:
    """A broadcast-soft black-and-white striped shirt, cap, and pants."""

    official = np.zeros((320, 140, 3), dtype=np.uint8)
    official[:, :] = (40, 140, 40)
    official[20:300, 28:112] = (18, 18, 18)
    official[16:48, 40:100] = (12, 12, 12)
    for column in range(32, 108, 14):
        official[36:175, column : column + 7] = (245, 245, 245)
    official[180:300, 68:74] = (240, 240, 240)
    return cv2.GaussianBlur(official, (0, 0), 2.5)


def _jpeg(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_lone_players_crops_one_white_player_and_one_colored_player(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((400, 320, 3), dtype=np.uint8)
    image[:, :] = (40, 140, 40)
    image[40:220, 30:100] = (240, 240, 240)
    image[40:240, 190:270] = (40, 40, 200)
    monkeypatch.setattr("film_analysis.web.players._alone", lambda crop: True)
    monkeypatch.setattr(
        "film_analysis.web.players._detected_people",
        lambda frame: [PlayerBox(30, 40, 70, 180), PlayerBox(190, 40, 80, 200)],
    )

    white, colored = lone_players([_jpeg(image)])

    assert white is not None and colored is not None
    white_image = cv2.imdecode(np.frombuffer(white, dtype=np.uint8), cv2.IMREAD_COLOR)
    colored_image = cv2.imdecode(np.frombuffer(colored, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert white_image is not None and colored_image is not None
    assert white_image.reshape(-1, 3).mean() > 180
    assert colored_image[:, :, 2].mean() > colored_image[:, :, 0].mean() + 40


def test_a_referee_and_a_gray_coach_are_not_team_players(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    referee = np.zeros((320, 140, 3), dtype=np.uint8)
    referee[:, :] = (40, 140, 40)
    for column in range(24, 120, 8):
        referee[20:280, column : column + 4] = (250, 250, 250)
    coach = np.zeros((320, 140, 3), dtype=np.uint8)
    coach[:, :] = (40, 140, 40)
    coach[20:280, 30:110] = (140, 140, 140)
    monkeypatch.setattr("film_analysis.web.players._alone", lambda crop: True)
    monkeypatch.setattr(
        "film_analysis.web.players._detected_people",
        lambda frame: [PlayerBox(20, 20, 100, 260)],
    )

    assert lone_players([_jpeg(referee)]) == (None, None)
    assert lone_players([_jpeg(coach)]) == (None, None)


def test_a_soft_striped_official_is_not_a_team(monkeypatch: pytest.MonkeyPatch) -> None:
    official = _blurred_official()
    white = np.zeros((320, 140, 3), dtype=np.uint8)
    white[:, :] = (40, 140, 40)
    white[20:300, 30:110] = (245, 245, 245)

    assert _striped(official)
    assert not _striped(white)

    monkeypatch.setattr("film_analysis.web.players._alone", lambda crop: True)
    monkeypatch.setattr(
        "film_analysis.web.players._detected_people",
        lambda frame: [PlayerBox(20, 16, 100, 280)],
    )
    assert lone_players([_jpeg(official)]) == (None, None)

    monkeypatch.setattr("film_analysis.web.players._bgr", lambda jpeg: official)
    monkeypatch.setattr(
        "film_analysis.web.players._hog_people",
        lambda frame: [PlayerBox(24, 16, 92, 280)],
    )
    monkeypatch.setattr(
        "film_analysis.web.players._uniform_colors",
        lambda frame, box: ((248, 246, 242), (20, 20, 20), (20, 20, 20)),
    )
    monkeypatch.setattr(
        "film_analysis.web.players.find_players",
        lambda jpeg: [PlayerBox(24, 16, 92, 280)],
    )

    assert full_body_shot([b"\xff\xd8\xffframe"], "#f8f6f2", "#b42318") is None
    assert team_colors_from_players([b"\xff\xd8\xffframe"]) is None


def test_a_crop_that_contains_another_player_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((400, 320, 3), dtype=np.uint8)
    image[:, :] = (40, 140, 40)
    image[30:300, 20:160] = (240, 240, 240)
    monkeypatch.setattr("film_analysis.web.players._alone", lambda crop: True)
    monkeypatch.setattr(
        "film_analysis.web.players._detected_people",
        lambda frame: [PlayerBox(10, 20, 180, 300), PlayerBox(120, 40, 50, 160)],
    )

    white, colored = lone_players([_jpeg(image)])

    assert colored is None
    assert white is not None
    found = cv2.imdecode(np.frombuffer(white, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert found is not None
    assert found.shape[1] / found.shape[0] < 0.45


def test_a_player_cut_off_at_the_waist_or_standing_on_the_sideline_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legs = np.zeros((260, 120, 3), dtype=np.uint8)
    legs[:, :] = (40, 140, 40)
    legs[:220, 30:90] = (40, 40, 200)
    sideline = np.zeros((260, 120, 3), dtype=np.uint8)
    sideline[:, :] = (20, 20, 20)
    sideline[30:230, 30:90] = (40, 40, 180)
    monkeypatch.setattr(
        "film_analysis.web.players._detected_people",
        lambda frame: [PlayerBox(20, 10, 80, 240)],
    )

    assert lone_players([_jpeg(legs)]) == (None, None)
    assert lone_players([_jpeg(sideline)]) == (None, None)
