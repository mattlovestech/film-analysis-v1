"""Player boxes separate the two uniform colors."""

import cv2
import numpy as np
import pytest

from film_analysis.web.players import (
    PlayerBox,
    PlayerColors,
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
