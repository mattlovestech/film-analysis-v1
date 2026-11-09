"""Coach corrections become notes for the next cut and tag."""

from pathlib import Path

from fastapi.testclient import TestClient

from film_analysis.learning import apply_coach_notes, guidance_from_reply, publish_guidance
from film_analysis.storage.catalog import FilmTeams, MemoryCatalog
from film_analysis.web.app import create_app


class _Agent:
    name = "test"

    def segment(self, video_path: Path):
        assert video_path.is_file()
        return ()


def test_coach_notes_are_added_to_a_prompt() -> None:
    publish_guidance("")
    assert apply_coach_notes("Cut at the snap.") == "Cut at the snap."
    publish_guidance("Hold a run until the tackle.")
    try:
        noted = apply_coach_notes("Cut at the snap.")
    finally:
        publish_guidance("")
    assert "Cut at the snap." in noted
    assert "Hold a run until the tackle." in noted


def test_guidance_is_read_from_the_model_reply() -> None:
    assert guidance_from_reply('{"guidance":"End the play when the ball is dead."}') == (
        "End the play when the ball is dead."
    )
    assert guidance_from_reply("no json") == ""


def test_a_retag_is_kept_and_can_teach_the_next_read(tmp_path: Path) -> None:
    store = MemoryCatalog(tmp_path)
    project = store.create_project("Friday night", None)
    store.set_progress(project, status="ready", duration_seconds=40, scanned_seconds=40, error=None)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    store.add_play(project, 1, 12, 20, clip, "scrimmage")
    store.set_film_teams(project, FilmTeams("Home", "Away", "left"))

    def learn(guidance: str, edits: list) -> str:
        assert guidance == ""
        assert edits[0].action == "unit"
        assert "untagged" in edits[0].before
        return "When the opponent has the ball, tag your team as defense."

    client = TestClient(create_app(agent=_Agent(), store=store, learner=learn))
    tagged = client.post("/api/plays/1-1", json={"unit": "defense"})
    assert tagged.status_code == 200

    log = client.get("/api/learning")
    assert log.status_code == 200
    body = log.json()
    assert body["pending"] == 1
    assert body["edits"][0]["action"] == "unit"
    assert "defense" in body["edits"][0]["after"]

    taught = client.post("/api/learning")
    assert taught.status_code == 200
    assert taught.json()["guidance"] == "When the opponent has the ball, tag your team as defense."
    assert taught.json()["learned"] == 1
    again = client.get("/api/learning").json()
    assert again["pending"] == 0
    assert again["edits"][0]["learned"] is True

    empty = client.post("/api/learning")
    assert empty.status_code == 400
