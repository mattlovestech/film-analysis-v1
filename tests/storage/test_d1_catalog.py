"""The D1 catalog makes as few round trips as it can."""

import threading
from pathlib import Path

import pytest

from film_analysis.storage import catalog as catalog_module
from film_analysis.storage.catalog import ClipDirectory, D1Catalog, _missing_schema


class _FakeD1(D1Catalog):
    """Records each request instead of calling Cloudflare."""

    def __init__(self, database_id: str, master: list[dict[str, object]], tmp: Path) -> None:
        self.requests: list[dict[str, object]] = []
        self.master = master
        self.settings = {"workers": "6", "formations": "1"}
        super().__init__("account", "token", database_id, ClipDirectory(tmp))

    def _post(self, body: dict[str, object]) -> list[dict[str, object]]:
        self.requests.append(body)
        statements = body.get("batch") or [body]
        results: list[dict[str, object]] = []
        for statement in statements:
            sql = str(statement["sql"])
            if sql.startswith("select type, name, sql from sqlite_master"):
                rows: list[dict[str, object]] = self.master
            elif sql.startswith("select key, value from settings"):
                rows = [{"key": key, "value": value} for key, value in self.settings.items()]
            elif sql.startswith("select 1 as found from projects"):
                rows = [{"found": 1}]
            elif sql.startswith("insert into coach_edits"):
                rows = [{"id": 41}]
            else:
                rows = []
            results.append({"success": True, "results": rows, "meta": {"changes": 1}})
        return results


def _current_master() -> list[dict[str, object]]:
    """A database where every table, column, and index already exists."""

    tables = {
        "projects": "create table projects (id integer primary key, name text, status text, "
        "duration_seconds real, scanned_seconds real, error text, channel text, "
        "downloaded_bytes integer, total_bytes integer, archived integer, "
        "retries integer, retries_done integer)",
        "plays": "create table plays (project_id integer, play_number integer, family text, "
        "play_type text, reviewed integer, unit text, formation text, discarded integer)",
        "spans": "create table spans (project_id integer)",
        "activity": "create table activity (project_id integer primary key, steps text)",
        "settings": "create table settings (key text primary key, value text)",
        "coach_edits": "create table coach_edits (id integer primary key)",
        "reads": "create table reads (project_id integer, input_tokens integer, "
        "output_tokens integer, cost_usd real, model text)",
        "scouting_analysis": "create table scouting_analysis (project_id integer)",
        "teams": "create table teams (project_id integer primary key, uniforms text)",
    }
    rows: list[dict[str, object]] = [
        {"type": "table", "name": name, "sql": sql} for name, sql in tables.items()
    ]
    for sql in catalog_module._D1_INDEXES:
        name = sql.split("exists ")[1].split()[0]
        rows.append({"type": "index", "name": name, "sql": sql})
    return rows


@pytest.fixture(autouse=True)
def _fresh_schema_memo() -> None:
    catalog_module._SCHEMA_READY.clear()


def test_a_current_database_is_checked_with_one_request(tmp_path: Path) -> None:
    store = _FakeD1("db-current", _current_master(), tmp_path)

    assert len(store.requests) == 1


def test_a_second_catalog_for_the_same_database_skips_the_check(tmp_path: Path) -> None:
    _FakeD1("db-same", _current_master(), tmp_path)
    again = _FakeD1("db-same", _current_master(), tmp_path)

    assert again.requests == []


def test_missing_columns_and_indexes_are_added_in_one_batch(tmp_path: Path) -> None:
    master = [
        row
        for row in _current_master()
        if not (row["type"] == "index" and row["name"] == "spans_by_project")
    ]
    for row in master:
        if row["name"] == "teams":
            row["sql"] = "create table teams (project_id integer primary key)"

    store = _FakeD1("db-old", master, tmp_path)

    assert len(store.requests) == 2
    batch = store.requests[1]["batch"]
    assert [item["sql"] for item in batch] == [
        "alter table teams add column uniforms text",
        "create index if not exists spans_by_project on spans (project_id, kind)",
    ]


def test_missing_schema_does_not_mistake_a_longer_column_for_a_short_one() -> None:
    master = [{"type": "table", "name": "plays", "sql": "create table plays (units text)"}]

    missing = _missing_schema(("alter table plays add column unit text",), master)

    assert missing == ["alter table plays add column unit text"]


def test_settings_are_read_together_and_kept_briefly(tmp_path: Path) -> None:
    store = _FakeD1("db-settings", _current_master(), tmp_path)
    store.requests.clear()

    assert store.worker_count() == 6
    assert store.formation_reads() is True
    assert store.coach_guidance() == ""
    assert len(store.requests) == 1


def test_changing_a_setting_is_seen_on_the_next_read(tmp_path: Path) -> None:
    store = _FakeD1("db-write", _current_master(), tmp_path)
    assert store.worker_count() == 6

    store.settings["workers"] = "9"
    store.set_worker_count(9)

    assert store.worker_count() == 9


def test_a_film_is_read_in_one_request(tmp_path: Path) -> None:
    store = _FakeD1("db-film", _current_master(), tmp_path)
    store.requests.clear()

    store.get_project(3)

    assert len(store.requests) == 1
    assert len(store.requests[0]["batch"]) == 3


def test_deleting_a_film_is_one_request(tmp_path: Path) -> None:
    store = _FakeD1("db-delete", _current_master(), tmp_path)
    store.requests.clear()

    store.delete_project(3)

    assert len(store.requests) == 1
    assert len(store.requests[0]["batch"]) == 8


def test_a_coach_edit_keeps_the_id_the_insert_returned(tmp_path: Path) -> None:
    store = _FakeD1("db-edit", _current_master(), tmp_path)
    store.requests.clear()

    edit = store.add_coach_edit(3, 1, "Film", "retag", "run", "pass")

    assert edit.edit_id == 41
    assert len(store.requests) == 1


def test_learned_edits_are_marked_in_one_request(tmp_path: Path) -> None:
    store = _FakeD1("db-learned", _current_master(), tmp_path)
    store.requests.clear()

    store.mark_coach_edits_learned([4, 5, 6])

    assert len(store.requests) == 1
    assert store.requests[0]["params"] == [4, 5, 6]


def test_a_film_page_checks_its_film_once(tmp_path: Path) -> None:
    store = _FakeD1("db-images", _current_master(), tmp_path)
    store.requests.clear()

    for number in range(1, 6):
        store.play_image(3, number, "formation")

    assert len(store.requests) == 1


def test_a_deleted_film_is_not_remembered(tmp_path: Path) -> None:
    store = _FakeD1("db-gone", _current_master(), tmp_path)
    store.play_image(3, 1, "formation")
    store.delete_project(3)
    store.requests.clear()

    store.play_image(3, 1, "formation")

    assert len(store.requests) == 1


def test_replacing_a_team_image_never_leaves_it_missing(tmp_path: Path) -> None:
    clips = ClipDirectory(tmp_path)
    clips.save_team_image(1, "left-player", b"old", ".jpg")
    seen: list[bool] = []
    stop = threading.Event()

    def watch() -> None:
        while not stop.is_set():
            seen.append(clips.team_image(1, "left-player") is not None)

    watcher = threading.Thread(target=watch)
    watcher.start()
    for round_number in range(300):
        clips.save_team_image(1, "left-player", f"new-{round_number}".encode(), ".jpg")
    stop.set()
    watcher.join()

    assert seen and all(seen)
    assert sorted(path.name for path in (tmp_path / "projects" / "1" / "teams").iterdir()) == [
        "left-player.jpg"
    ]
