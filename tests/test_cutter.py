from pathlib import Path

from film_analysis.cutter.server import _apply_worker_count, _worker_limit
from film_analysis.storage.catalog import MemoryCatalog


def test_a_cloud_film_starts_at_the_requested_worker_count(tmp_path: Path) -> None:
    catalog = MemoryCatalog(tmp_path)
    slots, baseline = _worker_limit(catalog, 8)

    assert slots.get() == 8
    assert baseline == 4
    assert _apply_worker_count(catalog, slots, baseline) == 4
    assert slots.get() == 8


def test_a_running_cloud_film_picks_up_a_new_worker_count(tmp_path: Path) -> None:
    catalog = MemoryCatalog(tmp_path)
    catalog.set_worker_count(4)
    slots, baseline = _worker_limit(catalog, 4)

    catalog.set_worker_count(12)
    seen = _apply_worker_count(catalog, slots, baseline)

    assert seen == 12
    assert slots.get() == 12
    assert _apply_worker_count(catalog, slots, seen) == 12
    assert slots.get() == 12
