"""A small pool of diagram workers fed from a queue.

Each queued play becomes a run in the RunStore before a worker takes it, so the
page shows it waiting, then running step by step, then finished.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from film_analysis.diagram.pipeline import DiagramPipeline
from film_analysis.diagram.trace import Run, RunStore

DEFAULT_WORKERS = 2
MAX_WORKERS = 8


def worker_count() -> int:
    raw = os.environ.get("FILM_DIAGRAM_WORKERS", "").strip()
    try:
        value = int(raw) if raw else DEFAULT_WORKERS
    except ValueError:
        value = DEFAULT_WORKERS
    return max(1, min(MAX_WORKERS, value))


class DiagramWorkers:
    def __init__(self, pipeline: DiagramPipeline, workers: int | None = None) -> None:
        self.pipeline = pipeline
        self.store: RunStore = pipeline.store
        self.size = workers or worker_count()
        self._pool = ThreadPoolExecutor(self.size, thread_name_prefix="diagram-worker")
        self._lock = threading.Lock()
        self._busy: set[str] = set()

    def submit(self, plays: Sequence[tuple[int, int]]) -> list[Run]:
        """Queue each (project, play). A play already queued or running keeps its run."""

        runs: list[Run] = []
        active = {
            run.uid: run for run in self.store.recent(10_000) if run.status in {"queued", "running"}
        }
        seen: set[str] = set()
        for project, play in plays:
            uid = f"{project}-{play}"
            if uid in seen:
                continue
            seen.add(uid)
            if uid in active:
                runs.append(active[uid])
                continue
            run = self.store.create(uid, project, play)
            active[uid] = run
            runs.append(run)
            self._pool.submit(self._work, run)
        return runs

    def stats(self) -> dict[str, object]:
        with self._lock:
            busy = sorted(self._busy)
        payload = self.store.stats()
        payload.update({"workers": self.size, "busy": len(busy), "busy_on": busy})
        return payload

    def _work(self, run: Run) -> None:
        name = threading.current_thread().name
        with self._lock:
            self._busy.add(run.uid)
        try:
            self.pipeline.run(run, worker=name)
        finally:
            with self._lock:
                self._busy.discard(run.uid)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
