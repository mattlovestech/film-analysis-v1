"""Runs, steps, and saved images for the diagram workers.

A run is one play going through the pipeline once. Each step records when it
started, how long it took, what it found, and why it stopped. Images a step
produces (the snap frames, the frame with players circled) are kept beside the
run so the page can show exactly what the worker looked at.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

_LOG = logging.getLogger("film_analysis.diagram")

RUN_STATES = ("queued", "running", "done", "skipped", "failed")
STEP_STATES = ("running", "ok", "warn", "skipped", "failed")
_KEEP_RUNS = 200


class StepSkipped(Exception):
    """A step decided this play is not for the diagram, such as a punt."""


class StepFailed(Exception):
    """A step could not go on. The message is shown on the page."""


@dataclass
class Step:
    name: str
    code: str
    status: str = "running"
    started: float = 0.0
    ended: float | None = None
    detail: str = ""
    data: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)

    @property
    def ms(self) -> int | None:
        if self.ended is None:
            return None
        return int(round((self.ended - self.started) * 1000))


@dataclass
class Run:
    run_id: str
    uid: str
    project: int
    play: int
    status: str = "queued"
    queued_at: float = field(default_factory=time.time)
    started_at: float | None = None
    ended_at: float | None = None
    worker: str = ""
    detail: str = ""
    error: str = ""
    steps: list[Step] = field(default_factory=list)
    log: list[dict[str, object]] = field(default_factory=list)
    diagram: dict[str, object] | None = None

    def summary(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "uid": self.uid,
            "project": self.project,
            "play": self.play,
            "status": self.status,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "wait_ms": _ms(self.queued_at, self.started_at),
            "total_ms": _ms(self.started_at, self.ended_at),
            "worker": self.worker,
            "detail": self.detail,
            "error": self.error.splitlines()[-1] if self.error else "",
            "step": next((step.name for step in self.steps if step.status == "running"), ""),
            "steps": {step.name: step.status for step in self.steps},
            "offense": _count(self.diagram, "offense"),
            "defense": _count(self.diagram, "defense"),
        }

    def full(self) -> dict[str, object]:
        payload = self.summary()
        payload["error"] = self.error
        payload["steps"] = [{**asdict(step), "ms": step.ms} for step in self.steps]
        payload["log"] = list(self.log)
        payload["diagram"] = self.diagram
        return payload


class RunStore:
    """Every run in memory, and on disk when a folder is given.

    The folder keeps one ``run.json`` and its images per run, so a restart
    still shows yesterday's traces.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root
        self._runs: dict[str, Run] = {}
        self._images: dict[tuple[str, str], bytes] = {}
        self._lock = threading.RLock()
        if root is not None:
            self._load(root)

    # Runs ---------------------------------------------------------------

    def create(self, uid: str, project: int, play: int) -> Run:
        run = Run(uuid.uuid4().hex[:12], uid, project, play)
        with self._lock:
            self._runs[run.run_id] = run
            self._trim()
        self.event(run, "info", "queued")
        self.save(run)
        return run

    def get(self, run_id: str) -> Run | None:
        with self._lock:
            return self._runs.get(run_id)

    def recent(self, limit: int = 50) -> list[Run]:
        with self._lock:
            runs = sorted(self._runs.values(), key=lambda run: run.queued_at, reverse=True)
        return runs[:limit]

    def latest_by_uid(self) -> dict[str, Run]:
        latest: dict[str, Run] = {}
        with self._lock:
            for run in self._runs.values():
                seen = latest.get(run.uid)
                if seen is None or run.queued_at > seen.queued_at:
                    latest[run.uid] = run
        return latest

    def active_uids(self) -> set[str]:
        with self._lock:
            return {run.uid for run in self._runs.values() if run.status in {"queued", "running"}}

    def stats(self) -> dict[str, object]:
        with self._lock:
            runs = list(self._runs.values())
        counts = {state: 0 for state in RUN_STATES}
        step_ms: dict[str, list[int]] = {}
        for run in runs:
            counts[run.status] = counts.get(run.status, 0) + 1
            for step in run.steps:
                if step.ms is not None and step.status in {"ok", "warn"}:
                    step_ms.setdefault(step.name, []).append(step.ms)
        totals = [
            int((run.ended_at - run.started_at) * 1000)
            for run in runs
            if run.started_at and run.ended_at and run.status == "done"
        ]
        finished = counts["done"] + counts["failed"]
        return {
            "runs": counts,
            "success_rate": round(counts["done"] / finished, 3) if finished else None,
            "avg_total_ms": int(sum(totals) / len(totals)) if totals else None,
            "p95_total_ms": _percentile(totals, 0.95),
            "avg_step_ms": {
                name: int(sum(values) / len(values)) for name, values in step_ms.items()
            },
        }

    # Steps and events ---------------------------------------------------

    @contextmanager
    def step(self, run: Run, name: str, code: str) -> Iterator[Step]:
        """Time one step. Raise StepSkipped or StepFailed inside to stop the run."""

        current = Step(name, code, started=time.time())
        with self._lock:
            run.steps.append(current)
        self.event(run, "info", f"{name} started", step=name)
        self.save(run)
        try:
            yield current
        except StepSkipped as reason:
            current.status = "skipped"
            current.detail = current.detail or str(reason)
            raise
        except StepFailed as reason:
            current.status = "failed"
            current.detail = current.detail or str(reason)
            raise
        except Exception as error:
            current.status = "failed"
            current.detail = current.detail or f"{type(error).__name__}: {error}"
            raise
        else:
            current.status = "warn" if current.warnings else "ok"
        finally:
            current.ended = time.time()
            level = "error" if current.status == "failed" else "info"
            self.event(
                run,
                level,
                f"{name} {current.status} in {current.ms} ms"
                + (f": {current.detail}" if current.detail else ""),
                step=name,
            )
            for warning in current.warnings:
                self.event(run, "warn", warning, step=name)
            self.save(run)

    def event(self, run: Run, level: str, message: str, **fields: object) -> None:
        entry: dict[str, object] = {"t": time.time(), "level": level, "message": message, **fields}
        with self._lock:
            run.log.append(entry)
        log = _LOG.error if level == "error" else _LOG.warning if level == "warn" else _LOG.info
        log("diagram run=%s uid=%s %s", run.run_id, run.uid, message)

    def set_state(self, run: Run, status: str, **values: object) -> None:
        with self._lock:
            run.status = status
            for key, value in values.items():
                setattr(run, key, value)
        self.save(run)

    # Images -------------------------------------------------------------

    def put_image(self, run: Run, step: Step, name: str, jpeg: bytes) -> None:
        key = f"{len(run.steps):02d}-{name}.jpg"
        with self._lock:
            self._images[(run.run_id, key)] = jpeg
            step.artifacts.append(key)
        if self.root is not None:
            folder = self.root / run.run_id
            folder.mkdir(parents=True, exist_ok=True)
            (folder / key).write_bytes(jpeg)

    def image(self, run_id: str, name: str) -> bytes | None:
        if "/" in name or "\\" in name or ".." in name:
            return None
        with self._lock:
            found = self._images.get((run_id, name))
        if found is not None:
            return found
        if self.root is None:
            return None
        path = self.root / run_id / name
        return path.read_bytes() if path.is_file() else None

    # Disk ---------------------------------------------------------------

    def save(self, run: Run) -> None:
        if self.root is None:
            return
        folder = self.root / run.run_id
        try:
            folder.mkdir(parents=True, exist_ok=True)
            with self._lock:
                text = json.dumps(run.full(), default=str)
            (folder / "run.json").write_text(text, encoding="utf-8")
        except OSError:
            _LOG.warning("could not save diagram run %s", run.run_id, exc_info=True)

    def _load(self, root: Path) -> None:
        if not root.is_dir():
            return
        for path in root.glob("*/run.json"):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                run = _run_from_json(raw)
            except (OSError, ValueError, TypeError, KeyError):
                continue
            if run.status in {"queued", "running"}:
                # The worker that had this run is gone with the old process.
                run.status = "failed"
                run.error = run.error or "The server restarted during this run."
            self._runs[run.run_id] = run

    def _trim(self) -> None:
        if len(self._runs) <= _KEEP_RUNS:
            return
        oldest = sorted(self._runs.values(), key=lambda run: run.queued_at)
        for run in oldest[: len(self._runs) - _KEEP_RUNS]:
            if run.status in {"queued", "running"}:
                continue
            self._runs.pop(run.run_id, None)
            for key in [key for key in self._images if key[0] == run.run_id]:
                self._images.pop(key, None)


def _run_from_json(raw: dict[str, object]) -> Run:
    steps = []
    for item in raw.get("steps") or []:
        if not isinstance(item, dict):
            continue
        steps.append(
            Step(
                name=str(item["name"]),
                code=str(item.get("code") or ""),
                status=str(item.get("status") or "ok"),
                started=float(item.get("started") or 0.0),
                ended=float(item["ended"]) if item.get("ended") is not None else None,
                detail=str(item.get("detail") or ""),
                data=dict(item.get("data") or {}),
                warnings=list(item.get("warnings") or []),
                artifacts=list(item.get("artifacts") or []),
            )
        )
    diagram = raw.get("diagram")
    return Run(
        run_id=str(raw["run_id"]),
        uid=str(raw["uid"]),
        project=int(raw["project"]),
        play=int(raw["play"]),
        status=str(raw.get("status") or "failed"),
        queued_at=float(raw.get("queued_at") or 0.0),
        started_at=_float(raw.get("started_at")),
        ended_at=_float(raw.get("ended_at")),
        worker=str(raw.get("worker") or ""),
        detail=str(raw.get("detail") or ""),
        error=str(raw.get("error") or ""),
        steps=steps,
        log=[entry for entry in raw.get("log") or [] if isinstance(entry, dict)],
        diagram=diagram if isinstance(diagram, dict) else None,
    )


def _float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _ms(start: float | None, end: float | None) -> int | None:
    if start is None or end is None:
        return None
    return int(round((end - start) * 1000))


def _count(diagram: dict[str, object] | None, side: str) -> int:
    if not isinstance(diagram, dict):
        return 0
    players = diagram.get(side)
    return len(players) if isinstance(players, list) else 0


def _percentile(values: list[int], share: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(share * (len(ordered) - 1))))
    return ordered[index]
