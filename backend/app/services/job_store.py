"""Per-migration job tracking.

The previous implementation kept a single ``progress`` field on a *singleton*
service, so every migration shared the same state and concurrent runs clobbered
each other. The status endpoint also ignored the ``migration_id`` it was given.

This module replaces that with a proper, self-contained job model:

* Each migration has its own :class:`MigrationProgress` keyed by id.
* Progress is expressed as an ordered list of :class:`Step` objects so the UI can
  render a real checklist ("each step"), not just a fuzzy percentage.
* :class:`MigrationJobStore` owns the lifecycle, is safe to touch from multiple
  coroutines, and evicts old/finished jobs so memory does not grow without bound.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Step model
# ---------------------------------------------------------------------------

class StepState(str, Enum):
    """Lifecycle of a single pipeline step."""
    PENDING = "pending"
    ACTIVE = "active"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Step:
    key: str
    label: str
    state: StepState = StepState.PENDING
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "label": self.label, "state": self.state.value, "detail": self.detail}


# Ordered pipeline. ``fetch`` is relabelled per source (clone vs. extract).
# The weight of each step feeds the overall percentage so the bar moves during
# clone/analyze/package, not only while files are being migrated.
_PIPELINE: List[tuple[str, str, float]] = [
    ("queue", "Queued", 0.0),
    ("fetch", "Fetching source", 0.10),
    ("analyze", "Analyzing project", 0.05),
    ("migrate", "Migrating files", 0.80),
    ("package", "Packaging output", 0.05),
]

_STEP_WEIGHTS = {key: weight for key, _, weight in _PIPELINE}


def _default_steps() -> List[Step]:
    return [Step(key=key, label=label) for key, label, _ in _PIPELINE]


# ---------------------------------------------------------------------------
# Per-migration progress
# ---------------------------------------------------------------------------

@dataclass
class MigrationProgress:
    """All state for one migration, owned by exactly one migration."""

    migration_id: str
    kind: str = "github"  # github | upload | local
    status: str = "queued"  # queued | running | completed | failed

    steps: List[Step] = field(default_factory=_default_steps)

    total_files: int = 0
    processed_files: int = 0
    total_chunks: int = 0
    processed_chunks: int = 0
    current_file: str = ""
    current_chunk: str = ""

    errors: List[str] = field(default_factory=list)

    # Populated when the job finishes.
    result: Optional[Dict[str, Any]] = None
    zip_path: Optional[str] = None

    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # -- step helpers -------------------------------------------------------

    def _step(self, key: str) -> Optional[Step]:
        return next((s for s in self.steps if s.key == key), None)

    @property
    def phase(self) -> str:
        """Key of the step currently active (or the last finished one)."""
        active = next((s for s in self.steps if s.state == StepState.ACTIVE), None)
        if active:
            return active.key
        done = [s for s in self.steps if s.state == StepState.DONE]
        return done[-1].key if done else "queue"

    def start_step(self, key: str, label: Optional[str] = None, detail: str = "") -> None:
        """Mark ``key`` active and every earlier step done."""
        reached = False
        for step in self.steps:
            if step.key == key:
                reached = True
                step.state = StepState.ACTIVE
                if label is not None:
                    step.label = label
                step.detail = detail
            elif not reached:
                if step.state in (StepState.PENDING, StepState.ACTIVE):
                    step.state = StepState.DONE
        self.touch()

    def complete_step(self, key: str, detail: str = "") -> None:
        step = self._step(key)
        if step:
            step.state = StepState.DONE
            if detail:
                step.detail = detail
        self.touch()

    def finish(self) -> None:
        """Close out the job: active steps become done, any never-reached steps
        become skipped (e.g. ``package`` for the local flow), status completed."""
        for step in self.steps:
            if step.state == StepState.ACTIVE:
                step.state = StepState.DONE
            elif step.state == StepState.PENDING:
                step.state = StepState.SKIPPED
        self.status = "completed"
        self.touch()

    def fail_current(self, message: str) -> None:
        """Mark the active step failed and record the error."""
        active = next((s for s in self.steps if s.state == StepState.ACTIVE), None)
        if active is None:
            active = self._step("migrate")
        if active:
            active.state = StepState.FAILED
            active.detail = message
        self.status = "failed"
        if message and message not in self.errors:
            self.errors.append(message)
        self.touch()

    def touch(self) -> None:
        self.updated_at = time.time()

    # -- derived ------------------------------------------------------------

    @property
    def percentage(self) -> float:
        """Weighted completion across the whole pipeline (0-100)."""
        pct = 0.0
        for step in self.steps:
            weight = _STEP_WEIGHTS.get(step.key, 0.0)
            if step.state in (StepState.DONE, StepState.SKIPPED):
                pct += weight
            elif step.state == StepState.ACTIVE and step.key == "migrate":
                # Scale the (large) migrate weight by chunk-level progress.
                if self.total_chunks > 0:
                    pct += weight * (self.processed_chunks / self.total_chunks)
                elif self.total_files > 0:
                    pct += weight * (self.processed_files / self.total_files)
        return round(min(pct, 1.0) * 100, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "migration_id": self.migration_id,
            "kind": self.kind,
            "status": self.status,
            "phase": self.phase,
            "progress": self.percentage,
            "steps": [s.to_dict() for s in self.steps],
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "total_chunks": self.total_chunks,
            "processed_chunks": self.processed_chunks,
            "current_file": self.current_file,
            "current_chunk": self.current_chunk,
            "errors": list(self.errors),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class MigrationJobStore:
    """In-memory registry of migrations.

    Good enough for a single-process dev tool. For a multi-worker deployment
    this is the seam where you would swap in Redis.
    """

    def __init__(self, ttl_seconds: int = 3600, max_jobs: int = 200):
        self._jobs: Dict[str, MigrationProgress] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
        self._max_jobs = max_jobs

    def create(self, migration_id: str, kind: str = "github") -> MigrationProgress:
        with self._lock:
            self._evict_locked()
            progress = MigrationProgress(migration_id=migration_id, kind=kind, status="queued")
            self._jobs[migration_id] = progress
            return progress

    def get(self, migration_id: str) -> Optional[MigrationProgress]:
        with self._lock:
            return self._jobs.get(migration_id)

    def _evict_locked(self) -> None:
        now = time.time()
        # Drop jobs that finished long ago.
        stale = [
            jid for jid, job in self._jobs.items()
            if job.status in ("completed", "failed") and (now - job.updated_at) > self._ttl
        ]
        for jid in stale:
            self._jobs.pop(jid, None)

        # Hard cap: drop the oldest finished jobs first, then oldest of any kind.
        if len(self._jobs) >= self._max_jobs:
            ordered = sorted(self._jobs.values(), key=lambda j: j.updated_at)
            for job in ordered:
                if len(self._jobs) < self._max_jobs:
                    break
                self._jobs.pop(job.migration_id, None)


_store: Optional[MigrationJobStore] = None


def get_job_store() -> MigrationJobStore:
    global _store
    if _store is None:
        _store = MigrationJobStore()
    return _store
