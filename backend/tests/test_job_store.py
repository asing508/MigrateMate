"""Tests for the per-migration job store and step model."""

from app.services.job_store import (
    MigrationJobStore, MigrationProgress, StepState,
)


def test_steps_start_pending_and_progress_is_zero():
    p = MigrationProgress(migration_id="a")
    assert p.status == "queued"
    assert p.percentage == 0.0
    assert all(s.state == StepState.PENDING for s in p.steps)


def test_start_step_marks_earlier_steps_done():
    p = MigrationProgress(migration_id="a")
    p.start_step("migrate")
    states = {s.key: s.state for s in p.steps}
    assert states["queue"] == StepState.DONE
    assert states["fetch"] == StepState.DONE
    assert states["analyze"] == StepState.DONE
    assert states["migrate"] == StepState.ACTIVE
    assert states["package"] == StepState.PENDING
    assert p.phase == "migrate"


def test_percentage_scales_with_chunk_progress():
    p = MigrationProgress(migration_id="a")
    p.start_step("migrate")
    p.total_chunks = 10
    p.processed_chunks = 5
    # queue(0) + fetch(.10) + analyze(.05) done = 15%, plus half of migrate(.80) = 40% -> 55%
    assert p.percentage == 55.0


def test_completion_reaches_100():
    p = MigrationProgress(migration_id="a")
    p.start_step("package")
    p.complete_step("package")
    p.status = "completed"
    assert p.percentage == 100.0


def test_fail_current_records_error_and_status():
    p = MigrationProgress(migration_id="a")
    p.start_step("migrate")
    p.fail_current("boom")
    assert p.status == "failed"
    assert "boom" in p.errors
    failed = [s for s in p.steps if s.state == StepState.FAILED]
    assert failed and failed[0].key == "migrate"


def test_store_isolates_jobs():
    store = MigrationJobStore()
    a = store.create("a", kind="github")
    b = store.create("b", kind="upload")
    a.start_step("migrate")
    a.total_chunks = 4
    a.processed_chunks = 4
    # b is untouched — proves migrations no longer share state.
    assert b.percentage == 0.0
    assert store.get("a") is a
    assert store.get("b") is b
    assert store.get("missing") is None


def test_store_evicts_when_over_capacity():
    store = MigrationJobStore(ttl_seconds=0, max_jobs=3)
    for i in range(5):
        job = store.create(str(i))
        job.status = "completed"
    # Never exceeds the cap.
    assert len(store._jobs) <= 3


def test_to_dict_shape():
    p = MigrationProgress(migration_id="a", kind="github")
    p.start_step("migrate")
    d = p.to_dict()
    assert d["migration_id"] == "a"
    assert d["phase"] == "migrate"
    assert isinstance(d["steps"], list)
    assert d["steps"][0]["key"] == "queue"
    assert {"key", "label", "state", "detail"} <= set(d["steps"][0].keys())
