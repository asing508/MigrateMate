"""End-to-end migration tests that run fully offline (pattern-based agent)."""

import os

import pytest

from app.services.migration_service import get_batch_migration_service
from app.services.job_store import MigrationProgress, StepState
from app.agents.migration_agent import get_migration_agent


FLASK_APP = '''from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/")
def index():
    return jsonify({"msg": "hello"})


@app.route("/echo", methods=["POST"])
def echo():
    data = request.json
    return jsonify(data)
'''

UTIL = '''def add(a, b):
    return a + b
'''


# ---------------------------------------------------------------------------
# Agent (pattern-based fallback)
# ---------------------------------------------------------------------------

async def test_agent_pattern_migration_produces_valid_fastapi():
    agent = get_migration_agent()
    agent._model = None  # force pattern-based path

    result = await agent.migrate_chunk(
        project_id=0, job_id=0,
        source_framework="flask", target_framework="fastapi",
        chunk_id=0,
        chunk_content=FLASK_APP,
        chunk_name="app",
        chunk_type="misc",
    )

    assert result["status"] == "completed"
    code = result["migrated_code"]
    compile(code, "<migrated>", "exec")          # valid Python
    assert "from flask import" not in code        # flask import gone
    assert "FastAPI()" in code                     # app converted
    assert "jsonify(" not in code                  # jsonify stripped


async def test_agent_injects_request_param_when_body_uses_request():
    agent = get_migration_agent()
    agent._model = None
    chunk = 'def echo():\n    data = request.json\n    return data\n'
    result = await agent.migrate_chunk(
        project_id=0, job_id=0,
        source_framework="flask", target_framework="fastapi",
        chunk_id=0, chunk_content=chunk, chunk_name="echo", chunk_type="function",
    )
    code = result["migrated_code"]
    compile(code, "<m>", "exec")
    assert "request: Request" in code
    assert "await request.json()" in code


# ---------------------------------------------------------------------------
# Full local-directory pipeline
# ---------------------------------------------------------------------------

@pytest.fixture
def flask_project(tmp_path):
    (tmp_path / "app.py").write_text(FLASK_APP, encoding="utf-8")
    (tmp_path / "util.py").write_text(UTIL, encoding="utf-8")
    return tmp_path


async def test_local_migration_completes_with_step_progress(flask_project, tmp_path):
    service = get_batch_migration_service()
    out_dir = str(tmp_path / "out")
    progress = MigrationProgress(migration_id="test", kind="local")

    result = await service.migrate_local_directory(
        source_dir=str(flask_project),
        output_dir=out_dir,
        progress=progress,
    )

    # Progress / steps
    assert progress.status == "completed"
    assert progress.percentage == 100.0
    migrate_step = next(s for s in progress.steps if s.key == "migrate")
    assert migrate_step.state == StepState.DONE

    # Summary
    summary = result["summary"]
    assert summary["total_files"] == 2
    assert summary["files_migrated"] >= 1
    assert 0.0 <= summary["average_confidence"] <= 1.0

    # Output is valid Python and de-Flasked
    app_out = os.path.join(out_dir, "app.py")
    assert os.path.exists(app_out)
    migrated = open(app_out, encoding="utf-8").read()
    compile(migrated, app_out, "exec")
    assert "from flask import" not in migrated
    assert "FastAPI()" in migrated

    # Non-Flask file is copied through unchanged.
    util_out = open(os.path.join(out_dir, "util.py"), encoding="utf-8").read()
    assert util_out == UTIL
