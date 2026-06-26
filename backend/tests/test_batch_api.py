"""API tests for the batch migration endpoints (offline)."""

import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

BASE = "/api/v1/batch"

FLASK_APP = (
    "from flask import Flask, jsonify\n\n"
    "app = Flask(__name__)\n\n"
    '@app.route("/")\n'
    "def index():\n"
    '    return jsonify({"ok": True})\n'
)


def _make_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("app.py", FLASK_APP)
    buf.seek(0)
    return buf.read()


def test_status_unknown_returns_404():
    resp = client.get(f"{BASE}/status/nope")
    assert resp.status_code == 404


def test_download_unknown_returns_404():
    # The path is derived from the id server-side, so arbitrary-file access
    # (the old `?path=` traversal) is impossible by construction.
    resp = client.get(f"{BASE}/download/nope")
    assert resp.status_code == 404


def test_upload_rejects_non_zip():
    resp = client.post(
        f"{BASE}/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400


def test_upload_rejects_invalid_zip():
    resp = client.post(
        f"{BASE}/upload",
        files={"file": ("fake.zip", b"not really a zip", "application/zip")},
    )
    assert resp.status_code == 400


def test_upload_flow_completes_and_downloads():
    # TestClient runs the BackgroundTask synchronously before returning.
    resp = client.post(
        f"{BASE}/upload",
        files={"file": ("project.zip", _make_zip(), "application/zip")},
    )
    assert resp.status_code == 202
    migration_id = resp.json()["migration_id"]

    status = client.get(f"{BASE}/status/{migration_id}").json()
    assert status["status"] == "completed"
    assert status["progress"] == 100.0
    assert [s for s in status["steps"] if s["key"] == "migrate"][0]["state"] == "done"

    result = client.get(f"{BASE}/result/{migration_id}").json()
    assert result["status"] == "completed"
    assert result["download_url"] == f"{BASE}/download/{migration_id}"
    assert result["summary"]["total_files"] == 1

    dl = client.get(result["download_url"])
    assert dl.status_code == 200
    assert dl.headers["content-type"] == "application/zip"
    # Downloaded payload is a real zip containing the migrated app + requirements.
    names = zipfile.ZipFile(io.BytesIO(dl.content)).namelist()
    assert "app.py" in names
    assert "requirements.txt" in names


def test_github_endpoint_starts_background_job(monkeypatch):
    from app.services import get_batch_migration_service

    service = get_batch_migration_service()

    async def fake_clone(*args, progress=None, **kwargs):
        progress.start_step("package")
        progress.complete_step("package")
        progress.status = "completed"
        return {"zip_path": None, "results": [], "summary": {"total_files": 0}}

    monkeypatch.setattr(service, "migrate_github_repo", fake_clone)

    resp = client.post(
        f"{BASE}/github",
        json={"repo_url": "https://github.com/x/y", "branch": "main"},
    )
    assert resp.status_code == 202
    migration_id = resp.json()["migration_id"]
    status = client.get(f"{BASE}/status/{migration_id}").json()
    assert status["status"] == "completed"
