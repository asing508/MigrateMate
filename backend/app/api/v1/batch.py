"""Batch migration API endpoints.

Every migration is tracked by id in a shared job store (see
``app/services/job_store.py``). Both the GitHub and ZIP-upload flows run in the
background and return a ``migration_id`` immediately; the client polls
``/status/{migration_id}`` for step-by-step progress and downloads the result
via ``/download/{migration_id}``.
"""

import os
import shutil
import tempfile
import uuid
import zipfile

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.services import get_batch_migration_service, get_job_store

router = APIRouter()


# ==================== Schemas ====================

class GitHubMigrateRequest(BaseModel):
    repo_url: str = Field(..., description="GitHub repository URL")
    branch: str = Field(default="main", description="Branch to migrate")
    source_framework: str = Field(default="flask")
    target_framework: str = Field(default="fastapi")


class StartMigrationResponse(BaseModel):
    migration_id: str
    status: str


class LocalMigrateRequest(BaseModel):
    source_dir: str = Field(..., description="Path to source Flask project directory")
    output_dir: str = Field(..., description="Path to output FastAPI project directory")
    source_framework: str = Field(default="flask")
    target_framework: str = Field(default="fastapi")


def _new_migration_id() -> str:
    return uuid.uuid4().hex[:12]


# ==================== GitHub ====================

@router.post("/github", response_model=StartMigrationResponse, status_code=202)
async def migrate_github_repo(request: GitHubMigrateRequest, background_tasks: BackgroundTasks):
    """Start migration of a GitHub repository (runs in the background)."""
    service = get_batch_migration_service()
    store = get_job_store()

    migration_id = _new_migration_id()
    progress = store.create(migration_id, kind="github")

    async def run_migration():
        try:
            result = await service.migrate_github_repo(
                repo_url=request.repo_url,
                branch=request.branch,
                source_framework=request.source_framework,
                target_framework=request.target_framework,
                progress=progress,
            )
            progress.result = result
            progress.zip_path = result.get("zip_path")
        except Exception as e:  # noqa: BLE001 - surfaced to the client via status
            progress.fail_current(str(e))

    background_tasks.add_task(run_migration)
    return StartMigrationResponse(migration_id=migration_id, status="started")


# ==================== Upload ====================

@router.post("/upload", response_model=StartMigrationResponse, status_code=202)
async def migrate_uploaded_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_framework: str = "flask",
    target_framework: str = "fastapi",
):
    """Migrate an uploaded ZIP file (runs in the background)."""
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Only .zip files are supported")

    service = get_batch_migration_service()
    store = get_job_store()
    migration_id = _new_migration_id()

    # Persist the upload to a server-controlled path. The client's filename is
    # never used for the path, so it can't traverse directories.
    temp_path = os.path.join(tempfile.gettempdir(), f"migratemate_upload_{migration_id}.zip")
    try:
        with open(temp_path, "wb") as out:
            shutil.copyfileobj(file.file, out)
    finally:
        await file.close()

    if not zipfile.is_zipfile(temp_path):
        os.remove(temp_path)
        raise HTTPException(400, "Uploaded file is not a valid ZIP archive")

    progress = store.create(migration_id, kind="upload")

    async def run_migration():
        try:
            result = await service.migrate_uploaded_zip(
                zip_path=temp_path,
                source_framework=source_framework,
                target_framework=target_framework,
                progress=progress,
            )
            progress.result = result
            progress.zip_path = result.get("zip_path")
        except Exception as e:  # noqa: BLE001
            progress.fail_current(str(e))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    background_tasks.add_task(run_migration)
    return StartMigrationResponse(migration_id=migration_id, status="started")


# ==================== Status / result / download ====================

@router.get("/status/{migration_id}")
async def get_migration_status(migration_id: str):
    """Get step-by-step status of a migration."""
    progress = get_job_store().get(migration_id)
    if progress is None:
        raise HTTPException(404, "Migration not found")
    return progress.to_dict()


@router.get("/result/{migration_id}")
async def get_migration_result(migration_id: str):
    """Get migration result with a download link."""
    progress = get_job_store().get(migration_id)
    if progress is None:
        raise HTTPException(404, "Migration not found")

    if progress.status != "completed" or not progress.result:
        return {
            "status": progress.status,
            "errors": progress.errors,
        }

    result = progress.result
    return {
        "status": "completed",
        "download_url": f"/api/v1/batch/download/{migration_id}",
        "summary": result.get("summary"),
        "files": result.get("results"),
    }


@router.get("/download/{migration_id}")
async def download_result(migration_id: str):
    """Download the migrated project ZIP for a completed migration.

    The path is looked up server-side from the migration id, so the client can
    never request an arbitrary file (no path traversal).
    """
    progress = get_job_store().get(migration_id)
    if progress is None or not progress.zip_path:
        raise HTTPException(404, "Result not found")

    zip_path = progress.zip_path
    # Defence in depth: the result must live in the system temp dir.
    temp_root = os.path.realpath(tempfile.gettempdir())
    real_zip = os.path.realpath(zip_path)
    if not real_zip.startswith(temp_root + os.sep) or not os.path.exists(real_zip):
        raise HTTPException(404, "File not found")

    return FileResponse(
        path=real_zip,
        filename=os.path.basename(real_zip),
        media_type="application/zip",
    )


# ==================== Local (dev helper) ====================

@router.post("/local")
async def migrate_local_directory(request: LocalMigrateRequest):
    """
    Migrate a local Flask project directory to FastAPI.

    Useful for local development and testing. Runs synchronously.
    """
    if not os.path.exists(request.source_dir):
        raise HTTPException(400, f"Source directory not found: {request.source_dir}")
    if not os.path.isdir(request.source_dir):
        raise HTTPException(400, f"Source path is not a directory: {request.source_dir}")

    try:
        service = get_batch_migration_service()
        result = await service.migrate_local_directory(
            source_dir=request.source_dir,
            output_dir=request.output_dir,
            source_framework=request.source_framework,
            target_framework=request.target_framework,
        )
        return {
            "status": "completed",
            "output_dir": result["output_dir"],
            "summary": result["summary"],
            "files": [
                {"path": r["output_path"], "confidence": r["confidence"]}
                for r in result["results"]
            ],
        }
    except Exception as e:
        raise HTTPException(500, f"Migration failed: {str(e)}")
