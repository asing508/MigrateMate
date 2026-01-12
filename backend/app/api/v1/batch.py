"""Batch migration API endpoints."""

import os
import shutil
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.services import get_batch_migration_service

router = APIRouter()


class GitHubMigrateRequest(BaseModel):
    repo_url: str = Field(..., description="GitHub repository URL")
    branch: str = Field(default="main", description="Branch to migrate")
    source_framework: str = Field(default="flask")
    target_framework: str = Field(default="fastapi")


class MigrationStatusResponse(BaseModel):
    status: str
    progress: float
    current_file: str
    current_chunk: str
    total_files: int
    processed_files: int
    errors: list


# Store for active migrations (in production, use Redis)
active_migrations = {}


@router.post("/github")
async def migrate_github_repo(request: GitHubMigrateRequest, background_tasks: BackgroundTasks):
    """Start migration of a GitHub repository."""
    service = get_batch_migration_service()
    
    # Generate migration ID
    import uuid
    migration_id = str(uuid.uuid4())[:8]
    
    async def run_migration():
        try:
            result = await service.migrate_github_repo(
                repo_url=request.repo_url,
                branch=request.branch,
                source_framework=request.source_framework,
                target_framework=request.target_framework
            )
            active_migrations[migration_id] = {
                "status": "completed",
                "result": result
            }
        except Exception as e:
            active_migrations[migration_id] = {
                "status": "failed",
                "error": str(e)
            }
    
    active_migrations[migration_id] = {"status": "starting"}
    background_tasks.add_task(run_migration)
    
    return {"migration_id": migration_id, "status": "started"}


@router.post("/upload")
async def migrate_uploaded_file(
    file: UploadFile = File(...),
    source_framework: str = "flask",
    target_framework: str = "fastapi"
):
    """Migrate an uploaded ZIP file."""
    if not file.filename.endswith('.zip'):
        raise HTTPException(400, "Only ZIP files are supported")
    
    # Save uploaded file
    import tempfile
    temp_path = os.path.join(tempfile.gettempdir(), file.filename)
    with open(temp_path, 'wb') as f:
        content = await file.read()
        f.write(content)
    
    try:
        service = get_batch_migration_service()
        result = await service.migrate_uploaded_zip(
            zip_path=temp_path,
            source_framework=source_framework,
            target_framework=target_framework
        )
        
        return {
            "status": "completed",
            "download_url": f"/api/v1/batch/download?path={result['zip_path']}",
            "summary": result['summary']
        }
    finally:
        os.remove(temp_path)


@router.get("/status/{migration_id}")
async def get_migration_status(migration_id: str):
    """Get status of a migration."""
    if migration_id not in active_migrations:
        raise HTTPException(404, "Migration not found")
    
    migration = active_migrations[migration_id]
    service = get_batch_migration_service()
    progress = service.progress
    
    if progress:
        return MigrationStatusResponse(
            status=progress.status,
            progress=progress.percentage,
            current_file=progress.current_file,
            current_chunk=progress.current_chunk,
            total_files=progress.total_files,
            processed_files=progress.processed_files,
            errors=progress.errors
        )
    
    return migration


@router.get("/download")
async def download_result(path: str):
    """Download migrated project ZIP."""
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    
    return FileResponse(
        path=path,
        filename=os.path.basename(path),
        media_type="application/zip"
    )


@router.get("/result/{migration_id}")
async def get_migration_result(migration_id: str):
    """Get migration result with download link."""
    if migration_id not in active_migrations:
        raise HTTPException(404, "Migration not found")
    
    migration = active_migrations[migration_id]
    
    if migration.get("status") != "completed":
        return {"status": migration.get("status"), "error": migration.get("error")}
    
    result = migration.get("result", {})
    return {
        "status": "completed",
        "download_url": f"/api/v1/batch/download?path={result.get('zip_path', '')}",
        "summary": result.get("summary"),
        "files": result.get("results")
    }


class LocalMigrateRequest(BaseModel):
    source_dir: str = Field(..., description="Path to source Flask project directory")
    output_dir: str = Field(..., description="Path to output FastAPI project directory")
    source_framework: str = Field(default="flask")
    target_framework: str = Field(default="fastapi")


@router.post("/local")
async def migrate_local_directory(request: LocalMigrateRequest):
    """
    Migrate a local Flask project directory to FastAPI.
    
    This endpoint is useful for local development and testing.
    """
    # Validate directories exist
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
            target_framework=request.target_framework
        )
        
        return {
            "status": "completed",
            "output_dir": result['output_dir'],
            "summary": result['summary'],
            "files": [{"path": r['output_path'], "confidence": r['confidence']} for r in result['results']]
        }
    except Exception as e:
        raise HTTPException(500, f"Migration failed: {str(e)}")
