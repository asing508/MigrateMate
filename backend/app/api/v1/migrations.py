"""Migration job API endpoints."""

from typing import List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, Field

from app.core.database import get_db
from app.models import Project, MigrationJob, JobStatus, CodeChunk, MigrationStatus

router = APIRouter()


# ==================== Schemas ====================

class JobCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    config: dict = Field(default_factory=dict)


class JobResponse(BaseModel):
    id: int
    project_id: int
    name: str
    status: JobStatus
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    total_files: int
    processed_files: int
    confidence_score: Optional[float]
    error_message: Optional[str]
    
    class Config:
        from_attributes = True


class JobDetailResponse(JobResponse):
    config: dict
    logs: Optional[str]
    test_results: Optional[dict]
    output_files: Optional[dict]


class JobListResponse(BaseModel):
    jobs: List[JobResponse]
    total: int


class ChunkResponse(BaseModel):
    id: int
    file_path: str
    name: Optional[str]
    chunk_type: str
    start_line: int
    end_line: int
    migration_status: MigrationStatus
    migration_confidence: Optional[float]
    
    class Config:
        from_attributes = True


class MigrationProgressResponse(BaseModel):
    job_id: int
    status: JobStatus
    progress_percentage: float
    total_chunks: int
    completed_chunks: int
    failed_chunks: int
    current_chunk: Optional[str]


# ==================== Endpoints ====================

@router.post("/{project_id}/jobs", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(project_id: int, job: JobCreate, db: AsyncSession = Depends(get_db)):
    """Create a new migration job for a project."""
    # Verify project exists
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    db_job = MigrationJob(
        project_id=project_id,
        name=job.name,
        config=job.config,
        status=JobStatus.QUEUED,
    )
    db.add(db_job)
    await db.commit()
    await db.refresh(db_job)
    return db_job


@router.get("/{project_id}/jobs", response_model=JobListResponse)
async def list_jobs(project_id: int, skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)):
    """List all migration jobs for a project."""
    query = select(MigrationJob).where(MigrationJob.project_id == project_id).offset(skip).limit(limit)
    result = await db.execute(query)
    jobs = result.scalars().all()
    return JobListResponse(jobs=jobs, total=len(jobs))


@router.get("/{project_id}/jobs/{job_id}", response_model=JobDetailResponse)
async def get_job(project_id: int, job_id: int, db: AsyncSession = Depends(get_db)):
    """Get a specific migration job."""
    result = await db.execute(
        select(MigrationJob).where(MigrationJob.id == job_id, MigrationJob.project_id == project_id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/{project_id}/jobs/{job_id}/start", response_model=JobResponse)
async def start_job(project_id: int, job_id: int, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)):
    """Start a migration job."""
    result = await db.execute(
        select(MigrationJob).where(MigrationJob.id == job_id, MigrationJob.project_id == project_id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status not in [JobStatus.QUEUED, JobStatus.FAILED]:
        raise HTTPException(status_code=400, detail=f"Cannot start job with status {job.status}")
    
    job.status = JobStatus.PLANNING
    job.started_at = datetime.utcnow()
    await db.commit()
    await db.refresh(job)
    
    # TODO: Add background task to run migration
    # background_tasks.add_task(run_migration, job_id, project_id)
    
    return job


@router.post("/{project_id}/jobs/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(project_id: int, job_id: int, db: AsyncSession = Depends(get_db)):
    """Cancel a running migration job."""
    result = await db.execute(
        select(MigrationJob).where(MigrationJob.id == job_id, MigrationJob.project_id == project_id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status in [JobStatus.COMPLETED, JobStatus.CANCELLED]:
        raise HTTPException(status_code=400, detail=f"Cannot cancel job with status {job.status}")
    
    job.status = JobStatus.CANCELLED
    job.completed_at = datetime.utcnow()
    await db.commit()
    await db.refresh(job)
    return job


@router.get("/{project_id}/jobs/{job_id}/progress", response_model=MigrationProgressResponse)
async def get_job_progress(project_id: int, job_id: int, db: AsyncSession = Depends(get_db)):
    """Get real-time progress of a migration job."""
    result = await db.execute(
        select(MigrationJob).where(MigrationJob.id == job_id, MigrationJob.project_id == project_id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Count chunks by status
    chunks_result = await db.execute(
        select(CodeChunk).where(CodeChunk.project_id == project_id)
    )
    chunks = chunks_result.scalars().all()
    
    total = len(chunks)
    completed = sum(1 for c in chunks if c.migration_status == MigrationStatus.MIGRATED)
    failed = sum(1 for c in chunks if c.migration_status == MigrationStatus.FAILED)
    
    progress = (completed / total * 100) if total > 0 else 0
    
    return MigrationProgressResponse(
        job_id=job_id,
        status=job.status,
        progress_percentage=progress,
        total_chunks=total,
        completed_chunks=completed,
        failed_chunks=failed,
        current_chunk=None,
    )


@router.get("/{project_id}/chunks", response_model=List[ChunkResponse])
async def list_chunks(project_id: int, status: Optional[MigrationStatus] = None, skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)):
    """List code chunks for a project."""
    query = select(CodeChunk).where(CodeChunk.project_id == project_id)
    if status:
        query = query.where(CodeChunk.migration_status == status)
    query = query.offset(skip).limit(limit)
    
    result = await db.execute(query)
    chunks = result.scalars().all()
    return chunks
