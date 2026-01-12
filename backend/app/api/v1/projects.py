"""Project API endpoints."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, Field

from app.core.database import get_db
from app.models import Project, ProjectStatus, FrameworkType

router = APIRouter()


# ==================== Schemas ====================

class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    source_framework: FrameworkType
    target_framework: FrameworkType
    repository_url: Optional[str] = None
    branch: str = "main"


class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    status: Optional[ProjectStatus] = None
    repository_url: Optional[str] = None
    branch: Optional[str] = None


class ProjectResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    source_framework: FrameworkType
    target_framework: FrameworkType
    repository_url: Optional[str]
    branch: str
    status: ProjectStatus
    total_files: int
    total_chunks: int
    
    class Config:
        from_attributes = True


class ProjectListResponse(BaseModel):
    projects: List[ProjectResponse]
    total: int


# ==================== Endpoints ====================

@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(project: ProjectCreate, db: AsyncSession = Depends(get_db)):
    """Create a new migration project."""
    db_project = Project(
        name=project.name,
        description=project.description,
        source_framework=project.source_framework,
        target_framework=project.target_framework,
        repository_url=project.repository_url,
        branch=project.branch,
        status=ProjectStatus.PENDING,
    )
    db.add(db_project)
    await db.commit()
    await db.refresh(db_project)
    return db_project


@router.get("/", response_model=ProjectListResponse)
async def list_projects(skip: int = 0, limit: int = 100, status: Optional[ProjectStatus] = None, db: AsyncSession = Depends(get_db)):
    """List all projects with optional filtering."""
    query = select(Project)
    if status:
        query = query.where(Project.status == status)
    query = query.offset(skip).limit(limit)
    
    result = await db.execute(query)
    projects = result.scalars().all()
    
    # Get total count
    count_query = select(Project)
    if status:
        count_query = count_query.where(Project.status == status)
    count_result = await db.execute(count_query)
    total = len(count_result.scalars().all())
    
    return ProjectListResponse(projects=projects, total=total)


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: int, db: AsyncSession = Depends(get_db)):
    """Get a project by ID."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(project_id: int, project_update: ProjectUpdate, db: AsyncSession = Depends(get_db)):
    """Update a project."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    update_data = project_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(project, key, value)
    
    await db.commit()
    await db.refresh(project)
    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(project_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a project and all associated data."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    await db.delete(project)
    await db.commit()
