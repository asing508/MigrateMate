"""API v1 routers."""

from app.api.v1.projects import router as projects_router
from app.api.v1.migrations import router as migrations_router
from app.api.v1.ai import router as ai_router
from app.api.v1.batch import router as batch_router

__all__ = ["projects_router", "migrations_router", "ai_router", "batch_router"]
