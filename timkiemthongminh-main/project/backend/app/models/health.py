"""Schemas (Pydantic) cho GET /health."""

from typing import Optional

from pydantic import BaseModel


class ComponentHealth(BaseModel):
    status: str  # "ok" | "error"
    detail: Optional[str] = None


class HealthStatus(BaseModel):
    status: str  # "ok" | "degraded"
    qdrant: ComponentHealth
    embedding_api: ComponentHealth
    version: str
    timestamp: str
