"""
Router: `GET /health`.

Kiểm tra tình trạng 2 phụ thuộc hạ tầng (Qdrant Cloud, Hugging Face
Inference API) bằng 1 lần thử NHẸ mỗi bên — KHÔNG build/encode lại
vector, KHÔNG load model nào, không đụng tới collection.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.config import APP_VERSION
from app.models.health import ComponentHealth, HealthStatus
from app.utils.time import utcnow_iso

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthStatus)
def health_check(request: Request) -> HealthStatus:
    qdrant_store = request.app.state.qdrant_store
    embedding_service = request.app.state.embedding_service

    qdrant_ok, qdrant_detail = qdrant_store.check_health()
    embedding_ok, embedding_detail = embedding_service.check_health()

    overall_status = "ok" if (qdrant_ok and embedding_ok) else "degraded"

    return HealthStatus(
        status=overall_status,
        qdrant=ComponentHealth(status="ok" if qdrant_ok else "error", detail=qdrant_detail),
        embedding_api=ComponentHealth(
            status="ok" if embedding_ok else "error", detail=embedding_detail
        ),
        version=APP_VERSION,
        timestamp=utcnow_iso(),
    )
