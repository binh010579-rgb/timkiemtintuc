"""
Exception handler tập trung cho FastAPI app.

Mọi lỗi tầng hạ tầng (Hugging Face Inference API, Qdrant Cloud) và lỗi
không lường trước được đều đi qua đây, trả về JSON có cấu trúc nhất quán
cho client — KHÔNG rò rỉ traceback nội bộ.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.database.qdrant_client import QdrantConnectionError
from app.services.embedding_service import EmbeddingServiceError
from app.utils.time import utcnow_iso

logger = logging.getLogger(__name__)


def _error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": message, "timestamp": utcnow_iso()},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Gọi 1 lần lúc khởi tạo `app` trong `app/main.py`."""

    @app.exception_handler(EmbeddingServiceError)
    async def embedding_error_handler(request: Request, exc: EmbeddingServiceError):
        logger.error("Embedding service error tại %s: %s", request.url.path, exc)
        return _error_response(
            502, "Không gọi được dịch vụ embedding (Hugging Face Inference API)."
        )

    @app.exception_handler(QdrantConnectionError)
    async def qdrant_error_handler(request: Request, exc: QdrantConnectionError):
        logger.error("Qdrant connection error tại %s: %s", request.url.path, exc)
        return _error_response(503, "Không kết nối được tới Qdrant Cloud.")

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Lỗi không lường trước tại %s", request.url.path)
        return _error_response(500, "Đã xảy ra lỗi nội bộ.")
