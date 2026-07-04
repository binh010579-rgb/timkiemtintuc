"""
Entrypoint FastAPI — kiến trúc cloud production.

Khi khởi động, server CHỈ:
1. Nạp dữ liệu bài báo (NewsRepository) vào RAM — vẫn cần để phục vụ
   danh sách/danh mục/tìm kiếm từ khoá và để hydrate content sau khi
   Qdrant trả kết quả semantic search.
2. Kết nối tới Qdrant Cloud và kiểm tra collection đã tồn tại (log cảnh
   báo nếu rỗng — KHÔNG tự build vector).
3. Khởi tạo EmbeddingService — đây chỉ là 1 HTTP client trỏ tới Hugging
   Face Inference API, KHÔNG load model, KHÔNG tốn RAM/GPU.

TUYỆT ĐỐI KHÔNG:
- Load SentenceTransformer/torch.
- Encode lại toàn bộ dữ liệu bài báo.
- Ghi/đọc vector ở đâu khác ngoài Qdrant Cloud.

Việc sinh vector cho toàn bộ dữ liệu là trách nhiệm của script độc lập
`build_vectors.py`, chạy OFFLINE, tách biệt hoàn toàn khỏi vòng đời server.

Chạy server (từ thư mục backend/):
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health, news, search
from app.config import ALLOWED_ORIGINS, APP_VERSION, QDRANT_VECTOR_SIZE
from app.core.exceptions import register_exception_handlers
from app.core.logging import setup_logging
from app.database.postgres_news_repository import news_repository
from app.database.qdrant_client import qdrant_store
from app.middleware.logging_middleware import RequestLoggingMiddleware
from app.services.embedding_service import embedding_service

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Dữ liệu bài báo: đọc 1 lần, giữ trong RAM ---
    news_repository.load()
    app.state.news_repository = news_repository
    logger.info("Đã nạp %d bài báo vào RAM.", news_repository.count())

    # --- Qdrant Cloud: connect (có retry) rồi ensure_collection.
    # ensure_collection CHỈ tạo collection nếu chưa tồn tại (rỗng, không
    # vector); nếu đã tồn tại thì chỉ kết nối, KHÔNG đụng dữ liệu cũ.
    # KHÔNG có bước nào ở đây encode/build lại vector. ---
    qdrant_store.connect()
    qdrant_store.ensure_collection(vector_size=QDRANT_VECTOR_SIZE)

    count = qdrant_store.count()
    if count == 0:
        logger.warning(
            "Qdrant Cloud collection chưa có vector nào. Chạy "
            "`python build_vectors.py` (offline, 1 lần) để sinh embedding "
            "cho toàn bộ dữ liệu trước khi dùng POST /search."
        )
    else:
        logger.info("Qdrant Cloud sẵn sàng với %d vector.", count)
    app.state.qdrant_store = qdrant_store

    # --- Embedding service: chỉ là HTTP client, không load model ---
    app.state.embedding_service = embedding_service
    logger.info("EmbeddingService sẵn sàng (Hugging Face Inference API, không model local).")

    yield  # server phục vụ request trong khoảng thời gian này

    # --- Shutdown: không có tài nguyên gì cần dọn ---


app = FastAPI(
    title="News Semantic Search API (Cloud)",
    description=(
        "Backend stateless: dữ liệu bài báo giữ trong RAM, embedding sinh "
        "qua Hugging Face Inference API (BAAI/bge-m3), vector lưu trên "
        "Qdrant Cloud. Không load model, không torch, không build lại "
        "vector lúc khởi động."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

register_exception_handlers(app)

app.include_router(health.router)
app.include_router(news.router)
app.include_router(search.router)


@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "message": "News Semantic Search API đang chạy."}
