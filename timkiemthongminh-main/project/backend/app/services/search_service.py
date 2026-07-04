"""
Search Pipeline (semantic search) — service điều phối, tách hoàn toàn
khỏi router (`app/api/search.py`).

Luồng DUY NHẤT của pipeline này, đúng yêu cầu kiến trúc cloud:

    User Query
      -> QueryEmbeddingCache.get()         (TTL 30 phút, RAM-only)
      -> EmbeddingService.embed_query()    (cache miss -> gọi Hugging Face
                                             Inference API, KHÔNG model local)
      -> QdrantVectorStore.search()        (k-NN trên Qdrant Cloud, lấy
                                             SEARCH_CANDIDATE_K ứng viên)
      -> Reranker.rerank()                 (tuỳ chọn — mặc định passthrough,
                                             xem app/services/rerank_service.py)
      -> Top SEARCH_TOP_K
      -> NewsRepository.get_by_ids()       (lấy content đầy đủ)
      -> SemanticSearchResultItem[]        (trả JSON)

Backend KHÔNG build/rebuild vector ở bước nào trong luồng này — vector
của toàn bộ bài báo đã được sinh sẵn và lưu trên Qdrant Cloud từ trước
bởi script độc lập `build_vectors.py` (chạy offline, không phải một phần
của server request-response).
"""

from __future__ import annotations

import pandas as pd

from app.config import SEARCH_CANDIDATE_K, SEARCH_SCORE_THRESHOLD, SEARCH_TOP_K
from app.core.cache import QueryEmbeddingCache, query_embedding_cache
from app.database.news_repository import NewsRepository
from app.database.qdrant_client import QdrantVectorStore
from app.models.search import SemanticSearchResultItem
from app.services.embedding_service import EmbeddingService
from app.services.rerank_service import Reranker, default_reranker


def _row_to_result_item(row: pd.Series, score: float) -> SemanticSearchResultItem:
    """
    `content` lấy từ cột `noi_dung` — nội dung ĐẦY ĐỦ của bài báo, KHÔNG
    phải `summary`. Đây là phần dùng làm context khi gửi cho AI.
    """
    return SemanticSearchResultItem(
        id=int(row["id"]),
        title=row["tieu_de"],
        summary=row["summary"],
        content=row.get("noi_dung"),
        url=row["link"],
        image=None,
        date=row["ngay_dang"],
        source=row["nguon"],
        score=score,
    )


def search_articles(
    query: str,
    embedding_service: EmbeddingService,
    qdrant_store: QdrantVectorStore,
    news_repository: NewsRepository,
    top_k: int = SEARCH_TOP_K,
    candidate_k: int = SEARCH_CANDIDATE_K,
    score_threshold: float | None = SEARCH_SCORE_THRESHOLD,
    cache: QueryEmbeddingCache = query_embedding_cache,
    reranker: Reranker = default_reranker,
) -> list[SemanticSearchResultItem]:
    """
    Thực thi Search Pipeline. Trả về danh sách rỗng (không lỗi) nếu:
    - Query rỗng/toàn khoảng trắng — không gọi cache/HF API/Qdrant.
    - Qdrant không có kết quả nào đạt `score_threshold` (hoặc collection
      chưa có vector).
    """
    if not query or not query.strip():
        return []

    # --- Cache: nếu query giống hệt đã hỏi trong 30 phút gần đây, dùng
    # lại vector cũ, KHÔNG gọi lại Hugging Face Inference API. ---
    query_vector = cache.get(query)
    if query_vector is None:
        query_vector = embedding_service.embed_query(query)
        cache.set(query, query_vector)

    # Lấy candidate_k ứng viên (mặc định 30) để bước rerank (nếu bật) có
    # đủ dữ liệu để sắp xếp lại trước khi cắt còn top_k (mặc định 10).
    hits = qdrant_store.search(
        query_vector, top_k=candidate_k, score_threshold=score_threshold
    )  # [(article_id, score), ...]

    if not hits:
        return []

    ids = [article_id for article_id, _ in hits]
    scores = {article_id: score for article_id, score in hits}

    rows = news_repository.get_by_ids(ids)
    items = [_row_to_result_item(row, scores[int(row["id"])]) for row in rows]

    items = reranker.rerank(query, items)
    return items[:top_k]
