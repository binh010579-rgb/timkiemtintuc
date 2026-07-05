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

import logging

import pandas as pd

from app.config import SEARCH_CANDIDATE_K, SEARCH_SCORE_THRESHOLD, SEARCH_TOP_K
from app.core.cache import QueryEmbeddingCache, query_embedding_cache
from app.database.news_repository import NewsRepository
from app.database.qdrant_client import QdrantVectorStore
from app.models.search import SemanticSearchResultItem
from app.services.embedding_service import EmbeddingService
from app.services.rerank_service import Reranker, default_reranker

logger = logging.getLogger(__name__)


def _row_to_result_item(
    row: pd.Series, score: float, content_by_id: dict[int, str | None]
) -> SemanticSearchResultItem:
    """
    `content` là nội dung ĐẦY ĐỦ của bài báo, KHÔNG phải `summary`. Đây là
    phần dùng làm context khi gửi cho AI.

    Với kiến trúc hiện tại (Postgres, dataset 50k+ dòng), `content` KHÔNG
    còn nằm sẵn trong `row` (DataFrame chỉ giữ metadata trong RAM) — phải
    lấy từ `content_by_id`, dict đã được fetch riêng theo đúng top-k id
    (xem `NewsRepository.get_content_by_ids()`), tránh phải giữ content
    của toàn bộ dataset trong RAM.
    """
    article_id = int(row["id"])
    return SemanticSearchResultItem(
        id=article_id,
        title=row["tieu_de"],
        summary=row["summary"],
        content=content_by_id.get(article_id),
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
        # Log để TINH CHỈNH score_threshold dựa trên dữ liệu thật, thay vì
        # đoán mò: gọi lại KHÔNG áp threshold (chỉ lấy 1 điểm) để biết điểm
        # cao nhất thực tế của query này là bao nhiêu — nếu nó chỉ nhỉnh
        # hơn 1 chút so với threshold đang đặt, có thể threshold hơi chặt;
        # nếu nó quá thấp, xác nhận query thật sự không liên quan dataset.
        if score_threshold is not None:
            probe = qdrant_store.search(query_vector, top_k=1, score_threshold=None)
            if probe:
                logger.info(
                    "Query %r: 0 kết quả đạt threshold=%.3f (điểm cao nhất "
                    "thực tế trong dataset chỉ %.3f).",
                    query, score_threshold, probe[0][1],
                )
            else:
                logger.info(
                    "Query %r: Qdrant collection rỗng, không có match nào.", query
                )
        return []

    logger.info(
        "Query %r: %d candidate qua threshold=%s — score min=%.3f, max=%.3f.",
        query,
        len(hits),
        score_threshold,
        min(score for _, score in hits),
        max(score for _, score in hits),
    )

    ids = [article_id for article_id, _ in hits]
    scores = {article_id: score for article_id, score in hits}

    rows = news_repository.get_by_ids(ids)
    # Fetch content ĐẦY ĐỦ chỉ cho đúng candidate_k bài này (thường 10-30
    # bài) — không bao giờ toàn bộ dataset. Xem docstring _row_to_result_item.
    content_by_id = news_repository.get_content_by_ids(ids)
    items = [
        _row_to_result_item(row, scores[int(row["id"])], content_by_id)
        for row in rows
    ]

    items = reranker.rerank(query, items)
    return items[:top_k]
