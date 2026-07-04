"""
Router: endpoint Search Pipeline (semantic search).

Router chỉ nhận request, gọi `services/search_service.py`, trả JSON.
Toàn bộ logic (gọi HF Inference API, tìm kiếm Qdrant Cloud, hydrate
content) nằm ở service layer — router không biết chi tiết bên trong.
"""

from fastapi import APIRouter, Request

from app.config import SEARCH_CANDIDATE_K, SEARCH_TOP_K
from app.models.search import SemanticSearchQuery, SemanticSearchResultItem
from app.services import search_service

router = APIRouter(tags=["search"])


@router.post("/search", response_model=list[SemanticSearchResultItem])
def semantic_search(request: Request, body: SemanticSearchQuery):
    """
    Search Pipeline:

        User Query -> Cache -> HF Inference API (embedding) -> Qdrant
        Cloud (k-NN, top {SEARCH_CANDIDATE_K}) -> (tuỳ chọn) Re-ranking
        -> Top {SEARCH_TOP_K} -> lấy content đầy đủ từ repository -> JSON.

    Backend KHÔNG load model, KHÔNG build lại vector ở endpoint này. Lỗi
    tầng hạ tầng (HF API, Qdrant) được xử lý bởi exception handler tập
    trung ở `app/core/exceptions.py`, không xử lý riêng ở đây.
    """
    return search_service.search_articles(
        query=body.query,
        embedding_service=request.app.state.embedding_service,
        qdrant_store=request.app.state.qdrant_store,
        news_repository=request.app.state.news_repository,
        top_k=SEARCH_TOP_K,
        candidate_k=SEARCH_CANDIDATE_K,
    )
