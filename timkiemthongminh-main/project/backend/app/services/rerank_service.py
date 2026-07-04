"""
Bước Re-ranking (TUỲ CHỌN) trong Search Pipeline:

    ... -> Qdrant Search -> Top {SEARCH_CANDIDATE_K} -> [Re-ranking] ->
    Top {SEARCH_TOP_K} -> JSON

Vì kiến trúc này CẤM load model cục bộ (không torch, không
SentenceTransformer/CrossEncoder chạy trên server), implementation mặc
định ở đây là "passthrough": giữ nguyên thứ tự Qdrant đã trả về (đã sắp
xếp giảm dần theo cosine score), chỉ cắt còn top K cuối cùng.

Đây là 1 điểm mở rộng (Protocol) — nếu sau này muốn bật re-ranking thật
sự, cách hợp lệ (không phá quy tắc "không model local") là viết thêm 1
class gọi qua HTTP tới một Inference API cross-encoder (ví dụ Hugging
Face Inference API với model dạng sentence-similarity), rồi truyền class
đó vào `search_service.search_articles(reranker=...)` thay cho
`default_reranker`.
"""

from __future__ import annotations

from typing import Protocol

from app.models.search import SemanticSearchResultItem


class Reranker(Protocol):
    def rerank(
        self, query: str, items: list[SemanticSearchResultItem]
    ) -> list[SemanticSearchResultItem]: ...


class PassthroughReranker:
    """Implementation mặc định: không rerank, chỉ trả nguyên danh sách đầu vào."""

    def rerank(
        self, query: str, items: list[SemanticSearchResultItem]
    ) -> list[SemanticSearchResultItem]:
        return items


# Instance dùng mặc định bởi search_service.py.
default_reranker = PassthroughReranker()
