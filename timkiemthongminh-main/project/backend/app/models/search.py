"""
Schemas (Pydantic) cho Search Pipeline (semantic search, POST /search).
"""

from typing import Optional

from pydantic import BaseModel


class SemanticSearchQuery(BaseModel):
    """Body của POST /search."""

    query: str


class SemanticSearchResultItem(BaseModel):
    """
    Một kết quả của Search Pipeline.

    - `summary`: phần đã được embed, chỉ phục vụ hiển thị nhanh.
    - `content`: nội dung ĐẦY ĐỦ của bài báo, lấy từ repository theo `id`
      sau khi Qdrant Cloud trả kết quả — dùng làm context khi gửi cho AI,
      KHÔNG dùng `summary` cho việc đó.
    """

    id: int
    title: Optional[str] = None
    summary: Optional[str] = None
    content: Optional[str] = None
    url: Optional[str] = None
    image: Optional[str] = None
    date: Optional[str] = None
    source: Optional[str] = None
    score: float
