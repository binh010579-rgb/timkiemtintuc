"""
Schemas (Pydantic) cho resource "news" — danh sách, phân trang, danh mục.

Được thiết kế để khớp 1-1 với `frontend/src/types/news.ts`, để frontend
không cần sửa logic xử lý response khi backend đổi kiến trúc nội bộ.
"""

from typing import List, Optional

from pydantic import BaseModel


class NewsItem(BaseModel):
    id: int
    title: Optional[str] = None
    publish_date: Optional[str] = None
    author: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    comments: Optional[int] = None
    summary: Optional[str] = None
    image: Optional[str] = None


class NewsList(BaseModel):
    total: int
    page: int
    limit: int
    total_pages: int
    items: List[NewsItem]


class CategoryItem(BaseModel):
    name: str
    count: int


class CategoriesList(BaseModel):
    total: int
    items: List[CategoryItem]


class KeywordSearchResult(BaseModel):
    """Kết quả tìm kiếm theo từ khoá (GET /api/search) — khác semantic search."""

    query: str
    total: int
    page: int
    limit: int
    total_pages: int
    items: List[NewsItem]
