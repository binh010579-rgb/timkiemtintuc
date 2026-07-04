"""
Router: các endpoint HTTP cho resource "news" (danh sách, danh mục, tìm
kiếm từ khoá). Router KHÔNG chứa business logic — chỉ nhận tham số từ
request, gọi xuống `services/news_service.py`, trả kết quả.
"""

from fastapi import APIRouter, Query, Request

from app.models.news import CategoriesList, KeywordSearchResult, NewsList
from app.services import news_service

router = APIRouter(prefix="/api", tags=["news"])


def _get_news_df(request: Request):
    """Lấy DataFrame đã nạp sẵn trong RAM từ NewsRepository (xem main.py)."""
    return request.app.state.news_repository.ensure_loaded()


@router.get("/news", response_model=NewsList)
def list_news(
    request: Request,
    page: int = Query(1, ge=1, description="Số trang, bắt đầu từ 1"),
    limit: int = Query(10, ge=1, le=100, description="Số tin mỗi trang"),
):
    """Danh sách tin mới nhất, có phân trang."""
    df = _get_news_df(request)
    return news_service.get_news_list(df, page=page, limit=limit)


@router.get("/news/featured", response_model=NewsList)
def featured_news(
    request: Request,
    limit: int = Query(5, ge=1, le=100, description="Số tin nổi bật cần lấy"),
):
    """Danh sách tin nổi bật (nhiều bình luận nhất)."""
    df = _get_news_df(request)
    return news_service.get_featured_news(df, limit=limit)


@router.get("/categories", response_model=CategoriesList)
def categories(request: Request):
    """Danh mục tin tức, nhóm theo nguồn báo."""
    df = _get_news_df(request)
    return news_service.get_categories(df)


@router.get("/search", response_model=KeywordSearchResult)
def keyword_search(
    request: Request,
    q: str = Query("", description="Từ khoá tìm kiếm"),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
):
    """Tìm kiếm tin tức theo từ khoá (tiêu đề + tóm tắt) — khác semantic search."""
    df = _get_news_df(request)
    return news_service.search_news(df, query=q, page=page, limit=limit)
