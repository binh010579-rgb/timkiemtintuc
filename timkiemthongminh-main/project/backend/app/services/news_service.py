"""
Service layer: business logic cho danh sách tin, danh mục, và tìm kiếm
từ khoá (keyword search) — thao tác trên DataFrame lấy từ
`NewsRepository`. KHÔNG đọc lại CSV, KHÔNG chứa route/HTTP logic.
"""

from __future__ import annotations

import math

import pandas as pd

from app.config import DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT
from app.database.news_repository import strip_accents_lower
from app.models.news import CategoriesList, CategoryItem, KeywordSearchResult, NewsItem, NewsList


def _clamp_limit(limit: int) -> int:
    return max(1, min(limit, MAX_PAGE_LIMIT))


def _clamp_page(page: int) -> int:
    return max(1, page)


def _row_to_news_item(row: pd.Series) -> NewsItem:
    comments = row["so_binh_luan"]
    return NewsItem(
        id=int(row["id"]),
        title=row["tieu_de"],
        publish_date=row["ngay_dang"],
        author=row["tac_gia"],
        source=row["nguon"],
        url=row["link"],
        comments=None if pd.isna(comments) else int(comments),
        summary=row["summary"],
        image=None,
    )


def _paginate_df(df: pd.DataFrame, page: int, limit: int) -> tuple[pd.DataFrame, int, int]:
    page = _clamp_page(page)
    limit = _clamp_limit(limit)
    start = (page - 1) * limit
    return df.iloc[start : start + limit], page, limit


def get_news_list(df: pd.DataFrame, page: int = 1, limit: int = DEFAULT_PAGE_LIMIT) -> NewsList:
    """Danh sách tin mới nhất, sắp xếp theo ngày đăng giảm dần."""
    sorted_df = df.sort_values(by="ngay_dang", ascending=False, kind="stable")
    page_df, page, limit = _paginate_df(sorted_df, page, limit)
    total = len(df)
    return NewsList(
        total=total,
        page=page,
        limit=limit,
        total_pages=max(1, math.ceil(total / limit)),
        items=[_row_to_news_item(row) for _, row in page_df.iterrows()],
    )


def get_featured_news(df: pd.DataFrame, limit: int = 5) -> NewsList:
    """Tin nổi bật: N tin có nhiều bình luận nhất."""
    limit = _clamp_limit(limit)
    featured_df = df.sort_values(by="so_binh_luan", ascending=False, kind="stable").head(limit)
    total = len(df)
    return NewsList(
        total=total,
        page=1,
        limit=limit,
        total_pages=max(1, math.ceil(total / limit)),
        items=[_row_to_news_item(row) for _, row in featured_df.iterrows()],
    )


def get_categories(df: pd.DataFrame) -> CategoriesList:
    """Danh mục = nhóm theo nguồn báo (cột `nguon`)."""
    counts = df["nguon"].fillna("Khác").value_counts()
    items = [CategoryItem(name=str(name), count=int(count)) for name, count in counts.items()]
    return CategoriesList(total=len(items), items=items)


def search_news(
    df: pd.DataFrame, query: str, page: int = 1, limit: int = DEFAULT_PAGE_LIMIT
) -> KeywordSearchResult:
    """Tìm theo tiêu đề + tóm tắt, không phân biệt hoa/thường, không dấu."""
    q = strip_accents_lower(query.strip())

    if not q:
        matched = df.iloc[0:0]
    else:
        mask = df["_search_blob"].str.contains(q, regex=False, na=False)
        matched = df[mask]

    page_df, page, limit = _paginate_df(matched, page, limit)
    total = len(matched)
    return KeywordSearchResult(
        query=query,
        total=total,
        page=page,
        limit=limit,
        total_pages=max(1, math.ceil(total / limit)),
        items=[_row_to_news_item(row) for _, row in page_df.iterrows()],
    )
