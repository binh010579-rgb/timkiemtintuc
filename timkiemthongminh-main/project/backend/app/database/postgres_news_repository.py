"""
Repository đọc dữ liệu bài báo từ Postgres (Neon), thay cho
`NewsRepository` (bản CSV) trong `news_repository.py`.

Kế thừa TOÀN BỘ logic xử lý chung (sinh id hash ổn định từ URL, chuẩn hoá
kiểu dữ liệu, tạo `_search_blob`) từ `NewsRepository._finalize()` — chỉ
override đúng phần đọc dữ liệu thô (`load()`). Nhờ vậy `id` sinh ra vẫn
giống hệt bản CSV cũ (cùng là hash(url)), nên KHÔNG làm lệch ánh xạ với
vector đã/sẽ upsert lên Qdrant Cloud (xem build_vectors.py).

QUAN TRỌNG: cột `id SERIAL` của bảng Postgres KHÔNG được dùng làm `id`
public — nó chỉ là khoá kỹ thuật nội bộ của bảng SQL. `id` mà backend trả
ra ngoài (và dùng làm khoá Qdrant) luôn là `stable_id_from_url(link)`.
"""

from __future__ import annotations

import logging

import pandas as pd

from app.config import DATABASE_URL
from app.database.news_repository import NewsRepository

logger = logging.getLogger(__name__)

# Ánh xạ tên cột trong bảng Postgres -> tên cột gốc mà phần còn lại của
# backend (services/api) đang dùng (giữ nguyên để không phải sửa gì khác).
_COLUMN_MAP = {
    "source": "nguon",
    "title": "tieu_de",
    "publish_date": "ngay_dang",
    "author": "tac_gia",
    "content": "noi_dung",
    "summary": "summary",
    "comments": "so_binh_luan",
    "url": "link",
}

_SELECT_SQL = f"""
    SELECT {', '.join(_COLUMN_MAP.keys())}
    FROM news
"""


class PostgresNewsRepository(NewsRepository):
    """Cùng interface với `NewsRepository`, chỉ đổi nguồn đọc sang Postgres."""

    def load(self, database_url: str = DATABASE_URL) -> None:
        import psycopg2

        logger.info("Đang kết nối Postgres (Neon) để nạp dữ liệu bài báo...")
        conn = psycopg2.connect(database_url)
        try:
            df = pd.read_sql(_SELECT_SQL, conn)
        finally:
            conn.close()

        if df.empty:
            raise RuntimeError(
                "Bảng 'news' trên Postgres không có dòng nào. Kiểm tra lại "
                "đã import dữ liệu vào Neon chưa (xem data_cleaning_pipeline.py)."
            )

        df = df.rename(columns=_COLUMN_MAP)

        # publish_date từ Postgres trả về kiểu datetime.date -> chuyển
        # thành string "YYYY-MM-DD" để khớp định dạng cột `ngay_dang` cũ
        # (NewsItem.publish_date là Optional[str], không phải date).
        df["ngay_dang"] = df["ngay_dang"].apply(
            lambda d: d.isoformat() if pd.notna(d) else None
        )

        logger.info("Đã đọc %d dòng từ Postgres, đang xử lý...", len(df))
        self.df = self._finalize(df)
        self.loaded = True
        logger.info("Đã nạp %d bài báo từ Postgres (Neon) vào RAM.", len(self.df))


# Instance toàn cục duy nhất — dùng thay cho `news_repository` (bản CSV)
# trong `main.py` khi chuyển hẳn sang Postgres.
news_repository = PostgresNewsRepository()
