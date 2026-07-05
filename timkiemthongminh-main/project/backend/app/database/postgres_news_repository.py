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

BỘ NHỚ (RAM) — THAY ĐỔI QUAN TRỌNG so với bản trước:
Cột `content` (nội dung ĐẦY ĐỦ của bài báo, `noi_dung`) KHÔNG còn được
SELECT và giữ trong RAM cho toàn bộ 50k+ dòng nữa — đây chính là nguyên
nhân gây "Ran out of memory" trên gói Free (512MB) của Render. Thay vào
đó:
- `load()` chỉ nạp metadata nhẹ (title, summary, date, author, source,
  url, số bình luận) vào RAM — đủ cho list/danh mục/tìm kiếm từ khoá.
- `content` được fetch TRỰC TIẾP từ Postgres theo đúng các `id` cần dùng,
  chỉ khi search_service cần hydrate top-k kết quả (xem
  `get_content_by_ids()` bên dưới) — không bao giờ load hết 1 lần.
"""

from __future__ import annotations

import logging

import pandas as pd
from psycopg2 import pool as psycopg2_pool

from app.config import DATABASE_URL
from app.database.news_repository import NewsRepository

logger = logging.getLogger(__name__)

# Ánh xạ tên cột trong bảng Postgres -> tên cột gốc mà phần còn lại của
# backend (services/api) đang dùng (giữ nguyên để không phải sửa gì khác).
# LƯU Ý: KHÔNG có "content" ở đây nữa — xem docstring ở đầu file.
_METADATA_COLUMN_MAP = {
    "source": "nguon",
    "title": "tieu_de",
    "publish_date": "ngay_dang",
    "author": "tac_gia",
    "summary": "summary",
    "comments": "so_binh_luan",
    "url": "link",
}

_METADATA_SELECT_SQL = f"""
    SELECT {', '.join(_METADATA_COLUMN_MAP.keys())}
    FROM news
"""

# Connection pool nhỏ, dùng riêng cho các query fetch content theo id lúc
# request (KHÔNG dùng cho load() lúc khởi động — load() vẫn mở 1 connection
# rời, chạy 1 lần). maxconn=5 là đủ cho traffic nhỏ/vừa trên gói free; tăng
# lên nếu cần nhiều connection đồng thời hơn. Tạo lười (lazy) để không mở
# connection nào cho tới khi thực sự có request cần content.
_content_pool: psycopg2_pool.SimpleConnectionPool | None = None


def _get_content_pool(database_url: str) -> psycopg2_pool.SimpleConnectionPool:
    global _content_pool
    if _content_pool is None:
        _content_pool = psycopg2_pool.SimpleConnectionPool(1, 5, database_url)
    return _content_pool


class PostgresNewsRepository(NewsRepository):
    """Cùng interface với `NewsRepository`, chỉ đổi nguồn đọc sang Postgres."""

    def __init__(self) -> None:
        super().__init__()
        self._database_url: str | None = None

    def load(self, database_url: str = DATABASE_URL) -> None:
        import psycopg2

        logger.info("Đang kết nối Postgres (Neon) để nạp metadata bài báo...")
        conn = psycopg2.connect(database_url)
        try:
            df = pd.read_sql(_METADATA_SELECT_SQL, conn)
        finally:
            conn.close()

        if df.empty:
            raise RuntimeError(
                "Bảng 'news' trên Postgres không có dòng nào. Kiểm tra lại "
                "đã import dữ liệu vào Neon chưa (xem data_cleaning_pipeline.py)."
            )

        df = df.rename(columns=_METADATA_COLUMN_MAP)

        # publish_date từ Postgres trả về kiểu datetime.date -> chuyển
        # thành string "YYYY-MM-DD" để khớp định dạng cột `ngay_dang` cũ
        # (NewsItem.publish_date là Optional[str], không phải date).
        df["ngay_dang"] = df["ngay_dang"].apply(
            lambda d: d.isoformat() if pd.notna(d) else None
        )

        logger.info("Đã đọc %d dòng metadata từ Postgres, đang xử lý...", len(df))
        self.df = self._finalize(df)
        self.loaded = True
        self._database_url = database_url
        logger.info(
            "Đã nạp %d bài báo (chỉ metadata, KHÔNG có content) vào RAM.",
            len(self.df),
        )

    def get_content_by_ids(self, ids: list[int]) -> dict[int, str | None]:
        """
        Fetch nội dung ĐẦY ĐỦ (`content`) từ Postgres cho đúng danh sách
        `ids` cần dùng — thường chỉ 10-30 bài (candidate_k của search
        pipeline), KHÔNG bao giờ toàn bộ dataset. Đây là điểm thay thế cho
        việc giữ sẵn cột content trong RAM của bản cũ.
        """
        if not ids:
            return {}

        df = self.ensure_loaded()
        indexed = df.set_index("id", drop=False)

        # id -> link (đã có sẵn trong metadata đang giữ ở RAM, không tốn
        # thêm 1 lần round-trip DB để tra ngược URL).
        links_by_id: dict[int, str] = {}
        for i in ids:
            try:
                links_by_id[i] = indexed.loc[i, "link"]
            except KeyError:
                continue

        if not links_by_id:
            return {}

        if self._database_url is None:
            raise RuntimeError(
                "PostgresNewsRepository.get_content_by_ids() được gọi trước "
                "khi load() chạy xong — kiểm tra lại thứ tự khởi động server."
            )

        pool = _get_content_pool(self._database_url)
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT url, content FROM news WHERE url = ANY(%s)",
                    (list(links_by_id.values()),),
                )
                rows = cur.fetchall()
        finally:
            pool.putconn(conn)

        content_by_link = {link: content for link, content in rows}
        return {i: content_by_link.get(link) for i, link in links_by_id.items()}


# Instance toàn cục duy nhất — dùng thay cho `news_repository` (bản CSV)
# trong `main.py` khi chuyển hẳn sang Postgres.
news_repository = PostgresNewsRepository()
