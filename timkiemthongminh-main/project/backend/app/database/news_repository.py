"""
Repository cho dữ liệu bài báo (title, summary, content, url, tác giả...).

THIẾT KẾ THEO REPOSITORY PATTERN: mọi module khác (services, api) chỉ gọi
qua `NewsRepository`, KHÔNG biết (và không quan tâm) dữ liệu vật lý đang
nằm ở CSV hay ở một database quan hệ thật. Hiện tại nguồn dữ liệu là CSV
đọc 1 lần vào RAM khi server khởi động — nếu sau này đổi sang
Postgres/MySQL, chỉ cần viết 1 class khác cùng interface
(`load`, `ensure_loaded`, `list_all`, `get_by_ids`, `count`) và inject vào
`main.py`, không phải sửa `services/` hay `api/`.

Đây KHÔNG phải vector store — vector embedding nằm hoàn toàn trên Qdrant
Cloud (xem `qdrant_client.py`), repository này chỉ giữ metadata/nội dung.
"""

from __future__ import annotations

import os
import unicodedata

import pandas as pd

from app.config import CSV_PATH

REQUIRED_COLUMNS = ["nguon", "tieu_de", "ngay_dang", "tac_gia", "summary", "so_binh_luan", "link"]


def strip_accents_lower(text: str) -> str:
    """Chuẩn hoá chuỗi tiếng Việt: bỏ dấu, chữ thường — dùng để so khớp tìm kiếm từ khoá."""
    normalized = unicodedata.normalize("NFD", text)
    no_accents = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return no_accents.lower()


class NewsRepository:
    """
    Giữ toàn bộ dữ liệu bài báo trong RAM (DataFrame), nạp MỘT LẦN khi
    server khởi động (xem `main.py` lifespan). Mọi request sau đó chỉ đọc
    từ DataFrame có sẵn, KHÔNG đọc lại CSV.
    """

    def __init__(self) -> None:
        self.df: pd.DataFrame | None = None
        self.loaded: bool = False

    def load(self, csv_path: str = CSV_PATH) -> None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Không tìm thấy file dữ liệu tại: {csv_path}. "
                "Đảm bảo file đã được đặt đúng vị trí backend/data/."
            )

        df = pd.read_csv(csv_path)

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Dữ liệu thiếu các cột bắt buộc: {missing}")

        df = df.reset_index(drop=True)
        df.insert(0, "id", df.index + 1)

        df["so_binh_luan"] = (
            pd.to_numeric(df["so_binh_luan"], errors="coerce").round().astype("Int64")
        )

        text_cols = ["nguon", "tieu_de", "ngay_dang", "tac_gia", "summary", "link", "noi_dung"]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].where(df[col].notna(), None)

        df["_search_blob"] = (
            df["tieu_de"].fillna("") + " " + df["summary"].fillna("")
        ).map(strip_accents_lower)

        self.df = df
        self.loaded = True

    def ensure_loaded(self) -> pd.DataFrame:
        if not self.loaded or self.df is None:
            raise RuntimeError(
                "NewsRepository chưa được nạp dữ liệu. Gọi load() khi server "
                "khởi động trước khi xử lý request."
            )
        return self.df

    def get_by_ids(self, ids: list[int]) -> list[pd.Series]:
        """Lấy các dòng theo đúng thứ tự `ids` (giá trị cột `id`, không phải vị trí)."""
        df = self.ensure_loaded()
        indexed = df.set_index("id", drop=False)
        return [indexed.loc[i] for i in ids]

    def count(self) -> int:
        return len(self.ensure_loaded())


# Instance toàn cục duy nhất — được nạp dữ liệu trong main.py (lifespan).
news_repository = NewsRepository()
