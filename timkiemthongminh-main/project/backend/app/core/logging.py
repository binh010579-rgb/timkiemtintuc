"""
Cấu hình logging tập trung cho backend runtime (`app/main.py`).

`build_vectors.py` là script offline, tự cấu hình logging riêng của nó —
KHÔNG dùng module này (giữ 2 vòng đời hoàn toàn tách biệt).
"""

from __future__ import annotations

import logging

from app.config import LOG_LEVEL


def setup_logging() -> None:
    """Gọi đúng 1 lần lúc import `app.main`, trước khi tạo FastAPI app."""
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
