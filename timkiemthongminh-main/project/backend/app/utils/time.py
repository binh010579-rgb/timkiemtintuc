"""
Helper thời gian dùng chung — hiện tại chỉ phục vụ `GET /health` và các
exception handler cần timestamp trong response JSON.
"""

from __future__ import annotations

from datetime import datetime, timezone


def utcnow_iso() -> str:
    """Timestamp ISO 8601 UTC (có timezone rõ ràng, ví dụ 2026-07-04T10:00:00+00:00)."""
    return datetime.now(timezone.utc).isoformat()
