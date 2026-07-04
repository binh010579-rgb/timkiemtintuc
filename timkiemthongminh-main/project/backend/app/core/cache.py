"""
Cache TTL trong RAM cho embedding của câu truy vấn tìm kiếm.

Vị trí trong Search Pipeline:

    User Query -> [Cache] -> Embedding API -> Qdrant Search -> ...

Nguyên tắc bắt buộc:
- Chỉ cache EMBEDDING CỦA QUERY (không cache embedding tài liệu, không
  cache kết quả search).
- Chỉ tồn tại trong RAM (`cachetools.TTLCache`) — KHÔNG ghi xuống file,
  KHÔNG ghi xuống Redis/DB nào. Mất sạch khi restart server, đúng như
  yêu cầu "không cache embedding trong file".
- TTL mặc định 30 phút (`CACHE_TTL_SECONDS`, cấu hình qua env).
- Thread-safe: nhiều request async có thể chạy xen kẽ trong cùng event
  loop (hoặc nhiều worker), nên mọi thao tác đọc/ghi đều qua `threading.Lock`.
"""

from __future__ import annotations

import threading
from typing import Optional

from cachetools import TTLCache

from app.config import CACHE_MAXSIZE, CACHE_TTL_SECONDS


class QueryEmbeddingCache:
    """Cache (query text đã chuẩn hoá) -> (vector embedding)."""

    def __init__(self, maxsize: int = CACHE_MAXSIZE, ttl: float = CACHE_TTL_SECONDS) -> None:
        self._cache: TTLCache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._lock = threading.Lock()

    @staticmethod
    def _normalize_key(query: str) -> str:
        # Chỉ strip khoảng trắng thừa — KHÔNG lowercase/bỏ dấu, vì embedding
        # model có thể nhạy với hoa/thường và dấu tiếng Việt.
        return query.strip()

    def get(self, query: str) -> Optional[list[float]]:
        key = self._normalize_key(query)
        with self._lock:
            return self._cache.get(key)

    def set(self, query: str, vector: list[float]) -> None:
        key = self._normalize_key(query)
        with self._lock:
            self._cache[key] = vector

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


# Instance toàn cục duy nhất — dùng bởi search_service.py.
query_embedding_cache = QueryEmbeddingCache()
