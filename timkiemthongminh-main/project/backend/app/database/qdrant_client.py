"""
Wrapper kết nối tới Qdrant Cloud — nơi lưu trữ DUY NHẤT cho vector
embedding của bài báo.

Nguyên tắc bắt buộc của module này:
- CHỈ Qdrant Cloud qua HTTPS — KHÔNG có chế độ embedded, KHÔNG lưu vector
  ở thư mục local nào (xem validate HTTPS ở `app/config.py`).
- URL + API key đọc từ biến môi trường (QDRANT_URL, QDRANT_API_KEY),
  KHÔNG hardcode.
- `connect()` tự retry (exponential backoff) khi kết nối thất bại — hữu
  ích với free-tier cluster cần "đánh thức" hoặc mạng chập chờn.
- `ensure_collection()`: nếu collection CHƯA tồn tại thì tạo mới; nếu ĐÃ
  tồn tại thì chỉ kết nối tới collection đó, KHÔNG đụng vào dữ liệu cũ.
  Đây là hàm DUY NHẤT được gọi lúc backend khởi động — không có đường nào
  dẫn tới việc build/encode lại vector từ startup của server.
- `recreate_collection()` (xoá + tạo lại) CHỈ được dùng bởi script offline
  `build_vectors.py` khi cố tình muốn build lại toàn bộ, KHÔNG bao giờ
  được gọi từ `app/main.py`.

Toàn bộ việc tìm kiếm/so khớp độ tương đồng do Qdrant đảm nhiệm nội bộ —
KHÔNG cosine similarity tự viết, KHÔNG numpy brute-force, KHÔNG sklearn,
KHÔNG faiss.
"""

from __future__ import annotations

import logging
import time

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import (
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_CONNECT_MAX_RETRIES,
    QDRANT_CONNECT_RETRY_BACKOFF_SECONDS,
    QDRANT_CONNECT_TIMEOUT_SECONDS,
    QDRANT_URL,
)

logger = logging.getLogger(__name__)


class QdrantConnectionError(Exception):
    """Không kết nối được tới Qdrant Cloud sau khi đã retry hết số lần cho phép."""


class QdrantVectorStore:
    """Singleton bọc QdrantClient, dùng chung bởi search_service.py và build_vectors.py."""

    def __init__(self) -> None:
        self.client: QdrantClient | None = None
        self.collection_name = QDRANT_COLLECTION_NAME

    def connect(
        self,
        max_retries: int = QDRANT_CONNECT_MAX_RETRIES,
        backoff_seconds: float = QDRANT_CONNECT_RETRY_BACKOFF_SECONDS,
    ) -> None:
        """
        Tạo kết nối HTTPS tới Qdrant Cloud nếu chưa có (no-op nếu đã kết
        nối). Retry với backoff tăng dần nếu request kiểm tra kết nối
        (`get_collections`) thất bại — ví dụ cluster free-tier đang "thức
        dậy" sau thời gian ngủ đông, hoặc lỗi mạng tạm thời.

        Raise `QdrantConnectionError` nếu vẫn thất bại sau `max_retries` lần.
        """
        if self.client is not None:
            return

        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                client = QdrantClient(
                    url=QDRANT_URL,
                    api_key=QDRANT_API_KEY,
                    https=True,
                    timeout=QDRANT_CONNECT_TIMEOUT_SECONDS,
                )
                # Gọi thử 1 request nhẹ để xác nhận kết nối thật sự sống,
                # không chỉ khởi tạo object client thành công.
                client.get_collections()
                self.client = client
                if attempt > 1:
                    logger.info("Kết nối thành công sau %d lần thử.", attempt)
                else:
                    logger.info("Đã kết nối tới Qdrant Cloud (%s).", QDRANT_URL)
                return
            except Exception as exc:  # noqa: BLE001 - muốn bắt mọi lỗi mạng/HTTP để retry
                last_error = exc
                logger.warning("Kết nối thất bại (lần %d/%d): %s", attempt, max_retries, exc)
                if attempt < max_retries:
                    sleep_for = backoff_seconds * attempt
                    logger.info("Thử lại sau %.1fs...", sleep_for)
                    time.sleep(sleep_for)

        raise QdrantConnectionError(
            f"Không kết nối được tới Qdrant Cloud sau {max_retries} lần thử: {last_error}"
        )

    def _client(self) -> QdrantClient:
        if self.client is None:
            raise RuntimeError("QdrantVectorStore chưa connect(). Gọi connect() trước khi dùng.")
        return self.client

    def collection_exists(self) -> bool:
        names = [c.name for c in self._client().get_collections().collections]
        return self.collection_name in names

    def ensure_collection(self, vector_size: int) -> None:
        """
        Nếu collection CHƯA tồn tại: tạo mới (rỗng, chưa có vector nào).
        Nếu ĐÃ tồn tại: không làm gì cả, chỉ coi như đã sẵn sàng để dùng.

        Đây là hàm DUY NHẤT liên quan tới collection được gọi khi backend
        khởi động (`app/main.py`) — KHÔNG bao giờ xoá hay ghi đè dữ liệu
        vector đã có sẵn.
        """
        if self.collection_exists():
            logger.info("Collection '%s' đã tồn tại — chỉ kết nối.", self.collection_name)
            return
        logger.info("Collection '%s' chưa tồn tại — đang tạo mới...", self.collection_name)
        self._client().create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=vector_size, distance=qmodels.Distance.COSINE
            ),
        )
        logger.info("Đã tạo collection '%s' (dim=%d).", self.collection_name, vector_size)

    def recreate_collection(self, vector_size: int) -> None:
        """
        Xoá collection cũ (nếu có) và tạo lại trống.

        CHỈ dùng bởi script offline `build_vectors.py` khi build lại toàn
        bộ vector từ đầu — KHÔNG bao giờ được gọi từ vòng đời server
        (`app/main.py`).
        """
        if self.collection_exists():
            self._client().delete_collection(self.collection_name)
        self._client().create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=vector_size, distance=qmodels.Distance.COSINE
            ),
        )

    def count(self) -> int:
        if not self.collection_exists():
            return 0
        info = self._client().get_collection(self.collection_name)
        return int(info.points_count or 0)

    def check_health(self) -> tuple[bool, str | None]:
        """
        Kiểm tra nhẹ dùng bởi `GET /health` — KHÔNG phải `connect()` đầy
        đủ với retry/backoff (không muốn health check bị treo lâu), chỉ
        thử 1 lần và trả kết quả ngay. KHÔNG tạo/xoá/đụng collection.
        """
        try:
            if self.client is None:
                self.connect(max_retries=1, backoff_seconds=0)
            else:
                self.client.get_collections()
            return True, None
        except Exception as exc:  # noqa: BLE001 - muốn bắt mọi lỗi để báo cáo trạng thái
            return False, str(exc)

    def upsert(
        self,
        ids: list[int],
        vectors: list[list[float]],
        payloads: list[dict] | None = None,
        batch_size: int = 128,
    ) -> None:
        """Đẩy (id, vector, payload) vào collection theo batch. Dùng bởi build_vectors.py."""
        client = self._client()
        total = len(ids)
        payloads = payloads if payloads is not None else [{} for _ in range(total)]
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            client.upsert(
                collection_name=self.collection_name,
                points=qmodels.Batch(
                    ids=[int(i) for i in ids[start:end]],
                    vectors=vectors[start:end],
                    payloads=payloads[start:end],
                ),
            )

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        score_threshold: float | None = None,
    ) -> list[tuple[int, float]]:
        """
        Tìm top_k điểm gần nhất bằng Qdrant Search API. Chỉ lấy id + score
        (with_payload=False) — hydrate nội dung đầy đủ là việc của
        `search_service.py`, đọc từ `NewsRepository`, không phải từ
        payload Qdrant.

        Trả về list[(article_id, score)] giảm dần theo score.
        """
        hits = self._client().search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=False,
            with_vectors=False,
        )
        return [(int(hit.id), float(hit.score)) for hit in hits]


# Instance toàn cục duy nhất.
qdrant_store = QdrantVectorStore()
