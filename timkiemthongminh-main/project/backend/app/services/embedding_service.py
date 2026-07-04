"""
Service DUY NHẤT chịu trách nhiệm sinh embedding — gọi Hugging Face
Inference API (model BAAI/bge-m3, multilingual, hỗ trợ tốt tiếng Việt).

QUY TẮC BẮT BUỘC của kiến trúc này:
- KHÔNG import torch, KHÔNG import sentence-transformers.
- KHÔNG load bất kỳ model nào vào RAM/GPU của backend.
- Toàn bộ việc suy luận (inference) mô hình embedding diễn ra ở phía
  Hugging Face — backend chỉ là HTTP client gọi tới đó.

Model BGE-M3 là model ĐỐI XỨNG (symmetric retrieval): không yêu cầu
instruction prefix khác nhau giữa document và query như Qwen3-Embedding.
Nếu sau này đổi sang một model bất đối xứng, chỉ cần set
`EMBEDDING_QUERY_PREFIX` trong config — hàm `embed_query()` đã sẵn chỗ
áp dụng prefix đó mà không phải sửa code gọi.
"""

from __future__ import annotations

import time

import httpx

from app.config import (
    EMBEDDING_QUERY_PREFIX,
    HF_API_BASE_URL,
    HF_API_MAX_RETRIES,
    HF_API_RETRY_BACKOFF_SECONDS,
    HF_API_TIMEOUT_SECONDS,
    HF_API_TOKEN,
    HF_EMBEDDING_MODEL,
)


class EmbeddingServiceError(Exception):
    """Lỗi khi gọi Hugging Face Inference API để sinh embedding."""


class EmbeddingService:
    """
    HTTP client gọi Hugging Face Inference API (feature-extraction pipeline).

    Stateless — không giữ model, không giữ vector trong RAM giữa các lần
    gọi. Mỗi request `/search` gọi `embed_query()` đúng 1 lần.
    """

    def __init__(self) -> None:
        self._endpoint = f"{HF_API_BASE_URL}/{HF_EMBEDDING_MODEL}"
        self._headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    def _call_api(self, inputs: list[str]) -> list[list[float]]:
        """
        Gọi HF Inference API, tự retry khi model đang "cold start"
        (HF trả 503 kèm `estimated_time` trong lúc load model lên server
        của họ — bình thường với free/serverless inference, không phải lỗi).
        """
        last_error: Exception | None = None

        with httpx.Client(timeout=HF_API_TIMEOUT_SECONDS) as client:
            for attempt in range(1, HF_API_MAX_RETRIES + 1):
                try:
                    response = client.post(
                        self._endpoint,
                        headers=self._headers,
                        json={"inputs": inputs, "options": {"wait_for_model": True}},
                    )
                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPStatusError as exc:
                    last_error = exc
                    status = exc.response.status_code
                    # 503: model đang được nạp phía HF — retry có backoff.
                    # 429: rate limit — cũng đáng để retry.
                    if status in (503, 429) and attempt < HF_API_MAX_RETRIES:
                        time.sleep(HF_API_RETRY_BACKOFF_SECONDS * attempt)
                        continue
                    raise EmbeddingServiceError(
                        f"Hugging Face Inference API trả lỗi {status}: {exc.response.text}"
                    ) from exc
                except httpx.RequestError as exc:
                    last_error = exc
                    if attempt < HF_API_MAX_RETRIES:
                        time.sleep(HF_API_RETRY_BACKOFF_SECONDS * attempt)
                        continue
                    raise EmbeddingServiceError(
                        f"Không gọi được Hugging Face Inference API: {exc}"
                    ) from exc

        raise EmbeddingServiceError(f"Gọi Hugging Face Inference API thất bại: {last_error}")

    @staticmethod
    def _normalize_output(raw: list, expected_count: int) -> list[list[float]]:
        """
        HF feature-extraction có thể trả về:
        - list[list[float]]        (đã pooled sẵn, 1 vector / câu — trường
          hợp thường gặp với model có cấu hình sentence-embedding).
        - list[list[list[float]]]  (embedding theo từng token, chưa pool)

        Ở trường hợp thứ 2, tự mean-pooling theo token để ra 1 vector/câu —
        đây là phép tính hậu xử lý số học đơn thuần trên kết quả API trả
        về, KHÔNG phải chạy lại mô hình.
        """
        if len(raw) != expected_count:
            raise EmbeddingServiceError(
                f"Số vector trả về ({len(raw)}) không khớp số input ({expected_count})."
            )

        vectors: list[list[float]] = []
        for item in raw:
            if item and isinstance(item[0], list):
                # Chưa pooled: mean-pooling thủ công theo token.
                dim = len(item[0])
                sums = [0.0] * dim
                for token_vec in item:
                    for i, v in enumerate(token_vec):
                        sums[i] += v
                vectors.append([s / len(item) for s in sums])
            else:
                vectors.append(item)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Sinh embedding cho một batch văn bản tài liệu (dùng bởi build_vectors.py)."""
        if not texts:
            return []
        raw = self._call_api(texts)
        return self._normalize_output(raw, expected_count=len(texts))

    def embed_query(self, text: str) -> list[float]:
        """Sinh embedding cho 1 câu truy vấn tìm kiếm (dùng bởi search_service.py)."""
        prefixed = f"{EMBEDDING_QUERY_PREFIX}{text}" if EMBEDDING_QUERY_PREFIX else text
        vectors = self.embed_documents([prefixed])
        return vectors[0]

    def check_health(self, timeout_seconds: float = 5.0) -> tuple[bool, str | None]:
        """
        Kiểm tra nhẹ dùng bởi `GET /health` — gọi 1 request rất nhỏ, timeout
        ngắn, KHÔNG retry nhiều lần (khác `embed_query`, cần phản hồi
        nhanh). `wait_for_model=False` nên nếu model HF đang "cold start",
        health check có thể báo lỗi tạm thời — đây là trạng thái hạ tầng
        thật sự tại thời điểm kiểm tra, không phải lỗi code.
        """
        try:
            with httpx.Client(timeout=timeout_seconds) as client:
                response = client.post(
                    self._endpoint,
                    headers=self._headers,
                    json={"inputs": ["ping"], "options": {"wait_for_model": False}},
                )
                response.raise_for_status()
            return True, None
        except Exception as exc:  # noqa: BLE001 - muốn bắt mọi lỗi để báo cáo trạng thái
            return False, str(exc)


# Instance toàn cục duy nhất — stateless, không tốn RAM để giữ.
embedding_service = EmbeddingService()
