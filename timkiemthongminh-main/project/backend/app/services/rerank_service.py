"""
Bước Re-ranking trong Search Pipeline:

    ... -> Qdrant Search -> Top {SEARCH_CANDIDATE_K} -> [Re-ranking] ->
    Top {SEARCH_TOP_K} -> JSON

Có 2 implementation của `Reranker` (Protocol):

1. `PassthroughReranker` — giữ nguyên thứ tự Qdrant đã trả về (cosine).
   Dùng làm FALLBACK khi gọi API rerank thật bị lỗi — KHÔNG xoá.

2. `HFCrossEncoderReranker` (mặc định hiện tại) — gọi 1 cross-encoder
   THẬT qua Hugging Face Inference API (provider "hf-inference"),
   KHÔNG load model cục bộ. Model mặc định: BAAI/bge-reranker-v2-m3
   (cùng "họ" BGE với BAAI/bge-m3 đang dùng để embedding — multilingual,
   hỗ trợ tốt tiếng Việt).

--------------------------------------------------------------------------
GHI CHÚ QUAN TRỌNG — ĐỌC TRƯỚC KHI TIN TƯỞNG 100% (đã kiểm tra thực tế,
không đoán mò):

- BAAI/bge-reranker-v2-m3 trên HF Hub có pipeline_tag = "text-classification"
  (đây là model AutoModelForSequenceClassification, num_labels=1 — không
  phải "sentence-similarity"/feature-extraction như BAAI/bge-m3). Model
  này CÓ được deploy qua provider "hf-inference" (xác nhận trên trang
  model: badge "Inference Providers" + "HF Inference API").

- Trang tài liệu HỢP NHẤT nhiều-provider hiện tại của Inference Providers
  (huggingface.co/docs/inference-providers/tasks/text-classification) chỉ
  mô tả payload dạng `{"inputs": "<1 chuỗi>"}` — tài liệu đó KHÔNG nhắc gì
  tới input dạng CẶP (query, passage) mà 1 cross-encoder cần để hoạt động
  đúng. Task kiểu cũ "sentence-similarity" (source_sentence + sentences)
  cũng đã biến mất khỏi danh sách task Inference Providers hỗ trợ — mà
  bản chất task đó cũng chỉ là cosine similarity giữa 2 embedding riêng
  biệt (bi-encoder), không phải cross-encoder thật, nên dù còn tồn tại
  cũng không dùng được cho model này.
- Một "/rerank" endpoint chuẩn (nhận thẳng {query, texts}) CÓ tồn tại,
  nhưng chỉ ở dạng Hugging Face Inference ENDPOINTS (dedicated, TRẢ PHÍ,
  do người dùng tự deploy 1 container Text Embeddings Inference riêng) —
  KHÔNG phải Inference API serverless miễn phí mà kiến trúc hiện tại của
  dự án này đang dùng (embedding_service.py cũng chỉ dùng serverless).

- Cách gọi implement dưới đây — gửi `{"text": query, "text_pair": passage}`
  cho từng cặp (đúng định dạng mà `transformers.TextClassificationPipeline`
  hiểu để encode 1 CẶP câu, đây là hành vi có trong mã nguồn thư viện
  `transformers` mà provider "hf-inference" chạy bên dưới) — dựa trên
  hành vi THỰC TẾ của thư viện `transformers`, không phải suy đoán, NHƯNG
  không được liệt kê trong trang tài liệu hợp nhất multi-provider mới
  (trang đó rút gọn về mẫu số chung nhỏ nhất giữa các provider). Vì vậy
  đây KHÔNG phải một API được cam kết ổn định lâu dài 100%.

  => Do đó, class này bắt buộc phải có fallback về `PassthroughReranker`
     cho MỌI lỗi (HTTP lỗi, response sai định dạng, timeout...). Khuyến
     nghị: sau khi điền HF_API_TOKEN thật, gọi thử `POST /search` 1 lần và
     xem log — nếu thấy dòng log "Rerank qua HF Inference API lỗi..." thì
     rerank đang fallback về passthrough (search vẫn chạy bình thường,
     chỉ là chưa có rerank thật), cần xem lỗi cụ thể trong log để xử lý
     tiếp (ví dụ đổi sang model reranker khác, hoặc cân nhắc trả phí dùng
     Inference Endpoints dedicated nếu muốn rerank ổn định 100%).
--------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

import httpx

from app.config import (
    HF_API_BASE_URL,
    HF_API_MAX_RETRIES,
    HF_API_RETRY_BACKOFF_SECONDS,
    HF_API_TIMEOUT_SECONDS,
    HF_API_TOKEN,
    HF_RERANKER_MODEL,
)
from app.models.search import SemanticSearchResultItem

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    def rerank(
        self, query: str, items: list[SemanticSearchResultItem]
    ) -> list[SemanticSearchResultItem]: ...


class PassthroughReranker:
    """Implementation fallback: không rerank, chỉ trả nguyên danh sách đầu vào."""

    def rerank(
        self, query: str, items: list[SemanticSearchResultItem]
    ) -> list[SemanticSearchResultItem]:
        return items


class RerankServiceError(Exception):
    """Lỗi khi gọi Hugging Face Inference API để rerank (cross-encoder)."""


class HFCrossEncoderReranker:
    """
    Reranker THẬT — gọi cross-encoder qua Hugging Face Inference API
    (KHÔNG load model cục bộ). Xem ghi chú đầu file về giới hạn/rủi ro của
    cách gọi này.

    An toàn khi lỗi: MỌI exception trong quá trình gọi API hoặc parse kết
    quả đều được bắt lại, log cảnh báo, rồi fallback nguyên vẹn sang
    `PassthroughReranker` — không bao giờ để lỗi rerank làm hỏng luôn cả
    response của `POST /search`.

    Khi rerank THÀNH CÔNG: field `score` của từng `SemanticSearchResultItem`
    được GHI ĐÈ bằng điểm liên quan do cross-encoder chấm (đã qua sigmoid,
    nằm trong khoảng 0-1) — đây là tín hiệu liên quan chính xác hơn cosine
    similarity gốc từ Qdrant, phù hợp để hiển thị "độ liên quan" ở frontend
    (xem việc #5). Khi fallback về PassthroughReranker (lỗi API), `score`
    giữ nguyên là cosine similarity gốc từ Qdrant.
    """

    def __init__(self, model: str = HF_RERANKER_MODEL) -> None:
        self._model = model
        self._endpoint = f"{HF_API_BASE_URL}/models/{model}/pipeline/text-classification"
        self._headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        self._fallback = PassthroughReranker()

    def _call_api(self, pairs: list[dict[str, str]]) -> list:
        """Gọi HF Inference API, tự retry khi model đang cold start (503) hoặc rate limit (429)."""
        last_error: Exception | None = None

        with httpx.Client(timeout=HF_API_TIMEOUT_SECONDS) as client:
            for attempt in range(1, HF_API_MAX_RETRIES + 1):
                try:
                    response = client.post(
                        self._endpoint,
                        headers=self._headers,
                        json={
                            "inputs": pairs,
                            "parameters": {"function_to_apply": "sigmoid", "top_k": 1},
                            "options": {"wait_for_model": True},
                        },
                    )
                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPStatusError as exc:
                    last_error = exc
                    status = exc.response.status_code
                    if status in (503, 429) and attempt < HF_API_MAX_RETRIES:
                        time.sleep(HF_API_RETRY_BACKOFF_SECONDS * attempt)
                        continue
                    raise RerankServiceError(
                        f"HF Inference API (rerank) trả lỗi {status}: {exc.response.text}"
                    ) from exc
                except httpx.RequestError as exc:
                    last_error = exc
                    if attempt < HF_API_MAX_RETRIES:
                        time.sleep(HF_API_RETRY_BACKOFF_SECONDS * attempt)
                        continue
                    raise RerankServiceError(
                        f"Không gọi được HF Inference API (rerank): {exc}"
                    ) from exc

        raise RerankServiceError(f"Gọi HF Inference API (rerank) thất bại: {last_error}")

    @staticmethod
    def _extract_score(raw_item: object) -> float:
        """
        Response mong đợi cho MỖI cặp (query, passage) — theo pipeline
        text-classification chuẩn của `transformers` khi `top_k=1`:
            {"label": "LABEL_0", "score": 0.87}
        hoặc bọc trong list 1 phần tử: [{"label": ..., "score": 0.87}].
        Xử lý cả 2 dạng cho chắc chắn.
        """
        item = raw_item[0] if isinstance(raw_item, list) else raw_item
        if not isinstance(item, dict) or "score" not in item:
            raise RerankServiceError(f"Không tìm thấy field 'score' trong response: {raw_item!r}")
        return float(item["score"])

    def rerank(
        self, query: str, items: list[SemanticSearchResultItem]
    ) -> list[SemanticSearchResultItem]:
        if not items:
            return items

        pairs = [
            {"text": query, "text_pair": (item.summary or item.title or "")} for item in items
        ]

        try:
            raw = self._call_api(pairs)
            if not isinstance(raw, list) or len(raw) != len(items):
                raise RerankServiceError(
                    f"Response rerank không đúng định dạng mong đợi "
                    f"(kỳ vọng list dài {len(items)}, nhận: {type(raw).__name__})."
                )
            ce_scores = [self._extract_score(r) for r in raw]
        except Exception as exc:  # noqa: BLE001 - bất kỳ lỗi nào cũng phải fallback, không để /search sập
            logger.warning(
                "Rerank qua HF Inference API lỗi (model=%s): %s — fallback về "
                "PassthroughReranker (giữ nguyên thứ tự cosine từ Qdrant).",
                self._model,
                exc,
            )
            return self._fallback.rerank(query, items)

        reranked = sorted(zip(items, ce_scores), key=lambda pair: pair[1], reverse=True)
        return [item.model_copy(update={"score": score}) for item, score in reranked]


# Instance dùng mặc định bởi search_service.py — cross-encoder thật qua
# HF Inference API, tự fallback về passthrough nếu lỗi.
default_reranker = HFCrossEncoderReranker()
