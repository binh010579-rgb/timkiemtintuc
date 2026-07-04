#!/usr/bin/env python3
"""
Script OFFLINE, ĐỘC LẬP HOÀN TOÀN với backend runtime (`app/main.py`).

Mục đích DUY NHẤT: đọc `data/cleaned_news.csv`, sinh embedding cho
title+summary, rồi upload (upsert) vector lên Qdrant Cloud.

    KHÔNG được import bởi `app/main.py`.
    KHÔNG chạy tự động khi server khởi động.
    CHỈ chạy TAY (hoặc qua CI/cron) mỗi khi `cleaned_news.csv` thay đổi.

Hai chế độ sinh embedding (chọn qua --mode):
- `hf`    (mặc định): gọi Hugging Face Inference API — KHÔNG cần cài
           torch/sentence-transformers, dùng chung EmbeddingService của
           backend (app/services/embedding_service.py).
- `local`: chạy model SentenceTransformer NGAY TRÊN MÁY BẠN, hoàn toàn
           offline, không cần API/internet sau khi đã tải model. Yêu cầu
           cài thêm `pip install -r requirements-build-local.txt`
           (torch + sentence-transformers) — CÁC GÓI NÀY KHÔNG NẰM TRONG
           `requirements.txt` của backend, vì backend runtime tuyệt đối
           không cần chúng.

CHIẾN LƯỢC INCREMENTAL (chạy nhiều lần khi crawl dữ liệu theo đợt):
- Trước khi encode, script hỏi thẳng Qdrant Cloud (`existing_ids()`) xem
  ID nào đã có sẵn trong collection.
- CHỈ encode + upsert các bài có ID CHƯA có trên Qdrant (bài mới cào
  thêm). Bài đã upsert ở lần chạy trước — dù CSV giờ có thêm hàng chục
  nghìn dòng mới — sẽ KHÔNG bị encode lại, không tốn thêm gọi API.
- Vì mỗi batch được upsert xong ngay lập tức, nếu script bị ngắt giữa
  chừng (mất mạng, Ctrl+C...), chạy lại lệnh cũ sẽ tự động chỉ xử lý
  tiếp phần còn thiếu — Qdrant Cloud chính là "checkpoint", không cần
  file checkpoint riêng trên máy.
- Dùng `--force` nếu muốn ép encode lại TOÀN BỘ (kể cả ID đã có sẵn) —
  hữu ích khi bạn đổi model embedding, hoặc sửa lại nội dung bài cũ.

Cách chạy (từ thư mục backend/):

    python build_vectors.py                        # mode=hf (mặc định)
    python build_vectors.py --mode local --local-model BAAI/bge-m3
    python build_vectors.py --force                # ép encode lại toàn bộ
    python build_vectors.py --batch-size 16
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from tqdm import tqdm

from app.config import HF_EMBEDDING_MODEL
from app.database.news_repository import NewsRepository
from app.database.qdrant_client import qdrant_store

DEFAULT_BATCH_SIZE = 32

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_vectors")


# ------------------------------------------------------------------------
# Chuẩn bị text + payload từ DataFrame
# ------------------------------------------------------------------------


def _safe_str(value) -> str:
    if value is None:
        return ""
    return str(value)


def build_embedding_text(row) -> str:
    """Ghép title + summary — KHÔNG bao giờ thêm content (noi_dung)."""
    title = _safe_str(row.get("tieu_de"))
    summary = _safe_str(row.get("summary"))
    return f"{title}\n{summary}".strip()


def build_payload(row) -> dict:
    return {
        "id": int(row["id"]),
        "title": _safe_str(row.get("tieu_de")) or None,
        "summary": _safe_str(row.get("summary")) or None,
        "publish_date": _safe_str(row.get("ngay_dang")) or None,
    }


# ------------------------------------------------------------------------
# Hai chiến lược sinh embedding: Hugging Face API / model local
# ------------------------------------------------------------------------


class BaseEmbedder:
    """Interface chung: embed_batch(texts) -> list[list[float]]."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        raise NotImplementedError


class HuggingFaceApiEmbedder(BaseEmbedder):
    """Dùng lại EmbeddingService của backend — gọi Hugging Face Inference API."""

    def __init__(self) -> None:
        from app.services.embedding_service import embedding_service

        self._service = embedding_service

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._service.embed_documents(texts)

    @property
    def model_name(self) -> str:
        return HF_EMBEDDING_MODEL


class LocalModelEmbedder(BaseEmbedder):
    """
    Chạy model SentenceTransformer ngay trên máy — CHỈ dùng offline, cho
    script này. KHÔNG import từ `app/`, KHÔNG có đường nào khiến backend
    runtime phải cài torch/sentence-transformers.

    Yêu cầu: pip install -r requirements-build-local.txt
    """

    def __init__(self, model_name: str) -> None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Chế độ --mode local cần torch + sentence-transformers, chưa được "
                "cài. Chạy: pip install -r requirements-build-local.txt"
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Đang load model local '%s' trên device='%s'...", model_name, device)
        self._model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=len(texts),
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    @property
    def model_name(self) -> str:
        return self._model_name


# ------------------------------------------------------------------------
# Encode + upsert theo batch, có progress bar
# ------------------------------------------------------------------------


def encode_and_upsert(
    embedder: BaseEmbedder,
    ids: list[int],
    texts: list[str],
    payloads: list[dict],
    batch_size: int,
) -> int:
    """
    Encode theo batch rồi upsert thẳng vào Qdrant Cloud.

    Vì mỗi batch được upsert ngay sau khi encode xong, nếu script bị ngắt
    giữa chừng thì các bài đã upsert vẫn nằm an toàn trên Qdrant Cloud —
    chạy lại `main()` sẽ tự lọc lại danh sách "còn thiếu" từ đầu.

    Trả về số chiều (dim) của vector (0 nếu không có gì để encode).
    """
    total = len(ids)
    if total == 0:
        return 0

    logger.info(
        "Bắt đầu encode %d bài báo (mode=%s, model=%s, batch_size=%d)...",
        total,
        embedder.__class__.__name__,
        embedder.model_name,
        batch_size,
    )

    dim: int | None = None

    with tqdm(total=total, unit="bài", desc="Embedding + upsert") as pbar:
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_ids = ids[start:end]
            batch_texts = texts[start:end]
            batch_payloads = payloads[start:end]

            t0 = time.time()
            vectors = embedder.embed_batch(batch_texts)
            elapsed = time.time() - t0

            if dim is None:
                dim = len(vectors[0])
                qdrant_store.ensure_collection(vector_size=dim)

            qdrant_store.upsert(ids=batch_ids, vectors=vectors, payloads=batch_payloads)

            pbar.update(len(batch_ids))
            pbar.set_postfix(
                {
                    "batch_s": f"{elapsed:.1f}s",
                    "bài/s": f"{len(batch_texts) / max(elapsed, 1e-6):.1f}",
                }
            )
            logger.debug("Batch [%d:%d] xong trong %.2fs.", start, end, elapsed)

    logger.info("Đã upsert xong %d bài báo (dim=%d) vào Qdrant Cloud.", total, dim or 0)
    return dim or 0


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Script OFFLINE: đọc cleaned_news.csv, sinh embedding, upload lên "
            "Qdrant Cloud. Không liên quan tới backend runtime."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["hf", "local"],
        default="hf",
        help="hf = Hugging Face Inference API (mặc định). local = model chạy trên máy, offline.",
    )
    parser.add_argument(
        "--local-model",
        default=HF_EMBEDDING_MODEL,
        help=f"Tên model dùng khi --mode local (mặc định: {HF_EMBEDDING_MODEL}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ép encode lại TOÀN BỘ, kể cả ID đã có sẵn trên Qdrant Cloud.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help=(
            "XOÁ SẠCH collection hiện tại trên Qdrant Cloud rồi tạo lại trống, "
            "trước khi encode lại toàn bộ từ đầu. Dùng đúng 1 LẦN khi đổi cách "
            "sinh ID (ví dụ: chuyển từ ID theo vị trí dòng sang ID theo hash URL) "
            "-- lúc đó ID cũ trên Qdrant không còn khớp với ID mới, phải xoá "
            "sạch chứ không thể chỉ 'thêm mới'."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Log chi tiết từng batch (DEBUG level)."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("=== build_vectors.py bắt đầu (mode=%s) ===", args.mode)

    logger.info("Đang đọc dữ liệu từ cleaned_news.csv...")
    repo = NewsRepository()
    repo.load()
    df = repo.df
    logger.info("Đã đọc %d bài báo trong CSV.", len(df))

    all_ids = [int(x) for x in df["id"].tolist()]

    qdrant_store.connect()

    if args.recreate:
        from app.config import QDRANT_VECTOR_SIZE

        logger.warning(
            "--recreate: XOÁ SẠCH collection '%s' hiện tại trên Qdrant Cloud rồi "
            "tạo lại trống (dim=%d)...",
            qdrant_store.collection_name,
            QDRANT_VECTOR_SIZE,
        )
        qdrant_store.recreate_collection(vector_size=QDRANT_VECTOR_SIZE)
        logger.info("Đã xoá + tạo lại collection trống — sẽ encode lại TOÀN BỘ %d bài.", len(all_ids))
        ids_to_process = set(all_ids)
    elif args.force:
        logger.info("--force: sẽ encode lại TOÀN BỘ %d bài, kể cả ID đã có sẵn.", len(all_ids))
        ids_to_process = set(all_ids)
    else:
        existing = qdrant_store.existing_ids()
        logger.info("Qdrant Cloud hiện đã có %d bài.", len(existing))
        ids_to_process = {i for i in all_ids if i not in existing}
        logger.info(
            "-> %d bài MỚI cần encode (bỏ qua %d bài đã có sẵn).",
            len(ids_to_process),
            len(all_ids) - len(ids_to_process),
        )

    if not ids_to_process:
        logger.info("Không có bài nào cần encode. Dùng --force nếu muốn build lại toàn bộ.")
        return 0

    rows = [row for _, row in df.iterrows() if int(row["id"]) in ids_to_process]
    ids = [int(row["id"]) for row in rows]
    texts = [build_embedding_text(row) for row in rows]
    payloads = [build_payload(row) for row in rows]

    empty_count = sum(1 for t in texts if t == "")
    if empty_count:
        logger.warning("%d bài báo có title+summary rỗng — vẫn encode bình thường.", empty_count)

    try:
        embedder: BaseEmbedder
        if args.mode == "hf":
            embedder = HuggingFaceApiEmbedder()
        else:
            embedder = LocalModelEmbedder(model_name=args.local_model)
    except RuntimeError as exc:
        logger.error(str(exc))
        return 1

    try:
        dim = encode_and_upsert(
            embedder,
            ids=ids,
            texts=texts,
            payloads=payloads,
            batch_size=args.batch_size,
        )
    except Exception:
        logger.exception(
            "Lỗi giữa chừng khi encode/upsert. Các bài ĐÃ upsert thành công trước khi "
            "lỗi xảy ra vẫn an toàn trên Qdrant Cloud — chạy lại lệnh này để tiếp tục "
            "với các bài còn thiếu, không cần encode lại từ đầu."
        )
        return 1

    logger.info("=== Hoàn tất: đã xử lý %d bài mới, dim=%d. ===", len(ids), dim)
    return 0


if __name__ == "__main__":
    sys.exit(main())
