#!/usr/bin/env python3
"""
Script OFFLINE, ĐỘC LẬP HOÀN TOÀN với backend runtime (`app/main.py`).

Mục đích DUY NHẤT: đọc `data/cleaned_news.csv`, sinh embedding cho
title+summary, rồi upload (upsert) toàn bộ vector lên Qdrant Cloud.

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

Đặc điểm production của script:
- Logging có timestamp/level (module `logging`), không dùng print rời rạc.
- Progress bar (tqdm) theo batch.
- RESUME khi bị lỗi giữa chừng: mỗi batch được encode + upsert xong mới
  ghi checkpoint (`data/build_vectors_checkpoint.json`). Nếu script bị
  ngắt (lỗi mạng, Ctrl+C, mất điện...), chạy lại sẽ tự bỏ qua các bài đã
  upsert thành công, chỉ xử lý tiếp phần còn lại — KHÔNG encode lại từ đầu.
- Upsert theo ID: nếu ID đã tồn tại trong collection, Qdrant tự UPDATE
  point đó (ghi đè vector + payload) — đây là hành vi mặc định của
  `client.upsert()`, không cần logic riêng để phân biệt insert/update.
- Bỏ qua toàn bộ nếu dữ liệu + model không đổi so với lần chạy thành công
  gần nhất (so khớp hash nội dung) — dùng `--force` để ép chạy lại.

Cách chạy (từ thư mục backend/):

    python build_vectors.py                        # mode=hf (mặc định)
    python build_vectors.py --mode local --local-model BAAI/bge-m3
    python build_vectors.py --force                # ép encode lại toàn bộ
    python build_vectors.py --batch-size 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field

from tqdm import tqdm

from app.config import BASE_DIR, HF_EMBEDDING_MODEL
from app.database.news_repository import NewsRepository
from app.database.qdrant_client import qdrant_store

# ------------------------------------------------------------------------
# Cấu hình đường dẫn state/checkpoint + logging
# ------------------------------------------------------------------------

EMBEDDING_STATE_PATH = os.path.join(BASE_DIR, "data", "embedding_state.json")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "data", "build_vectors_checkpoint.json")
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


def texts_hash(texts: list[str], model_name: str) -> str:
    """Hash nội dung + tên model — biết khi nào cần encode lại từ đầu."""
    hasher = hashlib.sha256()
    hasher.update(model_name.encode("utf-8"))
    for t in texts:
        hasher.update(t.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


# ------------------------------------------------------------------------
# State (toàn bộ dữ liệu đã encode xong ở lần chạy TRƯỚC, thành công)
# ------------------------------------------------------------------------


def load_state() -> dict | None:
    if not os.path.exists(EMBEDDING_STATE_PATH):
        return None
    try:
        with open(EMBEDDING_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Không đọc được state (%s) — sẽ encode lại từ đầu.", exc)
        return None


def save_state(hash_value: str, count: int, dim: int) -> None:
    os.makedirs(os.path.dirname(EMBEDDING_STATE_PATH), exist_ok=True)
    with open(EMBEDDING_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump({"texts_hash": hash_value, "count": count, "dim": dim}, f)


# ------------------------------------------------------------------------
# Checkpoint (RESUME khi chạy giữa chừng bị lỗi/ngắt)
# ------------------------------------------------------------------------


@dataclass
class Checkpoint:
    content_hash: str = ""
    completed_ids: set[int] = field(default_factory=set)

    @classmethod
    def load(cls, expected_hash: str) -> "Checkpoint":
        """
        Đọc checkpoint cũ. Nếu hash nội dung không khớp (dữ liệu/model đã
        đổi so với lần chạy dở dang trước), checkpoint cũ không còn hợp lệ
        -> bắt đầu lại từ đầu (completed_ids rỗng).
        """
        if not os.path.exists(CHECKPOINT_PATH):
            return cls(content_hash=expected_hash)
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("content_hash") != expected_hash:
                logger.info("Checkpoint cũ thuộc lần chạy khác (dữ liệu đã đổi) — bỏ qua.")
                return cls(content_hash=expected_hash)
            completed = set(data.get("completed_ids", []))
            if completed:
                logger.info(
                    "Tìm thấy checkpoint dở dang: %d bài đã upsert thành công trước đó — "
                    "sẽ RESUME, bỏ qua các bài này.",
                    len(completed),
                )
            return cls(content_hash=expected_hash, completed_ids=completed)
        except Exception as exc:
            logger.warning("Không đọc được checkpoint (%s) — bắt đầu lại từ đầu.", exc)
            return cls(content_hash=expected_hash)

    def save(self) -> None:
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {"content_hash": self.content_hash, "completed_ids": sorted(self.completed_ids)},
                f,
            )

    def mark_done(self, ids: list[int]) -> None:
        self.completed_ids.update(ids)
        self.save()

    def clear(self) -> None:
        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)


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
# Encode + upsert theo batch, có progress bar + resume + logging
# ------------------------------------------------------------------------


def encode_and_upsert(
    embedder: BaseEmbedder,
    ids: list[int],
    texts: list[str],
    payloads: list[dict],
    batch_size: int,
    checkpoint: Checkpoint,
) -> int:
    """
    Encode theo batch rồi upsert thẳng vào Qdrant Cloud. Bỏ qua các ID đã
    có trong checkpoint (đã upsert thành công ở lần chạy trước bị ngắt).

    Nếu ID đã tồn tại trong collection, `qdrant_store.upsert()` tự UPDATE
    (ghi đè) point đó — hành vi mặc định của Qdrant upsert.

    Trả về số chiều (dim) của vector.
    """
    pending = [
        (i, t, p) for i, t, p in zip(ids, texts, payloads) if i not in checkpoint.completed_ids
    ]
    if not pending:
        logger.info("Không còn bài nào cần encode (đã hoàn tất từ checkpoint trước).")
        return 0

    total_all = len(ids)
    total_pending = len(pending)
    logger.info(
        "Bắt đầu encode %d/%d bài báo (mode=%s, model=%s, batch_size=%d)...",
        total_pending,
        total_all,
        embedder.__class__.__name__,
        embedder.model_name,
        batch_size,
    )

    qdrant_store.connect()
    dim: int | None = None

    with tqdm(total=total_pending, unit="bài", desc="Embedding + upsert") as pbar:
        for start in range(0, total_pending, batch_size):
            batch = pending[start : start + batch_size]
            batch_ids = [b[0] for b in batch]
            batch_texts = [b[1] for b in batch]
            batch_payloads = [b[2] for b in batch]

            t0 = time.time()
            vectors = embedder.embed_batch(batch_texts)
            elapsed = time.time() - t0

            if dim is None:
                dim = len(vectors[0])
                qdrant_store.ensure_collection(vector_size=dim)

            qdrant_store.upsert(ids=batch_ids, vectors=vectors, payloads=batch_payloads)

            # Chỉ đánh dấu "đã xong" SAU KHI upsert thành công -> nếu lỗi
            # xảy ra ở batch này (encode hoặc upsert), checkpoint vẫn giữ
            # nguyên trạng thái của các batch trước đó, không mất tiến độ.
            checkpoint.mark_done(batch_ids)

            pbar.update(len(batch))
            pbar.set_postfix(
                {
                    "batch_s": f"{elapsed:.1f}s",
                    "bài/s": f"{len(batch_texts) / max(elapsed, 1e-6):.1f}",
                }
            )
            logger.debug(
                "Batch [%d:%d] xong trong %.2fs.", start, start + len(batch), elapsed
            )

    logger.info("Đã upsert xong %d bài báo (dim=%d) vào Qdrant Cloud.", total_pending, dim or 0)
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
    parser.add_argument("--force", action="store_true", help="Ép encode lại toàn bộ dữ liệu.")
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
    logger.info("Đã đọc %d bài báo.", len(df))

    texts = [build_embedding_text(row) for _, row in df.iterrows()]
    ids = df["id"].tolist()
    payloads = [build_payload(row) for _, row in df.iterrows()]

    empty_count = sum(1 for t in texts if t == "")
    if empty_count:
        logger.warning("%d bài báo có title+summary rỗng — vẫn encode bình thường.", empty_count)

    model_name = args.local_model if args.mode == "local" else HF_EMBEDDING_MODEL
    hash_value = texts_hash(texts, f"{args.mode}:{model_name}")

    # --- Kiểm tra: dữ liệu + model có đổi so với lần chạy THÀNH CÔNG gần
    # nhất không? Nếu không đổi và Qdrant đã đủ vector -> bỏ qua toàn bộ. ---
    state = load_state()
    qdrant_store.connect()
    qdrant_count = qdrant_store.count()

    data_unchanged = (
        not args.force
        and state is not None
        and state.get("texts_hash") == hash_value
        and state.get("count") == len(texts)
        and qdrant_count == len(texts)
    )
    if data_unchanged:
        logger.info(
            "Dữ liệu + model không đổi, Qdrant Cloud đã có đủ %d vector — bỏ qua. "
            "Dùng --force để ép chạy lại.",
            qdrant_count,
        )
        return 0

    if args.force:
        logger.info("--force: bỏ checkpoint cũ, encode lại toàn bộ từ đầu.")
        Checkpoint(content_hash=hash_value).clear()

    checkpoint = Checkpoint.load(expected_hash=hash_value)

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
            checkpoint=checkpoint,
        )
    except Exception:
        logger.exception(
            "Lỗi giữa chừng khi encode/upsert. Tiến độ đã hoàn thành (%d/%d bài) được "
            "giữ lại trong checkpoint — chạy lại lệnh này để RESUME, không cần encode "
            "lại từ đầu.",
            len(checkpoint.completed_ids),
            len(ids),
        )
        return 1

    # Hoàn tất toàn bộ -> lưu state chính thức + xoá checkpoint dở dang.
    final_dim = dim or state.get("dim", 0) if state else dim
    save_state(hash_value, count=len(texts), dim=final_dim)
    checkpoint.clear()

    logger.info("=== Hoàn tất: %d bài báo, dim=%d. ===", len(texts), final_dim)
    return 0


if __name__ == "__main__":
    sys.exit(main())
