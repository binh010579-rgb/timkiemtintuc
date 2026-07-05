"""
Cấu hình tập trung cho backend (kiến trúc cloud, stateless embedding).

Toàn bộ giá trị nhạy cảm (API key, URL Qdrant Cloud, HF token) PHẢI đến từ
biến môi trường — không hardcode, không có giá trị "embedded/local" mặc
định như bản cũ. Thiếu biến bắt buộc sẽ raise lỗi ngay khi import module
này (fail fast khi server khởi động, thay vì lỗi mập mờ lúc có request).
"""

import os

try:
    # Chỉ có tác dụng khi chạy LOCAL và có file .env trong thư mục backend/.
    # Trên production (Render), Anthropic/Render tự bơm biến môi trường thật
    # vào process trước khi Python chạy, nên `override=False` đảm bảo KHÔNG
    # bao giờ ghi đè lên giá trị thật đã có sẵn trong môi trường production —
    # load_dotenv() ở đây chỉ điền vào chỗ trống (fallback cho local dev).
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:
    # python-dotenv không có trong requirements.txt của backend runtime
    # (chỉ cần cho local dev) — nếu chưa cài, coi như không có .env,
    # dựa hoàn toàn vào biến môi trường hệ thống (đúng hành vi production).
    pass


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Thiếu biến môi trường bắt buộc: {name}. "
            "Set biến này trong file .env (local) hoặc Secrets (production) "
            "trước khi khởi động server."
        )
    return value


# Thư mục gốc của backend (.../backend)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Nguồn dữ liệu nội dung bài báo (metadata: title/summary/content/url...) ---
# ĐÃ CHUYỂN sang Postgres (Neon) — xem app/database/postgres_news_repository.py.
# CSV_PATH giữ lại (legacy) cho app/database/news_repository.py (bản CSV cũ),
# phòng khi cần fallback/so sánh cục bộ, không còn được main.py sử dụng.
CSV_PATH = os.path.join(BASE_DIR, "data", "cleaned_news.csv")

# Connection string Postgres (Neon), dạng:
# postgresql://user:password@host/dbname?sslmode=require
DATABASE_URL = _require_env("DATABASE_URL")

# --- CORS ---
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

# --- Phân trang ---
MAX_PAGE_LIMIT = 100
DEFAULT_PAGE_LIMIT = 10

# --- Hugging Face Inference API (embedding) ---
# KHÔNG load model nào ở local — mọi việc sinh embedding đều gọi qua API
# này. Model multilingual, hỗ trợ tốt tiếng Việt.
HF_API_TOKEN = _require_env("HF_API_TOKEN")
HF_EMBEDDING_MODEL = os.environ.get("HF_EMBEDDING_MODEL", "BAAI/bge-m3")

# Model cross-encoder dùng để RERANK (xem app/services/rerank_service.py).
# Mặc định BAAI/bge-reranker-v2-m3 — cùng họ BGE với model embedding,
# multilingual, hỗ trợ tốt tiếng Việt. KHÔNG load model này cục bộ, chỉ
# gọi qua HF Inference API (pipeline text-classification).
HF_RERANKER_MODEL = os.environ.get("HF_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
# LƯU Ý: HF đã khai tử hoàn toàn domain "api-inference.huggingface.co"
# (không còn resolve DNS -> lỗi "No address associated with hostname").
# Endpoint mới là "router.huggingface.co" (Inference Providers router).
HF_API_BASE_URL = os.environ.get(
    "HF_API_BASE_URL", "https://router.huggingface.co/hf-inference"
)
HF_API_TIMEOUT_SECONDS = float(os.environ.get("HF_API_TIMEOUT_SECONDS", "30"))
HF_API_MAX_RETRIES = int(os.environ.get("HF_API_MAX_RETRIES", "3"))
HF_API_RETRY_BACKOFF_SECONDS = float(os.environ.get("HF_API_RETRY_BACKOFF_SECONDS", "2"))

# BGE-M3 không yêu cầu instruction prefix cho query (khác Qwen3-Embedding) —
# đây là model đối xứng (symmetric retrieval), document và query dùng
# chung một cách encode. Giữ hằng số này để dễ đổi model khác sau này nếu
# model đó lại yêu cầu instruction.
EMBEDDING_QUERY_PREFIX = os.environ.get("EMBEDDING_QUERY_PREFIX", "")

# --- Qdrant Cloud (vector store — BẮT BUỘC dùng Qdrant Cloud, không có
# chế độ embedded/local nào trong kiến trúc này) ---
QDRANT_URL = _require_env("QDRANT_URL")
if not QDRANT_URL.lower().startswith("https://"):
    raise RuntimeError(
        f"QDRANT_URL phải dùng HTTPS (nhận được: '{QDRANT_URL}'). "
        "Qdrant Cloud luôn cấp URL dạng https://xxxxx.cloud.qdrant.io — "
        "kiểm tra lại giá trị biến môi trường QDRANT_URL."
    )
QDRANT_API_KEY = _require_env("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "news_articles")

# Số chiều vector của model embedding đang dùng (BAAI/bge-m3 = 1024).
# Cần biết trước để tạo collection mới mà KHÔNG phải encode thử 1 câu lúc
# khởi động server (điều đó sẽ vi phạm nguyên tắc "không build vector lúc
# khởi động"). Đổi model khác thì nhớ đổi luôn giá trị này.
QDRANT_VECTOR_SIZE = int(os.environ.get("QDRANT_VECTOR_SIZE", "1024"))

# Retry khi kết nối Qdrant Cloud thất bại (mạng chập chờn, cluster đang
# resume sau khi ngủ đông ở free tier...).
QDRANT_CONNECT_MAX_RETRIES = int(os.environ.get("QDRANT_CONNECT_MAX_RETRIES", "5"))
QDRANT_CONNECT_RETRY_BACKOFF_SECONDS = float(
    os.environ.get("QDRANT_CONNECT_RETRY_BACKOFF_SECONDS", "2")
)
QDRANT_CONNECT_TIMEOUT_SECONDS = float(os.environ.get("QDRANT_CONNECT_TIMEOUT_SECONDS", "10"))

# --- Search Pipeline ---
# Query -> HF Inference API (embedding) -> Qdrant Cloud (k-NN) -> Top K
# -> hydrate content đầy đủ từ repository -> trả JSON.
SEARCH_TOP_K = int(os.environ.get("SEARCH_TOP_K", "10"))

_threshold_raw = os.environ.get("SEARCH_SCORE_THRESHOLD", "0.4")
# Ngưỡng cosine similarity tối thiểu (BGE-M3, đã normalize) để 1 kết quả
# được coi là "đủ liên quan" và trả về. KHÔNG để None/trống mặc định nữa —
# lý do: nếu không có ngưỡng, Qdrant luôn trả về top_k điểm GẦN NHẤT hiện
# có trong collection dù độ gần đó thực chất rất thấp (VD: query gõ không
# dấu, hoặc query lệch hẳn chủ đề dataset) — gây ra kết quả "trớt quớt"
# hiển thị cho người dùng thay vì báo "không tìm thấy".
#
# 0.4 là điểm khởi đầu hợp lý cho BGE-M3 (dựa trên phân phối cosine
# similarity thường gặp: match tốt thường >= 0.5-0.6, match yếu/nhiễu
# thường < 0.3), NHƯNG cần tinh chỉnh dựa trên dữ liệu thật — xem log
# "Query %r: %d candidate, score min=...max=..." ở search_service.py sau
# khi deploy để biết phân phối score thực tế của dataset bạn, rồi chỉnh
# lại giá trị này qua biến môi trường SEARCH_SCORE_THRESHOLD trên Render
# (không cần sửa code/redeploy). Đặt SEARCH_SCORE_THRESHOLD="" (chuỗi
# rỗng) nếu muốn tắt hẳn ngưỡng lọc (quay về hành vi cũ).
SEARCH_SCORE_THRESHOLD: float | None = float(_threshold_raw) if _threshold_raw.strip() else None

# Số ứng viên lấy từ Qdrant TRƯỚC bước re-ranking (tuỳ chọn) — xem
# app/services/rerank_service.py. Luôn >= SEARCH_TOP_K.
SEARCH_CANDIDATE_K = int(os.environ.get("SEARCH_CANDIDATE_K", "30"))

# --- Cache TTL trong RAM cho embedding của query (app/core/cache.py) ---
CACHE_TTL_SECONDS = float(os.environ.get("CACHE_TTL_SECONDS", str(30 * 60)))
CACHE_MAXSIZE = int(os.environ.get("CACHE_MAXSIZE", "512"))

# --- Logging & version (dùng bởi app/core/logging.py và GET /health) ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
APP_VERSION = os.environ.get("APP_VERSION", "2.1.0")
