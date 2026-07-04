# News Semantic Search — Backend (Cloud, Production-Ready)

Backend FastAPI cho hệ thống tìm kiếm ngữ nghĩa (semantic search) tin
tức, thiết kế để chạy **stateless** và deploy miễn phí trên **Render**.
Xem `../README.md` (gốc project) để biết hướng dẫn tổng quan cả
frontend + backend + deploy.

## Kiến trúc

```
React (Vercel)
     │
     ▼
FastAPI (Render)
     │
     ▼
TTL Cache (RAM, 30 phút)
     │  (cache miss)
     ▼
Embedding Service — Hugging Face Inference API (BAAI/bge-m3)
     │
     ▼
Qdrant Cloud (k-NN, top 30 ứng viên)
     │
     ▼
(Tuỳ chọn) Re-ranking
     │
     ▼
Top 10 — JSON Response
```

Nguyên tắc bắt buộc (đã tuân thủ trong toàn bộ code):

- **Không** load `SentenceTransformer`/`torch` trên server runtime.
- **Không** build/encode lại vector khi backend khởi động.
- **Không** tạo collection Qdrant mới nếu đã tồn tại — chỉ kết nối.
- Toàn bộ vector nằm trên **Qdrant Cloud**, không có bản sao local.
- Semantic Search luôn dùng **vector search** thật (không SQL `LIKE`,
  không Full Text Search thay thế).

## Cấu trúc thư mục

```
backend/
├── app/
│   ├── api/            # Router — chỉ gọi service, không chứa business logic
│   │   ├── health.py   # GET /health
│   │   ├── news.py     # GET /api/news, /api/news/featured, /api/categories, /api/search
│   │   └── search.py   # POST /search (semantic search)
│   ├── core/
│   │   ├── cache.py        # TTLCache (RAM) cho embedding của query
│   │   ├── exceptions.py   # Exception handler tập trung
│   │   └── logging.py      # Cấu hình logging tập trung
│   ├── database/
│   │   ├── news_repository.py  # Metadata bài báo (CSV -> RAM)
│   │   └── qdrant_client.py    # Wrapper Qdrant Cloud (connect/ensure/search/upsert)
│   ├── middleware/
│   │   └── logging_middleware.py  # Log method/path/status/thời gian mỗi request
│   ├── models/          # Pydantic Request/Response models
│   ├── services/
│   │   ├── embedding_service.py  # HTTP client gọi Hugging Face Inference API
│   │   ├── news_service.py       # Business logic danh sách/danh mục/từ khoá
│   │   ├── rerank_service.py     # Re-ranking (tuỳ chọn, mặc định passthrough)
│   │   └── search_service.py     # Điều phối Search Pipeline
│   ├── utils/
│   │   └── time.py       # Helper timestamp (dùng bởi /health, error response)
│   ├── config.py         # Cấu hình tập trung, đọc từ biến môi trường
│   └── main.py           # FastAPI app, lifespan, middleware, router
├── build_vectors.py       # Script OFFLINE — sinh & upload vector (KHÔNG chạy bởi server)
├── data/cleaned_news.csv  # Dữ liệu nguồn
├── Dockerfile
├── requirements.txt        # Dependency backend runtime (nhẹ, không torch)
├── requirements-build.txt        # Dependency thêm cho build_vectors.py (mode hf)
├── requirements-build-local.txt  # Dependency thêm cho build_vectors.py (mode local, có torch)
└── .env.example
```

## Chạy local

```bash
cd project/backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Điền HF_API_TOKEN, QDRANT_URL, QDRANT_API_KEY trong .env

uvicorn main:app --reload --port 8000
```

Swagger UI: `http://localhost:8000/docs` · Health check: `http://localhost:8000/health`

## Sinh & upload vector (offline, tách biệt khỏi server)

`build_vectors.py` **không** được gọi bởi backend runtime — chạy tay mỗi
khi `cleaned_news.csv` thay đổi:

```bash
pip install -r requirements.txt -r requirements-build.txt   # mode hf (mặc định)

python build_vectors.py                # sinh embedding qua Hugging Face API, upsert Qdrant Cloud
python build_vectors.py --force        # ép encode lại toàn bộ
python build_vectors.py --batch-size 16
```

Script có: progress bar (tqdm), retry, logging có timestamp, resume khi
lỗi giữa chừng (checkpoint theo ID), và tự bỏ qua nếu dữ liệu không đổi.
Nếu ID đã tồn tại trên Qdrant, upsert sẽ tự **update** thay vì tạo trùng.

## Biến môi trường

| Biến | Bắt buộc | Mô tả |
|---|---|---|
| `HF_API_TOKEN` | ✅ | Token Hugging Face (quyền Read) |
| `HF_EMBEDDING_MODEL` | | Mặc định `BAAI/bge-m3` |
| `QDRANT_URL` | ✅ | URL Qdrant Cloud (bắt buộc `https://`) |
| `QDRANT_API_KEY` | ✅ | API key Qdrant Cloud |
| `QDRANT_COLLECTION_NAME` | | Mặc định `news_articles` |
| `QDRANT_VECTOR_SIZE` | | Mặc định `1024` (dim của BGE-M3) |
| `ALLOWED_ORIGINS` | | Domain frontend, phân cách dấu phẩy |
| `SEARCH_TOP_K` | | Số kết quả cuối cùng trả về (mặc định `10`) |
| `SEARCH_CANDIDATE_K` | | Số ứng viên lấy từ Qdrant trước khi rerank (mặc định `30`) |
| `CACHE_TTL_SECONDS` | | TTL cache embedding query, giây (mặc định `1800` = 30 phút) |
| `LOG_LEVEL` | | Mặc định `INFO` |

Xem đầy đủ trong `.env.example`.

## API chính

- `GET /health` — trạng thái `status`, `qdrant`, `embedding_api`, `version`, `timestamp`.
- `POST /search` — semantic search: `{"query": "..."}` → danh sách `title, summary, content, url, image, date, source, score`.
- `GET /api/search?q=...` — tìm kiếm từ khoá (khác semantic search).
- `GET /api/news`, `GET /api/news/featured`, `GET /api/categories`.

## Ghi chú kiến trúc / quyết định thiết kế

- **TTL Cache**: nằm ở tầng service (`search_service.py`), chỉ cache
  vector của **query**, TTL 30 phút, chỉ trong RAM (`cachetools.TTLCache`),
  KHÔNG ghi file — mất khi restart, đúng yêu cầu.
- **Re-ranking**: là bước tuỳ chọn theo pipeline yêu cầu. Vì kiến trúc
  cấm chạy model cục bộ, implementation mặc định (`PassthroughReranker`)
  chỉ giữ nguyên thứ tự Qdrant trả về. Đây là điểm mở rộng — nếu cần
  rerank thật, chỉ nên gọi qua 1 Inference API khác (không load model
  trên server), tương tự cách `EmbeddingService` gọi Hugging Face.
- **Exception Handling**: tập trung ở `app/core/exceptions.py` (lỗi
  Hugging Face → 502, lỗi Qdrant → 503, lỗi khác → 500), router không
  cần tự bắt lỗi lặp lại.
- **Health Check**: `GET /health` gọi 1 request rất nhẹ tới Qdrant
  (`get_collections`) và Hugging Face (`wait_for_model=False`, timeout
  ngắn) — không build/encode gì, không đụng dữ liệu.
