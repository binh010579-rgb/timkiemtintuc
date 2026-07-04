# Tìm Kiếm Tin Tức Thông Minh

Ứng dụng tìm kiếm tin tức bằng **semantic search**: frontend React + Vite,
backend FastAPI, vector embedding lưu trên **Qdrant Cloud**, embedding
sinh qua **Hugging Face Inference API** (không chạy model AI nào trên
server). Thiết kế để deploy miễn phí: **Render** (backend) + **Vercel**
(frontend) + **Qdrant Cloud** (vector store).

> Phiên bản trước dùng Qdrant chạy local qua Docker + model Qwen3 Embedding
> tải về máy. Kiến trúc đó đã được thay thế hoàn toàn bởi bản cloud này —
> không cần Docker cho Qdrant nữa, không cần tải model nào về máy.

## Kiến trúc

```
React (Vercel)
     │  fetch JSON
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

- **Không có database quan hệ.** `backend/data/cleaned_news.csv` là nguồn
  dữ liệu duy nhất, đọc 1 lần khi server khởi động, giữ trong RAM.
- **Qdrant Cloud** lưu vector embedding (title + summary) của từng bài
  báo — tìm k-NN hoàn toàn do Qdrant Search API đảm nhiệm, backend không
  tự viết cosine similarity, không cache `.npz`.
- **Sinh embedding là bước riêng, offline** (`build_vectors.py`), tách
  khỏi vòng đời server — server chỉ gọi Hugging Face API để embed câu
  query, không bulk-encode lại dữ liệu mỗi lần restart, không load
  `SentenceTransformer`/`torch`.

## Cấu trúc thư mục

```
project/
├── backend/            # FastAPI — xem backend/README chi tiết bên dưới
│   ├── app/
│   ├── build_vectors.py
│   ├── data/cleaned_news.csv
│   └── Dockerfile
├── frontend/            # React + Vite + Tailwind + TanStack Query
│   └── src/
├── docker-compose.yml   # Chạy backend local bằng Docker
└── render.yaml           # Blueprint deploy backend lên Render
```

## Chạy local

### 1. Backend

```bash
cd project/backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Điền HF_API_TOKEN, QDRANT_URL, QDRANT_API_KEY trong .env
# (tạo cluster miễn phí tại https://cloud.qdrant.io, tạo token đọc tại
#  https://huggingface.co/settings/tokens)

uvicorn main:app --reload --port 8000
```

Swagger UI: `http://localhost:8000/docs` · Health check: `http://localhost:8000/health`

Sinh & upload vector lên Qdrant Cloud (chạy tay, mỗi khi CSV thay đổi):

```bash
pip install -r requirements.txt -r requirements-build.txt
python build_vectors.py
```

Script có progress bar, retry, logging, resume khi lỗi giữa chừng — xem
`project/backend/README` (docstring đầu file `build_vectors.py`) để biết
thêm chi tiết.

### 2. Frontend

```bash
cd project/frontend
npm install
cp .env.example .env   # tuỳ chọn — mặc định đã trỏ tới http://localhost:8000
npm run dev
```

Mở `http://localhost:5173`. Frontend gọi backend qua `VITE_API_BASE_URL`
(mặc định `http://localhost:8000` khi không set — xem
`src/services/apiClient.ts`).

### Chạy backend bằng Docker (tuỳ chọn)

```bash
cp project/backend/.env.example project/backend/.env   # điền giá trị thật
docker compose up --build
```

## Biến môi trường

### Backend (`project/backend/.env`)

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
| `CACHE_TTL_SECONDS` | | TTL cache embedding query, giây (mặc định `1800`) |
| `LOG_LEVEL` | | Mặc định `INFO` |

Xem đầy đủ (kèm giá trị mặc định) trong `project/backend/.env.example`.

### Frontend (`project/frontend/.env`)

| Biến | Bắt buộc | Mô tả |
|---|---|---|
| `VITE_API_BASE_URL` | | URL backend. Mặc định `http://localhost:8000` khi không set |

## API chính (backend)

- `GET /health` — trạng thái `status`, `qdrant`, `embedding_api`, `version`, `timestamp`.
- `POST /search` — semantic search: `{"query": "..."}` → `title, summary, content, url, image, date, source, score`.
- `GET /api/search?q=...` — tìm kiếm từ khoá (khác semantic search).
- `GET /api/news`, `GET /api/news/featured`, `GET /api/categories`.

## Deploy

### 1. Qdrant Cloud
Tạo cluster miễn phí tại [cloud.qdrant.io](https://cloud.qdrant.io), lấy
`Cluster URL` + `API Key`. Không cần tạo collection tay — backend tự tạo
nếu chưa có, sau đó chạy `build_vectors.py` (từ máy local, trỏ tới cùng
`QDRANT_URL`/`QDRANT_API_KEY`) để nạp dữ liệu.

### 2. Render (backend)
- Dùng `render.yaml` ở gốc repo (Blueprint), hoặc tạo Web Service thủ
  công: **Docker**, Root Directory = `project/backend`.
- Set các biến môi trường bắt buộc (xem bảng trên).
- Health Check Path: `/health`.

### 3. Vercel (frontend)
- Import repo, **Root Directory** = `project/frontend` (Vercel tự nhận
  diện Vite qua `vercel.json`).
- Set `VITE_API_BASE_URL` = domain Render.
- Sau khi có domain Vercel, cập nhật `ALLOWED_ORIGINS` trên Render bằng
  đúng domain đó (CORS).

## Ghi chú kiến trúc / quyết định thiết kế

- **TTL Cache**: chỉ cache vector của **query** tìm kiếm, TTL 30 phút,
  toàn bộ trong RAM (`cachetools.TTLCache`) — không ghi file, mất khi
  restart server.
- **Re-ranking**: bước tuỳ chọn trong pipeline. Vì kiến trúc cấm chạy
  model cục bộ, mặc định (`PassthroughReranker`) giữ nguyên thứ tự Qdrant
  trả về — là điểm mở rộng nếu sau này cần rerank thật qua 1 Inference
  API khác (không load model trên server).
- **Exception Handling**: tập trung ở `app/core/exceptions.py` (lỗi
  Hugging Face → 502, lỗi Qdrant → 503, lỗi khác → 500).
- **Health Check**: `GET /health` gọi 1 request rất nhẹ tới Qdrant và
  Hugging Face, không build/encode gì, không đụng dữ liệu.
