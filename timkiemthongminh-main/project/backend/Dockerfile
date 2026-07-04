FROM python:3.11.9-slim

WORKDIR /app

# Cài đặt dependencies trước để tận dụng cache layer khi code thay đổi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ backend (bao gồm data/cleaned_news.csv)
COPY . .

# Render tự set biến PORT lúc chạy container (thường 10000) — mặc định
# 8000 ở đây chỉ dùng khi chạy local qua `docker run`/docker-compose mà
# không set PORT. Dùng CMD dạng shell để expand ${PORT} lúc container
# khởi động, KHÔNG hardcode port cụ thể như bản trước (7860, dành riêng
# cho Hugging Face Spaces).
ENV PORT=8000
EXPOSE 8000

# QDRANT_URL, QDRANT_API_KEY, HF_API_TOKEN... sẽ được set qua Environment
# Variables trên Render (hoặc file .env khi chạy local) — KHÔNG hardcode.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
