# NYC Event Scout — single-container image (API + static frontend).
# Build/run via docker-compose up (see docker-compose.yml).
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Keep HF model cache inside the image/container, not the host user dir.
    HF_HOME=/app/.hf-cache

# CPU-only torch first: sentence-transformers would otherwise pull the default
# CUDA build (~6 GB of GPU libraries we can't use in this container).
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so the first /search doesn't stall
# on a ~90 MB download (and the demo works without Hugging Face reachable).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
