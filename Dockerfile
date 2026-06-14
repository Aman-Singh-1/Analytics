FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# bake the embedding + reranker models into the image so cold start is fast
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('BAAI/bge-small-en-v1.5'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY app ./app
COPY data/faiss_index ./data/faiss_index

# HF Spaces (Docker SDK) expects the app on 7860
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
