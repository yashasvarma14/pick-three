# Pick Three on Hugging Face Spaces (Docker Space).
# One container serves both the API and the website on port 7860.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface

# Spaces run containers as user 1000.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

USER user
# Bake the embedding model into the image so the first visitor doesn't wait for a download.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY --chown=user . .

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
