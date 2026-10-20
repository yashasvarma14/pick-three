# Pick Three on Google Cloud Run.
# One container serves both the API and the website, on the port Cloud Run gives it ($PORT).
FROM python:3.12-slim

# FASTEMBED_CACHE_PATH is where the embedding model gets stored inside the image.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed

# Run as a regular user, not root.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

USER user
# Bake the embedding model into the image so the first visitor doesn't wait for a download.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2', cache_dir='/home/user/.cache/fastembed')"

COPY --chown=user . .

EXPOSE 8080
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers --forwarded-allow-ips '*'"]
