# Inference-only container for Hugging Face Spaces (Docker SDK).
# Serves the trained models via gunicorn; performs NO training at runtime.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    PORT=7860

WORKDIR /app

# Install lean, CPU-only dependencies first for better layer caching.
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy the app, the inference code, and the committed trained models.
COPY main.py .
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY templates/ ./templates/
COPY static/ ./static/

EXPOSE 7860

# 1 worker keeps the (small) models in memory once; 2 threads for concurrency.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "120", "main:app"]
