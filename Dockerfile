# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

# System dependencies required by the ingestion/parsing stack:
# - poppler-utils / ghostscript: PDF rendering & table extraction (camelot, pdf2image)
# - tesseract-ocr: OCR (pytesseract / unstructured)
# - libgl1: required by opencv-python at import time
# - build-essential: native extensions for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    ghostscript \
    tesseract-ocr \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code.
COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
