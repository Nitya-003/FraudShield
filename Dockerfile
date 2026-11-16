# Multi-stage Dockerfile for FraudShield Unified Platform

FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend-web


COPY frontend-web/package*.json ./
RUN npm ci

COPY frontend-web/ ./
RUN npm run build

FROM python:3.11-slim AS runtime
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ml/ ./ml/
COPY models/ ./models/
COPY data/processed/ ./data/processed/
COPY reports/ ./reports/
COPY run.py ./

COPY --from=frontend-builder /app/frontend-web/dist ./frontend-web/dist

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch FraudShield server
CMD ["python", "run.py", "--no-browser", "--host", "0.0.0.0", "--port", "8000"]

