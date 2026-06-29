FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim as runner

WORKDIR /app

COPY --from=builder /root/.local /root/.local

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

COPY api/ api/
COPY src/ src/
COPY config/ config/
COPY params.yaml .

EXPOSE 7860

CMD ["sh", "-c", "celery -A api.workers.celery_app worker --loglevel=info & uvicorn api.main:app --host 0.0.0.0 --port 7860"]
