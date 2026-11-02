FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/vellum

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 vellum \
    && mkdir -p /home/vellum/.cache \
    && chown -R vellum:vellum /home/vellum /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY frontend ./frontend

RUN pip install --no-cache-dir ".[embeddings]"

USER vellum
EXPOSE 8000

CMD ["uvicorn", "research_engine.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
