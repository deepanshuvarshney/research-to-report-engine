# Public demo image: Postgres + pgvector, Redis, and the API in one container.
# Render's free instance has 512 MB of RAM, so this image does not install
# fastembed and the entrypoint starts a single inline worker.
FROM pgvector/pgvector:pg16

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH" \
    MALLOC_ARENA_MAX=2 \
    DEMO_MODE=true \
    JOB_EXECUTOR=inline \
    EMBEDDING_PROVIDER=hash \
    LLM_PROVIDER=mock \
    FRONTEND_DIR=/app/frontend

RUN printf '#!/bin/sh\nexit 101\n' > /usr/sbin/policy-rc.d \
    && chmod +x /usr/sbin/policy-rc.d \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        redis-server \
        redis-tools \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY frontend ./frontend
COPY deploy/render/entrypoint.sh /entrypoint.sh

RUN python3 -m venv /opt/venv \
    && pip install --no-cache-dir . \
    && chmod +x /entrypoint.sh \
    && mkdir -p /var/lib/postgresql/data /var/run/postgresql \
    && chown -R postgres:postgres /var/lib/postgresql /var/run/postgresql

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
