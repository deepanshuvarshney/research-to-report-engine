#!/bin/bash
# Start the local database and cache, then the API. The filesystem is ephemeral.
set -euo pipefail

export PGDATA="${PGDATA:-/var/lib/postgresql/data}"
export DATABASE_URL="postgresql+asyncpg://vellum:vellum@127.0.0.1:5432/vellum"
export REDIS_URL="redis://127.0.0.1:6379/0"
export DEMO_MODE="${DEMO_MODE:-true}"
export JOB_EXECUTOR="${JOB_EXECUTOR:-inline}"
export EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-hash}"
export LLM_PROVIDER="${LLM_PROVIDER:-mock}"
export FRONTEND_DIR="${FRONTEND_DIR:-/app/frontend}"
export PORT="${PORT:-8000}"

mkdir -p "$PGDATA" /var/run/postgresql /tmp
chown -R postgres:postgres "$PGDATA" /var/run/postgresql

if [ ! -s "$PGDATA/PG_VERSION" ]; then
  gosu postgres initdb -D "$PGDATA" --encoding=UTF8 --locale=C --username=postgres >/tmp/initdb.log
  cat >> "$PGDATA/postgresql.conf" <<'EOF'
listen_addresses = '127.0.0.1'
port = 5432
shared_buffers = 16MB
max_connections = 20
work_mem = 2MB
maintenance_work_mem = 16MB
fsync = off
synchronous_commit = off
full_page_writes = off
EOF
  cat > "$PGDATA/pg_hba.conf" <<'EOF'
local all all trust
host all all 127.0.0.1/32 scram-sha-256
host all all ::1/128 scram-sha-256
EOF
  chown postgres:postgres "$PGDATA/postgresql.conf" "$PGDATA/pg_hba.conf"
fi

if ! gosu postgres pg_ctl -D "$PGDATA" -w -l /tmp/postgres.log start; then
  echo "postgres failed to start" >&2
  cat /tmp/postgres.log >&2 || true
  exit 1
fi

if ! gosu postgres psql -d postgres -tAc "SELECT 1 FROM pg_roles WHERE rolname='vellum'" | grep -q 1; then
  gosu postgres psql -d postgres -v ON_ERROR_STOP=1 \
    -c "CREATE ROLE vellum LOGIN SUPERUSER PASSWORD 'vellum'"
fi
if ! gosu postgres psql -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='vellum'" | grep -q 1; then
  gosu postgres psql -d postgres -v ON_ERROR_STOP=1 \
    -c "CREATE DATABASE vellum OWNER vellum"
fi

redis-server \
  --bind 127.0.0.1 \
  --port 6379 \
  --daemonize yes \
  --maxmemory 32mb \
  --maxmemory-policy allkeys-lru \
  --save "" \
  --appendonly no \
  --protected-mode yes \
  --dir /tmp

for _ in $(seq 1 20); do
  if redis-cli -h 127.0.0.1 ping | grep -q PONG; then
    break
  fi
  sleep 0.2
done
redis-cli -h 127.0.0.1 ping | grep -q PONG

cd /app
exec python -m uvicorn research_engine.main:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --workers 1 \
  --no-access-log
