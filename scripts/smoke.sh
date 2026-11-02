#!/usr/bin/env bash
# End-to-end smoke against a running Vellum stack. No API key required when LLM_PROVIDER=mock.
set -euo pipefail

BASE="${1:-http://localhost:8000}"

echo "waiting for ${BASE}/health"
ready=0
for _ in $(seq 1 60); do
  if curl -fsS "${BASE}/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "${ready}" -ne 1 ]]; then
  echo "health check failed" >&2
  exit 1
fi

QUESTION="What are the main ideas behind retrieval-augmented generation?"
RESP="$(curl -fsS -X POST "${BASE}/api/jobs" \
  -H "content-type: application/json" \
  -d "{\"question\": \"${QUESTION}\"}")"
echo "${RESP}"
if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi
ID="$("${PY}" -c 'import json,sys; print(json.loads(sys.argv[1])["id"])' "${RESP}")"

for _ in $(seq 1 90); do
  BODY="$(curl -fsS "${BASE}/api/jobs/${ID}")"
  STATUS="$("${PY}" -c 'import json,sys; print(json.loads(sys.argv[1])["status"])' "${BODY}")"
  echo "status=${STATUS}"
  if [[ "${STATUS}" == "completed" ]]; then
    echo "----- report -----"
    curl -fsS "${BASE}/api/jobs/${ID}/report?format=md" | head -n 60
    exit 0
  fi
  if [[ "${STATUS}" == "failed" ]]; then
    echo "${BODY}" >&2
    exit 1
  fi
  sleep 2
done

echo "timed out waiting for job ${ID}" >&2
exit 1
