#!/usr/bin/env bash
# Start the OpenEnv pilot as a SEPARATE process (not inside the Arena client).
#
# Foreground (default):
#   ./examples/openenv/separate_service/run_service.sh
#
# Background:
#   ./examples/openenv/separate_service/run_service.sh --daemon
#   export ARENA_OPENENV_BASE_URL=http://127.0.0.1:8000
#
# Docker alternative:
#   docker compose -f docker/openenv/docker-compose.yml up --build -d

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

HOST="${OPENENV_HOST:-127.0.0.1}"
PORT="${OPENENV_PORT:-8000}"
ENV_KIND="${OPENENV_ENV:-rps}"
PYTHON_BIN="${ARENA_PYTHON:-${ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

DAEMON=0
if [[ "${1:-}" == "--daemon" ]]; then
  DAEMON=1
fi

LOG_DIR="${ARENA_OPENENV_LOG_DIR:-${ROOT}/.arena/openenv-service}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/server-${PORT}.log"
PID_FILE="${LOG_DIR}/server-${PORT}.pid"

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}" || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "OpenEnv service already running (pid=${old_pid}) on ${HOST}:${PORT}" >&2
    echo "ARENA_OPENENV_BASE_URL=http://${HOST}:${PORT}"
    exit 0
  fi
fi

CMD=("${PYTHON_BIN}" -m arena.adapters.task_openenv.server --host "${HOST}" --port "${PORT}" --env "${ENV_KIND}")

if [[ "${DAEMON}" -eq 1 ]]; then
  "${CMD[@]}" >"${LOG_FILE}" 2>&1 &
  echo $! >"${PID_FILE}"
  echo "Started OpenEnv separate service pid=$(cat "${PID_FILE}") log=${LOG_FILE}"
  echo "ARENA_OPENENV_BASE_URL=http://${HOST}:${PORT}"
  deadline=$((SECONDS + 30))
  while (( SECONDS < deadline )); do
    if "${PYTHON_BIN}" -c "from urllib.request import urlopen; urlopen('http://${HOST}:${PORT}/health', timeout=0.5)" 2>/dev/null; then
      exit 0
    fi
    sleep 0.1
  done
  echo "OpenEnv service did not become healthy within 30s; see ${LOG_FILE}" >&2
  exit 1
fi

echo "ARENA_OPENENV_BASE_URL=http://${HOST}:${PORT}"
exec "${CMD[@]}"
