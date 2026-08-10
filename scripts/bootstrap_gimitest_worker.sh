#!/usr/bin/env bash
# Build a fresh resolver-isolated Gimitest worker and print ARENA_GIMITEST_PYTHON.
#
# Usage:
#   scripts/bootstrap_gimitest_worker.sh
#   eval "$(scripts/bootstrap_gimitest_worker.sh --export)"
#   ARENA_GIMITEST_VENV=/tmp/arena-gimitest scripts/bootstrap_gimitest_worker.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ARENA_GIMITEST_VENV:-${ROOT}/.venv-gimitest}"
BASE_PYTHON="${ARENA_GIMITEST_BASE_PYTHON:-python3}"
CONSTRAINTS="${ROOT}/constraints/gimitest-worker.txt"
# Editable checkout by default; override for wheel/CI:
#   ARENA_GIMITEST_ARENA_SPEC='/path/to/arena-1.0.0rc1-py3-none-any.whl[torch,pettingzoo,gimitest]'
ARENA_SPEC="${ARENA_GIMITEST_ARENA_SPEC:-${ROOT}[torch,pettingzoo,gimitest]}"
EXPORT=0

for arg in "$@"; do
  case "${arg}" in
    --export) EXPORT=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: ${arg}" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${CONSTRAINTS}" ]]; then
  echo "missing constraints file: ${CONSTRAINTS}" >&2
  exit 1
fi

rm -rf "${VENV}"
"${BASE_PYTHON}" -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip
# Editable / wheel Arena stack first (keeps current Gymnasium for PettingZoo).
if [[ "${ARENA_SPEC}" == /* ]] || [[ "${ARENA_SPEC}" == ./* ]] || [[ "${ARENA_SPEC}" == "${ROOT}"* ]]; then
  python -m pip install -c "${CONSTRAINTS}" -e "${ARENA_SPEC}"
else
  python -m pip install -c "${CONSTRAINTS}" "${ARENA_SPEC}"
fi
# Provider package only — do not honor its obsolete Gymnasium pin.
python -m pip install --no-deps -c "${CONSTRAINTS}" gimitest==1.0

WORKER_PYTHON="$(cd "${VENV}/bin" && pwd)/python"
export ARENA_GIMITEST_PYTHON="${WORKER_PYTHON}"

python - <<'PY'
import json
import os
from importlib.metadata import version

names = ["arena", "gimitest", "torch", "pettingzoo", "pillow"]
print(json.dumps({name: version(name) for name in names}, sort_keys=True, indent=2))
print("ARENA_GIMITEST_PYTHON=" + os.environ["ARENA_GIMITEST_PYTHON"])
PY

if [[ "${EXPORT}" -eq 1 ]]; then
  printf 'export ARENA_GIMITEST_PYTHON=%q\n' "${WORKER_PYTHON}"
else
  echo
  echo "Worker ready. In this shell:"
  echo "  export ARENA_GIMITEST_PYTHON=${WORKER_PYTHON}"
  echo "Then:"
  echo "  arena doctor --capability gimitest --json"
  echo "  python scripts/qualify_gimitest_isolated.py"
fi
