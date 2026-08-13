#!/usr/bin/env bash
# R-14 signing rehearsal with ephemeral keys only.
#
# This script NEVER invents gate passes, NEVER uses production keys, and NEVER
# writes a self-binding arena.release-evidence/v1 index into the repo.
#
# Success path (all R-01…R-14 evidence files present, non-template, status=pass):
#   1. Generate ephemeral Ed25519 keys under KEY_DIR
#   2. arena release assemble → KEY_DIR/release-index.json
#   3. arena release sign / verify with those keys
#
# Failure path (default until gates are filled):
#   Exit non-zero and print every missing / incomplete / template gate.
#
# Usage (from repository root):
#   bash scripts/rehearse_release_sign.sh
#   KEY_DIR=/tmp/arena-rehearse-$$ bash scripts/rehearse_release_sign.sh
#   EVIDENCE_DIR=evidence bash scripts/rehearse_release_sign.sh
#
# Optional:
#   ARTIFACT_GLOB='dist/arena-*.whl dist/arena-*.tar.gz'
#   RELEASE / TAG / COMMIT overrides
#
# Compatible with macOS /bin/bash 3.2 (no associative arrays).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

EVIDENCE_DIR="${EVIDENCE_DIR:-evidence}"
KEY_DIR="${KEY_DIR:-}"
PYTHON="${PYTHON:-python3}"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
fi
if command -v arena >/dev/null 2>&1; then
  ARENA=(arena)
else
  ARENA=("${PYTHON}" -m arena.cli.main)
fi

GATE_IDS="R-01 R-02 R-03 R-04 R-05 R-06 R-07 R-08 R-09 R-10 R-11 R-12 R-13 R-14"

gate_filename() {
  case "$1" in
    R-01) echo "R-01-platform-ci.json" ;;
    R-02) echo "R-02-hermetic.json" ;;
    R-03) echo "R-03-human-handoff.json" ;;
    R-04) echo "R-04-stores.json" ;;
    R-05) echo "R-05-openenv.json" ;;
    R-06) echo "R-06-gimitest.json" ;;
    R-07) echo "R-07-adversarial.json" ;;
    R-08) echo "R-08-security.json" ;;
    R-09) echo "R-09-compatibility.json" ;;
    R-10) echo "R-10-performance.json" ;;
    R-11) echo "R-11-public-distribution.json" ;;
    R-12) echo "R-12-recovery.json" ;;
    R-13) echo "R-13-docs.json" ;;
    R-14) echo "R-14-integrity.json" ;;
    *) return 1 ;;
  esac
}

_die() {
  echo "ERROR: $*" >&2
  exit 1
}

_is_safe_key_dir() {
  local path="$1"
  case "${path}" in
    /tmp/*|/private/tmp/*)
      return 0
      ;;
  esac
  "${PYTHON}" - "${ROOT}" "${path}" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
path = Path(sys.argv[2]).resolve()
try:
    rel = path.relative_to(root).as_posix()
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if rel == "evidence/local" or rel.startswith("evidence/local/") else 1)
PY
}

if [[ -z "${KEY_DIR}" ]]; then
  KEY_DIR="$(mktemp -d /tmp/arena-rehearse-sign.XXXXXX)"
fi
mkdir -p "${KEY_DIR}"
KEY_DIR="$(cd "${KEY_DIR}" && pwd)"
_is_safe_key_dir "${KEY_DIR}" || _die \
  "KEY_DIR must be under /tmp or gitignored evidence/local/ (got: ${KEY_DIR})"

COMMIT="${COMMIT:-$(git rev-parse HEAD)}"
if [[ -z "${RELEASE:-}" || -z "${TAG:-}" ]]; then
  VERSION="$("${PYTHON}" - <<'PY'
from pathlib import Path
ns = {}
exec(Path("arena/_version.py").read_text(encoding="utf-8"), ns)
print(ns["VERSION"])
PY
)"
  RELEASE="${RELEASE:-${VERSION}}"
  TAG="${TAG:-v${VERSION}}"
fi

echo "==> R-14 signing rehearsal (ephemeral keys only; no invented gate passes)"
echo "    root=${ROOT}"
echo "    evidence=${EVIDENCE_DIR}"
echo "    key_dir=${KEY_DIR}"
echo "    release=${RELEASE} tag=${TAG}"
echo "    commit=${COMMIT}"

MISSING=""
TEMPLATE=""
INCOMPLETE=""
READY=""
READY_COUNT=0

for gate in ${GATE_IDS}; do
  file="${EVIDENCE_DIR}/$(gate_filename "${gate}")"
  if [[ ! -f "${file}" ]]; then
    MISSING="${MISSING}
  - ${gate}=${file}"
    continue
  fi
  verdict="$("${PYTHON}" - "${file}" "${gate}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_gate = sys.argv[2]
text = path.read_text(encoding="utf-8")
markers = ("REPLACE_WITH_40_CHAR_LOWERCASE_SHA", "sha256:REPLACE", "REPLACE")
if any(marker in text for marker in markers):
    print("template")
    raise SystemExit(0)
try:
    doc = json.loads(text)
except json.JSONDecodeError as exc:
    print(f"invalid-json:{exc}")
    raise SystemExit(0)
status = doc.get("status")
gate_id = doc.get("gate")
if gate_id not in (None, expected_gate):
    print(f"gate-mismatch:{gate_id}")
elif status != "pass":
    print(f"status:{status!r}")
else:
    print("ready")
PY
)"
  case "${verdict}" in
    ready)
      READY="${READY}
  - ${gate}=${file}"
      READY_COUNT=$((READY_COUNT + 1))
      ;;
    template)
      TEMPLATE="${TEMPLATE}
  - ${gate}=${file}"
      ;;
    *)
      INCOMPLETE="${INCOMPLETE}
  - ${gate}=${file} (${verdict})"
      ;;
  esac
done

echo
echo "==> Gate inventory"
echo "    ready: ${READY_COUNT}/14"
if [[ -n "${READY}" ]]; then
  echo "    ready gates:${READY}"
fi
if [[ -n "${MISSING}" ]]; then
  echo "    missing:${MISSING}"
fi
if [[ -n "${TEMPLATE}" ]]; then
  echo "    template/placeholder (not attachable):${TEMPLATE}"
fi
if [[ -n "${INCOMPLETE}" ]]; then
  echo "    incomplete:${INCOMPLETE}"
fi

if [[ -n "${MISSING}${TEMPLATE}${INCOMPLETE}" ]]; then
  {
    echo
    echo "ERROR: arena release assemble is blocked — refusing to invent gate passes."
    echo "Missing or incomplete mandatory gates:"
    [[ -n "${MISSING}" ]] && echo "missing:${MISSING}"
    [[ -n "${TEMPLATE}" ]] && echo "template:${TEMPLATE}"
    [[ -n "${INCOMPLETE}" ]] && echo "incomplete:${INCOMPLETE}"
    echo
    echo "Fill real evidence files under ${EVIDENCE_DIR}/ (start from"
    echo "evidence/templates/, then replace every REPLACE_* field with measured"
    echo "results). Local dry-runs, hermetic-capable inventories, and collector"
    echo "skeletons are not assemble inputs."
    echo
    echo "Ephemeral key dir prepared at: ${KEY_DIR}"
    echo "(no release-index was written)"
  } >&2
  exit 2
fi

# shellcheck disable=SC2086
if [[ -n "${ARTIFACT_GLOB:-}" ]]; then
  # Intentional word-splitting of operator-supplied glob list.
  set -- ${ARTIFACT_GLOB}
else
  set -- dist/arena-*.whl dist/arena-*.tar.gz
fi
ARTIFACTS=()
for candidate in "$@"; do
  if [[ -f "${candidate}" ]]; then
    ARTIFACTS+=("${candidate}")
  fi
done
((${#ARTIFACTS[@]})) || _die "no release artifacts found under dist/ (run bash scripts/pypi_dry_run.sh first)"

PRIVATE_KEY="${KEY_DIR}/release-private.pem"
PUBLIC_KEY="${KEY_DIR}/release-public.pem"
INDEX_OUT="${KEY_DIR}/release-index.json"
SIG_OUT="${KEY_DIR}/release-index.sig.json"
rm -f "${PRIVATE_KEY}" "${PUBLIC_KEY}" "${INDEX_OUT}" "${SIG_OUT}"

echo
echo "==> Generating ephemeral signing keys"
"${ARENA[@]}" attest keygen --private "${PRIVATE_KEY}" --public "${PUBLIC_KEY}"

GATE_ARGS=()
for gate in ${GATE_IDS}; do
  GATE_ARGS+=(--gate "${gate}=${EVIDENCE_DIR}/$(gate_filename "${gate}")")
done
ARTIFACT_ARGS=()
for artifact in "${ARTIFACTS[@]}"; do
  ARTIFACT_ARGS+=(--artifact "${artifact}")
done

echo "==> arena release assemble (all gates present)"
"${ARENA[@]}" release assemble \
  --release "${RELEASE}" --tag "${TAG}" --commit "${COMMIT}" \
  "${GATE_ARGS[@]}" \
  "${ARTIFACT_ARGS[@]}" \
  --out "${INDEX_OUT}"

echo "==> arena release sign"
"${ARENA[@]}" release sign "${INDEX_OUT}" \
  --key "${PRIVATE_KEY}" --out "${SIG_OUT}"

echo "==> arena release verify"
VERIFY_ARGS=(release verify "${INDEX_OUT}" --signature "${SIG_OUT}" --key "${PUBLIC_KEY}")
if "${ARENA[@]}" release verify -h 2>&1 | grep -q -- '--at-release'; then
  VERIFY_ARGS+=(--at-release)
fi
"${ARENA[@]}" "${VERIFY_ARGS[@]}"

cat <<EOF

==> Rehearsal OK
    index=${INDEX_OUT}
    signature=${SIG_OUT}
    public_key=${PUBLIC_KEY}
    private_key=${PRIVATE_KEY}  (ephemeral; do not commit)
EOF
