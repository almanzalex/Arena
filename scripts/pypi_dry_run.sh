#!/usr/bin/env bash
# R-11 rehearsal: build sdist + wheel and run twine check.
#
# This script NEVER uploads to TestPyPI or PyPI. It does not read or require
# TWINE_USERNAME, TWINE_PASSWORD, API tokens, or any other publish secrets.
#
# Usage (from repository root):
#   bash scripts/pypi_dry_run.sh
#
# Optional: PYPI_DRY_RUN_DIST=path  (default: dist/)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DIST_DIR="${PYPI_DRY_RUN_DIST:-dist}"
PYTHON="${PYTHON:-python3}"

echo "==> R-11 dry-run: build + twine check (no upload, no secrets)"
echo "    root=${ROOT}"
echo "    dist=${DIST_DIR}"
echo "    python=$("${PYTHON}" -c 'import sys; print(sys.executable)')"

"${PYTHON}" -m pip install --upgrade pip >/dev/null
"${PYTHON}" -m pip install --upgrade 'build>=1.2' 'twine>=5.0' >/dev/null

rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}"

echo "==> python -m build --sdist --wheel --outdir ${DIST_DIR}"
"${PYTHON}" -m build --sdist --wheel --outdir "${DIST_DIR}"

echo "==> twine check --strict ${DIST_DIR}/*"
"${PYTHON}" -m twine check --strict "${DIST_DIR}"/*

echo "==> Artifacts ready for Trusted Publishing (upload is intentionally omitted):"
ls -la "${DIST_DIR}"
echo "==> Dry-run OK. See docs/pypi-trusted-publishing.md for OIDC publisher setup."
