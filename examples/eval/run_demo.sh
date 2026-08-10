#!/usr/bin/env bash
# Hermetic checked-in cyclic RPS eval demo (no trainer repo, no mutation of demo/).
# Prerequisites: pip install 'arena[torch,pettingzoo]' (or editable install from this repo).
#
# Env knobs:
#   ARENA_EVAL_DEMO_WORK  — work directory (default: mktemp under TMPDIR)
#   ARENA_EVAL_DEMO_KEEP  — if 1, keep work dir when auto-created
#   ARENA_BIN             — arena CLI (default: arena, else python -m arena)
#   ARENA_PYTHON          — python used for digest helpers / fallback CLI
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DEMO_SRC="${ROOT}/demo"
if [[ ! -f "${DEMO_SRC}/evaluation.yaml" ]]; then
  echo "demo missing; run: python ${ROOT}/generate_demo.py" >&2
  exit 2
fi

PYTHON_BIN="${ARENA_PYTHON:-python3}"
if [[ -n "${ARENA_BIN:-}" ]]; then
  # shellcheck disable=SC2206
  ARENA_CLI=(${ARENA_BIN})
elif command -v arena >/dev/null 2>&1; then
  ARENA_CLI=(arena)
else
  ARENA_CLI=("${PYTHON_BIN}" -m arena)
fi

AUTO_WORK=0
if [[ -n "${ARENA_EVAL_DEMO_WORK:-}" ]]; then
  WORK="${ARENA_EVAL_DEMO_WORK}"
else
  WORK="$(mktemp -d "${TMPDIR:-/tmp}/arena-eval-demo.XXXXXX")"
  AUTO_WORK=1
fi

cleanup() {
  if [[ "${AUTO_WORK}" -eq 1 && "${ARENA_EVAL_DEMO_KEEP:-0}" != "1" ]]; then
    rm -rf "${WORK}"
  fi
}
trap cleanup EXIT

# Fresh sandbox copy — never write into the checked-in demo pack.
mkdir -p "${WORK}"
# Clear prior contents of the work dir, then copy the demo pack in.
find "${WORK}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
if command -v rsync >/dev/null 2>&1; then
  rsync -a \
    --exclude '.arena' \
    --exclude 'eval-run' \
    --exclude 'slice' \
    --exclude 'bundle' \
    "${DEMO_SRC}/" "${WORK}/"
else
  cp -R "${DEMO_SRC}/." "${WORK}/"
  rm -rf "${WORK}/.arena" "${WORK}/eval-run" "${WORK}/slice" "${WORK}/bundle"
fi

cd "${WORK}"
"${ARENA_CLI[@]}" init
# BSD mktemp (macOS) requires the X's at the end of the template.
POP_JSON="$(mktemp "${TMPDIR:-/tmp}/arena-pop.XXXXXX")"
REPORT_JSON="$(mktemp "${TMPDIR:-/tmp}/arena-report.XXXXXX")"
"${ARENA_CLI[@]}" population create ./population.yaml --ref populations/opp --json > "${POP_JSON}"
ROCK="$("${PYTHON_BIN}" -c "from arena.core.sdk import Policy; print(Policy.load('rock.arena').digest)")"
PAPER="$("${PYTHON_BIN}" -c "from arena.core.sdk import Policy; print(Policy.load('paper.arena').digest)")"
SCISSORS="$("${PYTHON_BIN}" -c "from arena.core.sdk import Policy; print(Policy.load('scissors.arena').digest)")"
"${ARENA_CLI[@]}" eval validate ./evaluation.yaml --population ./population.yaml
"${ARENA_CLI[@]}" eval run ./evaluation.yaml \
  --policy "${ROCK}=./rock.arena" \
  --policy "${PAPER}=./paper.arena" \
  --policy "${SCISSORS}=./scissors.arena" \
  --population ./population.yaml \
  --out ./eval-run \
  --json
"${ARENA_CLI[@]}" eval report ./eval-run --out ./eval-run --json | tee "${REPORT_JSON}"
"${PYTHON_BIN}" - "${REPORT_JSON}" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
assert r.get("nontransitivity_warning"), r
assert r["metrics"]["payoff_matrix"].get("ranking") is None
# Bound digests must be present on the non-transitivity-aware report.
for key in (
    "evaluation_digest",
    "evaluation_intent_digest",
    "execution_binding_digest",
    "semantic_result_digest",
    "eval_run_digest",
    "sampling_ledger_digest",
):
    assert r.get(key), f"missing bound digest {key}: {r.keys()}"
assert isinstance(r.get("sampling_ledger"), list) and r["sampling_ledger"], r
print("ok: nontransitivity_warning present; ranking null; digests bound")
PY
"${ARENA_CLI[@]}" data select ./eval-run --out ./slice --outcome loss --role player_0 --json
"${ARENA_CLI[@]}" eval bundle ./eval-run --out ./bundle --report ./eval-run/report.json
echo "Demo complete (hermetic work dir: ${WORK}):"
echo "  eval-run/  report + cells"
echo "  slice/     loss episodes"
echo "  bundle/    locked digests"
if [[ "${AUTO_WORK}" -eq 1 && "${ARENA_EVAL_DEMO_KEEP:-0}" != "1" ]]; then
  echo "  (work dir will be removed on exit; set ARENA_EVAL_DEMO_KEEP=1 to retain)"
fi
