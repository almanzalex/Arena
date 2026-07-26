#!/usr/bin/env bash
# Run the checked-in cyclic RPS eval demo end-to-end (no trainer repo).
# Prerequisites: pip install 'arena[torch,pettingzoo]' (or editable install from this repo).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DEMO="${ROOT}/demo"
if [[ ! -f "${DEMO}/evaluation.yaml" ]]; then
  echo "demo missing; run: python ${ROOT}/generate_demo.py" >&2
  exit 2
fi
cd "${DEMO}"
rm -rf .arena eval-run slice bundle
arena init
arena population create ./population.yaml --ref populations/opp --json > /tmp/arena-pop.json
ROCK=$(python -c "from arena.core.sdk import Policy; print(Policy.load('rock.arena').digest)")
PAPER=$(python -c "from arena.core.sdk import Policy; print(Policy.load('paper.arena').digest)")
SCISSORS=$(python -c "from arena.core.sdk import Policy; print(Policy.load('scissors.arena').digest)")
arena eval validate ./evaluation.yaml --population ./population.yaml
arena eval run ./evaluation.yaml \
  --policy "${ROCK}=./rock.arena" \
  --policy "${PAPER}=./paper.arena" \
  --policy "${SCISSORS}=./scissors.arena" \
  --population ./population.yaml \
  --out ./eval-run \
  --json
arena eval report ./eval-run --out ./eval-run --json | tee /tmp/arena-report.json
python - <<'PY'
import json
r=json.load(open("/tmp/arena-report.json"))
assert r.get("nontransitivity_warning"), r
assert r["metrics"]["payoff_matrix"].get("ranking") is None
print("ok: nontransitivity_warning present; ranking null")
PY
arena data select ./eval-run --out ./slice --outcome loss --role player_0 --json
arena eval bundle ./eval-run --out ./bundle --report ./eval-run/report.json
echo "Demo complete:"
echo "  eval-run/  report + cells"
echo "  slice/     loss episodes"
echo "  bundle/    locked digests"
