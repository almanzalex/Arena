#!/usr/bin/env bash
# Run the checked-in cyclic RPS eval demo end-to-end (no trainer repo).
# Prerequisites: pip install 'rlx[torch,pettingzoo]' (or editable install from this repo).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DEMO="${ROOT}/demo"
if [[ ! -f "${DEMO}/evaluation.yaml" ]]; then
  echo "demo missing; run: python ${ROOT}/generate_demo.py" >&2
  exit 2
fi
cd "${DEMO}"
rm -rf .rlx eval-run slice bundle
rlx init
rlx population create ./population.yaml --ref populations/opp --json > /tmp/rlx-pop.json
ROCK=$(python -c "from rlx.core.sdk import Policy; print(Policy.load('rock.rlx').digest)")
PAPER=$(python -c "from rlx.core.sdk import Policy; print(Policy.load('paper.rlx').digest)")
SCISSORS=$(python -c "from rlx.core.sdk import Policy; print(Policy.load('scissors.rlx').digest)")
rlx eval validate ./evaluation.yaml --population ./population.yaml
rlx eval run ./evaluation.yaml \
  --policy "${ROCK}=./rock.rlx" \
  --policy "${PAPER}=./paper.rlx" \
  --policy "${SCISSORS}=./scissors.rlx" \
  --population ./population.yaml \
  --out ./eval-run \
  --json
rlx eval report ./eval-run --out ./eval-run --json | tee /tmp/rlx-report.json
python - <<'PY'
import json
r=json.load(open("/tmp/rlx-report.json"))
assert r.get("nontransitivity_warning"), r
assert r["metrics"]["payoff_matrix"].get("ranking") is None
print("ok: nontransitivity_warning present; ranking null")
PY
rlx data select ./eval-run --out ./slice --outcome loss --role player_0 --json
rlx eval bundle ./eval-run --out ./bundle --report ./eval-run/report.json
echo "Demo complete:"
echo "  eval-run/  report + cells"
echo "  slice/     loss episodes"
echo "  bundle/    locked digests"
