#!/usr/bin/env bash
# End-to-end handoff demo for the RPS pilot (no external training repo).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEMO="${ROOT}/examples/handoff/.demo"
rm -rf "${DEMO}"
mkdir -p "${DEMO}/artifacts" "${DEMO}/runs"

DEMO_DIR="${DEMO}" python - <<'PY'
import os
from pathlib import Path
from rlx.conformance.fixtures import build_rps_policy

demo = Path(os.environ["DEMO_DIR"])
build_rps_policy(demo / "artifacts" / "player_0.rlx", role="player_0", seed=1)
build_rps_policy(demo / "artifacts" / "player_1.rlx", role="player_1", seed=2)
print("exported policies ->", demo / "artifacts")
PY

cat > "${DEMO}/match.yaml" <<'EOF'
schema: rlx.match/v0alpha1
task:
  adapter: pettingzoo-parallel
  env: rlx/competitive_rps_v0
assignments:
  player_0: ./artifacts/player_0.rlx
  player_1: ./artifacts/player_1.rlx
seeds: {start: 0, count: 10}
action_mode: deterministic
record:
  trajectories: all
failure_policy:
  timeout_seconds: 60
  retain_incomplete: true
  retry: 0
EOF

cd "${DEMO}"
rlx init
rlx inspect ./artifacts/player_0.rlx
rlx check rlx/competitive_rps_v0 ./artifacts/player_0.rlx --role player_0
rlx check rlx/competitive_rps_v0 ./artifacts/player_1.rlx --role player_1
rlx match run ./match.yaml --record --out ./runs/baseline-match
rlx data inspect ./runs/baseline-match/trajectories
echo "Demo complete: ${DEMO}/runs/baseline-match"
