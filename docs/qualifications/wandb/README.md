# W&B store qualification (preview)

Status: **preview**. Do not claim stable until a live W&B qualification report
is attached here as `live-qualification.json` with
`mode: live` and `counts_as_live_evidence: true`.

Simulation (`?simulate=`) never counts as live evidence.

## Prerequisites

```bash
python -m pip install 'arena[wandb]'
wandb login   # or export WANDB_API_KEY
```

## Simulation qualify (CI / local rehearsal)

```bash
mkdir -p /tmp/arena-wandb-sim
arena store qualify examples/eval/demo/rock.arena \
  "wandb://entity/project/artifact?simulate=/tmp/arena-wandb-sim" \
  --out docs/qualifications/wandb/simulation-qualification.json
```

Expect `mode: "simulation"` and `counts_as_live_evidence: false`.

## Live qualify (required for stable)

```bash
arena doctor --capability wandb
arena store qualify examples/eval/demo/rock.arena \
  'wandb://ENTITY/PROJECT/ARTIFACT' \
  --out docs/qualifications/wandb/live-qualification.json
```

## Fail-loud without credentials

Live qualify without `WANDB_API_KEY` / `wandb login` fails with
`STORE_CREDENTIALS_REQUIRED` before contacting the service. Simulation is never
accepted as a live claim.
