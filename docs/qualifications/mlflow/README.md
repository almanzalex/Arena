# MLflow store qualification (preview)

Status: **preview**. Do not claim stable until a live MLflow qualification
report is attached here as `live-qualification.json` with
`mode: live` and `counts_as_live_evidence: true`.

Simulation (`?simulate=`) never counts as live evidence. A default local
`./mlruns` directory is also not live evidence.

## Prerequisites

```bash
python -m pip install 'arena[mlflow]'
export MLFLOW_TRACKING_URI='https://mlflow.example'   # remote tracking server
# or pass ?tracking_uri=https%3A%2F%2Fmlflow.example on the destination URI
```

Authenticate with whatever the tracking server requires (basic auth, token, etc.).

## Simulation qualify (CI / local rehearsal)

```bash
mkdir -p /tmp/arena-mlflow-sim
arena store qualify examples/eval/demo/rock.arena \
  "mlflow://arena-experiment?simulate=/tmp/arena-mlflow-sim" \
  --out docs/qualifications/mlflow/simulation-qualification.json
```

Expect `mode: "simulation"` and `counts_as_live_evidence: false`.

## Live qualify (required for stable)

```bash
arena doctor --capability mlflow
arena store qualify examples/eval/demo/rock.arena \
  'mlflow://arena-experiment?tracking_uri=https%3A%2F%2Fmlflow.example' \
  --out docs/qualifications/mlflow/live-qualification.json
```

## Fail-loud without credentials / remote tracking

Live qualify without `MLFLOW_TRACKING_URI` / `?tracking_uri=` fails with
`STORE_CREDENTIALS_REQUIRED`. A `file:` tracking URI is refused for live mode;
use `?simulate=/absolute/path` for local protocol rehearsal instead.
