# Qualification evidence (one screen)

Local qualification JSON lives here. **Preview stays preview** until evidence is
bound into the release index on the exact release commit. Simulation /
`?simulate=` / missing credentials never become a live claim. Do not invent
passes.

## Status + paths + how to run

| Cap | Matrix | On-disk evidence | How to (re)run |
|---|---|---|---|
| [OpenEnv](openenv/README.md) | preview | `openenv/R-05-openenv-separate-service.json` (`separately_deployed: true`) | `docker compose -f docker/openenv/docker-compose.yml up --build -d` → `ARENA_OPENENV_BASE_URL=http://127.0.0.1:8000 python scripts/qualify_openenv_separate_service.py --out docs/qualifications/openenv` |
| [Gimitest](gimitest/README.md) | preview | `gimitest/R-06-gimitest.json` (`stable_claim: false`) | `scripts/bootstrap_gimitest_worker.sh` → `export ARENA_GIMITEST_PYTHON=…` → `python scripts/qualify_gimitest_isolated.py --out docs/qualifications/gimitest/R-06-gimitest.json` |
| [Hugging Face](hf/README.md) | preview | **none** — need `HF_TOKEN` + live immutable revision | `HF_TOKEN=… ARENA_HF_LIVE_DEST='hf://models/ORG/REPO/arena' python scripts/qualify_hf_live.py examples/eval/demo/rock.arena "$ARENA_HF_LIVE_DEST" --report docs/qualifications/hf/live-qualification.json` (or Actions: `HF live qualify (R-04)`) |
| [OCI](oci/README.md) | preview | **none** live (`oras` + login) | Live: `arena store qualify … 'oci://REGISTRY/…' --out docs/qualifications/oci/live-qualification.json` · Sim: `?simulate=/abs` only |
| [W&B](wandb/README.md) | preview | **none** live | Same pattern under `wandb/`; `?simulate=` ≠ live |
| [MLflow](mlflow/README.md) | preview | **none** live | Same pattern under `mlflow/`; `?simulate=` ≠ live |

`arena doctor` may list OpenEnv/Gimitest as `usable_today=preview` when deps (and
for Gimitest, `ARENA_GIMITEST_PYTHON`) are ready, and may point at the on-disk
evidence paths above. That is **not** a `v1.0.0` stable claim.

## Attach (does not flip the matrix to stable)

```bash
python scripts/collect_release_evidence.py \
  --attach R-04=docs/qualifications/hf/live-qualification.json \
  --attach R-05=docs/qualifications/openenv/R-05-openenv-separate-service.json \
  --attach R-06=docs/qualifications/gimitest/R-06-gimitest.json
```

Pointing matrix `evidence` at a real file while `status` stays `preview` is
allowed (local proof on disk). Flipping `status` → `stable` requires release-CI
binding (+ credentials for HF).

## Rules

1. Missing credentials or an unavailable service → stay preview / no fake live file.
2. Loopback OpenEnv without `separately_deployed: true`, in-process Gimitest, and
   store `?simulate=` do **not** satisfy the 1.0 external floor.
3. Store reports need `arena.store-qualification/v1` with `mode: live` and
   `simulation: false` (or `counts_as_live_evidence: true` where used).
