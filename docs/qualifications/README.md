# Qualification evidence drop zone

Other streams emit **qualification JSON** here. Stream D (R-gates) is the only
owner that may later propose `arena/support-matrix.json` `evidence` field updates
after real qualification files exist.

## Preview store qualifications

| Backend | Status | Live evidence path | Simulation never live |
|---------|--------|--------------------|------------------------|
| [OCI](oci/README.md) | preview | `oci/live-qualification.json` | yes |
| [W&B](wandb/README.md) | preview | `wandb/live-qualification.json` | yes |
| [MLflow](mlflow/README.md) | preview | `mlflow/live-qualification.json` | yes |
| [Hugging Face](hf/README.md) | preview | `hf/live-qualification.json` | yes |

Stable promotion requires a checked-in live report with
`counts_as_live_evidence: true`. Absence of that file means the backend stays
preview.

## Rules

1. **Do not invent passes.** Missing credentials or an unavailable service means
   the capability stays `preview` / evidence `none-attached`.
2. **Simulation is not live.** `?simulate=` store reports, local OpenEnv loopback
   without `separately_deployed: true`, and in-process Gimitest lineage do **not**
   satisfy the 1.0 external floor.
3. **File naming.** Prefer:

   | Capability | Suggested path |
   |---|---|
   | Hugging Face live round-trip | `docs/qualifications/hf/live-qualification.json` |
   | Separately deployed OpenEnv | `docs/qualifications/openenv/R-05-openenv-separate-service.json` |
   | Isolated-interpreter Gimitest | `docs/qualifications/gimitest/R-06-gimitest.json` |
   | OCI / W&B / MLflow (optional) | `docs/qualifications/<backend>/live-qualification.json` |

4. **Attach into the R-gate skeleton** (does not mutate the support matrix):

   ```bash
   python scripts/collect_release_evidence.py \
     --attach R-04=docs/qualifications/hf/live-qualification.json \
     --attach R-05=docs/qualifications/openenv/R-05-openenv-separate-service.json \
     --attach R-06=docs/qualifications/gimitest/R-06-gimitest.json
   ```

   Sibling stream READMEs may live under `docs/qualifications/openenv/` and
   `docs/qualifications/gimitest/`; this top-level README is the drop-zone
   contract owned by stream D.

5. **Support-matrix updates.** Only stream D proposes flipping
   `evidence: none-attached` → a concrete evidence pointer / `stable` after the
   qualification JSON is reviewed and bound into the release evidence index.
   Other streams open PRs that add the JSON + docs only. Pointing `evidence`
   at a real file while `status` stays `preview` is allowed (local proof on
   disk; not yet a release-stable claim).

## Schema expectations

- Stores: `arena.store-qualification/v1` with `"mode": "live"` and
  `"simulation": false`.
- OpenEnv / Gimitest adapters: qualification report with explicit deployment /
  isolated-interpreter fields (see `evidence/templates/R-05-openenv.json` and
  `R-06-gimitest.json`).
- The collector **rejects** simulated HF and loopback OpenEnv attachments for
  R-04 / R-05.

## Current status

| Capability | Matrix status | On-disk evidence |
|---|---|---|
| OpenEnv | preview | `openenv/R-05-openenv-separate-service.json` (separate-service, `ok: true`) |
| Gimitest | preview | `gimitest/R-06-gimitest.json` (isolated worker, `stable_claim: false`) |
| Hugging Face | preview | no `hf/live-qualification.json` — needs `HF_TOKEN` + live round-trip |
| OCI / W&B / MLflow | preview | no live reports yet |

`arena doctor` may report preview caps as locally usable when deps are present;
that is not a `v1.0.0` stable claim. HF remains the non-waivable live-store floor.
