# Preview store qualifications

| Backend | Status | Live evidence path | Simulation never live |
|---------|--------|--------------------|------------------------|
| [OCI](oci/README.md) | preview | `oci/live-qualification.json` | yes |
| [W&B](wandb/README.md) | preview | `wandb/live-qualification.json` | yes |
| [MLflow](mlflow/README.md) | preview | `mlflow/live-qualification.json` | yes |

Stable promotion requires a checked-in live report with
`counts_as_live_evidence: true`. Absence of that file means the backend stays
preview.
