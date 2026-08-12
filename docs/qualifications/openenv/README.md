# OpenEnv R-05 qualification evidence

Machine-readable records for a **separately operated** OpenEnv service. These
files are evidence only; they do not flip `arena/support-matrix.json`.

| Artifact | Meaning |
| --- | --- |
| `R-05-openenv-separate-service.json` | Gate summary (base_url, `separately_deployed`, digests, claim boundary) |
| `openenv-separate-qualification.json` | `arena.adapter-qualification/v1` task report |
| `task-rps-openenv-separate.yaml` | Imported task pin used for the run |

The collector refuses loopback OpenEnv attachments unless
`separately_deployed: true`. That flag is set by
`scripts/qualify_openenv_separate_service.py` because the script never starts
the service — only an already-running operated process/container qualifies.
Attaching the file fills the R-05 evidence slot; it does not flip the support
matrix to stable.

Regenerate:

```bash
docker compose -f docker/openenv/docker-compose.yml up --build -d
export ARENA_OPENENV_BASE_URL=http://127.0.0.1:8000
.venv/bin/python scripts/qualify_openenv_separate_service.py \
  --out docs/qualifications/openenv
```
