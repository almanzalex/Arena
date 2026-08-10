# OpenEnv separate service (Docker)

Runs Arena's frozen OpenEnv pilot (`arena.adapters.task_openenv.server`) as a
**separately operated** HTTP service. The Arena client process must connect only
via `packaging.base_url` / `ARENA_OPENENV_BASE_URL` — it must not spawn this
container inline and then claim R-05.

```bash
# from repository root
docker compose -f docker/openenv/docker-compose.yml up --build -d
export ARENA_OPENENV_BASE_URL=http://127.0.0.1:8000
.venv/bin/python scripts/qualify_openenv_separate_service.py \
  --out docs/qualifications/openenv
```

Process-only alternative (no Docker): see `examples/openenv/separate_service/`.
