# OpenEnv separate-service qualification

R-05 requires a **separately operated** OpenEnv service. Spawning the pilot
inside the same pytest fixture that runs the Arena client is loopback evidence
only — useful for contracts, insufficient for a stable support claim.

## 1. Start the service (pick one)

Docker:

```bash
docker compose -f docker/openenv/docker-compose.yml up --build -d
export ARENA_OPENENV_BASE_URL=http://127.0.0.1:8000
```

Process:

```bash
./examples/openenv/separate_service/run_service.sh --daemon
export ARENA_OPENENV_BASE_URL=http://127.0.0.1:8000
```

## 2. Qualify from the Arena client

```bash
.venv/bin/python scripts/qualify_openenv_separate_service.py \
  --out docs/qualifications/openenv
```

## 3. Pytest gate

```bash
.venv/bin/pytest -m docker tests/integrations/test_openenv_separate_service.py -q
```

If `ARENA_OPENENV_BASE_URL` is unset or `/health` fails, the test **fails loud**
with this recipe — it never marks success or flips the support matrix.
