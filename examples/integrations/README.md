# Integration examples (real free environments)

Local qualification paths that do **not** require cloud credentials.

## Environment support inventory

| Surface | Extra / install | Status (support matrix) | Real-env smoke |
| --- | --- | --- | --- |
| **Gymnasium** | via `arena[pettingzoo]` | spaces / classic control used by adapters | CartPole Parallel wrapper + Match API (`run_real_env_match.py`) |
| **PettingZoo** | `arena[pettingzoo]` | stable path for native Parallel/AEC pilots | RPS pilots elsewhere; CartPole wraps Gymnasium as Parallel |
| **OpenSpiel** | `arena[openspiel]` | **stable** | Frozen games under `examples/tasks/openspiel-*.yaml`; Match smoke in this folder |
| **OpenEnv** | `arena[openenv]` | preview (loopback service) | Local `python -m arena.adapters.task_openenv.server`; never fake remote success |
| **Gimitest** | isolated `ARENA_GIMITEST_PYTHON` + `--no-deps gimitest==1.0` | preview | Fail-loud when worker unset; no credential pretence |

Credential-backed stores (`hf`, `oci`, `wandb`, `mlflow`) are out of scope here: `arena doctor` reports `credentials_required` and `authentication_attempted: false` without treating missing live auth as success.

## Run

```bash
# from repo root, with arena[pettingzoo,openspiel,torch] installed
python examples/integrations/run_real_env_match.py --out /tmp/arena-env-smoke

# OpenEnv loopback (optional; needs openenv + free port)
python -m arena.adapters.task_openenv.server --port 8000
arena doctor --capability openenv

# Gimitest: only after isolating a worker Python
arena doctor --capability gimitest   # must be ready; otherwise repair and exit non-zero from this script
```

See `tests/integrations/` for automated smoke and fail-loud qualification.
