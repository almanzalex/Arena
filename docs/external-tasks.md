# External tasks in RLX 0.5

RLX executes external tasks only through the `task_packaging` registry. A packager
must implement both `make_env(spec)` and `describe_task(spec)`. The second method is
mandatory because remote/container tasks otherwise hide role spaces, agent lifecycle,
runtime revision, and mask semantics from preflight checks.

## OpenEnv

The qualification set includes competitive RPS and vector coordination served over
OpenEnv 0.4.x. The server wraps PettingZoo tasks with OpenEnv's own
FastAPI/WebSocket transport; RLX does not replace OpenEnv containers or hosting.

```bash
pip install 'rlx[openenv]'
python -m rlx.adapters.task_openenv.server --port 8000
rlx task import openenv://127.0.0.1:8000/rlx/competitive_rps_v0 \
  --name task:rps-openenv@0.3 --source-revision openenv-0.4.1
rlx task verify-equivalence examples/tasks/native-rps.yaml task-rps-openenv-0.3.yaml \
  --trace-suite examples/tasks/rps-equivalence.yaml
rlx adapter qualify task-rps-openenv-0.3.yaml \
  --peer examples/tasks/native-rps.yaml \
  --trace-suite examples/tasks/rps-equivalence.yaml --out openenv-qualification.json
```

Import reads `/schema`, records its SHA-256 digest, pins the endpoint/revision, and
embeds the RLX per-role contract plus an `rlx.openenv-capabilities/v1` protocol
declaration. Each new client session refuses a changed schema or contract pin. For
non-pilot environments, pass `--contract`; an
OpenEnv action/observation JSON Schema does not by itself define multi-agent Gym spaces.

T-01 compares seeded observations, actions, rewards, terminations, truncations, masks,
and agent selection. T-02 crosses the actual WebSocket JSON boundary. T-03 records
`disconnect`, `container_crash`, `timeout`, and `protocol_error` distinctly.

## OpenSpiel

The qualified catalog is organized by OpenSpiel semantics rather than one generic
wrapper:

| Family | Qualified game | RLX interaction |
|---|---|---|
| Sequential, deterministic, perfect information | `tic_tac_toe`, `connect_four`, `breakthrough` | `aec` |
| Sequential with explicit chance and imperfect information | `kuhn_poker` | `aec` using information-state tensors |
| Simultaneous, deterministic | `matrix_rps` | `parallel` using one joint `apply_actions` |

Observation/action dimensions remain game-specific; legal actions are required masks.
Chance outcomes use the episode's seeded NumPy generator and are retained in task
infos. Each game support claim requires a checked trace generated from OpenSpiel's
authoritative state.

```bash
pip install 'rlx[openspiel]'
rlx task verify-equivalence examples/tasks/openspiel-tic-tac-toe.yaml \
  --trace-suite examples/tasks/openspiel-tic-tac-toe-trace.yaml
rlx adapter qualify examples/tasks/openspiel-tic-tac-toe.yaml \
  --trace-suite examples/tasks/openspiel-tic-tac-toe-trace.yaml \
  --out openspiel-qualification.json

rlx task verify-equivalence examples/tasks/openspiel-connect-four.yaml \
  --trace-suite examples/tasks/openspiel-connect-four-trace.yaml

rlx task verify-equivalence examples/tasks/openspiel-kuhn-poker.yaml \
  --trace-suite examples/tasks/openspiel-kuhn-poker-trace.yaml
rlx task verify-equivalence examples/tasks/openspiel-matrix-rps.yaml \
  --trace-suite examples/tasks/openspiel-matrix-rps-trace.yaml
```

Exploitability metrics, unlisted game IDs, unsupported player counts, and semantic
mismatches remain rejected and require a new contract plus frozen qualification fixture.
