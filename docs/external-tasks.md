# External tasks in Arena 0.5

Arena executes external tasks only through the `task_packaging` registry. A packager
must implement both `make_env(spec)` and `describe_task(spec)`. The second method is
mandatory because remote/container tasks otherwise hide role spaces, agent lifecycle,
runtime revision, and mask semantics from preflight checks.

## OpenEnv

The qualification set includes competitive RPS and vector coordination served over
OpenEnv 0.4.x. The server wraps PettingZoo tasks with OpenEnv's own
FastAPI/WebSocket transport; Arena does not replace OpenEnv containers or hosting.

```bash
pip install 'arena[openenv]'
python -m arena.adapters.task_openenv.server --port 8000
arena task import openenv://127.0.0.1:8000/arena/competitive_rps_v0 \
  --name task:rps-openenv@0.3 --source-revision openenv-0.4.1
arena task verify-equivalence examples/tasks/native-rps.yaml task-rps-openenv-0.3.yaml \
  --trace-suite examples/tasks/rps-equivalence.yaml
arena adapter qualify task-rps-openenv-0.3.yaml \
  --peer examples/tasks/native-rps.yaml \
  --trace-suite examples/tasks/rps-equivalence.yaml --out openenv-qualification.json
```

Import reads `/schema`, records its SHA-256 digest, pins the endpoint/revision, and
embeds the Arena per-role contract plus an `arena.openenv-capabilities/v1` protocol
declaration. Each new client session refuses a changed schema or contract pin. For
non-pilot environments, pass `--contract`; an
OpenEnv action/observation JSON Schema does not by itself define multi-agent Gym spaces.

T-01 compares seeded observations, actions, rewards, terminations, truncations, masks,
and agent selection. T-02 crosses the actual WebSocket JSON boundary. T-03 records
`disconnect`, `container_crash`, `timeout`, and `protocol_error` distinctly.


## PettingZoo classic RPS

Arena's `pettingzoo-parallel` adapter already loads PettingZoo's upstream
`classic/rps_v2` environment for both `parallel` and `aec` interactions (in
addition to the packaged pilot RPS envs). Checked-in task YAMLs and a demo make
that path usable without rewriting OpenSpiel fixtures:

```bash
pip install 'arena[pettingzoo,torch]'
arena demo multiagent --out /tmp/arena-ma-demo --json
# or
python examples/multiagent/run_demo.py --out /tmp/arena-ma-demo
```

Task manifests:

- `examples/tasks/pettingzoo-classic-rps.yaml` — simultaneous `parallel` play
- `examples/tasks/pettingzoo-classic-rps-aec.yaml` — turn-based `aec` twin

Both export portable Discrete(4)/Discrete(3) policies, record trajectories, and
emit stable policy/outcome digests. Acceptance coverage lives in
`tests/acceptance/test_pettingzoo_classic_multiagent.py`.

## OpenSpiel

The qualified catalog is organized by OpenSpiel semantics rather than one generic
wrapper:

| Family | Qualified game | Arena interaction |
|---|---|---|
| Sequential, deterministic, perfect information | `tic_tac_toe`, `connect_four`, `breakthrough` | `aec` |
| Sequential with explicit chance and imperfect information | `kuhn_poker`, `leduc_poker` | `aec` using information-state tensors |
| Simultaneous, deterministic | `matrix_rps` | `parallel` using one joint `apply_actions` |

Observation/action dimensions remain game-specific; legal actions are required masks.
Chance outcomes use the episode's seeded NumPy generator and are retained in task
infos. Each game support claim requires a checked trace generated from OpenSpiel's
authoritative state.

```bash
pip install 'arena[openspiel]'
arena task verify-equivalence examples/tasks/openspiel-tic-tac-toe.yaml \
  --trace-suite examples/tasks/openspiel-tic-tac-toe-trace.yaml
arena adapter qualify examples/tasks/openspiel-tic-tac-toe.yaml \
  --trace-suite examples/tasks/openspiel-tic-tac-toe-trace.yaml \
  --out openspiel-qualification.json

arena task verify-equivalence examples/tasks/openspiel-connect-four.yaml \
  --trace-suite examples/tasks/openspiel-connect-four-trace.yaml

arena task verify-equivalence examples/tasks/openspiel-kuhn-poker.yaml \
  --trace-suite examples/tasks/openspiel-kuhn-poker-trace.yaml
arena task verify-equivalence examples/tasks/openspiel-leduc-poker.yaml \
  --trace-suite examples/tasks/openspiel-leduc-poker-trace.yaml
arena task verify-equivalence examples/tasks/openspiel-matrix-rps.yaml \
  --trace-suite examples/tasks/openspiel-matrix-rps-trace.yaml
```

Exploitability metrics, unlisted game IDs, unsupported player counts, and semantic
mismatches remain rejected and require a new contract plus frozen qualification fixture.
