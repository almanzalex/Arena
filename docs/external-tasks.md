# External tasks in RLX 0.3

RLX executes external tasks only through the `task_packaging` registry. A packager
must implement both `make_env(spec)` and `describe_task(spec)`. The second method is
mandatory because remote/container tasks otherwise hide role spaces, agent lifecycle,
runtime revision, and mask semantics from preflight checks.

## OpenEnv

The frozen pilot is RLX competitive RPS served over OpenEnv 0.4.x. The server wraps
the existing PettingZoo pilot with OpenEnv's own FastAPI/WebSocket transport; RLX does
not replace OpenEnv containers or hosting.

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
embeds the RLX per-role contract. Each new client session refuses a changed schema pin.
For non-pilot environments, pass `--contract`; an
OpenEnv action/observation JSON Schema does not by itself define multi-agent Gym spaces.

T-01 compares seeded observations, actions, rewards, terminations, truncations, masks,
and agent selection. T-02 crosses the actual WebSocket JSON boundary. T-03 records
`disconnect`, `container_crash`, `timeout`, and `protocol_error` distinctly.

## OpenSpiel

The support claim is exactly OpenSpiel `tic_tac_toe`, exposed as AEC. Observation
tensors have shape 27, actions are `Discrete(9)`, and legal actions are required masks.
The checked trace digest is a reference generated from OpenSpiel's authoritative state.

```bash
pip install 'rlx[openspiel]'
rlx task verify-equivalence examples/tasks/openspiel-tic-tac-toe.yaml \
  --trace-suite examples/tasks/openspiel-tic-tac-toe-trace.yaml
rlx adapter qualify examples/tasks/openspiel-tic-tac-toe.yaml \
  --trace-suite examples/tasks/openspiel-tic-tac-toe-trace.yaml \
  --out openspiel-qualification.json
```

Chance nodes, simultaneous games, imperfect-information claims, exploitability metrics,
and other game IDs require a new frozen fixture and qualification report.
