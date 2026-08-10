# Multi-agent PettingZoo classic RPS

This example runs a real two-player match against PettingZoo `classic/rps_v2`
through Arena's existing `pettingzoo-parallel` adapter (parallel and AEC).

## Quick path

```bash
pip install -e '.[pettingzoo,torch,dev]'
python -m arena demo multiagent --out /tmp/arena-ma-demo --json
# or
python examples/multiagent/run_demo.py --out /tmp/arena-ma-demo
```

Artifacts written under `--out`:

- `policies/rock.arena`, `policies/paper.arena` — portable fixed-action policies
- `parallel/` and `aec/` — seeded match runs with trajectories
- `summary.json` — policy digests, episode returns, and a stable outcome digest

Task YAMLs live in `examples/tasks/pettingzoo-classic-rps.yaml` and
`pettingzoo-classic-rps-aec.yaml`. OpenSpiel fixtures are untouched.
