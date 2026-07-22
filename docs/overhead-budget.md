# S-01 external-runtime overhead budget

Measured 2026-07-21 on local macOS, CPython 3.12, OpenEnv 0.4.1, persistent WebSocket,
100 repetitions of reset plus one joint RPS step:

| Path | Mean | Median | p95 |
|---|---:|---:|---:|
| Native in-process | 0.017 ms | 0.012 ms | 0.033 ms |
| OpenEnv loopback | 0.974 ms | 0.908 ms | 1.254 ms |

Mean transport overhead was 0.957 ms per reset+step (57.8× this deliberately tiny
native task). Reproduce with:

```bash
python -m rlx.adapters.task_openenv.server --port 8000
python examples/tasks/benchmark_openenv.py --base-url http://127.0.0.1:8000 --iterations 100
```

Budget: use external transport when an environment step is at least roughly 10 ms or
when isolation/reproducibility matters more than sub-millisecond throughput. Do not use
the OpenEnv adapter for microsecond-scale simulators or high-frequency vectorized stepping
without batching; network/container latency will dominate. Measure the actual deployment,
because loopback is a lower bound rather than a remote-service promise.
