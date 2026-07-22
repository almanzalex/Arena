# Expected handoff artifacts

After `examples/handoff/run_demo.sh`:

```text
.demo/
  artifacts/player_0.rlx/policy.yaml
  artifacts/player_0.rlx/payloads/weights.pt
  artifacts/player_1.rlx/policy.yaml
  artifacts/player_1.rlx/payloads/weights.pt
  match.yaml
  runs/baseline-match/run.yaml
  runs/baseline-match/trajectories/bundle.yaml
  runs/baseline-match/trajectories/episode_XXXX.json
```

Digests are content-addressed and printed by `rlx inspect` (`digest: sha256:…`). Compare those values across machines after transfer; they must match for identical bundles.
