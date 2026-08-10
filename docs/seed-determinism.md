# Seed determinism (evaluation science)

Arena treats **seeded native Discrete matches** as reproducible science: the same
episode seeds, policies, and task binding must replay identical action streams.
The contract lives in [RFC 001](../rfcs/001-portable-policy-contract.md) and is
implemented by [`arena.runtime.seed_protocol`](../arena/runtime/seed_protocol.py):

```
policy_rng = default_rng(episode_seed + role_salt(role) + step_index)
```

All match runners (Parallel, AEC, dynamic AEC) derive policy RNGs through this
helper so co-acting agents never share a stream at the same step index.

Hardening tests: [`tests/test_seed_protocol_hardening.py`](../tests/test_seed_protocol_hardening.py).

## What is guaranteed

| Surface | Guarantee |
|---------|-----------|
| Native PettingZoo Discrete / MultiDiscrete pilots | Same seeds → byte-identical trajectories (CPU NumPy categorical) |
| Stochastic mode with explicit match RNG | Exact action equality across processes (M-01 / C3) |
| Evaluation cells | Locked suite seeds map stably to cells (EV-04) |

## Where nondeterminism is expected

Do **not** claim scientific reproducibility without caveats when any of these apply:

| Source | Why |
|--------|-----|
| **GPU / CUDA / MPS** | Kernel scheduling and nondeterministic PyTorch ops can change logits |
| **External task services** | OpenEnv remotes, network adapters, and third-party hosts are outside Arena's RNG |
| **`rng=None` stochastic decode** | Action cases fall back to an unseeded generator if a caller omits RNG |
| **Envs that ignore `reset(seed=...)`** | Some dynamic pilot envs discard the episode seed for dynamics |
| **Global RNG mutation** | Code that touches global NumPy / PyTorch RNG outside the match runner |

## Operator check

Run the same match twice with identical seeds and compare trajectory action
sequences (see [clean-room.md](clean-room.md) reproducibility check). If they
diverge on a native Discrete CPU task, that is a bug — file it against the seed
protocol, not "RL noise."
