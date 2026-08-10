"""Episode seed → per-agent policy RNG derivation (RFC 001).

Match runners MUST seed stochastic policies via :func:`policy_rng` so Discrete
categorical sampling is bit-reproducible across processes for native envs.

Determinism guarantees (native Discrete / MultiDiscrete, CPU NumPy RNG):
same episode seed + role + step index → identical Generator stream → identical
actions when the policy weights and observations match.

Expected nondeterminism (do not publish as science without caveats):

- GPU / CUDA / MPS kernels and nondeterministic PyTorch ops
- External task services (OpenEnv remote endpoints, network adapters)
- Stochastic decode with ``rng=None`` (action-case fallback uses an unseeded
  generator — callers outside match runners must pass an explicit RNG)
- Env adapters that ignore ``reset(seed=...)`` (some dynamic pilot envs)
- Concurrent mutation of global NumPy / PyTorch RNG state outside Arena
"""

from __future__ import annotations

import numpy as np

from arena.core.identity import sha256_canonical

_MOD = 2**31 - 1


def role_salt(role: str) -> int:
    """Stable positive int salt so co-acting roles do not share RNG streams."""
    return int(sha256_canonical({"role": role})[:8], 16) % _MOD


def policy_rng_seed(episode_seed: int, role: str, step_index: int) -> int:
    """Integer seed for ``np.random.default_rng`` (episode + role salt + step)."""
    return int(episode_seed) + role_salt(role) + int(step_index)


def policy_rng(episode_seed: int, role: str, step_index: int) -> np.random.Generator:
    """NumPy Generator for one policy ``act`` call under the match seed protocol."""
    return np.random.default_rng(policy_rng_seed(episode_seed, role, step_index))
