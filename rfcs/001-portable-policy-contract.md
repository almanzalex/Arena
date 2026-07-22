# RFC 001 — Portable Policy Contract (RLX 0.1)

**Status:** Accepted for MVP 0.1  
**Date:** 2026-07-16  
**Depends on:** RFC 000

## Goals

Define the minimum executable contract so a custom-PyTorch policy can leave its training repository, pass `rlx check` against a PettingZoo Parallel task role, and produce source-equivalent actions under conformance fixtures F1–F4.

## Manifest schema (`rlx.policy/v0alpha1`)

Required fields:

| Field | Semantics |
|-------|-----------|
| `schema` | Must be `rlx.policy/v0alpha1` |
| `name` | Human-readable identity (not content identity) |
| `roles.allowed` | Roles this policy may control |
| `runtime.adapter` | `custom-pytorch` for MVP |
| `runtime.python` | Minimum Python version string |
| `observation` | Space contract (dtype, shape, bounds, optional `schema_ref`) |
| `action` | Space contract + `masks` (`none` \| `optional` \| `required`) |
| `state.recurrent` | `false` or recurrent config |
| `state.reset_on` | When recurrent: subset of `episode_start`, `agent_termination` |
| `inference.modes` | Subset of `deterministic`, `stochastic` |
| `preprocessing` | Declared transforms included in the bundle (`included: true` + params) |
| `payloads` | Digest map (`weights`, optional `reference_cases`) |
| `architecture` | Adapter-specific declarative network (no training-repo import) |
| `lineage` | Optional source run/checkpoint metadata |
| `conformance` | Optional status + suite digest |

Large payloads are referenced by digest; they are never embedded in the manifest.

## Observation / action spaces

- Spaces are Gymnasium-compatible descriptors: `Box`, `Discrete`, `MultiDiscrete`, `MultiBinary`.
- Compatibility compares: space type, dtype, shape/n, and finite bounds when present.
- Role-specific schemas are taken from the task adapter for the assigned role/agent.

## Preprocessing

- Must be **included** in the policy bundle for 0.1 (no external trainer hooks).
- Supported transforms: identity, mean/std normalization (broadcastable vectors), optional clip.
- Transform order is fixed: `normalize` then `clip` when both are set.
- Preprocessing identifiers appear on the manifest; `rlx check` requires matching identifiers when the task declares expectations.

## Roles

- A policy may only be assigned to a role listed in `roles.allowed`.
- Mismatch → structured `INCOMPATIBLE` report before match start (never a mid-run tensor error for role errors).

## Stochastic sampling

- Deterministic mode: argmax over logits (discrete) or mean action (continuous; continuous deferred beyond pilot Discrete).
- Stochastic mode: categorical sample from logits.
- RNG contract: policies accept an explicit `numpy.random.Generator` or integer seed per call; match runner seeds policies from the episode seed + role salt.
- P-02: with matched seeds, sampled actions must match exactly for Discrete categorical policies.

## Action masks

- When `action.masks: required`, the runtime must receive a boolean mask of length `n` (or broadcastable) each step.
- Illegal actions (mask `False`) must never be selected in either inference mode.
- Missing mask when required → pre-run / pre-step error (`P-04`).
- When masks are `none`, providing a mask is ignored with a warning in verify only.

## Recurrent state

- When `state.recurrent: true`, architecture declares hidden size and the payload includes recurrent weights.
- `reset()` clears hidden state at every boundary listed in `reset_on`.
- Match runner calls `policy.reset(agent_id)` on episode start and when the task marks that agent terminated.
- P-03: hidden-state trajectories must match the source policy on identical observation streams.

## Weight / runtime payloads

- Weights: PyTorch `state_dict` serialized with `torch.save` under a content-addressed object.
- Architecture templates live in `rlx.adapters.policy_custom_torch` (`mlp_categorical`, `gru_categorical`).
- Loading reconstitutes the module from architecture + weights only — **no** `sys.path` mutation toward a training repo.

## Provenance

- Policy digest = SHA-256 of canonical JSON manifest with payload digests substituted (payloads hashed separately).
- Match run records store policy digests, task identity/version, seeds, role map, and failure ledger.
- Trajectory steps store policy digest per agent/role.

## Conformance tolerances

| Case | Tolerance |
|------|-----------|
| Discrete deterministic actions | Exact equality |
| Discrete seeded stochastic actions | Exact equality under RNG contract |
| Logits / distribution params (optional diagnostics) | `atol=1e-5`, `rtol=1e-5` |
| Recurrent hidden states | `atol=1e-5`, `rtol=1e-5` |
| Continuous actions (if present) | Declared per-manifest; not required for pilot |

## Match schema (`rlx.match/v0alpha1`)

| Field | Semantics |
|-------|-----------|
| `task` | Task ref (`adapter`, `env`, `version`, optional config) |
| `assignments` | Map role/agent → policy path or ref |
| `seeds` | `{start, count}` or explicit list |
| `action_mode` | `deterministic` \| `stochastic` |
| `record` | Trajectory / input recording flags |
| `failure_policy` | `timeout_seconds`, `retain_incomplete`, `retry` |

## Trajectory schema (`rlx.trajectory/v0alpha1`)

Per episode:

- `seed`, `episode_index`, `agents`, `role_map`, policy digests, task identity
- Per step: joint `observations`, `actions`, `rewards`, `terminations`, `truncations`, optional `action_masks`, `infos`

D-01 requires every transition to include task, agent, role, policy, seed, observation, action, reward, and terminal flags.

## Negative compatibility dimensions

`rlx check` must fail clearly for mismatches in: observation dtype/shape/bounds, action space, role eligibility, preprocessing id, mask requirement, recurrent requirement, and inference mode availability.
