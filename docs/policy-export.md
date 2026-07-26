# Exporting a custom-PyTorch policy

## Contract

Policies are declarative: architecture template + `state_dict` + preprocessing, **or**
a bring-your-own scriptable `nn.Module` captured as TorchScript. Load time never
imports a training repository.

Supported templates:

- `mlp_categorical`
- `gru_categorical`

Custom modules use the BYO TorchScript path when `torch.jit.script` can capture
their explicit tensor-only `forward(obs[, hidden][, action_mask])` contract. Arena stores
the TorchScript archive and source-captured reference cases, not the trainer module.
Trace is an explicit, opt-in fallback for control-flow-free actors only; it must not be
used to freeze an example path for dynamic Python control flow.

## Action-space boundary

Action schemas are a **discriminated union of typed cases**. Incomplete claims fail
before publish; Arena never flattens, reorders, or coerces action types.

- **Templates** (`mlp_categorical` / `gru_categorical`): `Discrete` only.
- **BYO TorchScript** supported cases when the contract is complete:
  - `Discrete` — integer action, optional masks, argmax / seeded categorical sample.
  - `MultiDiscrete` — `nvec` + `logit_layout: {kind: concatenated}` (+ optional
    contiguous `slices`) + `sampling_order` (`sequential` or permutation) + `masks`.
    Actor emits concatenated per-factor logits; trajectory stores an int vector of
    length `len(nvec)`.
  - Deterministic `Box` — exact shape/bounds, finite floats, no clipping.
  - Stochastic `Box` (`distribution: diagonal_gaussian`) — `param_layout`
    `{kind: mean_log_std_concat}`, `transform.order: [sample, tanh, affine]`,
    `rng.algorithm: numpy_generator`, `deterministic_mode: mean`.
  - Typed `Dict` — canonical `key_order`, nested typed `spaces`, and
    `param_layout: {kind: concatenated_fields, fields: {...}}`.
- **Still rejected:** untyped/open Dict; incomplete MultiDiscrete/Dict/Gaussian
  claims; arbitrary mixture distributions or alternate transform orders; unbounded
  or auto-clipped Box outputs; unscriptable Python actors (orthogonal axis).

`torch.export` is not a shipping runtime tier in this release. Bundling arbitrary Python
source is also intentionally absent: an integrity digest and allowlist do not sandbox
Python. An actor that cannot be scripted fails before export with a recommendation to
refactor its inference path to explicit tensor operations. No partial `.arena` bundle is
published.

## CLI — template export

```bash
arena policy export \
  --adapter custom-pytorch \
  --source ./checkpoint.pt \
  --role player_0 \
  --spec examples/handoff/export_spec_player_0.yaml \
  --out ./artifacts/player_0.arena

arena policy verify ./artifacts/player_0.arena
```

## CLI — BYO TorchScript export

Exporter-side only: Arena imports `package.module:factory` to build the live module,
optionally loads `--source` weights (`weights_only=True` by default), scripts the
module, and embeds source-captured reference cases. The receiver loads only the
TorchScript payload + preprocess IR (no trainer imports).

```bash
arena policy export \
  --adapter custom-pytorch \
  --module mylab.actors:build_pistonball_actor \
  --source ./checkpoint.pt \
  --role piston_0 \
  --spec ./export_spec.yaml \
  --reference-cases ./cases.json \
  --source-revision 36bba61 \
  --wrappers-identity 'color_reduction(full)>resize(64,64)>frame_stack(4)' \
  --out ./artifacts/piston_0.arena
```

`--reference-cases` is required (JSON list of `{observation: ...}` objects, or
`{cases: [...]}`). Expected actions/logits are captured from the live module before
scripting. Optional lineage fields (`source_revision`, checkpoint digest, wrapper
identity) are recorded on the manifest and excluded from the content digest.

## Task wrappers (PettingZoo / SuperSuit)

Image / stacked-observation stacks that were applied in training must be declared on
the **task**, not guessed at match time:

```yaml
task:
  adapter: pettingzoo-parallel
  env: butterfly/pistonball_v6
  config: { continuous: false, max_cycles: 125 }
  observation_layout: HWC
  wrappers:
    - { op: color_reduction, mode: full }
    - { op: resize, x_size: 64, y_size: 64 }
    - { op: frame_stack, stack_size: 4 }
```

Supported wrapper ops: `color_reduction`, `resize`, `frame_stack` (SuperSuit aliases
`*_v0` / `*_v1` accepted). Unknown ops fail before env construction with an
extension recipe. Missing wrappers when the policy expects wrapped spaces fail at
`arena check` with an observation-shape mismatch — Arena never silently compares
against the unwrapped env.

## Axes + case registry

Dispatch for action types, Box distributions, preprocess ops, wrapper ops, actor
payloads, and task packaging goes through `arena.core.registry` / `arena.plugins`.
Unknown kinds fail loud. To add a case: implement the axis interface, register it,
add incomplete + complete tests, and run `arena adapter qualify` before claiming
support. `arena capture --task …` drafts spaces/action cases from a live env
(best-effort; human confirms; stochastic Box / Dict `param_layout` are never invented).

### Opt-in trust tiers (not sandboxed)

- **Payload `trusted_source`**: digest-pinned `.py` + `weights_only` state_dict.
  Refused without `--trust-source`. Prefer TorchScript.
- **Task packaging `entrypoint_bundle`**: digest-pinned env entrypoint. Refused
  without `--trust-task-code`. Prefer `pettingzoo_wrappers`.

Policy-side preprocess (`arena.preprocess/v1`) covers transforms **outside** the env
wrappers (e.g. HWC→CHW layout). Ops that live inside the network (e.g. `/255`) stay
in the scripted module.

`arena policy export` / `verify` captures the exported policy's behavior as reference cases
(`payloads/reference_cases.json`) on the exporting machine. `arena policy verify`
replays them and fails if the bundle no longer reproduces the same
actions/logits (e.g. on a different torch version or platform). To verify
against externally supplied cases, pass `--source-test <cases.json>`.

Checkpoint formats accepted: raw `state_dict`, or dict with `state_dict` / `model` keys matching the declared architecture.

## SDK

```python
from arena import Policy, Task, Match, check
from arena.conformance.fixtures import build_rps_policy

bundle = build_rps_policy("./player_0.arena", role="player_0", seed=1)
policy = Policy.load(bundle)
task = Task.load({"adapter": "pettingzoo-parallel", "env": "arena/competitive_rps_v0"})
check(task, policy.as_role("player_0")).raise_for_errors()
```

```python
from arena.adapters.policy_custom_torch import export_module_policy

export_module_policy(
    out_dir="./actor.arena",
    name="actor",
    roles=["piston_0"],
    module=my_module,
    observation={"type": "Box", "shape": [64, 64, 4], "layout": "HWC", "dtype": "uint8"},
    action={"type": "Discrete", "n": 3},
    preprocessing={"pipeline": {"version": "arena.preprocess/v1", "steps": [
        {"op": "layout", "from": "HWC", "to": "CHW"},
    ]}},
    reference_cases=cases,
    source_act_fn=source_fn,
)
```
