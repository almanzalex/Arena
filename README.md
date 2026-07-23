# RLX 0.5

Local-first Python CLI/SDK for **portable RL policy handoff**, **versioned
evaluation**, reproducible trajectory training, and verified external
runtime/provider/store integrations.

Export a custom PyTorch policy, hand it to a collaborator without the training repository, run a seeded PettingZoo Parallel (or AEC) match, evaluate populations with cross-play, and inspect complete joint trajectories.

## MVP workflow (0.1 portable policy)

```bash
pip install 'rlx[all]'

# Researcher A
rlx init
rlx policy export --adapter custom-pytorch --source checkpoint.pt \
  --role player_0 --spec examples/handoff/export_spec_player_0.yaml \
  --out ./artifacts/player_0.rlx
rlx policy verify ./artifacts/player_0.rlx

# Researcher B (clean machine, no training repo)
rlx inspect ./artifacts/player_0.rlx
rlx check rlx/competitive_rps_v0 ./artifacts/player_0.rlx --role player_0
rlx match run match.yaml --record --out ./runs/baseline-match
rlx data inspect ./runs/baseline-match/trajectories
```

## Evaluation workflow (0.2)

Runnable cyclic RPS demo (checked in under `examples/eval/demo/`):

```bash
pip install -e '.[torch,pettingzoo]'   # or 'rlx[torch,pettingzoo]'
bash examples/eval/run_demo.sh
```

Manual flow:

```bash
rlx population create ./population.yaml --ref populations/opponents
rlx eval run ./evaluation.yaml --policy … --population … --out ./eval-runs/x
rlx eval report ./eval-runs/x --json
rlx data select ./eval-runs/x --out ./datasets/losses --outcome loss
rlx eval bundle ./eval-runs/x --out ./bundles/x
```

See [docs/populations.md](docs/populations.md), [docs/evaluation.md](docs/evaluation.md),
[docs/eval-clean-room.md](docs/eval-clean-room.md), [docs/eval-usability-signoff.md](docs/eval-usability-signoff.md).
**0.2 sealed:** [docs/0.2-complete.md](docs/0.2-complete.md).
**0.3 complete:** [docs/0.3-complete.md](docs/0.3-complete.md).
**Release evidence:** [docs/0.3-evidence.md](docs/0.3-evidence.md).
**0.4 boundary expansion:** [docs/0.4-boundaries.md](docs/0.4-boundaries.md).
**0.4 evidence:** [docs/0.4-evidence.md](docs/0.4-evidence.md).
**0.5 generalization boundary:** [docs/0.5-boundaries.md](docs/0.5-boundaries.md).
**0.5 evidence:** [docs/0.5-evidence.md](docs/0.5-evidence.md).
**Before 1.0:** [docs/1.0-readiness.md](docs/1.0-readiness.md).
Deferred items: [docs/0.2-revisit.md](docs/0.2-revisit.md).

## External integration workflow (0.3)

```bash
# OpenEnv: launch/import the frozen RPS pilot, then prove native↔remote semantics.
pip install 'rlx[openenv]'
python -m rlx.adapters.task_openenv.server --port 8000
rlx task import openenv://127.0.0.1:8000/rlx/competitive_rps_v0 \
  --name task:rps-openenv@0.3 --source-revision openenv-0.4.1
rlx task verify-equivalence examples/tasks/native-rps.yaml task-rps-openenv-0.3.yaml \
  --trace-suite examples/tasks/rps-equivalence.yaml

# OpenSpiel: deterministic frozen catalog with checked reference traces.
pip install 'rlx[openspiel]'
rlx task verify-equivalence examples/tasks/openspiel-connect-four.yaml \
  --trace-suite examples/tasks/openspiel-connect-four-trace.yaml

# Artifact mirrors preserve the policy's sha256 identity.
rlx push examples/eval/demo/rock.rlx file:///tmp/rlx-mirror --verify
rlx pull 'file:///tmp/rlx-mirror#sha256:…' --verify
```

See [external tasks](docs/external-tasks.md), [evaluation providers](docs/eval-providers.md),
and [external stores](docs/external-stores.md).

## Generalized boundary workflow (0.5)

```bash
# Match/eval with forced agent birth + removal and explicit digest eligibility.
rlx match run dynamic-match.yaml --out dynamic-run

# Turn selected trajectories into a source-independent dataset, then train/reuse.
# Content-stable train/validation splits.
rlx data materialize selected/dataset.yaml --out portable-dataset \
  --split train=.8 --split validation=.2 --split-seed 17
rlx train behavior-cloning.yaml --out training-run
rlx policy verify training-run/policy.rlx

# Produce machine-readable push/pull evidence for any store implementation.
rlx store qualify training-run/policy.rlx \
  'oci://registry.example/lab/rlx?simulate=/tmp/rlx-oci' \
  --out store-qualification.json

# Sign the artifact identity with a user-owned key; the detached signature remains
# valid after any identity-preserving mirror round trip.
rlx attest keygen --private lab-private.pem --public lab-public.pem
rlx attest sign training-run/policy.rlx --key lab-private.pem \
  --issuer my-lab --out policy.attestation.json
rlx attest verify training-run/policy.rlx policy.attestation.json \
  --key lab-public.pem

# Run lifecycle re-entry, split/train/resume/reuse, three game semantics,
# signing, and four simulated stores as one checked journey.
bash examples/boundaries/run_demo.sh
```

See [training recipes](docs/training.md), [dynamic agents](docs/dynamic-agents.md),
and the [0.5 boundary record](docs/0.5-boundaries.md).

## Pilot pair (frozen in RFCs)

| | |
|--|--|
| **Task** | Bundled PettingZoo Parallel competitive RPS (`rlx/competitive_rps_v0`); AEC twin `rlx/competitive_rps_aec_v0` |
| **Policies** | Declarative custom-PyTorch categorical actors (`mlp_categorical` / `gru_categorical`) |

See [rfcs/000-product-boundary.md](rfcs/000-product-boundary.md), [rfcs/001-portable-policy-contract.md](rfcs/001-portable-policy-contract.md), [rfcs/003-populations.md](rfcs/003-populations.md), [rfcs/004-evaluation.md](rfcs/004-evaluation.md).

## Portable actor boundaries (0.1)

RLX supports fixed categorical templates and a narrow bring-your-own subset via an
**axes + case registry**: new messy lab scenarios are handled by registering
cases, not by reactive core patches. Dispatch is always `registry.get(kind)`;
unknown kinds fail loud with an extension recipe (interface, tests, and
`rlx adapter qualify` before claiming support). No silent coerce/flatten/pad.

A scriptable TorchScript tensor actor is the preferred BYO payload (script-first;
trace is explicit opt-in). The actor must declare `forward(obs[, hidden][, action_mask])`,
recurrence/hidden shape, preprocessing/layout, and action semantics. The receiver
imports no trainer package.

- Image observations must declare `layout: CHW|HWC`; preprocessing uses a
  serializable `rlx.preprocess/v1` pipeline (layout, running normalization, clipping,
  frame stack, flatten) via registered preprocess ops. Shape changes are errors.
- PettingZoo tasks may declare a SuperSuit wrapper chain (`color_reduction`,
  `resize`, `frame_stack`) plus `observation_layout` so `check`/`match` use the
  **wrapped** spaces. Unknown/missing wrappers fail loud via the wrapper registry.
- BYO TorchScript export is available via `rlx policy export --module pkg:factory`
  or `export_module_policy()`. Opt-in `trusted_source` (digest-pinned `.py`,
  `--trust-source`) exists but is **not sandboxed** and is not the default.
- Discrete actors support explicit in-graph masks. Deterministic bounded `Box`
  and complete BYO cases for `MultiDiscrete`, recursive typed `Dict`, and
  diagonal-Gaussian stochastic `Box` are registered cases when declared fully.
  Template actors stay Discrete-only.
- Task packaging defaults to `pettingzoo_wrappers`; opt-in `entrypoint_bundle`
  (digest-pinned env entrypoint, `--trust-task-code`) is registry-backed and
  refused by default.
- `rlx capture --task …` drafts spaces/action cases from a live env (best-effort;
  human confirms before publish).
- `rlx policy verify` requires source-captured evidence by default.

TorchScript is a portability format, not a malware sandbox: only load policy bundles
from a trusted lab/source. Full-module pickle checkpoints remain refused by default.

### Capability matrix

Capability claims are limited to **registry-registered + qualified** cases.

| Status | Policy/action subset | Exact contract and rationale |
|---|---|---|
| Supported now | `Discrete` categorical template actors and scripted custom `nn.Module` actors | Integer action, optional declared mask, deterministic argmax or NumPy `Generator` categorical sampling. Scripted actors require a source-captured reference suite and load without the trainer repository. CLI: template `--source` or BYO `--module`. |
| Supported now | Deterministic bounded `Box` scripted actors | Exact shape/dtype/bounds; finite output only; no clipping. |
| Supported now | `MultiDiscrete` BYO TorchScript (complete case) | `nvec` + `logit_layout: {kind: concatenated}` + `sampling_order` + `masks`. Never flattened to Discrete. |
| Supported now | Recursive typed `Dict` BYO TorchScript (complete case) | Canonical `key_order`, typed `spaces`, `param_layout: {kind: concatenated_fields}`. |
| Supported now | Stochastic `Box` `diagonal_gaussian` BYO (complete case) | `param_layout`, `transform.order: [sample, tanh, affine]`, `rng.algorithm: numpy_generator`. |
| Supported now | Declarative PettingZoo SuperSuit task wrappers (`pettingzoo_wrappers`) | `color_reduction` / `resize` / `frame_stack` (+ layout). |
| Supported only with explicit trust | `trusted_source` payload; `entrypoint_bundle` task packaging | Digest-pinned Python; `--trust-source` / `--trust-task-code`; **not sandboxed**. Prefer TorchScript / pettingzoo_wrappers. |
| Deliberately rejected | Incomplete claims; unknown registry kinds; untyped Dict; arbitrary mixtures | Fail loud with repair guidance or an extension recipe. No silent coercion. |

### Integration capability matrix

| Axis | Supported | Boundary |
|---|---|---|
| Task runtime | OpenEnv 0.4.x; RPS plus typed vector-coordination qualification tasks | Endpoint, schema, role contract, source revision, and protocol capabilities are pinned; transport failures remain distinct. This is not a claim over every OpenEnv environment. |
| Agent lifecycle | `dynamic_aec`; explicit-agent and role resolvers | Join eligibility, compatibility recheck, recurrent-state reset, removal/re-entry segment history, and joint boundary events. Unknown agents still fail loud. |
| Training | Registry cases: behavior cloning and return-weighted regression | Materialized verified episodes, deterministic digest-bucket splits, exact seeded checkpoint resume; categorical Discrete actions with Discrete/Box observations. |
| Game runtime | OpenSpiel 2.x semantic-family fixtures | Sequential perfect-information (`connect_four`), chance/imperfect-information (`kuhn_poker`), and simultaneous (`matrix_rps`) paths with legal masks and frozen traces. Game IDs remain qualification-scoped. |
| Eval provider | Native; Gimitest 1.0 provider | Content-addressed lineage; Gimitest may execute through an explicit separate-Python subprocess. Native cells can run concurrently with stable result order. |
| Artifact store | `file://`, `hf://`, `oci://`, `wandb://`, `mlflow://` | Machine-readable verified qualification, simulation/live labels, identity-preserving mirrors, and optional detached Ed25519 authenticity. |

### How to add a case

1. Implement the axis interface under `rlx/plugins/` (e.g. `ActionCase`, `DistributionCase`, `PreprocessOp`, `WrapperOp`, `PayloadCase`, `TaskPackager`).
2. Register it (`register_action_case(kind, case)`, etc., or an entry point group `rlx.plugins`).
3. Add fail-loud incomplete-claim tests and a complete end-to-end export/verify/act test.
4. Run `rlx adapter qualify <fixture>` on a fixture that exercises the new case before claiming support.

Every rejected category fails before a final bundle is published. The error names the
missing semantic contract and a safe repair; no action is coerced, flattened, reordered,
clipped, or approximated.

## Install

```bash
pip install -e '.[dev]'   # from a checkout
# or
pip install 'rlx[torch,pettingzoo]'
pip install 'rlx[openenv]'    # optional external task runtime
pip install 'rlx[openspiel]'  # optional frozen game adapter
pip install 'rlx[hf]'         # optional Hugging Face mirror
pip install 'rlx[wandb]'      # optional W&B artifact mirror
pip install 'rlx[mlflow]'     # optional MLflow artifact mirror
# OCI uses the ORAS CLI and its normal `oras login` credentials.
```

Core stays small (`pyyaml`, `numpy`). Heavy deps are optional extras.

## Acceptance gates

| ID | Gate |
|----|------|
| P-01…P-05 | Policy export fidelity, masks, recurrence, repo independence |
| M-01…M-02 | Seeded match reproducibility + failure accounting |
| D-01 | Trajectory completeness / provenance |
| U-01 | Clean-room handoff (scripted + hermetic + human checklist) |
| E-01…E-06 | Eval compose-check, sampling ledger, metrics, non-transitivity |
| A-01/A-02 | AEC runner + Parallel regression |
| D-02/D-03 | Dataset slice lineage + eval release bundles |
| Q-02 | Adapter qualify covers population/eval fixtures |
| U-02 | In-repo cross-play script replaced by population+eval |
| T-01…T-03 | External trace equivalence, serialization, and failure semantics |
| I-01…I-03 | Provider lineage, store round-trip, offline native core |
| S-01/U-03 | Overhead budget and integration-author qualification workflow |
| L-01…L-04 | Dynamic birth eligibility, compose-check/reset, lifecycle trajectories |
| TR-01…TR-04 | Materialized dataset integrity, seeded recipe, policy verify/reuse |
| ST-07 | Simulated and opt-in live OCI/W&B/MLflow/HF identity round trips |
| G-01…G-10 | Registry generalization, lifecycle re-entry, exact resume, semantic game families, external isolation, store qualification, authenticity, concurrency, composition |

```bash
pytest -q                    # fast default selection (slow/docker gates deselected)
```

### Hermetic U-01 clean-room gate

The strongest automated approximation of the human clean-room step lives in
`tests/acceptance/test_u01_hermetic.py`. It builds a real wheel + sdist
(`python -m build`), installs **only the wheel** into a throwaway
`python3.12 -m venv` with a scrubbed `HOME`/`XDG`/`PYTHONPATH` (no repo on the
import path), copies in **only** the `.rlx` bundles + `match.yaml` + the guide,
disables the network, and runs the commands **parsed out of** `docs/clean-room.md`
in order. If Docker is present it also runs the same flow in a minimal
`python:3.12-slim` image with `--network none`.

```bash
pytest -m slow -q            # wheel build + hermetic venv (+ Docker if available)
pytest -m docker -q          # only the Docker --network none variant
```

These gates are marked `slow`/`docker` and deselected from the default run. In
CI they run as a separate `hermetic` job (see `.github/workflows/ci.yml`). A CPU
torch index can be supplied via `RLX_TORCH_INDEX_URL` to avoid large CUDA wheels.
See [docs/clean-room.md](docs/clean-room.md) for the coverage matrix, the
machine-parseable command block, and the design of a third (LLM-reader) technique.
See [docs/adapter-qualification.md](docs/adapter-qualification.md) for the
evidence required before an adapter is called supported, and
[docs/usability-signoff.md](docs/usability-signoff.md) for the real-reader record.

## Layout

```text
rlx/
  core/           # manifests, store, compatibility, SDK, registry, capture, population, dataset
  plugins/        # axis case registrations (action, samplers, metrics, …)
  cli/            # rlx commands
  runtime/        # match + fixed/dynamic AEC + evaluation + training + trajectories
  conformance/    # fixtures F1–F6
  adapters/
    policy_custom_torch/
    task_pettingzoo/
    task_openenv/
    task_openspiel/
    eval_gimitest/
```

## Honest remaining boundaries (0.5)

No hosted RLX service/auth, provider billing/dashboard replacement, universal OpenSpiel
catalog, arbitrary online RL algorithm, silent lifecycle inference, artifact certificate
authority/revocation service, or silent Elo-only ranking. Live remote smokes require the
user's own credentials and are never conflated with `?simulate=` evidence. The exact
remaining release gates—not vague future scope—are tracked in
[docs/1.0-readiness.md](docs/1.0-readiness.md).
