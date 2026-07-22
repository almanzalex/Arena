# RFC 000 — Product Boundary (RLX)

**Status:** Accepted; Stages **0.1, 0.2, and 0.3 done**
**Date:** 2026-07-16 (updated 2026-07-21)

## Stage table

| Stage | Status | Focus |
|-------|--------|--------|
| **0.1** | Done | Portable policy handoff, Parallel matches, trajectories, registries |
| **0.2** | Done | Populations, versioned evaluation, AEC, slicing, eval bundles |
| **0.3** | Done | OpenEnv, one OpenSpiel game, Gimitest provider, file/HF stores ([docs/0.3-complete.md](../docs/0.3-complete.md)) |
| **0.4+** | Deferred | Training recipes, dataset export/reuse |

Handoff: [docs/0.2-complete.md](../docs/0.2-complete.md). Parked: [RFC 005 dynamic agents](005-dynamic-agents.md).

## Native pilot pair (frozen)

| Role | Choice | Rationale |
|------|--------|-----------|
| Task | Bundled PettingZoo **Parallel** competitive RPS (`rlx/competitive_rps_v0` in `rlx.adapters.task_pettingzoo.pilot_env`) | Self-contained zero-sum discrete Parallel env; no pygame/display; implements the PettingZoo Parallel API. Upstream `classic/rps_v2` remains optional. |
| Policies | Simple custom PyTorch categorical actors (feed-forward MLP templates shipped inside the `custom-pytorch` adapter) | No training-repo imports at load time; weights + architecture + preprocessing live in the policy bundle. |

Conformance fixtures F1–F4 exercise Gymnasium-compatible observation/action contracts against these templates. Fixture F5 runs the RPS Parallel task end-to-end. Fixture F6 is the AEC twin. A checked-in cyclic eval demo lives at `examples/eval/demo/` (`bash examples/eval/run_demo.sh`). Deferred maximal-claim items: `docs/0.2-revisit.md`.

## 0.3 external pilots (frozen)

| Integration | Frozen scope |
|---|---|
| OpenEnv | OpenEnv 0.4.x WebSocket transport serving the existing Parallel competitive RPS twin |
| OpenSpiel | OpenSpiel 2.x game id `tic_tac_toe`, AEC only |
| Gimitest | Gimitest 1.0 `GTest` provider hooks with content-addressed provider config |
| Stores | `file://` reference backend and Hugging Face Hub `hf://` backend |

## What RLX owns

- Artifact manifests and content-addressed identity (`sha256:…`)
- Portable-policy and match contracts (see RFC 001)
- Compatibility checks before execution
- Local filesystem workspace (`.rlx/`)
- Source-versus-exported conformance evidence
- Match run records and joint trajectory bundles
- Adapter boundaries (policy / task)

## What RLX does not own (MVP exclusions)

- Hosted services, auth, billing, telemetry, dashboards
- Training algorithms or recipe execution
- OpenEnv, OpenSpiel, and Gimitest internals (0.3 owns adapters only, not upstream cores)
- Hosted registries / cloud services (0.3 mirrors to user-owned file/HF stores; local `.rlx/` remains default)
- Populations / cross-play / AEC are **0.2** (done; see RFC 003/004)
- Dynamic agent lifecycle: **fail loud** until RFC 005 is implemented and qualified
- Training recipes: **0.4** (not 0.3)
- Guarantees of bit-identical numerics across arbitrary hardware (tolerances are declared)

## Artifact definitions

| Artifact | Meaning in 0.1 |
|----------|----------------|
| Policy | Manifest (`rlx.policy/v0alpha1`) + immutable weight/payload digests + declared inference contract |
| Task | Adapter reference + env identity/version + per-role observation/action schemas |
| Match | Manifest (`rlx.match/v0alpha1`) assigning policies to roles/agents over a fixed seed list |
| Run record | Digests, seeds, outcomes, failures, logs for one `rlx match run` |
| Trajectory bundle | Per-step joint transitions + full provenance |

## Trust boundary

- Manifests are human-readable and safe to commit.
- Payloads are immutable blobs addressed by SHA-256; refs may move, digests must not.
- Execution never imports the original training repository; inference dependencies must be declared on the policy manifest.
- Paths outside the workspace or bundle are rejected unless explicitly passed as CLI inputs for export.

## Local-first storage

```text
.rlx/
  objects/       # SHA-256 content-addressed blobs
  refs/          # names/tags → digests
  runs/          # match records, logs, trajectory outputs
  cache/         # rebuildable data
  workspace.toml
```

## Platforms

- Python **3.12+**
- macOS / Linux reference platforms
- Optional extras: `torch`, `pettingzoo` (+ Gymnasium)

## Versioning

- Schema ids: `rlx.policy/v0alpha1`, `rlx.match/v0alpha1`, `rlx.trajectory/v0alpha1`
- Unknown manifest fields are preserved where possible (forward compatibility)
- Package version tracks the CLI/SDK release (0.3.0 for external integrations)

## Acceptance gates (testable)

| ID | Criterion | Automated? |
|----|-----------|------------|
| P-01 | Source vs exported actions match on fixed obs/hidden states (deterministic) | Yes |
| P-02 | Seeded stochastic actions match under RNG contract (or declared distribution tolerance) | Yes |
| P-03 | Recurrent state init/reset boundaries match source | Yes |
| P-04 | Illegal actions never emitted; missing required masks fail pre-run | Yes |
| P-05 | Policy runs after training-repo path is removed from the environment | Yes |
| M-01 | Repeated seeded matches yield identical (or tolerance-compliant) trajectories | Yes |
| M-02 | Crash / timeout / invalid action / incomplete episode recorded, never silently dropped | Yes |
| D-01 | Every transition carries task, agent, role, policy, seed, obs, action, reward, terminal provenance | Yes |
| U-01 | Second user completes export→match from docs alone | Scripted clean-room + documented human checklist |

## Exit criterion

A consumer can install RLX, load an exported policy without the trainer repo, run a seeded PettingZoo Parallel match, and inspect complete joint trajectories.
