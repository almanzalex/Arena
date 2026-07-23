# Changelog

## 0.5.0 — 2026-07-22

- Replace one-off training dispatch with a trainer registry and qualify behavior
  cloning plus return-weighted regression; add deterministic dataset splits and
  mutation-checked exact checkpoint resume.
- Generalize dynamic AEC assignment through explicit-agent and stable-role lifecycle
  resolvers, including removal/re-entry and simultaneous leave/join evidence.
- Qualify three OpenSpiel semantic families: deterministic sequential perfect
  information, explicit chance/imperfect information, and simultaneous play.
- Add a second real OpenEnv task with typed Box contracts and pin an explicit
  transport capability contract.
- Isolate Gimitest providers behind an optional subprocess/Python boundary and make
  native evaluation workers execute concurrently without changing result order.
- Add store qualification for every registered backend and detached, user-owned
  Ed25519 attestations that remain valid across identity-preserving mirrors.
- Add a composed 0.5 journey and a narrow, evidence-backed 1.0 readiness ledger.

## 0.4.0 — folded into 0.5.0

- Add qualified `dynamic_aec` lifecycle execution with explicit policy-digest birth
  eligibility, compatibility rechecks, state resets, and join/leave trajectory evidence.
- Add portable dataset materialization and seeded behavior-cloning recipes through
  `rlx train`, including episode-integrity checks and reusable policy output.
- Expand the frozen OpenSpiel catalog to `connect_four` and `breakthrough` while
  continuing to reject chance, simultaneous, and unqualified games.
- Add OCI/ORAS, W&B, and MLflow verified mirrors, deterministic local simulations,
  and an opt-in authenticated live-store smoke.

## 0.3.0 — 2026-07-21

- Add identity-pinned OpenEnv task import, native/external trace equivalence,
  transport failure recording, and the frozen competitive-RPS pilot.
- Add the frozen OpenSpiel `tic_tac_toe` AEC adapter with legal-action masks.
- Add the Gimitest evaluation-provider axis with complete provider/task/policy lineage.
- Add verified `file://` and Hugging Face artifact mirrors without identity remapping.
- Preserve the offline native 0.2 path and keep dynamic agents and training outside scope.

## 0.2.0 — 2026-07-21

- Add populations, versioned evaluation, AEC execution, trajectory slicing, and eval bundles.
- Seal the checked demo and wheel-based offline clean-room gates.
