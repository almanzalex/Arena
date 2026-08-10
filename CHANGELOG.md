# Changelog


## Unreleased

- Add static shell completion (`arena completion bash|zsh|fish`), optional
  `arena[completion]` argcomplete wiring, and short `arena help` topics
  (overview / install / handoff / completion / naming). Strengthen the PyPI
  name-collision note with deferred rename candidates; do not rename the
  distribution yet.
- Document deferred hosted accounts/catalog/control plane in
  [RFC 013](rfcs/013-hosted-control-plane.md) and add a local-only
  `arena catalog local` stub that lists `file://` mirror descriptors
  (not a hosted SaaS product).
- Strengthen the PettingZoo `classic/rps_v2` multi-agent path with portable
  fixed-action policies, parallel/AEC task YAMLs, a packaged
  `arena demo multiagent` flow, and acceptance coverage that checks digests and
  interaction parity.

## 1.0.0rc1 — 2026-07-25

- Rename the product, Python package, CLI, artifact suffix, schema namespace,
  environment variables, plugin entry points, examples, and documentation to
  Arena. This is an intentional pre-1.0 identity break; predecessor and current
  artifact digests are not interchangeable.
- Freeze strict content identities, evaluation intent/binding/result identities,
  stable CLI JSON and exit contracts, bounded manifest parsing, transactional
  publication, supervised evaluation workers, and signed release evidence.
- Add a packaged support matrix and schema registry, source-free repeat-safe
  handoff demo, public CleanRL producer proof, and cross-runtime native/OpenEnv/
  Gimitest equivalence flow.
- Build and test a lazy out-of-tree store plugin, immutable HF revision handling,
  archive and mirror hardening, and GitHub release provenance/SBOM workflows.
- Add a narrow `gimitest` support extra so the isolated provider worker gets
  Pillow without accepting Gimitest 1.0's obsolete Gymnasium pin.

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
  `arena train`, including episode-integrity checks and reusable policy output.
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
