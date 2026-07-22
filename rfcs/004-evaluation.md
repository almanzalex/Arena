# RFC 004 — Versioned Evaluation (RLX 0.2)

**Status:** Accepted for 0.2  
**Date:** 2026-07-21  
**Depends on:** RFC 000, RFC 001, RFC 003

## Goal

Versioned **evaluation suites** that lock task(s), populations/policies, role maps/swaps, seeds, budgets, metrics, and recording policy. Suites expand to match jobs (Parallel via existing `run_match`, AEC via dedicated runner), produce reports with uncertainty and non-transitivity guards, and support trajectory slicing plus releaseable evaluation bundles.

## Schemas

| Schema | Purpose |
|--------|---------|
| `rlx.evaluation/v0alpha1` | Locked suite definition |
| `rlx.eval-run/v0alpha1` | Immutable run record + sampling ledger + cell→episode map |
| `rlx.eval-report/v0alpha1` | Metrics with evidence_refs; matrices primary |
| `rlx.dataset/v0alpha1` | Lineage-preserving trajectory slice |
| `rlx.eval-bundle/v0alpha1` | Locked digests for clean-room reproduce |

## Interaction

Task/suite MUST declare `interaction: parallel | aec`. Mismatch fails pre-run. Dynamic agent birth/removal is **unsupported** in 0.2.0 (fail loud with extension recipe).

## Cross-play (XP-*)

Role assignments may be fixed policies, population samples, or enumerated cartesian cells. Role swaps require declared transforms; incompatible swaps fail before any run directory is created.

## Metrics (MET-*)

At least: payoff/return matrix, win rates (when defined), episode counts, failure counts, uncertainty (CI/SE). **MUST NOT** silently collapse non-transitive payoffs to a single ranking; emit `nontransitivity_warning` and keep the matrix.

## Non-goals (0.2)

Training recipes, OpenEnv, OpenSpiel, Gimitest, external stores, hosted service, universal ranking oracles.
