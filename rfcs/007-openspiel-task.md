# RFC 007 — OpenSpiel Task Adapter (Arena 0.3)

**Status:** Accepted in 0.3; generalized by semantic family in 0.5
**Date:** 2026-07-21
**Depends on:** RFC 000, RFC 004, RFC 006

## User promise

Where OpenSpiel’s game representation fits better than PettingZoo (small perfect-information / classic games), expose it as an Arena **task** without erasing OpenSpiel semantics.

## Requirements (OS-*)

| ID | Requirement |
|----|-------------|
| OS-01 | Registered task packager `openspiel` with fail-loud unknown games |
| OS-02 | Map OpenSpiel players ↔ Arena roles; document imperfect-information limits |
| OS-03 | Same match/eval entrypoints as PettingZoo/OpenEnv (`interaction` declared) |
| OS-04 | Trace equivalence vs a declared reference (native OpenSpiel or twin) where claimed |
| OS-05 | Optional: exploitability / game-theoretic metrics as **eval metric plugins**, not core |

## Non-goals

- Full multi-agent deep RL on huge OpenSpiel suites in 0.3
- Replacing evaluation matrices with Elo

## Exit evidence

One small game runs `match` + `eval` with provenance; qualify report attached.

## Frozen implementation

The 0.3 game is `tic_tac_toe`. Version 0.5 qualifies three explicit semantic
families on OpenSpiel 2.x:

- sequential deterministic perfect information: `tic_tac_toe`, `connect_four`,
  `breakthrough`;
- sequential chance/imperfect information: `kuhn_poker`, `leduc_poker`, with
  seeded chance consumption and information-state tensors;
- simultaneous deterministic: `matrix_rps`, with one joint `apply_actions`.

All use roles `player_0`/`player_1`, game-specific tensors/actions, required legal
masks, and frozen traces. Other games or mismatched semantics fail with the extension
and qualification recipe.
