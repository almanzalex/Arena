# RFC 007 — OpenSpiel Task Adapter (RLX 0.3)

**Status:** Accepted and implemented in 0.3
**Date:** 2026-07-21
**Depends on:** RFC 000, RFC 004, RFC 006

## User promise

Where OpenSpiel’s game representation fits better than PettingZoo (small perfect-information / classic games), expose it as an RLX **task** without erasing OpenSpiel semantics.

## Requirements (OS-*)

| ID | Requirement |
|----|-------------|
| OS-01 | Registered task packager `openspiel` with fail-loud unknown games |
| OS-02 | Map OpenSpiel players ↔ RLX roles; document imperfect-information limits |
| OS-03 | Same match/eval entrypoints as PettingZoo/OpenEnv (`interaction` declared) |
| OS-04 | Trace equivalence vs a declared reference (native OpenSpiel or twin) where claimed |
| OS-05 | Optional: exploitability / game-theoretic metrics as **eval metric plugins**, not core |

## Non-goals

- Full multi-agent deep RL on huge OpenSpiel suites in 0.3
- Replacing evaluation matrices with Elo

## Exit evidence

One small game runs `match` + `eval` with provenance; qualify report attached.

## Frozen implementation

The only 0.3 game id is `tic_tac_toe` on OpenSpiel 2.x. It is mapped to AEC roles
`player_0`/`player_1`, observation tensors of length 27, `Discrete(9)` actions, and
required legal-action masks. `examples/tasks/openspiel-tic-tac-toe-trace.yaml` pins the
reference trace digest. Other games fail loud with the extension and qualification recipe.
