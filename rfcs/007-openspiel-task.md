# RFC 007 — OpenSpiel Task Adapter (RLX 0.3)

**Status:** Draft for 0.3  
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

One small game (e.g. tic-tac-toe or Kuhn poker—pick one and freeze) runs `match` + `eval` with provenance; qualify report attached.
