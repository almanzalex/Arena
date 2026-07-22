# RFC 005 — Dynamic Agent Lifecycle (parked)

**Status:** Parked beyond 0.3 (the frozen 0.3 pilots use fixed agents)
**Date:** 2026-07-21
**Depends on:** RFC 000, RFC 004, interaction registry (`rlx.plugins.interactions`)

## Decision

RLX **0.2.0 fails loud** when a task reports `dynamic_agents: true` (or when living agents diverge from the fixed assignment set mid-episode). Full birth/removal lifecycle is **not** required to close 0.2 or to open 0.3.

Implement only when a **concrete named external environment** cannot be represented with a fixed agent set.

## Required interface (when revived)

1. **Interaction case** `dynamic_aec` (or extension of `aec`) registered via `rlx.plugins.interactions`.
2. **Assignment model:** map `agent_id → policy digest` that can grow/shrink; births must declare which policy digests are eligible; removals must freeze that agent’s trajectory segment.
3. **Trajectory schema:** joint steps must record `agents_alive`, join/leave events, and per-agent reward accumulation across dead steps (PettingZoo AEC semantics).
4. **Compose check:** every newly born agent must pass `compose_check` before first `act`.
5. **Conformance + qualify:** fixture with forced birth/removal; `rlx adapter qualify` must include lifecycle evidence before any support claim.

## Non-goals

- Silent no-op when agents appear/disappear.
- Inferring policies for unknown agents.
- Claiming “PettingZoo complete” without a qualify report for the dynamic case.

## Extension recipe (current product text)

Dynamic agent birth/removal is outside RLX 0.3. To add support: implement agent lifecycle linked to policy state, register an interaction/task case, add conformance tests, and run `rlx adapter qualify` before claiming support.
