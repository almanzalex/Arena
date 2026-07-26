# RFC 005 — Dynamic Agent Lifecycle

**Status:** Implemented and generalized through lifecycle resolvers in 0.5
**Date:** 2026-07-21 (revived 2026-07-22)
**Depends on:** RFC 000, RFC 004, interaction registry (`arena.plugins.interactions`)

## Decision

Fixed-agent `parallel` and `aec` continue to fail loud when a task reports a
changing lifecycle. Qualified dynamic tasks use the separate `dynamic_aec`
interaction so existing execution semantics do not change silently.

The qualification environments force removal, birth, same-ID re-entry, and a joint
leave/join boundary. They are deliberately small, deterministic, and source-available;
additional dynamic environments need their own qualification evidence.

## Required interface (when revived)

1. **Interaction case** `dynamic_aec` registered via `arena.plugins.interactions`.
2. **Assignment model:** map `agent_id → policy digest` that can grow/shrink; births must declare which policy digests are eligible; removals must freeze that agent’s trajectory segment.
3. **Trajectory schema:** joint steps must record `agents_alive`, join/leave events, and per-agent reward accumulation across dead steps (PettingZoo AEC semantics).
4. **Compose check:** every newly born agent must pass `compose_check` before first `act`.
5. **Conformance + qualify:** fixture with forced birth/removal; `arena adapter qualify` must include lifecycle evidence before any support claim.

## Non-goals

- Silent no-op when agents appear/disappear.
- Inferring policies for unknown agents.
- Claiming “PettingZoo complete” without a qualify report for the dynamic case.

## Implemented contract

The `explicit` resolver preserves `agent_id → policy` assignments with
`task.lifecycle.birth_eligibility`. The `role` resolver maps declared concrete IDs
onto stable assignment roles and requires `join_eligibility` by role. Both plans are
immutable and refuse undeclared agents.

The runner refuses incomplete coverage before creating output, then re-runs
compose-check and resets policy state at every actual join boundary, including
same-ID re-entry. Trajectories record resolved bindings, `agents_alive`,
`join_events`, `leave_events`, and ordered `agent_segment_history`; the old
`agent_segments` view remains for compatibility.
