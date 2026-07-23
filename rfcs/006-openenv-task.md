# RFC 006 — OpenEnv Task Adapter (RLX 0.3)

**Status:** Accepted in 0.3; typed second-task qualification added in 0.5
**Date:** 2026-07-21
**Depends on:** RFC 000, RFC 001, RFC 004

## User promise

Use OpenEnv for **isolated / remote** task execution without changing RLX artifact identity. RLX does **not** reimplement OpenEnv’s container or service layer.

## Schema / CLI

```text
rlx task import openenv://… --name task:…@…
rlx task verify-equivalence <native> <openenv> --trace-suite <yaml>
```

Task manifests gain optional fields:

```yaml
adapter: openenv   # or pettingzoo-parallel | openspiel | …
env: openenv://…
interaction: parallel|aec
source_revision: …
equivalence:
  native_ref: task:…@…
  tolerances: { reward: 1e-6, obs: … }
```

## Requirements (OE-*)

| ID | Requirement |
|----|-------------|
| OE-01 | Import pins OpenEnv package/service identity + digest of declared entry metadata |
| OE-02 | `describe_task` / role spaces match native adapter contract shape |
| OE-03 | Match/eval runners call OpenEnv through a registered **task packager** case |
| OE-04 | Connection loss, container crash, timeout → recorded failures (M-02 semantics) |
| OE-05 | Offline: OpenEnv adapter absent → fail loud; core PettingZoo path still works (I-03) |

## Equivalence suite (T-01 / T-02)

Fixed action-trace suite compares native vs OpenEnv for the same seeds:

- observations, actions, rewards, terminations/truncations
- agent order / selection (AEC)
- masks when declared

Mismatches fail with a structured diff unless a **declared** tolerance is present in the suite.

## Non-goals

- Replacing PettingZoo pilot as the default local path
- Guaranteeing bit-identical floats across remote hardware without tolerances
- Hosting OpenEnv for users

## Exit evidence

Qualify fixture: native RPS (or chosen pilot) + OpenEnv twin; `verify-equivalence` green; one eval suite runs on both adapters with identical digests for policies/populations.

## Frozen implementation

- Pilot: `openenv://rlx/competitive_rps_v0`, OpenEnv 0.4.x, Parallel joint-action bridge.
- Second qualification task: `openenv://rlx/vector_coordination_v0`, typed Box
  observations and Discrete joint actions over the real service transport.
- Import pins `/schema`, endpoint, source revision, and an explicit RLX role-space contract.
- Import pins `rlx.openenv-capabilities/v1` features plus the contract digest.
- `examples/tasks/rps-equivalence.yaml` crosses the real WebSocket serialization boundary.
- Runtime errors preserve `disconnect`, `container_crash`, `timeout`, and `protocol_error`.
- Evidence: `tests/acceptance/test_openenv_equivalence.py` and `rlx adapter qualify --peer … --trace-suite …`.
