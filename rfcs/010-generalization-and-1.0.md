# RFC 010 — Generalization boundary and path to 1.0

**Status:** Accepted for 0.5
**Date:** 2026-07-22
**Depends on:** RFCs 000, 004–009

## Decision

Arena generalizes by stable axes and qualification evidence, not by absorbing every
upstream feature. Version 0.5 adds trainer and lifecycle-resolver axes, broadens task
adapters by semantic family, and makes provider/store evidence comparable.

A capability is supported only when all three are present:

1. a registered implementation behind a documented interface;
2. fail-loud validation for claims outside that implementation;
3. a reproducible qualification fixture that exercises the distinctive semantics.

## Axes

| Axis | 0.5 built-ins | Qualification distinction |
|---|---|---|
| Trainer | `behavior_cloning`, `return_weighted_regression` | Conflicting-return data proves the objectives differ |
| Lifecycle resolver | `explicit`, `role` | Re-entry and a simultaneous leave/join boundary |
| Interaction/task | Parallel, AEC, dynamic AEC, OpenEnv, OpenSpiel families | Frozen traces and failure taxonomy |
| Eval provider | Native, Gimitest | Lineage plus optional process/interpreter boundary |
| Store | File, HF, OCI, W&B, MLflow | Identical qualification schema with explicit simulation/live mode |

## Identity and authenticity

Content identity remains SHA-256 over canonical artifacts. Authenticity is a detached
Ed25519 statement over identity and artifact kind. The user supplies the trusted
public key; Arena does not create a global trust system.

This separation means an artifact may move across stores without being re-signed,
while a modified artifact cannot reuse the old attestation.

## Compatibility

- Existing fixed-agent interactions retain their old failure behavior.
- Existing explicit dynamic lifecycle manifests remain valid.
- Existing training recipes continue to dispatch as behavior cloning.
- Existing task/store URIs retain identity.
- New manifest fields are additive; unknown registry kinds fail with extension
  guidance rather than fallback behavior.

## Non-goals

- Universal task/game/trainer claims.
- Hosted control plane or credential management.
- Arbitrary-code sandboxing.
- Certificate authority or signature revocation service.
- Distributed online training in the 0.5 core.

## 1.0 policy

After 0.5, breadth pauses. The 1.0 release closes platform, live-integration,
clean-room, soak, security, schema-compatibility, and performance evidence listed in
`docs/1.0-readiness.md`.
