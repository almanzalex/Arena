# RFC 013 — Hosted accounts / catalog / control plane (deferred)

**Status:** Deferred (not a product; local foreshadowing stub only)
**Date:** 2026-08-10
**Depends on:** RFC 000, RFC 009, RFC 010

## Honest summary

Arena is a **local-first** protocol and CLI for portable policies, matches,
evaluations, and user-owned artifact mirrors. This RFC does **not** ship a
hosted SaaS product, Arena accounts, billing, cloud catalogs, dashboards, or a
multi-tenant control plane.

A future hosted plane remains a deliberate post-1.0 business change (see
`TODOS.md`). Until revisit criteria fire, the only catalog surface is a **local
stub** that lists `file://` mirror descriptors already on disk.

## Why deferred

| Reason | Detail |
|--------|--------|
| Product shape | Hosting accounts/catalog/control changes Arena from local protocol/tooling into a service business |
| Existing escape hatch | User-owned `file://`, HF, OCI, W&B, and MLflow stores already move digests without Arena credentials |
| Trust model | Identity is SHA-256 content addressing; authenticity is user-owned detached attestation — no Arena CA |
| 1.0 boundary | Evidence gates close local/offline claims; hosted scope is an explicit non-goal in RFC 000/009/010 |

**Revisit trigger** (from `TODOS.md`): repeated user demand that cannot be met by
user-owned stores.

## Local-first Arena owns today

- Artifact manifests and content-addressed identity (`sha256:…`)
- Portable policy / match / eval contracts and compatibility checks
- Local `.arena/` workspace and offline match/eval/train recipes
- User-selected mirrors (`arena push` / `arena pull`) that preserve digests
- Detached user-owned attestations over identity (no global trust service)

## What a future hosted plane *might* own

Only if a later RFC reopens this with clear demand and operators:

| Capability | Hosted plane (hypothetical) | Remains local-first |
|------------|----------------------------|---------------------|
| Account / org identity | AuthN/AuthZ, billing, quotas | Not required for push/pull/match |
| Shared catalog index | Searchable index of *published* digests + metadata | Source of truth stays content digests; local mirrors remain valid offline |
| Control plane | Job scheduling, fleet status, shared eval queues | Seeded local `match` / `eval` remain the reproducible unit |
| Hosted mirror | Optional `arena://…` backend behind the same store adapter contract | Digests must not remap; offline core must still work without the plane |

Non-negotiable if revisited: hosted URIs must not become a second identity system.
`sha256:…` remains the artifact identity; a catalog entry is a pointer, not a rewrite.

## Local catalog stub (this change)

```text
arena catalog local <dir|file:///absolute/path>
```

Lists `arena.mirror/v1` descriptors under `<root>/artifacts/*.json` from a
filesystem mirror produced by `arena push … file://…`. Output is a structured
catalog listing (`arena.catalog-list/v1`) with identity, kind, URI, and file
counts.

This is useful today for lab directories and foreshadows a hosted catalog shape
without claiming network discovery, accounts, or remote indexes.

### Non-goals for the stub

- No network calls, accounts, auth, billing, or telemetry
- No ranking, recommendations, or “official” public package index
- No rewrite of digests or silent remapping of URIs
- No claim that Arena operates a hosted control plane

## Acceptance

| ID | Criterion |
|----|-----------|
| H-01 | RFC states deferred status and ownership split honestly |
| H-02 | `arena catalog local` lists descriptors from a `file://` mirror directory |
| H-03 | Stub fails loud on missing/invalid roots; never invents remote entries |
| H-04 | Tests cover empty, populated, and invalid inputs |

## Exit / reopen

Do not expand this into accounts or a network catalog without a new RFC that
names operators, threat model, identity non-remapping proofs, and an offline
degradation story. Until then, document demand against user-owned stores first.
