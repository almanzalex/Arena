# RFC 003 — Policy Populations (Arena 0.2)

**Status:** Accepted for 0.2  
**Date:** 2026-07-21  
**Depends on:** RFC 000, RFC 001

## Goal

Define content-addressed **policy populations**: named/weighted sets of immutable 0.1 policy digests with optional generation/tags and role constraints. Populations enable cross-play and historical-checkpoint evaluation without trainer repositories.

## Schema: `arena.population/v0alpha1`

```yaml
schema: arena.population/v0alpha1
name: opponents-v3
members:
  - policy: sha256:…          # or path resolved to digest at create time
    weight: 1.0               # >= 0
    generation: 3             # optional int/string
    tags: [baseline]          # optional
    roles:
      allowed: [player_1]     # optional role constraints
```

## Identity (POP-01…05)

- Population digest is SHA-256 over canonical JSON of: schema, sorted members (policy digest, weight, generation, tags, roles.allowed).
- Human `name` and store refs are **not** part of identity.
- Members are immutable objects; editing creates a new digest. Refs may move.
- Role constraints are checked before evaluation assignments (`INCOMPATIBLE` if violated).
- Sampling (uniform/weighted) MUST record a **sampling ledger** (chosen digests + seed/stream) on the eval run.

## Non-goals

- Mutating members in place
- Loading trainer checkpoints into a population (only 0.1 policy artifacts)
