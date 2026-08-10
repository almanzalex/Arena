# RFC 011 — Streaming / sharded dataset materialization (spike)

**Status:** Spike (exploratory; not a 1.0 commitment)
**Date:** 2026-08-10
**Depends on:** RFC 000, atomic materialize (#4), provenance binding (#19)

## Problem

`arena data materialize` (and `materialize_dataset`) copies every selected episode
into one portable directory and publishes it atomically. That contract is correct
for small/medium lab slices and for offline training when the producer run may
disappear.

It becomes expensive or awkward when:

- the corpus is large enough that a full copy dominates disk or wall time;
- consumers (trainers, provenance checks) only need sequential verified reads;
- operators want episode files partitioned across N shard directories for parallel
  I/O or packaging without changing episode digests.

`docs/1.0-readiness.md` and `docs/training.md` already list large sharded /
streaming datasets as outside the built-in 1.0 trainer cases. This RFC spikes
library-level primitives so a later product decision can adopt, revise, or drop
them without rewriting atomic materialize.

## Decision (spike scope)

Ship two additive APIs under `arena.dataset` only:

1. **Stream-read** — verify digests while iterating episode bytes from an existing
   select/materialize manifest. No copy. Producer paths must still resolve.
2. **Sharded materialize** — same digest verification and atomic publish as
   `#4`, but episode files land under `episodes/shard_XXXX/` with deterministic
   index-modulo assignment.

Do **not** change `arena.core.dataset.materialize_dataset` behavior, CLI
`data materialize`, or the flat `episodes/episode_NNNNNN.json` layout that
training and soak tests rely on.

## Layout and identity

### Stream-read

```text
iter_verified_episodes(dataset.yaml|dict) → yields (index, entry, episode_dict)
```

- Rehashes each episode file; mismatch → `ConformanceError` (same posture as
  training `_load_samples`).
- Optional `split=` filter mirrors recipe `dataset_split`.
- Dataset content digest is **not** recomputed by the iterator; callers that need
  a portable identity still materialize.

### Sharded materialize

```text
episodes/
  shard_0000/episode_000000.json
  shard_0001/episode_000001.json
  …
dataset.yaml   # episodes[].path relative; lineage.materialized=true
dataset.json
```

- Shard assignment method: `index_mod/v1` — `shard_id = index % shard_count`.
- Optional train/validation/test splits remain `sha256_bucket/v1` (unchanged from
  core materialize).
- Lineage records `sharded: true`, `shard_count`, and `shard_method`.
- Dataset digest includes relative paths, so a sharded tree and a flat tree of
  the same episodes have **different** content digests by design.

Publication still uses `publish_directory` staging + verify: mid-write failures
must not leave a valid-looking final `out_dir` (same guarantee as `#4`).

## Determinism guarantees (in scope)

| Input held fixed | Guarantee |
|---|---|
| Source manifest + `shard_count` (+ optional splits/seed) | Identical episode digests, shard ids, split labels, and dataset content digest across two sharded materializations |
| Source manifest + stream iteration order | Same `(index, entry["digest"])` sequence; payload digests match entry digests |

## Explicit non-goals (this spike)

- Object-store / HF / OCI native streaming readers.
- Compression, columnar formats, or memory-mapped tensors.
- Changing CLI `arena data materialize` defaults or flags.
- Making sharded and flat materialize share a content digest.
- Wiring trainers to stream APIs (built-in BC still loads via flat/sharded
  manifests that list every episode).
- Distributed shard writers, resumable multi-host materialize, or partial
  publish of individual shards.
- Schema bump (`arena.dataset/v0alpha1` stays); shard metadata lives in
  `lineage` / relative paths only.

## Compatibility

- Existing flat materialize + soak tests remain the source of truth for atomic
  publish.
- Provenance bind/verify continue to resolve relative paths when
  `dataset_path` is passed.
- Unknown consumers that assume `episodes/episode_*.json` only must keep using
  flat materialize until they learn shard-relative paths from the manifest.

## Exit evidence for the spike

- Unit tests: stream digest verification + fail-loud mutation.
- Unit tests: sharded materialize digest/shard determinism across two runs.
- Unit tests: atomic materialize (`materialize_dataset`) still publishes a flat
  tree with `lineage.materialized` and without shard fields (regression guard).
- This RFC checked in under `rfcs/011-streaming-sharded-datasets.md`.

## Follow-ups (out of spike)

- Optional CLI `--shards N` once product wants the layout.
- Trainer sample loader that can prefer shard-local parallelism.
- Decide whether 1.0 readiness should promote streaming from “non-goal” to a
  tracked workstream.
