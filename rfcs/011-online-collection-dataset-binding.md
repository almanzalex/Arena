# RFC 011 — Bounded online collection → Arena dataset binding

**Status:** Proposed (CPU spike only; not a 1.0 product claim)
**Date:** 2026-08-10
**Depends on:** RFC 000, RFC 001, RFC 009 (dataset identity), RFC 010
**Revisit of:** `TODOS.md` “Distributed/online RL training”

## Decision

Arena will not absorb distributed/online RL as a core product surface in 1.0.
This RFC defines a **bounded wedge**: single-process episode collection that emits
existing `arena.trajectory/v0alpha1` runs, binds those episodes to
**policy + task identity** (`arena.dataset-binding/v1`), then hands the portable
dataset to the **already-qualified offline trainers**.

The wedge exists to prove failure semantics and lineage for “I collected, then
trained,” not to introduce Ray, PPO, async actors, or a replay service.

## User promise (bounded)

A lab can:

1. Roll out a fixed number of seeded episodes with a known policy digest on a
   known task.
2. Select those episodes and **stamp** `arena.dataset-binding/v1` (policy digest,
   task identity, optional role).
3. Fail loud if episode bytes disagree with the claimed binding.
4. Materialize a portable dataset and train with `behavior_cloning` or
   `return_weighted_regression` exactly as offline today.

Repeating steps 1–4 in one process is an **online collection loop**. It is still
offline training on frozen datasets between rounds. No algorithm claims beyond
the registered trainers.

## Schemas / artifacts (existing)

| Artifact | Schema | Role in the wedge |
|----------|--------|-------------------|
| Episode / run | `arena.trajectory/v0alpha1`, `arena.run/v0alpha1` | Collection output |
| Dataset slice | `arena.dataset/v0alpha1` | Selected episodes + digests |
| Provenance binding | `arena.dataset-binding/v1` | Policy + task (+ role) stamp |
| Train recipe / run | `arena.train/v1`, `arena.train-run/v1` | Unchanged offline path |

No new training algorithm schema is introduced by this RFC.

## Collection contract (COL-*)

| ID | Requirement |
|----|-------------|
| COL-01 | Collection is **single-process** and **bounded** (explicit episode/seed budget). |
| COL-02 | Episodes are written through the normal match/trajectory path (or an example that emits the same schema). Digests are content-addressed. |
| COL-03 | Incomplete / crashed / timed-out episodes follow existing match failure policy (`retain_incomplete`, recorded failures). They are **never silently dropped** from the run record. |
| COL-04 | A collection round that yields zero bindable episodes for the claimed policy+task+role **fails loud** (`DATASET_SELECT_EMPTY` or equivalent). |
| COL-05 | Online loop rounds are sequential in one process. Round *N+1* may load the policy exported by round *N*; it must not mutate round *N*'s dataset bytes. |

## Binding contract (BIN-*)

| ID | Requirement |
|----|-------------|
| BIN-01 | Before training, the dataset used as the recipe `dataset:` input MUST carry `provenance` with schema `arena.dataset-binding/v1` and a policy digest. |
| BIN-02 | Binding verification rehashes / reloads episode bytes and refuses mismatch (`DATASET_*` conformance errors). |
| BIN-03 | Task identity at least includes `env`; when `adapter` / `version` are bound, episodes must match those fields. |
| BIN-04 | Role-scoped binding (optional) requires the named role’s policy digest on each episode to equal the bound policy. |
| BIN-05 | Unbinding is explicit (`unbind_dataset_provenance`). Training recipes MUST NOT silently ignore a missing binding when the caller requested a bound path. |

### Known gap (honest)

`materialize_dataset` today copies episode digests and lineage but **does not
preserve** the `provenance` block. The wedge therefore **re-binds after
materialize** (and fails if re-bind verification fails). Preserving provenance
through materialize is a follow-up; until then, claiming a materialized dataset
is bound without re-verification is unsupported.

## Failure semantics

| Failure | Behavior |
|---------|----------|
| Env / policy crash mid-episode | Recorded on the run; incomplete retained per failure policy; not silently omitted |
| Seed / budget exhausted with zero matching episodes | Fail before training (`DATASET_SELECT_EMPTY`) |
| Bound policy digest absent from episode `policies` / assignments | Fail on bind/verify |
| Task env/adapter/version drift vs binding | Fail on bind/verify |
| Episode file mutated after select (digest drift) | Fail on materialize and/or train rehash |
| Unknown trainer / online algorithm id | Refuse (existing trainer registry); do not invent a fallback |
| Distributed worker / Ray / multi-host collect | Out of scope — fail by not offering the API |

Crash during materialize or train keeps existing atomicity / exact-resume
behavior for those subsystems; this RFC adds no new recovery protocol.

## Spike evidence (non-normative)

`examples/training/online_collect_loop.py` plus
`tests/training/test_online_collect_wedge.py` demonstrate:

- RPS Parallel collect via `Match.run` → trajectories with policy digests
- `select_bound_episodes` → materialize → **re-bind** → `run_training_recipe`
- A second round that collects with the newly exported policy

The spike is CPU-only, seconds-scale, and does **not** claim learning
improvement, sample efficiency, or distributed throughput.

## What is NOT in scope

Explicit exclusions (do not grow this wedge without a new RFC):

- **Distributed collection** (Ray, fleet workers, parameter servers, async actors)
- **Online RL algorithms** in the trainer registry (PPO, SAC, DQN, IMPALA, …)
- Replay buffers, prioritized replay, or streaming transition stores
- Sharded / compressed / remote live datasets as a collection substrate
- Multi-host orchestration, autoscaling, or hosted control planes
- Claiming “Arena does online RL” as a 1.0 product capability
- Changing match, eval, or store semantics for offline users
- Silent promotion of incomplete episodes into bound training datasets

`TODOS.md` remains correct: full distributed/online RL stays deferred. This RFC
only unlocks the **bounded collection → binding → offline train** revisit
trigger with documented failure semantics.

## Acceptance gates (for promoting beyond spike)

| ID | Criterion |
|----|-----------|
| OL-01 | Documented single-process loop meets COL-* and BIN-* |
| OL-02 | Adversarial tests: empty select, policy mismatch, task mismatch, post-select mutation |
| OL-03 | Docs state non-goals; README/training do not claim distributed/online RL |
| OL-04 | Optional: materialize preserves or explicitly strips+requires re-bind (no silent loss) |

## Exit criterion for this PR

RFC filed + minimal CPU spike + test green. No Ray. No new trainer algorithm.
No 1.0 scope expansion.
