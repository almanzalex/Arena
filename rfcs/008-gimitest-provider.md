# RFC 008 — Gimitest Evaluation Provider (Arena 0.3)

**Status:** Accepted in 0.3; dependency-isolation boundary added in 0.5
**Date:** 2026-07-21
**Depends on:** RFC 000, RFC 004

## User promise

Apply Gimitest (or compatible) robustness scenarios to an Arena policy/task pair and keep **lineage**: exact policy digests, task digest, perturbation config, seeds, and raw evidence.

## Interface

```yaml
# evaluation.yaml
schema: arena.evaluation/v0alpha1
provider: gimitest          # default: native
provider_config:
  suite: …
  …
```

```text
arena eval run eval/robustness.yaml --provider gimitest
```

Providers are a **registry axis** (like metrics/samplers): unknown provider → extension recipe.

## Requirements (GI-*)

| ID | Requirement |
|----|-------------|
| GI-01 | Provider registry; `native` remains default and offline |
| GI-02 | Every result cell carries policy/task/provider_config digests (I-01) |
| GI-03 | Perturbation config is content-addressed and frozen on the eval-run |
| GI-04 | Failures (unsupported perturbation, crash) use M-02 failure records |
| GI-05 | Report plugins may add robustness metrics; matrices/non-transitivity rules still apply when payoffs exist |

## Non-goals

- Reimplementing Gimitest inside Arena
- Requiring Gimitest for ordinary cross-play

## Exit evidence

One robustness eval fixture qualifies; disabling the extra leaves `native` eval green (I-03).

## Frozen implementation

`examples/eval/robustness.yaml` uses Gimitest 1.0 base hooks. Eval-run/report records
provider version, provider-config digest, task digest, and policy digests for every cell.
Gimitest 1.0's published Gymnasium 0.29.1 pin conflicts with current PettingZoo; the
documented install keeps current Gymnasium and installs the provider package with
`--no-deps`. In 0.5, `provider_config.isolation.mode: subprocess` may select an
absolute Python from a separate compatible environment and a timeout. The worker
exchanges versioned JSON and the parent verifies provider lineage. This is dependency
and failure isolation, not a hostile-code sandbox. The provider is never imported on
native evaluation paths.
