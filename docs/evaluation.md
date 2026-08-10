# Evaluation suites (legacy intent, stable 1.0 runs)

Versioned suites expand to match jobs (`run_match` or AEC), record a sampling ledger, and produce reports that keep payoff matrices primary.

The `arena.evaluation/v0alpha1` suite schema is legacy-frozen so earlier digests
remain valid. Arena 1.0 additionally projects it into
`arena.evaluation-intent/v1` and records `arena.evaluation-binding/v1`. Endpoint,
interpreter, worker count, and timeout implementation change the binding digest,
not semantic intent. Seeds, assignments, metrics, provider semantics, and
missingness policy change intent.

## Workflow

```bash
arena eval validate ./evaluation.yaml --population sha256:…=./population.yaml
arena eval run ./evaluation.yaml \
  --policy candidate=./player_0.arena \
  --policy <digest>=./rock.arena \
  --population <pop-digest>=./population.yaml \
  --out ./eval-runs/crossplay
arena eval report ./eval-runs/crossplay --json
arena eval bundle ./eval-runs/crossplay --out ./bundles/crossplay
# or: arena release build --eval ./eval-runs/crossplay --out ./bundles/crossplay
```

### One-shot cross-play matrix (single-lab shortcut)

When you already have two or more policy bundles and want a cartesian
population → cross-play matrix → non-transitivity-aware report without
hand-authoring YAML:

```bash
arena eval matrix \
  --policy ./rock.arena --policy ./paper.arena --policy ./scissors.arena \
  --env arena/competitive_rps_v0 \
  --config '{"max_cycles": 1}' \
  --out ./eval-runs/matrix \
  --json
```

This synthesizes a population, expands enumerated cross-play, writes
`eval_run.json` / `report.json`, and binds `evaluation_*`,
`execution_binding_digest`, `sampling_ledger_digest`, and population digests
onto the report.

## Semantics

- **Cells:** fixed policy assignments, population sample, or enumerated/cartesian cross-play.
- **Sampling ledger:** member digests + seed/stream for replay (E-02). Workers must not change seed→cell mapping (EV-04).
- **Compose check:** every assignment is checked before any run directory is written (E-01). Role swaps require a declared `transform`.
- **Metrics:** `mean_return`, `win_rate`, `payoff_matrix` (+ Wilson / bootstrap uncertainty). If a ranking is requested and a cycle is detected, emit `nontransitivity_warning` and keep the matrix primary (MET-03 / E-04).
- **Evidence:** summary cells carry `evidence_refs` to trajectory digests (E-05).
- **State:** `eval-run/v1` records `complete|incomplete|failed|cancelled` plus
  attempted/completed/failed denominators and a semantic result digest.
- **Missingness:** reports refuse non-complete runs unless
  `failure_policy.missingness: allow` and `max_failed_episodes` explicitly permit
  the recorded failures.
- **Hard budgets:** `budgets.timeout_seconds` defaults the cell executor to a
  supervised process. `budgets.executor: thread` is cooperative and should be
  used only for trusted calls that cannot cross JSON.
- **Interaction:** `parallel`, `aec`, or the qualified `dynamic_aec` lifecycle.

After `arena task verify-equivalence`, copy the returned
`shared_task_intent_digest` into both evaluation suites as
`task_intent_digest`. Verified native and external tasks can then share one
evaluation-intent digest while retaining different execution bindings.

See [RFC 004](../rfcs/004-evaluation.md) and [populations.md](populations.md).

## Trajectory slices

```bash
arena data select ./eval-runs/crossplay --out ./datasets/losses \
  --policy sha256:… --role player_0 --outcome loss
```

Datasets point at source digests; sources are not rewritten (SL-01…03).

## Hermetic reproduce (eval clean-room)

Sibling to [clean-room.md](clean-room.md): from a locked eval bundle, recompute metrics offline without trainer repos.

```bash
# On a fresh machine with Arena + extras installed (offline wheelhouse OK):
arena inspect ./bundles/crossplay/bundle.yaml --json
# Re-aggregate: arena eval report against a run dir restored from the bundle trajectories
```

Checklist:

1. Bundle contains `eval_run.json`, locked trajectories, and optional `report.json`.
2. Digests in `bundle.json` match on-disk files.
3. No trainer checkout on `PYTHONPATH`.
4. Report rebuild does not require re-running policies when only re-aggregation is needed.
