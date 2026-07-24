# Evaluation suites (legacy intent, stable 1.0 runs)

Versioned suites expand to match jobs (`run_match` or AEC), record a sampling ledger, and produce reports that keep payoff matrices primary.

The `rlx.evaluation/v0alpha1` suite schema is legacy-frozen so earlier digests
remain valid. RLX 1.0 additionally projects it into
`rlx.evaluation-intent/v1` and records `rlx.evaluation-binding/v1`. Endpoint,
interpreter, worker count, and timeout implementation change the binding digest,
not semantic intent. Seeds, assignments, metrics, provider semantics, and
missingness policy change intent.

## Workflow

```bash
rlx eval validate ./evaluation.yaml --population sha256:…=./population.yaml
rlx eval run ./evaluation.yaml \
  --policy candidate=./player_0.rlx \
  --policy <digest>=./rock.rlx \
  --population <pop-digest>=./population.yaml \
  --out ./eval-runs/crossplay
rlx eval report ./eval-runs/crossplay --json
rlx eval bundle ./eval-runs/crossplay --out ./bundles/crossplay
# or: rlx release build --eval ./eval-runs/crossplay --out ./bundles/crossplay
```

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

After `rlx task verify-equivalence`, copy the returned
`shared_task_intent_digest` into both evaluation suites as
`task_intent_digest`. Verified native and external tasks can then share one
evaluation-intent digest while retaining different execution bindings.

See [RFC 004](../rfcs/004-evaluation.md) and [populations.md](populations.md).

## Trajectory slices

```bash
rlx data select ./eval-runs/crossplay --out ./datasets/losses \
  --policy sha256:… --role player_0 --outcome loss
```

Datasets point at source digests; sources are not rewritten (SL-01…03).

## Hermetic reproduce (eval clean-room)

Sibling to [clean-room.md](clean-room.md): from a locked eval bundle, recompute metrics offline without trainer repos.

```bash
# On a fresh machine with RLX + extras installed (offline wheelhouse OK):
rlx inspect ./bundles/crossplay/bundle.yaml --json
# Re-aggregate: rlx eval report against a run dir restored from the bundle trajectories
```

Checklist:

1. Bundle contains `eval_run.json`, locked trajectories, and optional `report.json`.
2. Digests in `bundle.json` match on-disk files.
3. No trainer checkout on `PYTHONPATH`.
4. Report rebuild does not require re-running policies when only re-aggregation is needed.
