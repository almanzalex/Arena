# Offline training examples

Minimal CPU path for Arena's built-in offline trainers: record teacher rollouts,
select a role slice, materialize a portable dataset, train, verify, then evaluate.

See also [docs/training.md](../../docs/training.md).

## Quick path

```bash
# 1) Record a short teacher match (PettingZoo RPS pilot).
#    Edit match.teacher.yaml assignments to real .arena paths, then:
arena match run examples/training/match.teacher.yaml --out runs/teacher --record

# 2) Select + materialize a self-contained dataset
arena data select runs/teacher --out selected --role player_0
arena data materialize selected/dataset.yaml --out portable-dataset

# 3) Point a recipe at the portable dataset and train
#    Edit recipe.behavior_cloning.yaml: set dataset: portable-dataset/dataset.yaml
arena train examples/training/recipe.behavior_cloning.yaml --out training-run

# 4) Verify the exported bundle
arena policy verify training-run/policy.arena

# 5) Evaluate (assign digests / paths in your evaluation suite)
#    Candidate: training-run/policy.arena
#    Opponent: fixed-action rock (or any compatible policy)
```

## Recipes in this directory

| File | Algorithm | Notes |
| --- | --- | --- |
| `recipe.behavior_cloning.yaml` | `behavior_cloning` | Uniform transition weights; empty `hidden_dims` for a fast CPU smoke. |
| `recipe.return_weighted.yaml` | `return_weighted_regression` | Weights by episode return; set `temperature` / `max_weight`. |

Both recipes use `PLACEHOLDER_PORTABLE_DATASET` for `dataset`. Replace it with a
path to a materialized `dataset.yaml` (absolute or relative to the recipe file).

## Resume

Every training run writes `checkpoint.json` and `checkpoint.pt`. To continue:

1. Copy the recipe and increase `epochs`.
2. Set `resume_from` to the earlier run directory.
3. Keep algorithm, dataset, seed, batch size, learning rate, spaces, and
   architecture identical — Arena refuses contract mismatches.

## Hermetic test

The integration suite lives at `tests/training/test_offline_trainer_e2e.py`
(real env rollouts → train → resume parity → short eval). Default pytest skips
`slow`; include it with:

```bash
CUDA_VISIBLE_DEVICES= pytest tests/training/ -v --override-ini "addopts="
```
