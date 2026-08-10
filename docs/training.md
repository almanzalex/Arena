# Training recipes and portable datasets

0.5 keeps training downstream of recorded evidence and dispatches algorithms
through the trainer registry. The built-in qualified cases are seeded behavior
cloning and return-weighted regression over Arena trajectory datasets.

```bash
arena data select runs/teacher --out selected --role player_0
arena data materialize selected/dataset.yaml --out portable-dataset \
  --split train=.8 --split validation=.2 --split-seed 17
arena train recipe.yaml --out training-run
arena policy verify training-run/policy.arena
```

`data materialize` verifies every selected episode digest, copies episodes into a
self-contained directory, rewrites paths relative to that directory, and records
the parent dataset digest. The producer run may then be absent. Optional splits
use `sha256_bucket/v1`: the episode digest, occurrence index, and declared split
seed determine the assignment. The manifest records normalized weights and
observed counts, so the split is portable and content-addressed.

A recipe is explicit and content-addressed:

```yaml
schema: arena.train/v1
name: imitate-teacher
algorithm: behavior_cloning
dataset: portable-dataset/dataset.yaml
dataset_split: train
role: player_0
roles: [player_0, player_1]
seed: 23
epochs: 50
batch_size: 32
learning_rate: 0.01
observation: {type: Discrete, n: 4, dtype: int64}
action: {type: Discrete, n: 3, dtype: int64, masks: none}
architecture:
  type: mlp_categorical
  observation_dim: 4
  hidden_dims: [32, 32]
  action_n: 3
preprocessing: {id: normalize_v0, mean: 0.0, std: 1.0}
```

Before training, Arena rehashes every episode. The run record contains dataset and
recipe digests, seed, hyperparameters, example count, and the complete loss curve.
The output bundle embeds source-captured cases from the trained module and can be
verified or mirrored like any other policy.

Every training run also writes `checkpoint.json` and `checkpoint.pt`. To resume,
set `resume_from` to the earlier run directory and increase `epochs`. Arena verifies
the payload digest, loads tensor-only state with `weights_only=True`, and requires
an exact training-contract digest match before restoring model, optimizer, loss
history, and NumPy sampler state. The contract includes algorithm/config,
dataset/split, role, spaces, architecture, preprocessing, seed, batch size,
learning rate, and weighting.

The built-in cases support Discrete actions and Discrete/Box observations:

- `behavior_cloning` gives each transition uniform weight.
- `return_weighted_regression` weights transitions by their episode return and
  accepts positive `temperature` and `max_weight` settings in `algorithm_config`.

Arena refuses unknown algorithms rather than routing them through a similar case.
A new trainer implements `TrainingCase`, registers in `TRAINERS`, and supplies
distinct-objective, reproducibility, interruption, mutation, and policy-conformance
fixtures. Online/distributed RL and large sharded datasets remain outside the
built-in cases. A library spike for stream-read and sharded materialize lives
under `arena.dataset` and is documented in
`rfcs/012-streaming-sharded-datasets.md`; it does not change the default
atomic flat `data materialize` path.

## Mini lab loop (CartPole)

For a single-machine usefulness check that does not need an external trainer
checkout, run the CartPole collect → behavior-cloning → verify → match path:

```bash
python -m pip install -e '.[torch,pettingzoo]'
python examples/1.0/mini_train_cartpole.py --out ./arena-mini-train
arena policy verify ./arena-mini-train/train-run/policy.arena
```

The script rolls out a heuristic(+ε) teacher on Gymnasium `CartPole-v1`,
materializes an Arena trajectory dataset, trains with the built-in
`behavior_cloning` case, verifies the portable policy, runs a short seeded
Gymnasium eval, and (when PettingZoo is installed) a Match through the digest-
pinned `entrypoint_bundle` wrapper in `examples/1.0/cartpole_parallel.py`.
Digests and lineage land in `arena-mini-train/result.json`.

