# Evaluation providers

`provider` is a registry axis. `native` remains the default and imports no external
provider. Unknown names fail with the interface, registration function, test duties,
and qualification command.

## Gimitest

```yaml
provider: gimitest
provider_config:
  test_class: rlx.adapters.eval_gimitest.scenarios:RewardTransformScenario
  parameters: {reward_scale: -1.0}
```

Run and qualify the checked example:

```bash
# Gimitest 1.0 metadata pins Gymnasium 0.29.1, incompatible with current PettingZoo.
# Keep RLX's current Gymnasium, then install only the provider package itself.
pip install 'rlx[torch,pettingzoo]'
pip install --no-deps gimitest==1.0
rlx eval run examples/eval/robustness.yaml --out robustness-run
rlx adapter qualify examples/eval/robustness.yaml --out gimitest-qualification.json
```

The checked scenario is intentionally non-no-op: it transforms rewards, and the
semantic result digest must change. For a provider swap that should preserve
semantics, declare `provider_config.semantic: {}` and use
`gimitest.gtest:GTest`; the evaluation-intent and semantic-result digests then
match native while the execution-binding digest differs.

Every eval run and result cell records the exact policy digests, task digest, provider
version, worker Python/RLX versions, and content digest of `provider_config`. External test classes execute Python
and are refused unless `allow_external_test_class: true` is explicitly set. Ordinary
cross-play remains `provider: native` and does not import Gimitest.

When Gimitest's dependency set should not share the RLX interpreter, run the provider
through an explicitly selected Python:

```bash
python -m venv .venv-gimitest
.venv-gimitest/bin/python -m pip install 'rlx[torch,pettingzoo]==1.0.0rc1'
.venv-gimitest/bin/python -m pip install --no-deps gimitest==1.0
export RLX_GIMITEST_PYTHON="$PWD/.venv-gimitest/bin/python"
rlx doctor --capability gimitest
```

Install RLX's native stack first and Gimitest with `--no-deps`: Gimitest 1.0
publishes an obsolete Gymnasium pin. Doctor executes a bounded metadata-only
probe of the explicitly configured interpreter and requires matching RLX,
Gimitest, Torch, and PettingZoo distributions before reporting it ready.

```yaml
provider_config:
  suite: base-hooks
  test_class: gimitest.gtest:GTest
  isolation:
    mode: subprocess
    python: /absolute/path/to/gimitest-venv/bin/python
    timeout_seconds: 60
```

The parent and worker exchange a content-bound `rlx.eval-provider/v1`
request/result. Wall-time and stdout/stderr budgets supervise the whole process
group; request IDs and digests prevent stale responses. The path must be
absolute. This isolates dependency resolution and process failure; it is not an
arbitrary-code security sandbox.
