# Evaluation providers

`provider` is a registry axis. `native` remains the default and imports no external
provider. Unknown names fail with the interface, registration function, test duties,
and qualification command.

## Gimitest

```yaml
provider: gimitest
provider_config:
  suite: base-hooks
  test_class: gimitest.gtest:GTest
  parameters: {purpose: robustness}
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

Every eval run and result cell records the exact policy digests, task digest, provider
version, and content digest of `provider_config`. External test classes execute Python
and are refused unless `allow_external_test_class: true` is explicitly set. Ordinary
cross-play remains `provider: native` and does not import Gimitest.
