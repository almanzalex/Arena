# Documentation index

Arena’s detailed docs are intentionally long: they record product boundaries,
qualification evidence, and milestone claims. Use this page as a map.

## Start here (1.0)

| Doc | Purpose |
|---|---|
| [1.0-user-flows.md](1.0-user-flows.md) | Executable value flows |
| [1.0-readiness.md](1.0-readiness.md) | Release program; what blocks `v1.0.0` |
| [1.0-rc-local-evidence.md](1.0-rc-local-evidence.md) | Latest local RC proof |
| [1.0-test-plan.md](1.0-test-plan.md) | Test plan for 1.0 gates |
| [releasing.md](releasing.md) | Signed release procedure |
| [pypi-trusted-publishing.md](pypi-trusted-publishing.md) | R-11 TestPyPI dry-run + GitHub OIDC Trusted Publisher setup |
| [errors.md](errors.md) | Stable diagnostics / exit classes |

## Everyday guides

| Doc | Purpose |
|---|---|
| [clean-room.md](clean-room.md) | Second-machine policy handoff |
| [seed-determinism.md](seed-determinism.md) | Seed protocol + where nondeterminism is expected |
| [policy-export.md](policy-export.md) | Export / verify policies |
| [populations.md](populations.md) | Policy populations |
| [evaluation.md](evaluation.md) | Evaluation suites |
| [eval-clean-room.md](eval-clean-room.md) | Eval handoff checklist |
| [eval-providers.md](eval-providers.md) | Native / Gimitest providers |
| [external-tasks.md](external-tasks.md) | OpenEnv / OpenSpiel |
| [external-stores.md](external-stores.md) | `file://`, HF, OCI, W&B, MLflow |
| [training.md](training.md) | Offline trainer recipes |
| [dynamic-agents.md](dynamic-agents.md) | Dynamic AEC lifecycle |
| [adapter-qualification.md](adapter-qualification.md) | Qualify before claiming support |
| [integration-authoring.md](integration-authoring.md) | Authoring new integrations |
| [overhead-budget.md](overhead-budget.md) | Performance budgets |

## Usability / sign-off

| Doc | Purpose |
|---|---|
| [usability-signoff.md](usability-signoff.md) | Human clean-room record (0.1) |
| [eval-usability-signoff.md](eval-usability-signoff.md) | Human eval handoff record |
| [integration-usability-signoff.md](integration-usability-signoff.md) | Integration author record |

## Milestone archive

These documents are the sealed claim trail. Prefer them when auditing what was
promised at each stage; prefer the 1.0 docs above for current product truth.

| Stage | Complete / boundaries | Evidence |
|---|---|---|
| 0.2 | [0.2-complete.md](0.2-complete.md), [0.2-revisit.md](0.2-revisit.md) | — |
| 0.3 | [0.3-complete.md](0.3-complete.md), [0.3-delivery.md](0.3-delivery.md) | [0.3-evidence.md](0.3-evidence.md) |
| 0.4 | [0.4-boundaries.md](0.4-boundaries.md) | [0.4-evidence.md](0.4-evidence.md) |
| 0.5 | [0.5-boundaries.md](0.5-boundaries.md) | [0.5-evidence.md](0.5-evidence.md) |

RFCs live in [`../rfcs/`](../rfcs/).

## Examples

| Path | Purpose |
|---|---|
| `examples/eval/demo/` | Cyclic RPS population + eval pack |
| `examples/eval/run_demo.sh` | One-command 0.2-style eval journey |
| `examples/1.0/` | Local 1.0 boundary / CleanRL producer scripts |
| `examples/byo/` | Self-contained BYO TorchScript export (no CleanRL checkout) |
| `examples/boundaries/run_demo.sh` | Composed lifecycle / store / attest journey |
| `examples/plugins/` | Out-of-tree plugin wheel example |
