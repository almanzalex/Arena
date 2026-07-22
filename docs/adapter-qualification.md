# Adapter qualification (release gate)

An adapter may be described as supported only after it has a passing
machine-readable qualification report. The command is intentionally a release
gate, not a marketing probe:

```bash
rlx adapter qualify ./match.yaml --out ./qualification.json
```

The fixture is an ordinary match manifest whose assignments point at the
received policy bundles. The report records:

1. source-captured policy conformance (self-consistency is refused);
2. repeated seeded-match action-stream reproducibility;
3. malformed-contract rejection;
4. payload tamper detection;
5. trajectory run/bundle provenance; and
6. the required offline built-wheel handoff gate.

The last item is fulfilled by `pytest -m slow -q`, specifically the U-01
fresh-venv test. It builds the wheel, installs it from a local wheelhouse in a
new environment, copies only bundles/match/guide, and executes the documented
commands with the network disabled. Docker adds the stronger
`--network none` variant when available.

## 0.1 qualification record

The release fixture is the bundled `rlx/competitive_rps_v0` PettingZoo Parallel
task with two `custom-pytorch` fixed categorical policies. The fixture must be
exported with source-captured reference cases before running the command. Its
report is release evidence, not a permanent repository artifact (absolute
temporary paths and timestamps are intentionally included).

For a representative image stack (CleanRL / PettingZoo Pistonball), the task
manifest must declare the SuperSuit wrapper chain and `observation_layout`, and
the policy must be exported via the BYO TorchScript `--module` path (not the
template categorical exporter). See [policy-export.md](policy-export.md).

## 0.2 qualification (populations + evaluation)

`rlx adapter qualify` accepts either a match fixture (0.1 gates) or an
`rlx.evaluation/v0alpha1` suite. Evaluation fixtures must:

1. reference a population YAML with local policy bundle paths;
2. embed source-captured policy evidence;
3. reproduce sampling ledger + action streams across two runs;
4. emit report `evidence_refs`; and
5. when cyclic, keep `nontransitivity_warning` with `ranking: null`.

Companion hermetic gate: `pytest -m slow tests/acceptance/test_eval_hermetic.py`.

See [evaluation.md](evaluation.md) and [eval-clean-room.md](eval-clean-room.md).

## Registry cases and support claims

Portable behavior is an **axes + case registry** (`rlx.core.registry` /
`rlx.plugins`). Capability matrix rows may only name kinds that are both
registered and covered by a passing `rlx adapter qualify` fixture. Unknown
kinds raise `UnknownKindError` with an extension recipe (interface to
implement, tests to add, qualify required). Do not claim support from a
developer editable install alone.

### Adapter author checklist

- State the exact portable tensor/action contract, including image layout,
  preprocessing, masks, recurrence reset behavior, and RNG.
- Provide source-captured cases that exercise every declared path.
- Supply a seeded fixture match that records trajectories.
- Run the qualification command and preserve its JSON with the release.
- Run the clean-room wheel and Docker gates; do not call an adapter supported
  based on a developer editable install.
- Document every rejected input with an actionable repair and an RFC contract
  needed before it can become supported.
