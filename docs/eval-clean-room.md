# Eval clean-room checklist (U-02 companion)

Use after [clean-room.md](clean-room.md) when validating **evaluation** handoff (populations, suite digests, reports, bundles).

## Receive

1. Population YAML or digest ref
2. Evaluation suite YAML
3. Policy bundles for every digest in the suite/population
4. Optional: prior `eval_run` / `report` / `eval bundle`

## Commands

The fenced block below is the canonical eval clean-room flow. It is tagged
`rlx-eval-clean-room` so the hermetic gate can parse and execute these commands.
Friendly `--policy rock=./rock.rlx` names are rewritten to digests by the gate;
in a real lab, pass `sha256:…=./rock.rlx` (or generate digests via `rlx inspect`).

```bash rlx-eval-clean-room
rlx init
rlx population create ./population.yaml --ref populations/opp
rlx eval validate ./evaluation.yaml --population ./population.yaml
rlx eval run ./evaluation.yaml \
  --policy rock=./rock.rlx --policy paper=./paper.rlx --policy scissors=./scissors.rlx \
  --population ./population.yaml --out ./eval-run
rlx eval report ./eval-run --out ./eval-run
rlx data select ./eval-run --out ./slice --outcome loss --role player_0
rlx eval bundle ./eval-run --out ./bundle --report ./eval-run/report.json
```

## Pass criteria

- Suite digest run twice → identical sampling ledger + episode action streams (modulo timestamps).
- Cyclic RPS population → `nontransitivity_warning` present; matrix retained; `ranking` null.
- Bundle digests verify on a machine without the trainer repo.
- Retired hand-written `examples/eval/crossplay_script.py` exits non-zero and points here.

## Automated gate

```bash
pytest tests/acceptance/test_eval_hermetic.py -q          # doc drift (fast)
pytest -m slow tests/acceptance/test_eval_hermetic.py \
       tests/acceptance/test_u01_hermetic.py::test_u01_hermetic_venv -q
```

Author smoke (no wheel build):

```bash
bash examples/eval/run_demo.sh
```
