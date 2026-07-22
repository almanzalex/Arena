# Eval usability sign-off (U-02)

Companion to [usability-signoff.md](usability-signoff.md) for **population + evaluation**
handoff. Give the reader only:

- built RLX wheel (or `pip install 'rlx[torch,pettingzoo]'`)
- `examples/eval/demo/` (checked-in bundles + YAMLs) **or** an equivalent received pack
- [eval-clean-room.md](eval-clean-room.md)

Do **not** give them the trainer repo, this monorepo (except the demo pack), or author help.

## One-command smoke (author machine)

From a checkout with deps installed:

```bash
bash examples/eval/run_demo.sh
```

## Record (second-lab reader)

- Reader / machine / OS / Python:
- Wheel filename and hashes (or install method):
- Demo pack path and `sha256` of `population.yaml` / `evaluation.yaml` / each `*/DIGEST`:
- Start time (UTC):
- End time (UTC):
- Commands attempted (must match [eval-clean-room.md](eval-clean-room.md)), with exit codes:
- Did `population create`, `eval run`, `eval report`, `data select`, and `eval bundle` succeed? Yes / no:
- Was `nontransitivity_warning` present with `ranking: null`? Yes / no:
- Friction score (1 effortless — 5 blocked):
- Guide/CLI changes requested:

## Pass rule

Pass only if the reader finishes without source access or author intervention.
Automated hermetic (`pytest -m slow tests/acceptance/test_eval_hermetic.py`) proves
isolation + command parse; it does **not** replace this human record.
