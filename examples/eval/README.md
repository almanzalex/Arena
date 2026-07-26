# Example evaluation suite (Arena 0.2)

## Runnable demo (use this)

Checked-in cyclic RPS population + suite (no placeholders):

```bash
# from repo root, with 'arena[torch,pettingzoo]' installed
bash examples/eval/run_demo.sh
```

Or step-by-step from `examples/eval/demo/` following [eval-clean-room.md](../../docs/eval-clean-room.md).

Regenerate bundles (overwrites `demo/`):

```bash
python examples/eval/generate_demo.py --force
```

## Skeleton YAMLs

`population.yaml` / `evaluation.yaml` in this directory are **authoring templates**
with `REPLACE_*` digests. Prefer the `demo/` pack for a first run.

## Replacement for hand-rolled cross-play (U-02)

`crossplay_script.py` is **retired**. Use `run_demo.sh` or:

```bash
arena init
arena population create ./population.yaml --ref populations/rps-opp
arena eval run ./evaluation.yaml \
  --policy <digest>=./candidate.arena \
  --population ./population.yaml \
  --out ./eval-runs/crossplay
arena eval report ./eval-runs/crossplay --json
```

Docs: [populations](../../docs/populations.md), [evaluation](../../docs/evaluation.md),
[eval clean-room](../../docs/eval-clean-room.md), [eval usability sign-off](../../docs/eval-usability-signoff.md),
[0.2 revisit list](../../docs/0.2-revisit.md).
