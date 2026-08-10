# Gimitest isolated-interpreter qualification (R-06)

Arena keeps Gimitest as a **preview** capability until claimed-platform release CI
repeats the isolated-worker proof against the exact wheel and binds the report.
Local qualification here is necessary but not sufficient for a stable matrix claim.

## Bootstrap a separate worker

```bash
scripts/bootstrap_gimitest_worker.sh
export ARENA_GIMITEST_PYTHON="$PWD/.venv-gimitest/bin/python"
# or:
eval "$(scripts/bootstrap_gimitest_worker.sh --export)"
```

The bootstrap script:

1. Creates a fresh venv (default `.venv-gimitest/`)
2. Installs `arena[torch,pettingzoo,gimitest]` under `constraints/gimitest-worker.txt`
3. Installs `gimitest==1.0` with `--no-deps` (Gimitest's published Gymnasium pin is obsolete)
4. Prints `ARENA_GIMITEST_PYTHON`

Doctor without the env var must report `locally-unqualified`. With a ready worker:

```bash
arena doctor --capability gimitest --json
```

## Emit R-06 evidence

```bash
python scripts/qualify_gimitest_isolated.py \
  --out docs/qualifications/gimitest/R-06-gimitest.json
```

The script refuses to run when `ARENA_GIMITEST_PYTHON` is unset, non-absolute, or
identical to the parent interpreter. It proves:

- semantic-noop Gimitest matches native evaluation-intent and semantic-result digests
  while recording a distinct execution-binding digest
- non-no-op `RewardTransformScenario` changes **both** intent and result digests
- subprocess worker lineage is recorded on the provider envelope

`stable_claim` in the evidence JSON is always `false` for this local path.

## Release CI binding

Copy or regenerate the evidence as `evidence/R-06-gimitest.json` when assembling
the release index:

```bash
arena release assemble \
  … \
  --gate R-06=evidence/R-06-gimitest.json \
  …
```

See also `docs/eval-providers.md` and `examples/1.0/run_local_boundaries.py`.
