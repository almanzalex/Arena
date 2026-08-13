# Arena release evidence (stream D)

This directory holds **collector output**, **gate templates**, and (later)
attached qualification files for R-01…R-14.

## Important

- `release-index.json` produced by the collector is an
  `arena.release-evidence-skeleton/v1` document — **not** a signed
  `arena.release-evidence/v1` index.
- Live Hugging Face, separately deployed OpenEnv, and isolated Gimitest are
  **never** marked passed by local collection alone.
- Do not commit generated `local/` JSON or a completed signed index into the
  release commit that the index binds (self-reference hazard).

## Collect local proof

```bash
python scripts/collect_release_evidence.py
```

## Attach external evidence

```bash
python scripts/collect_release_evidence.py \
  --attach R-01=/path/to/ci-summary.json \
  --attach R-04=docs/qualifications/hf-live.json \
  --attach R-05=docs/qualifications/openenv/R-05-openenv-separate-service.json \
  --attach R-06=docs/qualifications/gimitest/R-06-gimitest.json
```

See [docs/releasing.md](../docs/releasing.md) and
[docs/qualifications/README.md](../docs/qualifications/README.md).

## Templates

Starter JSON shapes live in [`templates/`](templates/). Copy, fill with real
results, then `--attach` them. Simulation / loopback-only reports are rejected
for the external floor gates. Leave every `REPLACE_*` field unsubstituted until
the measurement exists — `scripts/rehearse_release_sign.sh` treats placeholders
as blockers, not passes.

## Signing rehearsal

```bash
bash scripts/rehearse_release_sign.sh
```

Fails loudly listing missing gates until all fourteen filled evidence files exist.
Ephemeral keys stay under `/tmp` or gitignored `evidence/local/` only.
