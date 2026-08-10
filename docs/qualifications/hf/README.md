# Hugging Face live store qualification (1.0 floor)

Hugging Face remains **preview** until a credentialed immutable-revision
push/pull produces `mode=live` evidence. Simulation and missing credentials
never satisfy that gate.

## Fail-closed qualify script

```bash
# Without HF_TOKEN / HUGGING_FACE_HUB_TOKEN → exit 1, mode=credential-missing
python scripts/qualify_hf_live.py examples/eval/demo/rock.arena \
  'hf://models/ORG/REPO/arena' \
  --report /tmp/hf-qualification.json
```

Expected without credentials:

| Field | Value |
|---|---|
| `schema` | `arena.store-qualification/v1` |
| `backend` | `hf` |
| `mode` | `credential-missing` |
| `ok` | `false` |
| `stable_claim_allowed` | `false` |
| exit code | non-zero |

`?simulate=` is refused by this script. Use `arena store qualify …?simulate=/abs`
only for local rehearsal; that report is labeled `simulation` and is not live
evidence.

## Live recipe (credentials required)

```bash
pip install 'arena[hf]'
export HF_TOKEN=<write-token>          # or HUGGING_FACE_HUB_TOKEN
export ARENA_HF_LIVE_DEST='hf://models/ORG/REPO/arena'
python scripts/qualify_hf_live.py examples/eval/demo/rock.arena \
  "$ARENA_HF_LIVE_DEST" \
  --out /tmp/hf-restored.arena \
  --report /tmp/hf-qualification.json
```

A passing live report must include:

- `mode: live`
- `ok: true`
- `checks.immutable_revision.ok: true` (40-hex commit on the returned URI)
- `stable_claim_allowed: true` only after those checks pass

Then attach the JSON to release evidence (R-04). Do **not** edit
`arena/support-matrix.json` to `stable` in the same change as a missing-token
run — flip only when live evidence exists and is bound to the release index.

## CI

Default CI has no Hugging Face secrets. The fail-closed path is always
exercised; optional live tests use `@pytest.mark.requires_hf` and skip with this
recipe when tokens are absent. Skipping is not a live pass.

## Related

- Adapter: `arena/core/store_hf.py`
- Boundary smoke (all credentialed stores): `examples/boundaries/live_store_smoke.py`
- Release-gate checklist: [release-gate.md](release-gate.md)
- Store overview: [../../external-stores.md](../../external-stores.md)
