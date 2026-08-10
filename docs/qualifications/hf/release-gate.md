# HF release-gate checklist (R-04)

Use this checklist before claiming Hugging Face as a 1.0 **stable** store.

## Must be true

- [ ] `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` was present for the qualifying run
- [ ] Qualifier was `scripts/qualify_hf_live.py` (or equivalent calling `qualify_hf_live`)
- [ ] Report `mode` is exactly `live` (not `simulation`, not `credential-missing`)
- [ ] Report `ok` is `true`
- [ ] Returned URI includes immutable `?revision=<40-hex>` and `#sha256:…`
- [ ] `checks.immutable_revision.ok` is `true`
- [ ] `checks.identity_preserved.ok` is `true`
- [ ] Evidence JSON is content-bound into the release index under R-04
- [ ] `arena/support-matrix.json` `hf.status` flipped to `stable` **only after** the above

## Explicit non-evidence

| Artifact | Counts as live stable evidence? |
|---|---|
| `mode=credential-missing` report | No |
| `mode=simulation` / `?simulate=` qualify | No |
| Unit tests with FakeApi / monkeypatch | No |
| Doctor capability probe | No (doctor does not authenticate) |
| CI skip of `@pytest.mark.requires_hf` | No |

## Fail-closed verify (no token)

```bash
env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN \
  python scripts/qualify_hf_live.py examples/eval/demo/rock.arena \
  'hf://models/ORG/REPO/arena' --report /tmp/hf-missing.json
# expect: exit != 0, mode=credential-missing, ok=false
```
