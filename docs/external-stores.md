# External artifact mirrors

Stores mirror bytes; they do not create a new RLX artifact identity. A directory mirror
stores a small `rlx.mirror/v1` descriptor plus every file under its SHA-256 object key.
`pull --verify` rehashes each blob and reloads policies to prove their original identity.

## Filesystem reference backend

```bash
uri=$(rlx push examples/eval/demo/rock.rlx file:///absolute/mirror --verify --json)
rlx pull 'file:///absolute/mirror#sha256:…' --out restored-rock.rlx --verify
```

The URI printed by `push` includes the immutable `#sha256:…` identity. If `--out` is
omitted, pull uses `pulled-<digest-prefix>.rlx`. Existing non-empty outputs are refused.

## Hugging Face Hub

```bash
pip install 'rlx[hf]'
hf auth login                         # normal backend credential flow
rlx push policy.rlx hf://models/org/repo/rlx --verify
rlx pull 'hf://models/org/repo/rlx#sha256:…' --verify
```

Use `datasets` or `spaces` instead of `models` for another repo type, and
`?revision=branch` when needed. RLX never stores tokens or remaps digests. The HF API
boundary is covered with a deterministic fake-backend round trip; a live authenticated
smoke is user/account-specific. OCI, W&B, and MLflow are follow-on adapters, not 0.3 claims.
