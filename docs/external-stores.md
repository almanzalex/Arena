# External artifact mirrors

Stores mirror bytes; they do not create a new RLX artifact identity. A directory mirror
stores a small `rlx.mirror/v1` descriptor plus every file under its SHA-256 object key.
Every pull rehashes each known-hash blob and validates artifact identity;
`pull --verify` additionally requests the backend's explicit verification flow.

## Filesystem reference backend

```bash
uri=$(rlx push examples/eval/demo/rock.rlx file:///absolute/mirror --verify --json)
rlx pull 'file:///absolute/mirror#sha256:…' --out restored-rock.rlx --verify
```

The URI printed by `push` includes the immutable `#sha256:…` identity. If `--out` is
omitted, pull uses `pulled-<digest-prefix>.rlx`. Existing non-empty outputs are refused.

## Credentialed backends

```bash
pip install 'rlx[hf]'
hf auth login                         # normal backend credential flow
rlx push policy.rlx hf://models/org/repo/rlx --verify
rlx pull 'hf://models/org/repo/rlx#sha256:…' --verify
```

Use `datasets` or `spaces` instead of `models` for another repo type, and
`?revision=branch` when needed. RLX resolves a movable ref exactly once, fetches
the descriptor and all blobs from that immutable 40-hex commit, and returns the
pinned revision in the artifact URI. RLX never stores tokens or remaps digests. The HF API
boundary is covered with deterministic simulation; a live authenticated smoke is
user/account-specific.

```bash
# OCI uses the ORAS CLI and `oras login`.
rlx push policy.rlx oci://registry.example/org/repo --verify

# W&B and MLflow use their normal SDK credentials/tracking configuration.
rlx push policy.rlx wandb://entity/project/artifact --verify
rlx push policy.rlx 'mlflow://experiment?tracking_uri=https%3A%2F%2Fmlflow.example' --verify
```

All credentialed schemes (`hf`, `oci`, `wandb`, `mlflow`) support an explicit local
simulation mode for CI and workflow rehearsal:

```bash
rlx push policy.rlx \
  'wandb://entity/project/artifact?simulate=/tmp/rlx-wandb' --verify
```

The returned URI retains `simulate=`, so it cannot be mistaken for remote evidence.
Use `examples/boundaries/live_store_smoke.py` after authenticating for an actual
push/pull verification; that script intentionally refuses simulation URIs.

OCI extraction rejects absolute/parent paths, links, special files, case and
Unicode-normalization collisions, excessive members, and expanded-byte bombs.
ORAS itself runs under the shared process supervisor.

## Comparable qualification evidence

```bash
rlx store qualify policy.rlx \
  'wandb://entity/project/artifact?simulate=/tmp/rlx-wandb' \
  --out qualification.json
```

The `rlx.store-qualification/v1` report records the backend, explicit
`simulation`/`live` mode, immutable returned URI, expected/restored identity, and
verified push/pull checks. A simulation report is never live-provider evidence.

## Detached authenticity

SHA-256 proves that bytes did not change; it does not identify who published them.
An optional user-owned Ed25519 signature covers the canonical artifact identity and
kind:

```bash
pip install 'rlx[signing]'
rlx attest keygen --private lab-private.pem --public lab-public.pem
rlx attest sign policy.rlx --key lab-private.pem \
  --issuer my-lab --out policy.attestation.json
rlx attest verify restored-policy.rlx policy.attestation.json \
  --key lab-public.pem
```

The private key is created with owner-only permissions and is never uploaded by RLX.
The verifier requires an explicit trusted public key. RLX does not provide a
certificate authority, revocation, transparency log, hardware-key manager, or malware
sandbox.
