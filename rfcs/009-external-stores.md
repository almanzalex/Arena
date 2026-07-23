# RFC 009 — External Artifact Stores

**Status:** Backends complete in 0.4; common qualification and authenticity added in 0.5
**Date:** 2026-07-21 (expanded 2026-07-22)
**Depends on:** RFC 000

## User promise

Mirror RLX objects to user-selected backends (Hugging Face Hub, OCI registries, W&B/MLflow artifacts, shared filesystems) **without** an RLX-hosted service. Digests do not change across push/pull.

## CLI

```text
rlx push <ref> hf://…|oci://…|file://…|wandb://…|mlflow://…
rlx pull <uri> --verify
```

## Requirements (ST-*)

| ID | Requirement |
|----|-------------|
| ST-01 | Store adapter registry; local `.rlx/` remains default and sufficient |
| ST-02 | Push uploads manifest + payloads by digest; pull restores byte-identical payloads |
| ST-03 | `--verify` recomputes digests and refuses mutation (I-02) |
| ST-04 | Auth is delegated to the backend’s normal credentials (HF token, docker login, etc.) |
| ST-05 | Offline / adapters disabled: push/pull fail loud; match/eval/population still work (I-03) |
| ST-06 | No silent remapping of `sha256:…` identities |

## Non-goals

- RLX accounts, billing, or public catalog hosting
- Replacing W&B/MLflow experiment tracking UX

## Exit evidence

Round-trip test: export policy → push → wipe local object → pull --verify → `policy verify` / match still green.

## Implemented backends

0.3 selects `file://` first and Hugging Face Hub `hf://` as the network backend. Both
store an `rlx.mirror/v1` descriptor plus digest-keyed bytes; verified pull rehashes every
file and reloads policies to confirm identity. HF delegates auth to normal Hub credentials.
0.4 adds OCI through the standard ORAS CLI, W&B Artifacts, and MLflow run
artifacts. They delegate authentication to their normal clients. Every credentialed
scheme also accepts an explicit `?simulate=/absolute/path` for deterministic local
workflow testing; simulation stays visibly present in the returned URI and is never
reported as a live remote operation.

0.5 adds one `rlx.store-qualification/v1` report for every registered scheme and
labels evidence `simulation` or `live`. Optional detached Ed25519 attestations bind a
user-owned issuer/key to the original artifact identity and remain valid across
verified mirrors. RLX deliberately does not own a certificate authority or revocation
service.
