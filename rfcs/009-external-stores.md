# RFC 009 — External Artifact Stores (RLX 0.3)

**Status:** Draft for 0.3  
**Date:** 2026-07-21  
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
