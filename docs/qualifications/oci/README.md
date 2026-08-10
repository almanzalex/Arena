# OCI store qualification (preview)

Status: **preview**. Do not claim stable until a live ORAS qualification report
is attached here as `live-qualification.json` with
`mode: live` and `counts_as_live_evidence: true`.

Simulation (`?simulate=`) never counts as live evidence.

## Prerequisites

1. Install Arena.
2. Install the [ORAS CLI](https://oras.land/) and ensure `oras` is on `PATH`.
3. Authenticate: `oras login <registry>`.

## Simulation qualify (CI / local rehearsal)

```bash
mkdir -p /tmp/arena-oci-sim
arena store qualify examples/eval/demo/rock.arena \
  "oci://registry.example/lab/arena?simulate=/tmp/arena-oci-sim" \
  --out docs/qualifications/oci/simulation-qualification.json
```

Expect `mode: "simulation"` and `counts_as_live_evidence: false`.

## Live qualify (required for stable)

```bash
arena doctor --capability oci
arena store qualify examples/eval/demo/rock.arena \
  'oci://REGISTRY/ORG/REPO' \
  --out docs/qualifications/oci/live-qualification.json
```

Refuse to promote while `live-qualification.json` is absent or reports
`counts_as_live_evidence: false`.

## Fail-loud without tooling/credentials

Live qualify without ORAS or registry auth must fail with
`STORE_CREDENTIALS_REQUIRED` (or an ORAS auth failure remapped to that code).
Appending `?simulate=/absolute/path` is the only offline rehearsal path.
