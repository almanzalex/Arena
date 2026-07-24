# RLX 1.0 release evidence

The release is not “green because CI said so.” Each mandatory gate produces a
durable file, and the exact wheel/sdist plus those files are content-bound into
one signed index.

The release-candidate workflow builds the distributions once, installs the
exact wheel, emits SHA-256 checksums, a reproducible CycloneDX SBOM, dependency,
Bandit, and secret-scan reports, and GitHub/Sigstore provenance plus SBOM
attestations over the wheel and sdist. Those repository-bound attestations
complement the RLX release-manager signature below; neither substitutes for the
other's trust channel.

```bash
rlx release assemble \
  --release 1.0.0 --tag v1.0.0 --commit "$(git rev-parse HEAD)" \
  --gate R-01=evidence/R-01-platform-ci.json \
  --gate R-02=evidence/R-02-hermetic.json \
  --gate R-03=evidence/R-03-human-handoff.json \
  --gate R-04=evidence/R-04-stores.json \
  --gate R-05=evidence/R-05-openenv.json \
  --gate R-06=evidence/R-06-gimitest.json \
  --gate R-07=evidence/R-07-adversarial.json \
  --gate R-08=evidence/R-08-security.json \
  --gate R-09=evidence/R-09-compatibility.json \
  --gate R-10=evidence/R-10-performance.json \
  --gate R-11=evidence/R-11-public-distribution.json \
  --gate R-12=evidence/R-12-recovery.json \
  --gate R-13=evidence/R-13-docs.json \
  --gate R-14=evidence/R-14-integrity.json \
  --artifact dist/rlx-1.0.0-py3-none-any.whl \
  --artifact dist/rlx-1.0.0.tar.gz \
  --out evidence/release-index.json

rlx attest keygen --private release-private.pem --public release-public.pem
rlx release sign evidence/release-index.json \
  --key release-private.pem --out evidence/release-index.sig.json
rlx release verify evidence/release-index.json \
  --signature evidence/release-index.sig.json \
  --key release-public.pem --at-release
```

`assemble` refuses missing R-01…R-14, non-pass gates, short commit IDs, malformed
tags, missing artifacts, and overwrite. `verify` rehashes every local artifact
and checks the detached Ed25519 signature against an independently supplied
public key. When gate evidence paths are available locally, `verify` also
rehashes every R-01…R-14 evidence file; the signature cannot turn missing or
modified local proof into a passing verification.

Current qualification is a separate signed ledger so an outage or expired
provider record can change today’s verdict without rewriting release history:

```bash
rlx release sign evidence/current-ledger.json \
  --key operations-private.pem --out evidence/current-ledger.sig.json
rlx release verify evidence/release-index.json \
  --signature evidence/release-index.sig.json --key release-public.pem \
  --current-ledger evidence/current-ledger.json \
  --ledger-signature evidence/current-ledger.sig.json \
  --ledger-key operations-public.pem
```

Records require one unique, non-empty capability and a `pass`, `stale`, or
`failed` status. Any `stale` or `failed` record makes current verification fail.
This repository does not contain production signing keys or credentials.
