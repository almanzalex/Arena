# Arena 1.0 release evidence

The release is not “green because CI said so.” Each mandatory gate produces a
durable file, and the exact wheel/sdist plus those files are content-bound into
one signed index.

## Local collector (skeleton only)

Automate as much **local** R-01…R-14 inventory as possible without inventing
live HF / separately deployed OpenEnv / release-CI Gimitest:

```bash
python scripts/collect_release_evidence.py
```

This writes:

- `evidence/local/` — doctor, schema-registry snapshot, golden fixture digests,
  hermetic-capable inventory, perf-smoke baselines, release-truth, and related
  inventories;
- `evidence/release-index.json` — an `arena.release-evidence-skeleton/v1`
  document with every R-01…R-14 slot marked `filled`, `local-partial`, or
  `missing`.

The skeleton is **not** `arena.release-evidence/v1` and must not be signed as a
release. External-floor gates stay `missing` until real files are attached.

### Attach CI / HF / OpenEnv / Gimitest evidence

Copy a template from `evidence/templates/`, fill it with real results (or drop
qualification JSON under `docs/qualifications/` — see
[qualifications/README.md](qualifications/README.md)), then:

```bash
python scripts/collect_release_evidence.py \
  --attach R-01=/path/to/ci-summary.json \
  --attach R-04=docs/qualifications/hf-live.json \
  --attach R-05=docs/qualifications/openenv/R-05-openenv-separate-service.json \
  --attach R-06=docs/qualifications/gimitest/R-06-gimitest-isolated.json
```

The collector refuses simulated store reports (`mode=simulation` / `?simulate=`)
and loopback OpenEnv attachments that lack `separately_deployed: true`.

| Gate | What to attach |
|---|---|
| R-01 | Clean-checkout CI summary for the exact commit (Linux/macOS × 3.12/3.13) |
| R-02 | Exact-wheel hermetic + Docker network-none report |
| R-03 | Two non-author handoff transcripts (content-digested) |
| R-04 | Live HF `arena.store-qualification/v1` (+ optional other stores or preview labels) |
| R-05 | Separately deployed OpenEnv qualification + failure drills |
| R-06 | Isolated-interpreter non-no-op Gimitest qualification |
| R-07…R-13 | Soak, SBOM/attestations, golden verify, perf baselines, recovery, docs truth |
| R-14 | Result of signed `arena release verify` after assemble |

Stream D owns `scripts/r_gates/**`, `evidence/templates/`, and this collector.
Only stream D may later propose `support-matrix.json` `evidence` field updates
after other streams emit qualification JSON.

## Signed assemble (final release asset)

The release-candidate workflow builds the distributions once, installs the
exact wheel, emits SHA-256 checksums, a reproducible CycloneDX SBOM, dependency,
Bandit, and secret-scan reports, and GitHub/Sigstore provenance plus SBOM
attestations over the wheel and sdist. Those repository-bound attestations
complement the Arena release-manager signature below; neither substitutes for the
other's trust channel.

```bash
arena release assemble \
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
  --artifact dist/arena-1.0.0-py3-none-any.whl \
  --artifact dist/arena-1.0.0.tar.gz \
  --out evidence/release-index.json

arena attest keygen --private release-private.pem --public release-public.pem
arena release sign evidence/release-index.json \
  --key release-private.pem --out evidence/release-index.sig.json
arena release verify evidence/release-index.json \
  --signature evidence/release-index.sig.json \
  --key release-public.pem --at-release
```

`assemble` refuses missing R-01…R-14, non-pass gates, short commit IDs, malformed
tags, missing artifacts, and overwrite. `verify` rehashes every local artifact
and checks the detached Ed25519 signature against an independently supplied
public key. When gate evidence paths are available locally, `verify` also
rehashes every R-01…R-14 evidence file; the signature cannot turn missing or
modified local proof into a passing verification.

## Lab artifact attestation (user-owned keys)

Release-index signing above is for the published distribution. Labs that only need
to prove *this artifact identity was signed by a key we trust* use
`arena attest` with their own Ed25519 PEM files. No certificate authority, no
Sigstore account, and no PyPI publisher identity is required—trust is whoever
you choose to distribute the public key to.

Install the optional crypto extra once, then:

```bash
pip install 'arena[signing]'

arena attest keygen --private lab-private.pem --public lab-public.pem
arena attest sign policy.arena \
  --key lab-private.pem --issuer example-lab --out policy.attestation.json
arena attest verify policy.arena policy.attestation.json --key lab-public.pem
```

`keygen` writes an unencrypted PKCS8 private PEM (`0600`) and SPKI public PEM
(`0644`) and refuses overwrite. `sign` binds the artifact's content-addressed
identity into a detached `arena.attestation/v1` JSON. `verify` rehashes the
local path, checks `key_id` against the supplied public key, and verifies the
Ed25519 signature over the canonical statement. Byte-tampered artifacts and
altered signature blobs fail closed.

Detached attestations survive identity-preserving `arena push` / `arena pull`
mirrors; authenticity is independent of where the bytes are stored. Arena does
not invent accounts, CAs, revocation, or transparency logs—operators keep and
distribute their own keys.

Current qualification is a separate signed ledger so an outage or expired
provider record can change today’s verdict without rewriting release history:

```bash
arena release sign evidence/current-ledger.json \
  --key operations-private.pem --out evidence/current-ledger.sig.json
arena release verify evidence/release-index.json \
  --signature evidence/release-index.sig.json --key release-public.pem \
  --current-ledger evidence/current-ledger.json \
  --ledger-signature evidence/current-ledger.sig.json \
  --ledger-key operations-public.pem
```

Records require one unique, non-empty capability and a `pass`, `stale`, or
`failed` status. Any `stale` or `failed` record makes current verification fail.
This repository does not contain production signing keys or credentials.

## R-11 public distribution rehearsal

Before any TestPyPI or PyPI upload, rehearse packaging without secrets:

```bash
bash scripts/pypi_dry_run.sh
```

That builds the sdist and wheel and runs `twine check --strict`. It never uploads
and never reads publish credentials. The manual GitHub Actions workflow
`pypi-dry-run.yml` runs the same check. Trusted Publisher (GitHub OIDC) setup
steps—registration fields, environments, and what a future upload job would
need—are documented in [pypi-trusted-publishing.md](pypi-trusted-publishing.md).
Do not add API tokens for the dry-run path.

## R-14 signing rehearsal (ephemeral keys)

Before every gate file exists, rehearse the assemble → sign → verify path without
inventing passes:

```bash
bash scripts/rehearse_release_sign.sh
```

By default the script:

- writes ephemeral Ed25519 keys only under `/tmp` (or gitignored
  `evidence/local/` if you set `KEY_DIR`);
- inventories `evidence/R-01-…R-14-*.json`;
- **fails loudly** listing every missing, template/`REPLACE_*`, or non-`pass`
  gate instead of fabricating assemble inputs;
- only when all fourteen real gate files are ready, runs `arena release assemble`,
  `sign`, and `verify` against artifacts under `dist/` (from the PyPI dry-run).

Do not commit rehearsal private keys or a self-bound `arena.release-evidence/v1`
index from a half-filled tree. The collector skeleton remains
`arena.release-evidence-skeleton/v1` and is not a substitute for this path.
