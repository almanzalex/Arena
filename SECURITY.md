# Security policy

## Supported versions

Security fixes are accepted against the current release candidate / main branch
(`1.0.0rc1` and forward). Older pre-1.0 milestone tags are historical.

## What Arena guarantees (and does not)

Arena provides **integrity, consent, and lineage** for RL artifacts:

- Content-addressed payloads and verify-before-execute paths.
- Explicit trust flags (`--trust-source`, `--trust-task-code`) that are **not**
  OS sandboxes.
- Optional detached user-owned signatures (`arena attest`).

Arena does **not** claim to safely execute untrusted third-party Python, Torch
modules, OpenEnv services, or provider workers. Treat policy bundles and
integration endpoints like any other code execution surface.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security-sensitive reports.

Email or privately message the repository owner listed on
[github.com/almanzalex/Arena](https://github.com/almanzalex/Arena) with:

1. A short description of the issue and impact.
2. Steps to reproduce (minimal fixture if possible).
3. Whether a fix or workaround is already known.

We will acknowledge receipt and coordinate disclosure. For dependency
vulnerabilities in optional extras, include the exact `pip freeze` of the
environment under test.

## Preferred hardening PRs

- Fail-closed digest / signature verification bugs.
- Path traversal or partial-publish races in store/demo writers.
- Cases where simulated store evidence could be mistaken for live qualification.
