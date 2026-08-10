# PyPI Trusted Publishing rehearsal (R-11)

This document is the **dry-run** path for Arena gate **R-11 (public
distribution)**. It proves that the distribution builds and passes metadata
checks, and records how to configure GitHub OIDC Trusted Publishers—without
uploading to TestPyPI or PyPI, and without CI secrets.

| Goal | Status in this rehearsal |
|---|---|
| Build sdist + wheel | Done locally and in CI |
| `twine check --strict` | Done locally and in CI |
| Upload to TestPyPI / PyPI | **Out of scope** (do not run) |
| API tokens / `TWINE_*` secrets | **Not required** and must not be added for the dry-run |

## Local dry-run

From the repository root (Python 3.12+):

```bash
bash scripts/pypi_dry_run.sh
```

The script:

1. Installs only `build` and `twine` (no publish credentials).
2. Builds `dist/*.whl` and `dist/*.tar.gz` via `python -m build`.
3. Runs `twine check --strict` on those artifacts.
4. Exits without calling `twine upload` or any Trusted Publisher upload action.

Optional: set `PYPI_DRY_RUN_DIST` to write artifacts somewhere other than `dist/`.

## CI dry-run (no secrets)

The workflow [`.github/workflows/pypi-dry-run.yml`](../.github/workflows/pypi-dry-run.yml)
is **manual only** (`workflow_dispatch`). It:

- checks out the tree;
- installs `build` + `twine`;
- runs `scripts/pypi_dry_run.sh`;
- uploads `dist/` as a workflow artifact for inspection.

It does **not**:

- request `id-token: write` (so GitHub cannot mint an OIDC token for PyPI);
- call `pypa/gh-action-pypi-publish` or `twine upload`;
- declare or consume repository secrets.

Run it from the Actions tab → **PyPI dry-run (R-11 rehearsal)** → **Run workflow**.

## Configure Trusted Publishers (GitHub OIDC) — setup only

Trusted Publishing replaces long-lived PyPI API tokens with a short-lived OIDC
identity bound to a specific GitHub repository, workflow file, and environment.
Configure this **before** any real upload workflow is enabled.

### 1. Decide the publisher targets

| Index | Project | Purpose |
|---|---|---|
| TestPyPI (`https://test.pypi.org`) | `arena` (or a rehearsal project name if `arena` is taken) | First live upload rehearsal |
| PyPI (`https://pypi.org`) | `arena` | Production releases only after TestPyPI + clean install |

Arena’s distribution name is declared in [`pyproject.toml`](../pyproject.toml)
as `name = "arena"`.

### 2. Create / claim the project on TestPyPI

1. Sign in at [https://test.pypi.org](https://test.pypi.org).
2. Create the project (or claim ownership of an existing empty project) named
   to match the distribution you will publish.
3. Open **Publishing** → **Add a new publisher** → **GitHub**.

### 3. Register the GitHub OIDC publisher (TestPyPI first)

Fill the Trusted Publisher form with values that will match the *future*
publish workflow (not the dry-run workflow):

| Field | Recommended value |
|---|---|
| Owner | `almanzalex` |
| Repository | `Arena` |
| Workflow name | e.g. `publish-pypi.yml` (the **upload** workflow filename, not `pypi-dry-run.yml`) |
| Environment name | e.g. `pypi` or `testpypi` (optional but recommended) |

Save the publisher. Repeat on production PyPI with a separate environment
(for example `pypi`) when you are ready for a real release.

### 4. Mirror the binding in GitHub

In the GitHub repository settings:

1. Create Environments matching the publisher form (`testpypi`, `pypi`).
2. Optionally require reviewers / wait timers on `pypi` before any upload job
   can run.
3. Do **not** store PyPI API tokens for Trusted Publishing flows. OIDC replaces
   them.

### 5. What a future upload workflow needs (do not enable in this PR)

A real Trusted Publishing upload job would need approximately:

```yaml
permissions:
  contents: read
  id-token: write   # required for OIDC; intentionally absent from the dry-run

jobs:
  publish:
    environment: testpypi   # must match the Trusted Publisher registration
    runs-on: ubuntu-24.04
    steps:
      # build once (or download the exact RC artifacts)
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          repository-url: https://test.pypi.org/legacy/
```

Until that workflow exists and is intentionally run, R-11 remains a
**rehearsal**: build + metadata check only.

### 6. After a real TestPyPI upload (later)

1. Clean-install from TestPyPI into a fresh venv.
2. Run the documented first-value / hermetic smoke against that install.
3. Only then promote the **same bytes** to PyPI (see
   [1.0-readiness.md](1.0-readiness.md) Phase 8).
4. Record hashes and install proof into `evidence/R-11-public-distribution.json`
   for [releasing.md](releasing.md).

## Security boundaries for this rehearsal

- Never commit PyPI tokens, `.pypirc` files with passwords, or `TWINE_PASSWORD`.
- The dry-run CI job must stay free of `id-token: write` and upload actions so
  a mis-click cannot publish.
- Prefer Trusted Publishing over API tokens for any future Arena release
  automation.

## Related

- Gate definition: [1.0-readiness.md](1.0-readiness.md) (R-11)
- Signed release index procedure: [releasing.md](releasing.md)
- Exact-artifact RC build: [`.github/workflows/release-candidate.yml`](../.github/workflows/release-candidate.yml)
