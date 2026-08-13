"""Mandatory R-01…R-14 gate metadata for the local evidence collector."""

from __future__ import annotations

from typing import Any

MANDATORY_GATE_IDS: tuple[str, ...] = tuple(f"R-{index:02d}" for index in range(1, 15))

# Gates that require live / remote / human evidence. The collector may record
# local contract proof for some of these, but must never auto-mark them `pass`.
EXTERNAL_FLOOR_GATES: frozenset[str] = frozenset({"R-04", "R-05", "R-06"})
NEVER_AUTO_PASS_GATES: frozenset[str] = frozenset(
    {
        "R-01",  # claimed-platform CI on the exact release commit
        "R-03",  # independent non-author handoff
        "R-04",  # live HF (and other stores labeled stable)
        "R-05",  # separately deployed OpenEnv
        "R-06",  # isolated Gimitest on release CI
        "R-11",  # TestPyPI / PyPI Trusted Publishing
        "R-14",  # signed release-index verify against the tag
    }
)

GATE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "R-01",
        "title": "Claimed platform CI",
        "owner": "release-engineering",
        "local_collectable": False,
        "evidence_filename": "R-01-platform-ci.json",
        "how_to_attach": (
            "Download the clean-checkout CI summary for the exact release commit "
            "(Linux x86_64 + macOS arm64, Python 3.12/3.13) and place it at "
            "evidence/R-01-platform-ci.json, or pass "
            "--attach R-01=/path/to/ci-summary.json to the collector."
        ),
    },
    {
        "id": "R-02",
        "title": "Hermetic package handoff",
        "owner": "release-engineering",
        "local_collectable": True,
        "local_kind": "hermetic-capable",
        "evidence_filename": "R-02-hermetic.json",
        "how_to_attach": (
            "Run `pytest -m slow` (and `pytest -m docker` where available) against "
            "the exact release wheel, then attach the report with "
            "--attach R-02=evidence/R-02-hermetic.json. Local hermetic-capable "
            "checks are not a substitute for the exact-artifact CI gate."
        ),
    },
    {
        "id": "R-03",
        "title": "Independent user handoff",
        "owner": "dx",
        "local_collectable": False,
        "evidence_filename": "R-03-human-handoff.json",
        "how_to_attach": (
            "Collect two non-author Linux/macOS timed handoff transcripts and "
            "content-bind them into evidence/R-03-human-handoff.json."
        ),
    },
    {
        "id": "R-04",
        "title": "Stable stores",
        "owner": "integration",
        "local_collectable": True,
        "local_kind": "file-store-only",
        "evidence_filename": "R-04-stores.json",
        "how_to_attach": (
            "Attach a live Hugging Face `arena.store-qualification/v1` report "
            "with mode=live (never ?simulate=). Optional OCI/W&B/MLflow live "
            "reports or explicit preview labels go in the same gate file. "
            "Example: --attach R-04=docs/qualifications/hf-live.json"
        ),
    },
    {
        "id": "R-05",
        "title": "OpenEnv",
        "owner": "runtime",
        "local_collectable": False,
        "evidence_filename": "R-05-openenv.json",
        "how_to_attach": (
            "Attach qualification JSON from a separately deployed/operated OpenEnv "
            "service (local loopback alone is not enough). "
            "Example: --attach R-05=docs/qualifications/openenv/"
            "R-05-openenv-separate-service.json"
        ),
    },
    {
        "id": "R-06",
        "title": "Gimitest",
        "owner": "provider",
        "local_collectable": False,
        "evidence_filename": "R-06-gimitest.json",
        "how_to_attach": (
            "Attach non-no-op qualification from a genuinely separate interpreter, "
            "repeated on claimed-platform release CI. "
            "Example: --attach R-06=docs/qualifications/gimitest/"
            "R-06-gimitest.json"
        ),
    },
    {
        "id": "R-07",
        "title": "Adversarial / soak",
        "owner": "quality",
        "local_collectable": True,
        "local_kind": "adversarial-inventory",
        "evidence_filename": "R-07-adversarial.json",
        "how_to_attach": (
            "Attach the release-scale soak/resource envelope report, or pass "
            "--attach R-07=/path/to/adversarial-soak.json."
        ),
    },
    {
        "id": "R-08",
        "title": "Security / supply chain",
        "owner": "security",
        "local_collectable": True,
        "local_kind": "supply-chain-scripts",
        "evidence_filename": "R-08-security.json",
        "how_to_attach": (
            "Attach SBOM, pip-audit, Bandit, secret-scan, and provenance "
            "artifacts from the release-candidate workflow for the exact commit."
        ),
    },
    {
        "id": "R-09",
        "title": "Compatibility",
        "owner": "core",
        "local_collectable": True,
        "local_kind": "schema-and-golden",
        "evidence_filename": "R-09-compatibility.json",
        "how_to_attach": (
            "Bind authentic 0.2/0.3/0.5 golden-fixture verify reports to the "
            "release index. Local golden digests are a partial inventory only."
        ),
    },
    {
        "id": "R-10",
        "title": "Performance",
        "owner": "performance",
        "local_collectable": True,
        "local_kind": "perf-smoke-baseline",
        "evidence_filename": "R-10-performance.json",
        "how_to_attach": (
            "Attach Linux/macOS five-run median baseline comparison within the "
            "declared relative/absolute thresholds. Perf-smoke is not R-10."
        ),
    },
    {
        "id": "R-11",
        "title": "Public distribution",
        "owner": "release-engineering",
        "local_collectable": True,
        "local_kind": "pypi-dry-run-inventory",
        "evidence_filename": "R-11-public-distribution.json",
        "how_to_attach": (
            "Attach TestPyPI → PyPI Trusted Publishing and clean post-publish "
            "install evidence. Local dry-run (build + twine) is rehearsal only "
            "and never counts as a TestPyPI/PyPI upload. Do not invent uploads."
        ),
    },
    {
        "id": "R-12",
        "title": "Failure recovery",
        "owner": "quality",
        "local_collectable": True,
        "local_kind": "recovery-inventory",
        "evidence_filename": "R-12-recovery.json",
        "how_to_attach": (
            "Attach the transaction fault matrix, cancellation/leak checks, and "
            "rollback rehearsal record for the release candidate."
        ),
    },
    {
        "id": "R-13",
        "title": "Documentation truth",
        "owner": "dx",
        "local_collectable": True,
        "local_kind": "release-truth",
        "evidence_filename": "R-13-docs.json",
        "how_to_attach": (
            "Re-run `python scripts/check_release_truth.py` on the final tag and "
            "attach the result; independently follow published docs for the handoff."
        ),
    },
    {
        "id": "R-14",
        "title": "Evidence integrity",
        "owner": "release-manager",
        "local_collectable": False,
        "evidence_filename": "R-14-integrity.json",
        "how_to_attach": (
            "After every other gate file exists, run `arena release assemble` and "
            "`arena release verify` with independently supplied keys; attach the "
            "verify result. This skeleton is not a signed release index."
        ),
    },
)


def gate_by_id(gate_id: str) -> dict[str, Any]:
    for spec in GATE_SPECS:
        if spec["id"] == gate_id:
            return spec
    raise KeyError(f"unknown release gate: {gate_id}")
