"""Hugging Face ``hf://`` mirror adapter and fail-closed live qualification.

Live evidence requires ``HF_TOKEN`` or ``HUGGING_FACE_HUB_TOKEN``. Missing
credentials produce ``mode=credential-missing`` with ``ok=false`` — never a
simulated live pass. Simulation (``?simulate=``) is rehearsal only.
"""

from __future__ import annotations

import io
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urldefrag, urlencode, urlparse

from arena.core.errors import StoreError
from arena.core.identity import canonical_json, digest_uri, parse_digest, sha256_bytes
from arena.core.manifests import dump_json, load_manifest

HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
HF_LIVE_RECIPE = (
    "pip install 'arena[hf]' && "
    "export HF_TOKEN=<write-token> && "
    "export ARENA_HF_LIVE_DEST='hf://models/ORG/REPO/arena' && "
    "python scripts/qualify_hf_live.py examples/eval/demo/rock.arena "
    "\"$ARENA_HF_LIVE_DEST\" --out /tmp/hf-restored.arena "
    "--report /tmp/hf-qualification.json"
)
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")
STORE_QUALIFICATION_SCHEMA = "arena.store-qualification/v1"


def hf_token_from_env() -> str | None:
    """Return the first non-empty HF token env var, or None."""
    for name in HF_TOKEN_ENV_VARS:
        value = os.environ.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def hf_live_credentials_present() -> bool:
    """True only when a live token is present in the process environment."""
    return hf_token_from_env() is not None


def is_immutable_hf_revision(revision: str | None) -> bool:
    return isinstance(revision, str) and bool(_IMMUTABLE_REVISION.fullmatch(revision))


def parse_hf_uri(uri: str) -> tuple[str, str, str, str | None]:
    parsed = urlparse(uri)
    if parsed.scheme != "hf":
        raise StoreError("Hugging Face store URI must start hf://")
    parts = [parsed.netloc, *[part for part in parsed.path.split("/") if part]]
    repo_type = "model"
    if parts and parts[0] in {"models", "datasets", "spaces"}:
        repo_type = {"models": "model", "datasets": "dataset", "spaces": "space"}[parts.pop(0)]
    if len(parts) < 2:
        raise StoreError("hf URI requires owner/repo, e.g. hf://models/lab/arena-artifacts")
    repo_id = "/".join(parts[:2])
    prefix = "/".join(parts[2:]).strip("/")
    revision = parse_qs(parsed.query).get("revision", [None])[0]
    return repo_id, repo_type, prefix, revision


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def credential_missing_report(
    *,
    source: Path | str | None = None,
    destination: str | None = None,
) -> dict[str, Any]:
    """Fail-closed report: missing token is never a live pass."""
    return {
        "schema": STORE_QUALIFICATION_SCHEMA,
        "backend": "hf",
        "mode": "credential-missing",
        "ok": False,
        "source": str(Path(source).resolve()) if source is not None else None,
        "destination": destination,
        "immutable_uri": None,
        "identity": None,
        "restored_out": None,
        "started_at": _utc_now(),
        "finished_at": _utc_now(),
        "checks": {
            "credentials": {
                "ok": False,
                "env_vars": list(HF_TOKEN_ENV_VARS),
                "present": False,
            },
            "live_round_trip": {
                "ok": False,
                "skipped": True,
                "reason": "HF_TOKEN / HUGGING_FACE_HUB_TOKEN absent",
            },
            "immutable_revision": {"ok": False, "skipped": True},
        },
        "repair": HF_LIVE_RECIPE,
        "stable_claim_allowed": False,
    }


def qualify_hf_live(
    source: Path | str,
    destination: str,
    *,
    report_path: Path | str | None = None,
    restored_out: Path | str | None = None,
) -> dict[str, Any]:
    """Live HF push/pull qualification, fail-closed without credentials.

    * Missing ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` → ``mode=credential-missing``,
      ``ok=false`` (never simulated as live).
    * ``?simulate=`` is refused — simulation is not live evidence.
    * With credentials, delegates to ``qualify_store`` and requires an immutable
      40-hex revision on the returned URI before ``ok`` can be true.
    """
    from arena.conformance.qualification import qualify_store

    parsed = urlparse(destination)
    if parsed.scheme != "hf":
        raise StoreError(
            "qualify_hf_live requires an hf:// destination",
            code="HF_DESTINATION_INVALID",
            repair="Pass hf://models/ORG/REPO/arena (or datasets/spaces).",
        )
    if "simulate" in parse_qs(parsed.query):
        raise StoreError(
            "qualify_hf_live refuses ?simulate=; simulation is never live HF evidence",
            code="HF_SIMULATE_REFUSED",
            repair=(
                "Remove ?simulate= and provide HF_TOKEN / HUGGING_FACE_HUB_TOKEN, "
                f"or rehearse with arena store qualify …?simulate=/abs/path. Recipe: {HF_LIVE_RECIPE}"
            ),
        )

    if not hf_live_credentials_present():
        report = credential_missing_report(source=source, destination=destination)
        if report_path is not None:
            dump_json(report, report_path)
        return report

    report = qualify_store(
        source,
        destination=destination,
        report_path=None,
        restored_out=restored_out,
    )
    if report.get("mode") != "live":
        report = {
            **report,
            "ok": False,
            "stable_claim_allowed": False,
            "checks": {
                **(report.get("checks") or {}),
                "live_mode": {
                    "ok": False,
                    "mode": report.get("mode"),
                    "reason": "HF live qualify produced a non-live mode",
                },
            },
        }
    else:
        revision = parse_qs(urlparse(urldefrag(report["immutable_uri"])[0]).query).get(
            "revision", [None]
        )[0]
        immutable_ok = is_immutable_hf_revision(revision)
        checks = dict(report.get("checks") or {})
        checks["credentials"] = {
            "ok": True,
            "env_vars": list(HF_TOKEN_ENV_VARS),
            "present": True,
        }
        checks["immutable_revision"] = {
            "ok": immutable_ok,
            "revision": revision,
        }
        report = {
            **report,
            "checks": checks,
            "ok": bool(report.get("ok")) and immutable_ok,
            "stable_claim_allowed": bool(report.get("ok")) and immutable_ok,
        }

    if report_path is not None:
        dump_json(report, report_path)
    return report


class HuggingFaceStoreAdapter:
    scheme = "hf"

    @staticmethod
    def _api() -> Any:
        try:
            from huggingface_hub import HfApi
        except ImportError as e:
            raise StoreError(
                "Hugging Face store requires optional extra 'hf'. "
                "Install with: python -m pip install 'arena[hf]'",
                code="CAPABILITY_MISSING",
                cause="optional extra 'hf' is not installed",
                repair=(
                    "Install the missing extra, then retry: python -m pip install 'arena[hf]'. "
                    "Confirm with `arena doctor --capability hf`."
                ),
                context={"extra": "hf", "capability": "hf"},
            ) from e
        return HfApi()

    @staticmethod
    def _with_revision(uri: str, revision: str) -> str:
        base, fragment = urldefrag(uri)
        parsed = urlparse(base)
        query = parse_qs(parsed.query)
        query["revision"] = [revision]
        pinned = parsed._replace(
            query=urlencode({key: values[-1] for key, values in query.items()})
        ).geturl()
        return f"{pinned}#{fragment}" if fragment else pinned

    def _pin_revision(self, source: str) -> str:
        base, _fragment = urldefrag(source)
        repo_id, repo_type, _prefix, revision = parse_hf_uri(base)
        if is_immutable_hf_revision(revision):
            return source
        try:
            info = self._api().repo_info(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
            )
            commit = str(info.sha)
        except Exception as exc:
            raise StoreError(
                "could not resolve the Hugging Face revision to an immutable commit",
                code="HF_REVISION_UNRESOLVED",
                repair=(
                    "Check repository access and pass a readable ?revision= branch, "
                    "tag, or commit before retrying."
                ),
            ) from exc
        if not is_immutable_hf_revision(commit):
            raise StoreError(f"Hugging Face returned a malformed commit revision: {commit!r}")
        return self._with_revision(source, commit)

    def push(self, artifact: Any, destination: str, *, verify: bool = False) -> str:
        from arena.core import mirror as mirror_mod

        simulated = mirror_mod._simulate_push(artifact, destination, verify=verify)
        if simulated is not None:
            return simulated
        if not hf_live_credentials_present():
            raise StoreError(
                "Hugging Face live push requires HF_TOKEN or HUGGING_FACE_HUB_TOKEN",
                code="HF_CREDENTIALS_MISSING",
                cause="no Hugging Face token in the process environment",
                repair=HF_LIVE_RECIPE,
                context={"env_vars": list(HF_TOKEN_ENV_VARS), "mode": "credential-missing"},
            )
        base = urldefrag(destination)[0]
        repo_id, repo_type, prefix, revision = parse_hf_uri(base)
        api = self._api()
        descriptor = artifact.descriptor()
        mirror_mod._validate_descriptor(descriptor, artifact.identity)

        def remote_path(path: str) -> str:
            return "/".join(part for part in (prefix, path) if part)

        for entry in descriptor["files"]:
            digest_hex = parse_digest(entry["digest"])
            api.upload_file(
                path_or_fileobj=io.BytesIO(artifact.files[entry["path"]]),
                path_in_repo=remote_path(f"objects/{digest_hex[:2]}/{digest_hex[2:]}"),
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                commit_message=f"Mirror Arena object {entry['digest']}",
            )
        committed = api.upload_file(
            path_or_fileobj=io.BytesIO(canonical_json(descriptor) + b"\n"),
            path_in_repo=remote_path(f"artifacts/{parse_digest(artifact.identity)}.json"),
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            commit_message=f"Mirror Arena artifact {artifact.identity}",
        )
        commit = getattr(committed, "oid", None)
        if not is_immutable_hf_revision(commit if isinstance(commit, str) else None):
            pinned_destination = self._pin_revision(destination)
        else:
            pinned_destination = self._with_revision(destination, commit)
        uri = mirror_mod._artifact_uri(pinned_destination, artifact.identity)
        if verify:
            loaded = self._download_descriptor(uri)
            mirror_mod._validate_descriptor(loaded, artifact.identity)
            for entry in loaded["files"]:
                actual = digest_uri(sha256_bytes(self._download_blob(uri, entry["digest"])))
                if actual != entry["digest"]:
                    raise StoreError(
                        f"push verification failed for {entry['path']}: "
                        f"expected {entry['digest']}, got {actual}"
                    )
        return uri

    def _download_descriptor(self, source: str) -> dict[str, Any]:
        from arena.core import mirror as mirror_mod

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise StoreError(
                "Hugging Face store requires optional extra 'hf'. "
                "Install with: python -m pip install 'arena[hf]'",
                code="CAPABILITY_MISSING",
                cause="optional extra 'hf' is not installed",
                repair=(
                    "Install the missing extra, then retry: python -m pip install 'arena[hf]'. "
                    "Confirm with `arena doctor --capability hf`."
                ),
                context={"extra": "hf", "capability": "hf"},
            ) from e
        base, identity = mirror_mod._identity_from_uri(source)
        repo_id, repo_type, prefix, revision = parse_hf_uri(base)
        filename = "/".join(
            part for part in (prefix, f"artifacts/{parse_digest(identity)}.json") if part
        )
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
        )
        descriptor = load_manifest(path)
        mirror_mod._validate_descriptor(descriptor, identity)
        return descriptor

    @staticmethod
    def _download_blob(source: str, digest: str) -> bytes:
        from arena.core import mirror as mirror_mod

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise StoreError(
                "Hugging Face store requires optional extra 'hf'. "
                "Install with: python -m pip install 'arena[hf]'",
                code="CAPABILITY_MISSING",
                cause="optional extra 'hf' is not installed",
                repair=(
                    "Install the missing extra, then retry: python -m pip install 'arena[hf]'. "
                    "Confirm with `arena doctor --capability hf`."
                ),
                context={"extra": "hf", "capability": "hf"},
            ) from e
        base, _identity = mirror_mod._identity_from_uri(source)
        repo_id, repo_type, prefix, revision = parse_hf_uri(base)
        digest_hex = parse_digest(digest)
        filename = "/".join(
            part for part in (prefix, f"objects/{digest_hex[:2]}/{digest_hex[2:]}") if part
        )
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
        )
        return Path(path).read_bytes()

    def pull(self, source: str, out: Path | str, *, verify: bool = False) -> dict[str, Any]:
        from arena.core import mirror as mirror_mod

        simulated = mirror_mod._simulate_pull(source, out, verify=verify)
        if simulated is not None:
            return simulated
        if not hf_live_credentials_present():
            raise StoreError(
                "Hugging Face live pull requires HF_TOKEN or HUGGING_FACE_HUB_TOKEN",
                code="HF_CREDENTIALS_MISSING",
                cause="no Hugging Face token in the process environment",
                repair=HF_LIVE_RECIPE,
                context={"env_vars": list(HF_TOKEN_ENV_VARS), "mode": "credential-missing"},
            )
        mirror_mod._identity_from_uri(source)
        pinned_source = self._pin_revision(source)
        descriptor = self._download_descriptor(pinned_source)

        def load_blob(digest: str) -> bytes:
            return self._download_blob(pinned_source, digest)

        return mirror_mod._write_restored(descriptor, load_blob, out, verify=verify)


# Backward-compatible alias used by older call sites / tests.
_parse_hf_uri = parse_hf_uri
