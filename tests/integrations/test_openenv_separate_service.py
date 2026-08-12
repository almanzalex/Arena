"""Separate-service OpenEnv qualification (R-05) — never fake live success."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from arena.adapters.task_openenv.service_recipe import (
    OPENENV_BASE_URL_ENV,
    SEPARATE_SERVICE_RECIPE,
    openenv_service_healthy,
    require_openenv_separate_service,
    resolve_openenv_base_url,
)

ROOT = Path(__file__).resolve().parents[2]


def _load_qualify_main():
    path = ROOT / "scripts" / "qualify_openenv_separate_service.py"
    spec = importlib.util.spec_from_file_location("qualify_openenv_separate_service", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_openenv_separate_service_recipe_is_actionable() -> None:
    assert "docker compose -f docker/openenv/docker-compose.yml" in SEPARATE_SERVICE_RECIPE
    assert "ARENA_OPENENV_BASE_URL" in SEPARATE_SERVICE_RECIPE
    assert "qualify_openenv_separate_service.py" in SEPARATE_SERVICE_RECIPE
    assert "support-matrix" in SEPARATE_SERVICE_RECIPE


def test_require_openenv_separate_service_fails_loud_when_unset(monkeypatch) -> None:
    monkeypatch.delenv(OPENENV_BASE_URL_ENV, raising=False)
    with pytest.raises(RuntimeError, match="docker compose -f docker/openenv"):
        require_openenv_separate_service()


def test_require_openenv_separate_service_fails_loud_when_unhealthy(monkeypatch) -> None:
    monkeypatch.setenv(OPENENV_BASE_URL_ENV, "http://127.0.0.1:9")
    with pytest.raises(RuntimeError, match="not healthy"):
        require_openenv_separate_service()


def test_qualify_script_fails_loud_without_service(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(OPENENV_BASE_URL_ENV, raising=False)
    main = _load_qualify_main()
    code = main(["--out", str(tmp_path / "out")])
    assert code == 2


@pytest.mark.docker
@pytest.mark.requires_openenv
def test_openenv_separate_service_live_qualification(tmp_path: Path) -> None:
    """Real R-05 gate: client connects only via base_url to an already-running service."""
    pytest.importorskip("openenv")
    pytest.importorskip("pettingzoo")

    base_url = resolve_openenv_base_url()
    if base_url is None or not openenv_service_healthy(base_url):
        pytest.fail(SEPARATE_SERVICE_RECIPE)

    from arena.conformance.qualification import qualify_task_fixture
    from arena.core.manifests import load_manifest
    from arena.core.tasks import import_openenv_task, verify_task_equivalence

    host_port = base_url.removeprefix("http://").removeprefix("https://")
    transport = "https" if base_url.startswith("https://") else "http"
    imported_path = tmp_path / "openenv-separate.yaml"
    imported = import_openenv_task(
        f"openenv://{host_port}/arena/competitive_rps_v0?transport={transport}",
        name="task:rps-openenv-separate-test@1.0",
        out=imported_path,
        source_revision="openenv-0.4.1",
    )
    assert imported["packaging"]["base_url"].rstrip("/") == base_url.rstrip("/")
    assert imported["packaging"]["schema_digest"].startswith("sha256:")

    native = load_manifest("examples/tasks/native-rps.yaml")
    suite = load_manifest("examples/tasks/rps-equivalence.yaml")
    equivalence = verify_task_equivalence(native, imported, suite)
    assert equivalence["ok"] is True

    qualification = qualify_task_fixture(
        imported_path,
        peer=Path("examples/tasks/native-rps.yaml"),
        trace_suite=Path("examples/tasks/rps-equivalence.yaml"),
        report_path=tmp_path / "openenv-separate-qualification.json",
    )
    assert qualification["ok"] is True
    assert qualification["adapter"] == "openenv"
    assert os.environ.get(OPENENV_BASE_URL_ENV, "").rstrip("/") == base_url.rstrip("/")


@pytest.mark.docker
@pytest.mark.requires_openenv
def test_openenv_separate_service_evidence_artifact_shape(tmp_path: Path) -> None:
    """When evidence already exists from the qualify script, assert R-05 shape."""
    evidence_path = Path("docs/qualifications/openenv/R-05-openenv-separate-service.json")
    if not evidence_path.is_file():
        base_url = resolve_openenv_base_url()
        if base_url is None or not openenv_service_healthy(base_url):
            pytest.fail(SEPARATE_SERVICE_RECIPE)
        main = _load_qualify_main()
        assert main(["--out", str(tmp_path)]) == 0
        evidence_path = tmp_path / "R-05-openenv-separate-service.json"

    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "arena.openenv-separate-qualification/v1"
    assert payload["gate"] == "R-05"
    assert payload["ok"] is True
    assert payload["mode"] == "separate-service"
    assert payload["separately_deployed"] is True
    assert payload["client"]["does_not_spawn_service"] is True
    assert payload["qualification"]["ok"] is True
    assert payload["stable_claim"]["support_matrix_flipped"] is False
