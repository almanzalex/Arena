from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import pytest

from arena.cli.main import main
from arena.conformance.qualification import qualify_store
from arena.core.errors import StoreError
from arena.core.mirror import pull_artifact, push_artifact
from arena.core.registry import capability_matrix
from arena.core.sdk import Policy


@pytest.mark.parametrize(
    "base",
    [
        "hf://models/lab/arena",
        "oci://registry.example/lab/arena",
        "wandb://lab/project/arena",
        "mlflow://arena-experiment",
    ],
)
def test_remote_store_simulations_preserve_policy_identity(tmp_path: Path, base: str) -> None:
    source = Path("examples/eval/demo/rock.arena").resolve()
    expected = Policy.load(source).digest
    mirror = tmp_path / "mirror"
    destination = f"{base}?simulate={quote(str(mirror), safe='/')}"
    pushed = push_artifact(source, destination, verify=True)
    assert pushed["identity"] == expected
    assert pushed["uri"].startswith(base)
    restored = tmp_path / "restored.arena"
    pulled = pull_artifact(pushed["uri"], restored, verify=True)
    assert pulled["verified"] is True
    assert Policy.load(restored).digest == expected


def test_store_simulation_path_must_be_absolute(tmp_path: Path) -> None:
    del tmp_path
    with pytest.raises(StoreError, match="simulate.*absolute"):
        push_artifact(
            "examples/eval/demo/rock.arena",
            "oci://registry.example/lab/arena?simulate=relative/path",
            verify=True,
        )


def test_boundary_stores_are_registered() -> None:
    assert {"file", "hf", "oci", "wandb", "mlflow"} <= set(
        capability_matrix()["external_store"]
    )


def test_store_qualification_labels_simulation_and_cli(
    tmp_path: Path,
) -> None:
    source = Path("examples/eval/demo/rock.arena").resolve()
    destination = (
        "oci://registry.example/lab/qualified"
        f"?simulate={quote(str((tmp_path / 'mirror').resolve()), safe='/')}"
    )
    report_path = tmp_path / "qualification.json"
    report = qualify_store(
        source,
        destination=destination,
        report_path=report_path,
    )
    assert report["schema"] == "arena.store-qualification/v1"
    assert report["mode"] == "simulation"
    assert report["counts_as_live_evidence"] is False
    assert report["ok"] is True
    assert report["checks"]["identity_preserved"]["ok"] is True
    assert report_path.exists()

    cli_destination = (
        "hf://models/lab/qualified"
        f"?simulate={quote(str((tmp_path / 'cli-mirror').resolve()), safe='/')}"
    )
    assert main(
        [
            "store",
            "qualify",
            str(source),
            cli_destination,
            "--out",
            str(tmp_path / "cli-qualification.json"),
        ]
    ) == 0


@pytest.mark.parametrize("kind", ["object", "directory"])
def test_store_qualification_is_not_policy_specific(
    tmp_path: Path,
    kind: str,
) -> None:
    if kind == "object":
        source = tmp_path / "evidence.json"
        source.write_text('{"result":"green"}\n', encoding="utf-8")
    else:
        source = tmp_path / "evidence"
        source.mkdir()
        (source / "report.json").write_text('{"result":"green"}\n', encoding="utf-8")
        (source / "notes.txt").write_text("qualified\n", encoding="utf-8")
    report = qualify_store(
        source,
        destination=(tmp_path / f"{kind}-mirror").as_uri(),
        report_path=tmp_path / f"{kind}-qualification.json",
    )
    assert report["ok"] is True
    assert report["checks"]["identity_preserved"]["kind"] == kind
