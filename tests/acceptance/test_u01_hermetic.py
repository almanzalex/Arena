"""Hermetic, doc-driven U-01 clean-room gate.

This strengthens the in-process ``test_u01_clean_room`` gate by reproducing the
conditions of the *human* U-01 step as closely as an automated test can:

* **Never-trained machine** — RLX is built into a real distributable wheel and
  installed into a throwaway ``python3.12 -m venv`` (or a minimal Docker image)
  with a scrubbed ``HOME``/``XDG``/``PYTHONPATH`` and no repository on the import
  path. The way a stranger installs it, *not* ``pip install -e .``.
* **Received bundles alone** — only the ``.rlx`` bundles, ``match.yaml``, and the
  clean-room doc are copied into the sandbox. The trainer package, checkpoints,
  export spec, and source repo are asserted absent/unimportable, with a negative
  control that fails iff a trainer import were required.
* **Follows docs/clean-room.md** — the commands are *parsed out of the document*
  and executed verbatim, in order. Documentation drift fails the test.
* **No hidden network** — the run phase disables the network (loopback-only
  socket guard + ``PIP_NO_INDEX`` for venv; ``--network none`` for Docker) so any
  hidden download fails.

The ``slow`` and ``docker`` markers keep the default ``pytest -q`` fast; run the
full gate with ``pytest -m slow`` (and ``pytest -m docker`` where Docker exists).
"""

from __future__ import annotations

import importlib.metadata
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from _cleanroom import (  # noqa: E402  (pytest puts the test dir on sys.path)
    EXPECTED_COMMANDS,
    parse_cleanroom_commands,
    validate_cleanroom_commands,
)
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from rlx.conformance.usability import run_blind_reader

REPO_ROOT = Path(__file__).resolve().parents[2]
CLEAN_ROOM_DOC = REPO_ROOT / "docs" / "clean-room.md"

_MATCH_YAML = """\
schema: rlx.match/v0alpha1
task:
  adapter: pettingzoo-parallel
  env: rlx/competitive_rps_v0
assignments:
  player_0: ./player_0.rlx
  player_1: ./player_1.rlx
seeds: {start: 0, count: 5}
action_mode: deterministic
record:
  trajectories: all
failure_policy:
  timeout_seconds: 30
  retain_incomplete: true
  retry: 0
"""

# Written into the venv so any hidden outbound connection during the run fails,
# while loopback stays usable. Gated on RLX_CLEANROOM_NO_NET so install is unaffected.
_NET_GUARD_SITECUSTOMIZE = '''\
import os

if os.environ.get("RLX_CLEANROOM_NO_NET") == "1":
    import socket

    _ALLOWED = {"127.0.0.1", "::1", "localhost", ""}
    _orig_connect = socket.socket.connect
    _orig_connect_ex = socket.socket.connect_ex

    def _host(address):
        if isinstance(address, (tuple, list)) and address:
            return address[0]
        return None

    def _connect(self, address, *a, **k):
        h = _host(address)
        if h in _ALLOWED or h is None:
            return _orig_connect(self, address, *a, **k)
        raise OSError(f"RLX clean-room: network access blocked (attempted {h!r})")

    def _connect_ex(self, address, *a, **k):
        h = _host(address)
        if h in _ALLOWED or h is None:
            return _orig_connect_ex(self, address, *a, **k)
        return 1

    socket.socket.connect = _connect
    socket.socket.connect_ex = _connect_ex
'''


# --------------------------------------------------------------------------- #
# Fast, unmarked gate: the documented commands must match the CLI flow.
# Runs in the default suite so doc drift is caught without the slow build.
# --------------------------------------------------------------------------- #
@pytest.mark.acceptance
def test_cleanroom_doc_matches_cli() -> None:
    """docs/clean-room.md must expose the canonical U-01 flow, in order."""
    commands = parse_cleanroom_commands(CLEAN_ROOM_DOC)
    validate_cleanroom_commands(commands)
    assert commands == list(EXPECTED_COMMANDS)


# --------------------------------------------------------------------------- #
# Shared build + handoff fixtures.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def hermetic_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a real wheel + sdist (as a second person would receive/install)."""
    pytest.importorskip("build", reason="`build` is needed for the hermetic gate")
    dist = tmp_path_factory.mktemp("dist")
    proc = subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--wheel", "--sdist",
         "--outdir", str(dist), str(REPO_ROOT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"wheel build failed:\n{proc.stdout}\n{proc.stderr}"
    wheels = list(dist.glob("rlx-*.whl"))
    sdists = list(dist.glob("rlx-*.tar.gz"))
    assert wheels, f"no wheel produced in {dist}"
    assert sdists, f"no sdist produced in {dist}"
    return wheels[0]


def _pack_installed_distribution(distribution: importlib.metadata.Distribution, wheelhouse: Path) -> None:
    """Turn an installed dependency into a local wheel for the nested fresh venv.

    The outer test environment has already resolved RLX's declared test extras.
    Repacking those installed distributions makes dependency resolution for the
    nested venv reproducible without contacting an index, while retaining the
    normal pip wheel-install path.
    """
    name = distribution.metadata["Name"]
    version = distribution.version
    wheel_metadata = distribution.read_text("WHEEL")
    files = tuple(distribution.files or ())
    if wheel_metadata:
        tags = [
            line.partition(":")[2].strip()
            for line in wheel_metadata.splitlines()
            if line.startswith("Tag:")
        ]
        assert tags, f"{name} has no wheel compatibility tag"
    else:
        # Conda can install pure-Python distributions without a WHEEL record.
        # Repackage those so dependencies such as setuptools remain available
        # to the fully offline nested pip install. Never mislabel compiled code
        # as a universal wheel.
        compiled = [
            member
            for member in files
            if str(member).endswith((".so", ".dylib", ".pyd"))
        ]
        assert not compiled, f"{name} has no WHEEL metadata and contains compiled extensions"
        tags = ["py3-none-any"]

    filename = f"{re.sub(r'[-.]+', '_', name)}-{version.replace('-', '_')}-{tags[0]}.whl"
    destination = wheelhouse / filename
    if destination.exists():
        return

    site_packages = Path(distribution.locate_file(".")).resolve()
    metadata_dir: str | None = None
    # Installed compiled packages such as torch are much larger than their
    # normal wheel archives. Compress the local wheelhouse so the slow gate does
    # not need several extra gigabytes while the nested venv is installed.
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        record_paths: list[str] = []
        for member in files:
            source = Path(distribution.locate_file(member)).resolve()
            # Console scripts belong to the outer venv. Pip regenerates them from
            # entry points when it installs this wheel into the nested venv.
            if not source.is_file() or not source.is_relative_to(site_packages):
                continue
            relative = source.relative_to(site_packages).as_posix()
            archive.write(source, relative)
            if not wheel_metadata:
                record_paths.append(relative)
            if len(Path(relative).parts) == 2 and relative.endswith(".dist-info/METADATA"):
                metadata_dir = relative.rsplit("/", 1)[0]
        if not wheel_metadata:
            if metadata_dir is None:
                metadata_dir = (
                    f"{re.sub(r'[-.]+', '_', name)}-"
                    f"{version.replace('-', '_')}.dist-info"
                )
                metadata = (
                    distribution.read_text("METADATA")
                    or distribution.read_text("PKG-INFO")
                    or str(distribution.metadata)
                )
                archive.writestr(f"{metadata_dir}/METADATA", metadata)
                record_paths.append(f"{metadata_dir}/METADATA")
            archive.writestr(
                f"{metadata_dir}/WHEEL",
                "Wheel-Version: 1.0\n"
                "Generator: rlx hermetic wheelhouse\n"
                "Root-Is-Purelib: true\n"
                "Tag: py3-none-any\n",
            )
            record_paths.append(f"{metadata_dir}/WHEEL")
            archive.writestr(
                f"{metadata_dir}/RECORD",
                "".join(f"{path},,\n" for path in record_paths)
                + f"{metadata_dir}/RECORD,,\n",
            )


@pytest.fixture(scope="session")
def hermetic_wheelhouse(
    hermetic_wheel: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Package the already-installed test dependencies for offline resolution."""
    wheelhouse = tmp_path_factory.mktemp("wheelhouse")
    installed = {
        canonicalize_name(distribution.metadata["Name"]): distribution
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }

    with zipfile.ZipFile(hermetic_wheel) as wheel:
        metadata_name = next(
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        )
        root_requirements = [
            Requirement(line.partition(":")[2].strip())
            for line in wheel.read(metadata_name).decode().splitlines()
            if line.startswith("Requires-Dist:")
        ]

    def enabled(requirement: Requirement, extras: set[str]) -> bool:
        if requirement.marker is None:
            return True
        environment = default_environment()
        return any(
            requirement.marker.evaluate({**environment, "extra": extra}) for extra in extras
        )

    pending = [
        requirement
        for requirement in root_requirements
        if enabled(requirement, {"", "torch", "pettingzoo"})
    ]
    packed: set[str] = set()
    while pending:
        requirement = pending.pop()
        name = canonicalize_name(requirement.name)
        if name in packed:
            continue
        distribution = installed.get(name)
        assert distribution is not None, f"required dependency is not installed: {requirement}"
        _pack_installed_distribution(distribution, wheelhouse)
        packed.add(name)
        pending.extend(
            Requirement(item)
            for item in distribution.requires or ()
            if enabled(Requirement(item), {""})
        )

    assert list(wheelhouse.glob("*.whl")), "offline wheelhouse is empty"
    return wheelhouse


@pytest.fixture(scope="session")
def handoff(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Produce the artifacts a recipient receives, plus trainer-side artifacts
    that must NEVER leak into the clean room."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("pettingzoo")
    from rlx.conformance.fixtures import build_rps_policy

    root = tmp_path_factory.mktemp("author")

    # --- Trainer side: a simulated private training repo + checkpoints + spec.
    trainer = root / "trainer_repo"
    trainer.mkdir()
    (trainer / "__init__.py").write_text(
        "SECRET = 'private trainer internals; must not reach the clean room'\n",
        encoding="utf-8",
    )
    torch.save({"state_dict": {}}, root / "ckpt_p0.pt")
    torch.save({"state_dict": {}}, root / "ckpt_p1.pt")
    (root / "export_spec.yaml").write_text("architecture: {type: mlp_categorical}\n", encoding="utf-8")

    # --- Recipient side: the portable bundles + match manifest.
    bundles = root / "bundles"
    bundles.mkdir()
    build_rps_policy(bundles / "player_0.rlx", role="player_0", seed=1)
    build_rps_policy(bundles / "player_1.rlx", role="player_1", seed=2)
    (bundles / "match.yaml").write_text(_MATCH_YAML, encoding="utf-8")

    return {"root": root, "bundles": bundles, "trainer_name": "trainer_repo"}


def _stage_sandbox(dest: Path, handoff: dict[str, Path]) -> None:
    """Copy ONLY what a recipient receives into the sandbox."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("player_0.rlx", "player_1.rlx"):
        shutil.copytree(handoff["bundles"] / name, dest / name)
    shutil.copy2(handoff["bundles"] / "match.yaml", dest / "match.yaml")
    # The received copy of the guide — parsed in-place, as a reader would use it.
    shutil.copy2(CLEAN_ROOM_DOC, dest / "clean-room.md")


def _assert_recipient_only(sandbox: Path, trainer_name: str) -> None:
    """Trainer/checkpoint/spec/repo artifacts must be absent from the sandbox."""
    assert not list(sandbox.glob(f"**/{trainer_name}")), "trainer package leaked into sandbox"
    assert not list(sandbox.glob("**/ckpt_*.pt")), "trainer checkpoints leaked into sandbox"
    assert not list(sandbox.glob("**/export_spec.yaml")), "export spec leaked into sandbox"
    # Legitimate: portable policy weights live inside the bundles.
    assert list(sandbox.glob("player_0.rlx/payloads/weights.pt")), "bundle weights missing"


# --------------------------------------------------------------------------- #
# Hermetic venv variant (default for CI).
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.acceptance
@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_u01_hermetic_venv(
    hermetic_wheel: Path, hermetic_wheelhouse: Path, handoff: dict[str, Path], tmp_path: Path
) -> None:
    venv_dir = tmp_path / "venv"
    sandbox = tmp_path / "sandbox"
    fake_home = tmp_path / "home"
    for d in (fake_home / ".cache", fake_home / ".config", fake_home / ".local" / "share"):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Fresh throwaway interpreter, no system site-packages (no repo leakage).
    r = subprocess.run(
        [sys.executable, "-m", "venv", "--clear", str(venv_dir)],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode == 0, r.stderr
    vbin = venv_dir / ("Scripts" if os.name == "nt" else "bin")
    vpython = vbin / ("python.exe" if os.name == "nt" else "python")

    # 2. Resolve and install ONLY the wheel + extras from the local wheelhouse.
    # This keeps the required fresh-venv wheel install while making the test
    # independent of PyPI availability and host CA-bundle configuration.
    r = subprocess.run(
        [
            str(vpython),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(hermetic_wheelhouse),
            f"{hermetic_wheel}[torch,pettingzoo]",
        ],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode == 0, f"wheel install failed:\n{r.stdout}\n{r.stderr}"

    # Install the network guard into the venv (active only when RLX_CLEANROOM_NO_NET=1).
    site = subprocess.run(
        [str(vpython), "-c", "import site; print(site.getsitepackages()[0])"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    (Path(site) / "sitecustomize.py").write_text(_NET_GUARD_SITECUSTOMIZE, encoding="utf-8")

    # 3. Copy ONLY the received artifacts into the sandbox.
    _stage_sandbox(sandbox, handoff)
    _assert_recipient_only(sandbox, handoff["trainer_name"])

    base_env = {
        "PATH": f"{vbin}{os.pathsep}/usr/bin:/bin",
        "HOME": str(fake_home),
        "XDG_CACHE_HOME": str(fake_home / ".cache"),
        "XDG_CONFIG_HOME": str(fake_home / ".config"),
        "XDG_DATA_HOME": str(fake_home / ".local" / "share"),
        "TMPDIR": str(tmp_path / "tmp"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "OMP_NUM_THREADS": "1",
        # deliberately NO PYTHONPATH — the repo must not be importable.
    }
    (tmp_path / "tmp").mkdir(exist_ok=True)
    run_env = {**base_env, "RLX_CLEANROOM_NO_NET": "1", "PIP_NO_INDEX": "1"}

    # 4. Isolation: rlx comes from the wheel; trainer/repo are unimportable.
    got = subprocess.run(
        [str(vpython), "-c", "import rlx; print(rlx.__file__)"],
        cwd=sandbox, env=base_env, capture_output=True, text=True, check=False,
    )
    assert got.returncode == 0, got.stderr
    rlx_file = got.stdout.strip()
    assert "site-packages" in rlx_file and str(venv_dir) in rlx_file, (
        f"rlx imported from outside the wheel install: {rlx_file}"
    )
    assert str(REPO_ROOT) not in rlx_file, f"rlx imported from the source repo: {rlx_file}"

    # Negative control: this MUST fail — it proves the trainer is truly absent.
    # If it ever passed, the clean room would not be a clean room.
    control = subprocess.run(
        [str(vpython), "-c", f"import {handoff['trainer_name']}"],
        cwd=sandbox, env=base_env, capture_output=True, text=True, check=False,
    )
    assert control.returncode != 0, (
        "trainer package is importable in the clean room — isolation is broken"
    )

    # 5. Doc-driven execution: run EXACTLY the documented commands, in order.
    commands = parse_cleanroom_commands(sandbox / "clean-room.md")
    validate_cleanroom_commands(commands)

    transcript = sandbox / "blind-reader-transcript.json"
    reader = run_blind_reader(
        sandbox / "clean-room.md",
        cwd=sandbox,
        env=run_env,
        transcript_path=transcript,
    )
    assert reader["success_from_docs_and_received_artifacts_only"], reader
    assert transcript.exists(), "blind-reader transcript missing"
    outputs = {attempt["command"]: attempt for attempt in reader["attempts"]}

    # 6. Success criteria from the guide.
    for cmd, proc in outputs.items():
        if cmd.startswith("rlx check"):
            assert "COMPATIBLE" in proc["stdout"], proc["stdout"]
        if cmd.startswith("rlx match run"):
            assert "failures=0" in proc["stdout"], proc["stdout"]

    out = sandbox / "runs" / "baseline-match"
    assert (out / "run.yaml").exists(), "run record missing"
    assert (out / "trajectories" / "bundle.yaml").exists(), "trajectory bundle missing"
    assert list((out / "trajectories").glob("episode_*.json")), "no episode trajectories recorded"


# --------------------------------------------------------------------------- #
# Hermetic Docker variant (real --network none; skipped if Docker is absent).
# --------------------------------------------------------------------------- #
def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=20, check=False
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.docker
@pytest.mark.slow
@pytest.mark.acceptance
@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_u01_hermetic_docker(
    hermetic_wheel: Path, handoff: dict[str, Path], tmp_path: Path
) -> None:
    if not _docker_available():
        pytest.skip("Docker is not available (`docker info` failed)")

    dockerfile = REPO_ROOT / "docker" / "clean-room.Dockerfile"
    assert dockerfile.exists(), "clean-room Dockerfile missing"

    # Build context contains ONLY the wheel (no source repo, no trainer).
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    shutil.copy2(hermetic_wheel, ctx / hermetic_wheel.name)

    image = f"rlx-cleanroom:{os.getpid()}"
    build = subprocess.run(
        ["docker", "build", "-f", str(dockerfile), "-t", image, str(ctx)],
        capture_output=True, text=True, check=False,
    )
    assert build.returncode == 0, f"docker build failed:\n{build.stdout}\n{build.stderr}"

    try:
        # Sandbox holds ONLY received artifacts + the generated command script.
        sandbox = tmp_path / "sandbox"
        _stage_sandbox(sandbox, handoff)
        _assert_recipient_only(sandbox, handoff["trainer_name"])

        commands = parse_cleanroom_commands(sandbox / "clean-room.md")
        validate_cleanroom_commands(commands)
        script = "set -euo pipefail\n" + "\n".join(commands) + "\n"
        (sandbox / "run_cleanroom.sh").write_text(script, encoding="utf-8")

        mount = f"{sandbox}:/work"

        # Isolation: rlx from the wheel, trainer unimportable — with --network none.
        rlx_where = subprocess.run(
            ["docker", "run", "--rm", "--network", "none", image,
             "python", "-c", "import rlx; print(rlx.__file__)"],
            capture_output=True, text=True, check=False,
        )
        assert rlx_where.returncode == 0, rlx_where.stderr
        assert "site-packages" in rlx_where.stdout, rlx_where.stdout

        control = subprocess.run(
            ["docker", "run", "--rm", "--network", "none", image,
             "python", "-c", f"import {handoff['trainer_name']}"],
            capture_output=True, text=True, check=False,
        )
        assert control.returncode != 0, "trainer importable inside clean-room image"

        # Doc-driven run, no network at all.
        run = subprocess.run(
            ["docker", "run", "--rm", "--network", "none", "-v", mount, "-w", "/work",
             image, "bash", "run_cleanroom.sh"],
            capture_output=True, text=True, check=False,
        )
        assert run.returncode == 0, f"clean-room run failed:\n{run.stdout}\n{run.stderr}"
        assert "failures=0" in run.stdout, run.stdout

        out = sandbox / "runs" / "baseline-match"
        assert (out / "run.yaml").exists()
        assert (out / "trajectories" / "bundle.yaml").exists()
        assert list((out / "trajectories").glob("episode_*.json"))
    finally:
        subprocess.run(["docker", "rmi", "-f", image], capture_output=True, text=True, check=False)
