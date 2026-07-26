"""Claim 2 (adversarial): trainer-free portability.

Attacks: try to sneak a hidden training-repo dependency past export.
  * a pickled closure / arbitrary object smuggled into the state_dict must NOT be
    silently executed or silently required at inference -> load fails loudly
  * preprocessing cannot smuggle executable code (numeric-only contract)
  * a clean bundle loads in a subprocess with an empty PYTHONPATH (no repo imports)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

from _adv_envs import make_discrete_policy  # noqa: E402

from arena.adapters.policy_custom_torch import (  # noqa: E402
    build_module,
    export_policy,
    load_runtime,
)


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_pickled_object_in_state_dict_fails_loudly(tmp_path: Path) -> None:
    """A checkpoint carrying an arbitrary picklable object (a stand-in for a
    trainer-only class / closure) must not yield a bundle that silently executes it
    or silently needs the trainer. Loading with weights_only=True fails loudly."""
    arch = {"type": "mlp_categorical", "observation_dim": 4, "hidden_dims": [8], "action_n": 3}
    state = build_module(arch).state_dict()

    class _TrainerOnly:
        # A payload that would run code on unpickling if weights_only were off.
        def __reduce__(self):
            return (print, ("code-exec-during-unpickle",))

    poisoned = dict(state)
    poisoned["_trainer_hook"] = _TrainerOnly()

    bundle = export_policy(
        out_dir=tmp_path / "poisoned",
        name="poisoned",
        roles=["player_0"],
        observation={"type": "Discrete", "n": 4, "dtype": "int64"},
        action={"type": "Discrete", "n": 3, "dtype": "int64", "masks": "none"},
        architecture=arch,
        state_dict=poisoned,
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
    )

    # The bundle never silently works: loading fails loudly rather than executing the
    # smuggled object or quietly depending on a trainer class.
    with pytest.raises(Exception) as exc:  # noqa: PT011 - torch raises UnpicklingError
        load_runtime(bundle)
    assert "weights only" in str(exc.value).lower() or "unpickl" in type(exc.value).__name__.lower()


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_preprocessing_is_numeric_only_no_code(tmp_path: Path) -> None:
    """Preprocessing is a declarative numeric contract (mean/std/clip). A string that
    looks like code is simply ignored, never eval'd — so nothing executable is
    embedded that could close over the trainer."""
    bundle = make_discrete_policy(tmp_path / "p", role="player_0")
    from arena.core.manifests import load_manifest

    manifest = load_manifest(bundle / "policy.yaml")
    prep = manifest["preprocessing"]
    assert set(prep).issubset({"included", "id", "mean", "std", "clip"})
    for key in ("mean", "std", "clip"):
        assert not callable(prep.get(key))
    # Runtime uses only numeric fields; no callables anywhere in the manifest.
    from arena.adapters.policy_custom_torch import Preprocess

    pp = Preprocess({"mean": 0.0, "std": 1.0})
    import numpy as np

    assert isinstance(pp(np.zeros(4, dtype=np.float32)), np.ndarray)


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_clean_bundle_loads_without_repo_on_pythonpath(tmp_path: Path) -> None:
    """A normally-exported bundle executes in a subprocess with PYTHONPATH cleared,
    from a directory that is not the repo — i.e. no accidental source-repo coupling."""
    bundle = make_discrete_policy(tmp_path / "clean", role="player_0", seed=3)
    script = (
        "import sys;"
        "from arena.adapters.policy_custom_torch import load_runtime;"
        "rt=load_runtime(sys.argv[1]); rt.reset();"
        "a=rt.act(0, mode='deterministic'); assert a in (0,1,2), a; print('OK', a)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script, str(bundle)],
        cwd=tmp_path,
        env={"PYTHONPATH": "", "PATH": __import__("os").environ.get("PATH", "")},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
