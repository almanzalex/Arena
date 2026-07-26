"""Registry axes: unknown kinds fail loud with extension recipes; trust defaults hold."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from arena.core.action_cases import box_distribution, validate_action_case
from arena.core.errors import SchemaError
from arena.core.registry import (
    ACTION_CASES,
    DISTRIBUTIONS,
    LIFECYCLE_RESOLVERS,
    PAYLOAD_LOADERS,
    PREPROCESS_OPS,
    TASK_PACKAGERS,
    TRAINERS,
    WRAPPER_OPS,
    UnknownKindError,
    capability_matrix,
    ensure_plugins_loaded,
)
from arena.plugins.wrappers import resolve_wrapper_kind


def test_capability_matrix_lists_registered_cases() -> None:
    matrix = capability_matrix()
    assert "Discrete" in matrix["action"]
    assert "diagonal_gaussian" in matrix["distribution"]
    assert "frame_stack" in matrix["preprocess"]
    assert "resize" in matrix["wrapper"]
    assert "torchscript" in matrix["payload"]
    assert "trusted_source" in matrix["payload"]
    assert "entrypoint_bundle" in matrix["task_packaging"]
    assert {
        "behavior_cloning",
        "return_weighted_regression",
    } <= set(matrix["trainer"])
    assert {"explicit", "role"} <= set(matrix["lifecycle_resolver"])


@pytest.mark.parametrize(
    ("getter", "kind"),
    [
        (lambda: ACTION_CASES.get("Tuple"), "Tuple"),
        (lambda: DISTRIBUTIONS.get("mixture"), "mixture"),
        (lambda: PREPROCESS_OPS.get("warp"), "warp"),
        (lambda: WRAPPER_OPS.get("grayscale_custom"), "grayscale_custom"),
        (lambda: PAYLOAD_LOADERS.get("onnx"), "onnx"),
        (lambda: TASK_PACKAGERS.get("ray"), "ray"),
        (lambda: TRAINERS.get("ppo"), "ppo"),
        (lambda: LIFECYCLE_RESOLVERS.get("matchmaker"), "matchmaker"),
    ],
)
def test_unknown_kinds_include_extension_recipe(getter, kind: str) -> None:
    ensure_plugins_loaded()
    with pytest.raises(UnknownKindError, match="extension|register|arena adapter qualify") as exc:
        getter()
    msg = str(exc.value)
    assert kind in msg
    assert "implement" in msg
    assert "qualify" in msg


def test_box_distribution_unknown_is_registry_recipe() -> None:
    with pytest.raises(SchemaError, match="Unknown distribution|mixture|qualify"):
        box_distribution({"distribution": "mixture_of_gaussians"})


def test_incomplete_multidiscrete_still_fails_loud() -> None:
    with pytest.raises(SchemaError, match="logit_layout"):
        validate_action_case({"type": "MultiDiscrete", "nvec": [2, 3], "masks": "none"})


def test_wrapper_alias_resolves_via_registry() -> None:
    ensure_plugins_loaded()
    assert resolve_wrapper_kind("resize_v1") == "resize"


@pytest.mark.requires_torch
def test_trusted_source_refused_by_default_and_digest_mismatch(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from arena.adapters.policy_custom_torch import load_runtime, verify_bundle_self
    from arena.plugins.payloads import export_trusted_source_bundle

    src = tmp_path / "actor.py"
    src.write_text(
        textwrap.dedent(
            """
            import torch
            import torch.nn as nn

            class Tiny(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lin = nn.Linear(2, 3)
                def forward(self, x):
                    return self.lin(x)

            def build_actor():
                return Tiny()
            """
        ),
        encoding="utf-8",
    )
    # Build weights from the same factory that trusted_source will import.
    import importlib.util

    spec = importlib.util.spec_from_file_location("tiny_actor_src", src)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    module = mod.build_actor()
    obs = {"type": "Box", "shape": [2], "dtype": "float32", "low": -1.0, "high": 1.0}
    action = {"type": "Discrete", "n": 3, "dtype": "int64", "masks": "none"}

    def source_act(case):
        x = torch.as_tensor(case["observation"], dtype=torch.float32).view(1, -1)
        logits = module(x).detach().numpy().reshape(-1)
        return int(np.argmax(logits)), logits.tolist()

    with pytest.raises(SchemaError, match="refusing to export trusted_source|trust_source"):
        export_trusted_source_bundle(
            out_dir=tmp_path / "no.arena",
            name="ts",
            roles=["agent"],
            source_py=src,
            state_dict=module.state_dict(),
            observation=obs,
            action=action,
            reference_cases=[{"observation": [0.1, -0.2], "mode": "deterministic"}],
            source_act_fn=source_act,
            trust_source=False,
        )

    bundle = export_trusted_source_bundle(
        out_dir=tmp_path / "yes.arena",
        name="ts",
        roles=["agent"],
        source_py=src,
        state_dict=module.state_dict(),
        observation=obs,
        action=action,
        reference_cases=[{"observation": [0.1, -0.2], "mode": "deterministic"}],
        source_act_fn=source_act,
        trust_source=True,
    )
    with pytest.raises(SchemaError, match="refused by default|trust-source|NOT sandboxed"):
        load_runtime(bundle, trust_source=False)

    rt = load_runtime(bundle, trust_source=True)
    act = rt.act(np.array([0.1, -0.2], dtype=np.float32), mode="deterministic")
    assert isinstance(act, (int, np.integer))
    assert verify_bundle_self(bundle, trust_source=True)["ok"]

    # Digest mismatch must fail.
    src_payload = bundle / "payloads" / "inference.py"
    src_payload.write_text(src_payload.read_text(encoding="utf-8") + "\n# tamper\n", encoding="utf-8")
    with pytest.raises(Exception, match="digest mismatch|integrity"):
        load_runtime(bundle, trust_source=True)


@pytest.mark.requires_torch
def test_entrypoint_bundle_refused_without_trust(tmp_path: Path) -> None:
    from arena.adapters.task_pettingzoo.adapter import make_env
    from arena.core.identity import digest_uri, sha256_file

    src = tmp_path / "env.py"
    src.write_text(
        textwrap.dedent(
            """
            def parallel_env(**kwargs):
                class _Env:
                    metadata = {"name": "tiny"}
                    possible_agents = ["a0"]
                    agents = ["a0"]
                    def reset(self, seed=None, options=None):
                        self.agents = ["a0"]
                        return {"a0": 0}, {}
                    def step(self, actions):
                        return {"a0": 0}, {"a0": 0.0}, {"a0": True}, {"a0": False}, {}
                    def observation_space(self, agent):
                        import gymnasium as gym
                        return gym.spaces.Discrete(2)
                    def action_space(self, agent):
                        import gymnasium as gym
                        return gym.spaces.Discrete(2)
                    def close(self):
                        pass
                return _Env()
            """
        ),
        encoding="utf-8",
    )
    digest = digest_uri(sha256_file(src))
    spec = {
        "adapter": "pettingzoo-parallel",
        "packaging": {
            "kind": "entrypoint_bundle",
            "root": str(tmp_path),
            "entrypoint": "env.py",
            "digest": digest,
            "factory": "parallel_env",
        },
    }
    with pytest.raises(SchemaError, match="refused by default|trust-task-code|NOT sandboxed"):
        make_env(spec, trust_task_code=False)

    env = make_env(spec, trust_task_code=True)
    try:
        obs, _ = env.reset(seed=0)
        assert "a0" in obs
    finally:
        env.close()

    # Digest mismatch
    bad = dict(spec)
    bad["packaging"] = dict(spec["packaging"])
    bad["packaging"]["digest"] = "sha256:" + ("0" * 64)
    with pytest.raises(SchemaError, match="digest mismatch"):
        make_env(bad, trust_task_code=True)


@pytest.mark.requires_torch
def test_capture_from_source_drafts_action_cases() -> None:
    gym = pytest.importorskip("gymnasium")
    from arena.core.capture import capture_action_case_from_space, capture_draft_from_actor

    md = capture_action_case_from_space(gym.spaces.MultiDiscrete([2, 3]))
    assert md["type"] == "MultiDiscrete"
    assert md["logit_layout"]["kind"] == "concatenated"
    draft = capture_draft_from_actor(
        observation={"type": "Box", "shape": [4], "dtype": "float32"},
        action=md,
        preferred_payload="torchscript",
    )
    assert draft["runtime"]["tier"] == "torchscript"
    assert draft["needs_human_confirm"] is True
