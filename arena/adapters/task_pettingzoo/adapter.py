"""PettingZoo Parallel task adapter (MVP: classic RPS + generic parallel envs)."""

from __future__ import annotations

from typing import Any

from arena.adapters.task_pettingzoo.wrappers import (
    apply_wrappers,
    normalize_wrappers,
    wrappers_provenance,
)
from arena.core.errors import missing_extra, ArenaError, SchemaError
from arena.core.spaces import gymnasium_space_to_dict

ADAPTER_NAME = "pettingzoo-parallel"
PILOT_ENV = "arena/competitive_rps_v0"
AEC_PILOT_ENV = "arena/competitive_rps_aec_v0"
DYNAMIC_PILOT_ENV = "arena/dynamic_lineup_aec_v0"
DYNAMIC_REENTRY_ENV = "arena/dynamic_reentry_aec_v0"
VECTOR_PILOT_ENV = "arena/vector_coordination_v0"
_PILOT_ALIASES = {
    PILOT_ENV,
    "competitive_rps_v0",
    "arena/competitive_rps",
    "arena/competitive_rps_v0",
}
_AEC_PILOT_ALIASES = {
    AEC_PILOT_ENV,
    "competitive_rps_aec_v0",
    "arena/competitive_rps_aec",
    "arena/competitive_rps_aec_v0",
}
_DYNAMIC_PILOT_ALIASES = {
    DYNAMIC_PILOT_ENV,
    "dynamic_lineup_aec_v0",
    "arena/dynamic_lineup_aec",
}
_DYNAMIC_REENTRY_ALIASES = {
    DYNAMIC_REENTRY_ENV,
    "dynamic_reentry_aec_v0",
    "arena/dynamic_reentry_aec",
}
_VECTOR_PILOT_ALIASES = {
    VECTOR_PILOT_ENV,
    "vector_coordination_v0",
    "arena/vector_coordination",
}
_VALID_LAYOUTS = frozenset({"HWC", "CHW"})


def env_id_is_pilot(spec: dict[str, Any]) -> bool:
    env = spec.get("env") or PILOT_ENV
    return (
        env in _PILOT_ALIASES
        or env in _AEC_PILOT_ALIASES
        or env in _DYNAMIC_PILOT_ALIASES
        or env in _DYNAMIC_REENTRY_ALIASES
        or env in _VECTOR_PILOT_ALIASES
    )


def _interaction(spec: dict[str, Any]) -> str:
    from arena.plugins.interactions import require_interaction_kind

    return require_interaction_kind(str(spec.get("interaction", "parallel")))


def _require_pz() -> None:
    try:
        import gymnasium  # noqa: F401
        import pettingzoo  # noqa: F401
    except ImportError as e:
        raise missing_extra(
            "pettingzoo",
            feature="PettingZoo task adapter",
            capability="pettingzoo",
        ) from e


def _observation_layout(spec: dict[str, Any]) -> str | None:
    layout = spec.get("observation_layout") or spec.get("layout")
    if layout is None:
        return None
    if layout not in _VALID_LAYOUTS:
        raise SchemaError(
            f"task.observation_layout must be HWC or CHW, got {layout!r}"
        )
    return str(layout)


def make_env(spec: dict[str, Any], *, trust_task_code: bool = False):
    """Create an env via the task-packaging registry.

    Default packaging is ``pettingzoo_wrappers`` (existing SuperSuit-declared
    PettingZoo path). ``entrypoint_bundle`` requires explicit trust via
    ``trust_task_code=True`` or ``spec["trust_task_code"]`` /
    ``spec["packaging"]["trust_task_code"]``.
    """
    from arena.core.registry import TASK_PACKAGERS, ensure_plugins_loaded
    from arena.plugins.tasks import resolve_packaging_kind

    ensure_plugins_loaded()
    packaging = spec.get("packaging") if isinstance(spec.get("packaging"), dict) else {}
    trust = bool(
        trust_task_code
        or spec.get("trust_task_code")
        or packaging.get("trust_task_code")
    )
    kind = resolve_packaging_kind(spec)
    env = TASK_PACKAGERS.get(kind).make_env(spec, trust_task_code=trust)
    provider = spec.get("_eval_provider")
    if isinstance(provider, dict):
        provider_kind = provider.get("kind")
        if provider_kind == "gimitest":
            from arena.adapters.eval_gimitest import decorate_env

            env = decorate_env(env, dict(provider.get("config") or {}))
        elif provider_kind:
            raise SchemaError(
                f"eval provider {provider_kind!r} requested task decoration but no "
                "registered environment hook exists"
            )
    return env


def _make_env_pettingzoo(spec: dict[str, Any]):
    """Create a PettingZoo Parallel or AEC env from a task spec.

    When ``task.wrappers`` is declared, SuperSuit wrappers are applied in order
    before the env is returned. Spaces discovered by ``describe_task`` /
    ``arena check`` therefore reflect the *wrapped* observation contract — never
    the silent unwrapped baseline.
    """
    _require_pz()
    normalize_wrappers(spec.get("wrappers"))
    _observation_layout(spec)
    interaction = _interaction(spec)

    env_id = spec.get("env") or PILOT_ENV
    config = dict(spec.get("config") or {})
    seed = config.pop("seed", None)

    if env_id in _VECTOR_PILOT_ALIASES:
        if interaction != "parallel":
            raise SchemaError(
                "arena/vector_coordination_v0 requires interaction=parallel"
            )
        from arena.adapters.task_pettingzoo.pilot_env import vector_parallel_env

        env = vector_parallel_env()
    elif env_id in _DYNAMIC_REENTRY_ALIASES:
        if interaction != "dynamic_aec":
            raise SchemaError(
                "arena/dynamic_reentry_aec_v0 requires interaction=dynamic_aec"
            )
        from arena.adapters.task_pettingzoo.pilot_env import dynamic_reentry_aec_env

        env = dynamic_reentry_aec_env()
    elif env_id in _DYNAMIC_PILOT_ALIASES:
        if interaction != "dynamic_aec":
            raise SchemaError(
                "arena/dynamic_lineup_aec_v0 requires interaction=dynamic_aec"
            )
        from arena.adapters.task_pettingzoo.pilot_env import dynamic_aec_env

        env = dynamic_aec_env()
    elif env_id in _AEC_PILOT_ALIASES or (
        interaction == "aec" and env_id in _PILOT_ALIASES
    ):
        from arena.adapters.task_pettingzoo.pilot_env import aec_env

        kwargs = {}
        if "max_cycles" in config:
            kwargs["max_cycles"] = config["max_cycles"]
        env = aec_env(**kwargs)
        interaction = "aec"
    elif env_id in _PILOT_ALIASES:
        if interaction in {"aec", "dynamic_aec"}:
            from arena.adapters.task_pettingzoo.pilot_env import aec_env

            kwargs = {}
            if "max_cycles" in config:
                kwargs["max_cycles"] = config["max_cycles"]
            env = aec_env(**kwargs)
        else:
            from arena.adapters.task_pettingzoo.pilot_env import parallel_env

            kwargs = {}
            if "max_cycles" in config:
                kwargs["max_cycles"] = config["max_cycles"]
            env = parallel_env(**kwargs)
    elif env_id in {"classic/rps_v2", "rps_v2"}:
        from pettingzoo.classic import rps_v2

        kwargs = {}
        if "max_cycles" in config:
            kwargs["max_cycles"] = config["max_cycles"]
        if interaction in {"aec", "dynamic_aec"}:
            env = rps_v2.env(**kwargs)
        else:
            env = rps_v2.parallel_env(**kwargs)
    else:
        if interaction in {"aec", "dynamic_aec"}:
            env = _load_aec_env(env_id, config)
        else:
            env = _load_parallel_env(env_id, config)

    # SuperSuit wrappers are Parallel-oriented; apply only for parallel interaction.
    if interaction == "parallel":
        env = apply_wrappers(env, spec.get("wrappers"))
    elif spec.get("wrappers"):
        raise SchemaError(
            "task.wrappers (SuperSuit) require interaction=parallel; "
            "AEC + SuperSuit is an unregistered combination — fail loud."
        )

    if seed is not None:
        env.reset(seed=int(seed))
    return env


def _load_aec_env(env_id: str, config: dict[str, Any]):
    import importlib

    parts = env_id.replace("-", "_").split("/")
    if len(parts) == 2:
        module_path = f"pettingzoo.{parts[0]}.{parts[1]}"
    else:
        module_path = f"pettingzoo.{env_id.replace('/', '.')}"
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        raise ArenaError(f"cannot load AEC env {env_id!r}: {e}") from e
    if not hasattr(mod, "env"):
        raise ArenaError(f"{module_path} does not expose env() for AEC")
    try:
        return mod.env(**config)
    except TypeError:
        return mod.env()


def _instantiate_parallel(mod: Any, config: dict[str, Any]):
    if not hasattr(mod, "parallel_env"):
        raise ArenaError(
            f"{getattr(mod, '__name__', mod)} does not expose parallel_env "
            "(AEC-only envs are out of MVP scope)"
        )
    try:
        return mod.parallel_env(**config)
    except TypeError:
        return mod.parallel_env()


def _load_parallel_env(env_id: str, config: dict[str, Any]):
    """Resolve env ids like classic/rps_v2, mpe/simple_tag_v3, or pettingzoo.*.

    PettingZoo ≥1.25/1.26 removed (or deprecated) ``pettingzoo.mpe`` — MPE lives
    in the standalone ``mpe2`` package. For ``mpe/*`` ids we try PettingZoo first,
    then ``mpe2``, then fail with an actionable install hint.
    """
    import importlib

    if env_id.startswith("pettingzoo."):
        mod = importlib.import_module(env_id)
        return _instantiate_parallel(mod, config)

    if env_id.startswith("mpe2."):
        mod = importlib.import_module(env_id)
        return _instantiate_parallel(mod, config)

    parts = env_id.replace("-", "_").split("/")
    family = parts[0] if len(parts) >= 1 else ""
    env_name = parts[1] if len(parts) == 2 else env_id.replace("/", ".")

    candidates: list[str] = []
    if len(parts) == 2:
        candidates.append(f"pettingzoo.{family}.{env_name}")
        if family == "mpe":
            candidates.append(f"mpe2.{env_name}")
    else:
        candidates.append(f"pettingzoo.{env_id.replace('/', '.')}")

    errors: list[str] = []
    for module_path in candidates:
        try:
            mod = importlib.import_module(module_path)
        except ImportError as e:
            errors.append(f"{module_path}: {e}")
            continue
        return _instantiate_parallel(mod, config)

    if family == "mpe" or env_id.startswith("mpe/"):
        raise ArenaError(
            f"cannot load MPE env {env_id!r}: PettingZoo no longer ships "
            f"pettingzoo.mpe on this install, and mpe2 is missing or failed to "
            f"import. Install with: pip install mpe2  (or pip install 'arena[pettingzoo]'). "
            f"Import path: from mpe2 import {env_name} / env id mpe/{env_name}. "
            f"Tried: {'; '.join(errors)}"
        )
    raise ArenaError(
        f"cannot load env {env_id!r}. Tried: {'; '.join(errors)}"
    )


def describe_task(spec: dict[str, Any]) -> dict[str, Any]:
    """Describe a task through its registered packaging case."""
    from arena.core.registry import TASK_PACKAGERS, ensure_plugins_loaded
    from arena.plugins.tasks import resolve_packaging_kind

    ensure_plugins_loaded()
    kind = resolve_packaging_kind(spec)
    packager = TASK_PACKAGERS.get(kind)
    if not hasattr(packager, "describe_task"):
        raise SchemaError(
            f"task packager {kind!r} does not implement describe_task(spec); "
            "remote/env-server task cases must expose immutable role spaces and runtime identity"
        )
    return packager.describe_task(spec)


def _describe_pettingzoo_task(spec: dict[str, Any]) -> dict[str, Any]:
    """Return role/agent observation and action schemas for compatibility checks.

    Observation spaces reflect the post-wrapper env. When ``observation_layout``
    is declared on the task, it is stamped onto each role observation dict so
    ``arena check`` can fail on layout mismatches (HWC vs CHW) rather than only
    on shape.
    """
    # Keep environment construction behind the public seam. Besides making
    # wrappers/provider decoration consistent with execution, this is the
    # supported injection point for embedders and conformance fixtures.
    env = make_env(spec)
    layout = _observation_layout(spec)
    wrapper_info = wrappers_provenance(spec.get("wrappers"))
    version = (
        f"pilot+pettingzoo-{__import__('pettingzoo').__version__}"
        if env_id_is_pilot(spec)
        else __import__("pettingzoo").__version__
    )
    return describe_env_contract(
        spec,
        env,
        adapter_name=ADAPTER_NAME,
        version=version,
        layout=layout,
        wrappers=wrapper_info,
    )


def describe_env_contract(
    spec: dict[str, Any],
    env: Any,
    *,
    adapter_name: str,
    version: str,
    layout: str | None = None,
    wrappers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe any PettingZoo-shaped Parallel/AEC adapter without changing identity."""
    interaction = _interaction(spec)
    try:
        if interaction in {"aec", "dynamic_aec"}:
            env.reset(seed=0)
            agents = list(env.agents)
            possible_agents = list(getattr(env, "possible_agents", agents))
            sample_obs = {a: env.observe(a) for a in possible_agents}
        else:
            sample_obs, _infos = env.reset(seed=0)
            agents = list(env.agents)
            possible_agents = list(getattr(env, "possible_agents", agents))
        roles: dict[str, Any] = {}
        provides_masks = False
        for agent in possible_agents:
            obs_space = env.observation_space(agent)
            act_space = env.action_space(agent)
            obs_dict = gymnasium_space_to_dict(obs_space)
            sample = sample_obs.get(agent)
            if isinstance(sample, dict) and "action_mask" in sample:
                provides_masks = True
                if hasattr(obs_space, "spaces") and "observation" in obs_space.spaces:
                    obs_dict = gymnasium_space_to_dict(obs_space.spaces["observation"])
            if layout is not None:
                shape = obs_dict.get("shape") or []
                if len(shape) >= 3:
                    obs_dict = dict(obs_dict)
                    obs_dict["layout"] = layout
            roles[agent] = {
                "agents": [agent],
                "observation": obs_dict,
                "action": gymnasium_space_to_dict(act_space),
            }
        # Dynamic agents: if possible_agents can differ from agents after reset, flag.
        dynamic_agents = False
        possible = list(getattr(env, "possible_agents", agents))
        if set(possible) != set(agents):
            dynamic_agents = True
        result: dict[str, Any] = {
            "adapter": adapter_name,
            "env": spec.get("env") or PILOT_ENV,
            "version": version,
            "agents": agents,
            "possible_agents": possible,
            "roles": roles,
            "provides_masks": provides_masks,
            "interaction": interaction,
            "dynamic_agents": dynamic_agents,
            "wrappers": wrappers or {"steps": [], "identity": "none"},
            "config": dict(spec.get("config") or {}),
        }
        if layout is not None:
            result["observation_layout"] = layout
        if spec.get("source_revision"):
            result["source_revision"] = spec["source_revision"]
        return result
    finally:
        env.close()


def extract_observation(raw_obs: Any) -> Any:
    """Unwrap dict observations that contain observation + action_mask."""
    if isinstance(raw_obs, dict) and "observation" in raw_obs:
        return raw_obs["observation"]
    return raw_obs


def extract_action_mask(raw_obs: Any) -> Any | None:
    if isinstance(raw_obs, dict) and "action_mask" in raw_obs:
        return raw_obs["action_mask"]
    return None
