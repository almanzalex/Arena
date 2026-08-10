"""Dataset/trajectory provenance binding (policy + task identity)."""

from arena.dataset.provenance import (
    PROVENANCE_BINDING_SCHEMA,
    bind_dataset_provenance,
    episode_policy_digests,
    episode_task_identity,
    load_episode,
    task_identity,
    unbind_dataset_provenance,
    verify_dataset_provenance,
)
from arena.dataset.select import select_bound_episodes
from arena.dataset.shard import (
    SHARD_METHOD,
    materialize_dataset_sharded,
    shard_dirname,
    shard_id_for_index,
)
from arena.dataset.stream import iter_verified_episodes

__all__ = [
    "PROVENANCE_BINDING_SCHEMA",
    "SHARD_METHOD",
    "bind_dataset_provenance",
    "episode_policy_digests",
    "episode_task_identity",
    "iter_verified_episodes",
    "load_episode",
    "materialize_dataset_sharded",
    "select_bound_episodes",
    "shard_dirname",
    "shard_id_for_index",
    "task_identity",
    "unbind_dataset_provenance",
    "verify_dataset_provenance",
]
