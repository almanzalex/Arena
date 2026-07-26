# Populations (Arena 0.2)

Named, content-addressed sets of **0.1 policy digests** for cross-play and historical evaluation.

## Create

```bash
arena init
arena population create ./population.yaml --ref populations/rps-opp --out ./population.resolved.yaml
arena population inspect populations/rps-opp --json
```

Member `policy` fields may be digests or local bundle paths; create resolves paths to digests. Population identity ignores the human `name` and store ref (see [RFC 003](../rfcs/003-populations.md)).

## SDK

```python
from arena.core.sdk import Population
from arena.core.store import LocalStore

store = LocalStore.find()
pop = Population.create(
    name="opponents",
    members=[{"policy": "./rock.arena", "weight": 1.0}],
    store=store,
    ref="populations/opponents",
)
print(pop.digest, pop.members)
```

## Constraints

- Members are immutable objects; moving a ref does not rewrite the blob.
- Optional `roles.allowed` is checked before eval assignment.
- Sampling strategies (`uniform`, `weighted`, `enumerated_crossplay`) are registry cases under `arena.plugins.samplers`.
