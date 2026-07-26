# Dynamic-agent lifecycle

Use `interaction: dynamic_aec` when agents may join, leave, or rejoin. Lifecycle
assignment is a registry axis and never inferred. The back-compatible `explicit`
resolver maps every concrete agent ID to a policy:

```yaml
task:
  adapter: pettingzoo-parallel
  env: arena/dynamic_lineup_aec_v0
  interaction: dynamic_aec
  lifecycle:
    birth_eligibility:
      agent_2: [sha256:...]
assignments:
  agent_0: ./policy.arena
  agent_1: ./policy.arena
  agent_2: ./policy.arena
```

Arena validates complete assignment/eligibility coverage before creating a run.
At birth it repeats compose-check, resets the born agent's policy state, and only
then permits inference. At removal it freezes the agent segment and applies the
declared termination reset.

For populations where many concrete IDs share one policy role, use `role`:

```yaml
task:
  adapter: pettingzoo-parallel
  env: arena/dynamic_reentry_aec_v0
  interaction: dynamic_aec
  lifecycle:
    resolver:
      kind: role
      agent_roles:
        agent_0: contestant
        agent_1: contestant
        agent_2: contestant
      join_eligibility:
        contestant: [sha256:...]
assignments:
  contestant: ./policy.arena
```

Role resolution is computed before the run and remains immutable. Eligibility is
still checked for each concrete join; it is not matchmaking or policy inference.

Each dynamic trajectory step includes `agents_alive_before`, `agents_alive`,
`join_events`, `leave_events`, and resolved assignment evidence. The episode
includes `initial_agents`, the back-compatible `agent_segments` map, and ordered
`agent_segment_history` so a removed ID can re-enter without erasing its earlier
lifetime.

Fixed `parallel` and `aec` modes retain their old behavior: a changing lifecycle
is an error. New dynamic environments require a forced lifecycle fixture and
qualification evidence; the built-in fixture is not a broad PettingZoo claim.

The built-in lineup and re-entry cases have frozen traces and pass the same public
qualification command used by other task adapters:

```bash
arena adapter qualify examples/tasks/dynamic-lineup.yaml \
  --trace-suite examples/tasks/dynamic-lineup-trace.yaml \
  --out dynamic-qualification.json

arena adapter qualify examples/tasks/dynamic-reentry.yaml \
  --trace-suite examples/tasks/dynamic-reentry-trace.yaml \
  --out dynamic-reentry-qualification.json
```
