# Clean-room handoff guide (U-01)

This guide is for a second lab member who **did not author the trainer**. Goal: load an exported Arena policy, run a seeded match, and inspect trajectories **without** the original training repository.

## Prerequisites

- Python 3.12+
- Fresh virtualenv on a machine that does **not** have the trainer checkout on `PYTHONPATH`
- Before isolating the run, install the received wheel and its declared extras:
  `python -m pip install './arena-0.2.0-py3-none-any.whl[torch,pettingzoo]'`.
  This bootstrap step may contact the configured package indexes to resolve
  PyTorch/PettingZoo/MPE dependencies. For an offline installation, Researcher A
  must also provide a compatible wheelhouse and use
  `--no-index --find-links ./wheelhouse`; a policy bundle is not a dependency
  bundle. After installation, the handoff commands below require no network.

## What you should receive

From Researcher A:

1. Policy bundle directories (e.g. `player_0.arena/`, `player_1.arena/`) containing `policy.yaml` + `payloads/`
2. A `match.yaml` assigning those policies to agents
3. Optional: expected digests listed in `examples/handoff/EXPECTED.md`

You should **not** need: trainer source, checkpoint loaders, private env wrappers,
or arbitrary Python source from Researcher A. Arena 0.1 receives only fixed
categorical templates or scriptable TorchScript tensor actors, explicit
preprocessing/layout, and source-captured evidence.

## Commands

The fenced block below is the canonical clean-room flow. It is tagged
`arena-clean-room` so the automated gate can parse and execute *exactly* these
commands, in order, inside a hermetic sandbox (see
[Machine-parseable command block](#machine-parseable-command-block)). The tag is
an ordinary Markdown info string — renderers still highlight it as `bash` — so
the human-readable instructions are unchanged. Run them from the directory that
holds the received `player_0.arena`, `player_1.arena`, and `match.yaml`.

```bash arena-clean-room
# 1. Workspace
arena init

# 2. Inspect without executing
arena inspect ./player_0.arena
arena inspect ./player_1.arena

# 3. Compatibility before a long run
arena check arena/competitive_rps_v0 ./player_0.arena --role player_0
arena check arena/competitive_rps_v0 ./player_1.arena --role player_1

# 4. Seeded match + trajectories
arena match run ./match.yaml --record --out ./runs/baseline-match

# 5. Inspect trajectories
arena data inspect ./runs/baseline-match/trajectories
```

## Success criteria

- `arena check` exits 0 for each assigned role
- `arena match run` completes requested episodes (`failure_count == 0` unless intentional)
- Trajectory bundle lists episodes with per-step joint observations/actions/rewards/terminals
- Run record (`runs/.../run.yaml`) stores policy digests, seeds, and task identity

## Reproducibility check

Run the same `match.yaml` twice with different `--out` dirs and confirm episode action sequences match (M-01).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ROLE_MISMATCH` | Policy `roles.allowed` ≠ assignment key | Assign to an allowed agent/role or re-export |
| `OBSERVATION_MISMATCH` / `ACTION_MISMATCH` | Space drift vs task | Re-export against current task schemas |
| `action mask required but missing` | Masked policy on unmasked task | Use a masked task or re-export with `masks: none` |
| ImportError for torch/pettingzoo | Extras not installed | `pip install 'arena[all]'` |
| Policy import tries training repo | Bundle incomplete / wrong adapter | Re-export a fixed categorical template or scriptable TorchScript tensor actor; do not ship trainer Python |

## Machine-parseable command block

The [Commands](#commands) section is tagged for automation. The clean-room gate
parses the fenced block whose info string is `bash arena-clean-room`, strips
comments and blank lines, and treats each remaining line as one command to run
in order. The gate fails if:

- the tagged block is missing or empty;
- the commands drift from the canonical U-01 flow (wrong command, wrong order,
  wrong arguments);
- any command shells out to something other than `arena` (e.g. re-installs, calls
  `python`, or references the trainer/checkpoints/source repo).

That makes **documentation drift a test failure**: if this guide and the CLI ever
disagree, CI goes red. Keep the block runnable and in sync with the CLI.

## Automated vs human U-01

Two complementary gates approximate the human clean-room step. Both drive the
**real CLI**; the hermetic gate additionally reproduces the "never-trained
machine, install like a stranger" conditions.

| Layer | Coverage |
|-------|----------|
| `tests/acceptance/test_u01_clean_room.py` | Fast in-process gate. Drives the real CLI end-to-end: `arena policy export`/`verify` from checkpoints, then—after copying only the `.arena` bundles into a fresh directory and deleting the trainer, checkpoints, and workspace (trainer off `PYTHONPATH`)—`arena init`/`inspect`/`check`/`match run --record`/`data inspect` from bundles alone. Uses the developer's editable install. |
| `tests/acceptance/test_u01_hermetic.py` (`slow`) | Hermetic "never-trained machine" gate. Builds a real `wheel` (`python -m build`), creates a throwaway `python3.12 -m venv` with a scrubbed `HOME`/`XDG`/`PYTHONPATH`, installs **only the wheel** (`arena[torch,pettingzoo]`) the way a stranger would (no editable repo path), copies **only** the `.arena` bundles + `match.yaml` + this doc into the sandbox, asserts the trainer package/checkpoints/spec/source repo are absent and unimportable, disables the network for the run, and then executes the commands **parsed out of this document** in order. |
| `tests/acceptance/test_u01_hermetic.py::...docker...` (`docker`) | Same doc-driven flow inside a minimal `python:3.12-slim` image that contains only the wheel, run with `--network none`. Skipped automatically when Docker is unavailable. |
| **Remaining human step** | A second person repeats this guide on a clean laptop/VM, *reads and interprets the prose*, times interventions, and notes any confusing or undocumented step. Automation proves the commands work from bundles alone; only a human can judge whether the prose is understandable and complete. |

Treat any undocumented manual step or trainer-repo dependency as a **release blocker**.
Use the concise real-reader record in
[usability-signoff.md](usability-signoff.md) for the eventual human sign-off.

### How the hermetic gate maps to each dimension of the human step

| Human-step dimension | How it is approximated | Residual (not automatable) |
|----------------------|------------------------|----------------------------|
| "A machine that never had the trainer" | Fresh `venv`/container, scrubbed env, no repo on `sys.path`; trainer package + checkpoints deleted and asserted unimportable; a negative control (`import trainer_repo`) must fail | A genuinely different physical host/OS/arch; real user account state |
| "Installs Arena like a second person" | Installs the built **wheel** (not `pip install -e .`), resolving `[torch,pettingzoo]` extras, into an empty environment | Whether public PyPI actually serves the pinned deps on release day |
| "From the received bundles alone" | Only `*.arena` + `match.yaml` + docs copied in; absence of trainer/spec/repo asserted | — |
| "Follows docs/clean-room.md" | Commands are **parsed from this file** and executed verbatim, in order; drift fails the test | Whether the surrounding prose is *understandable* to a human |
| "Without direct intervention from the author" | No chat, no hidden setup, no repo import; only what a recipient receives | Human judgment about ambiguity, missing context, friction, and timing |
| "No hidden network dependency" | Network disabled during the run (`--network none` in Docker; loopback-only socket guard + `PIP_NO_INDEX` in venv) so any hidden download fails | Perfectly airtight sandboxing on every OS |

## Technique 3 (design only): a fresh LLM agent as a clean-room reader

The hermetic gate proves the *commands* work from bundles alone, but it cannot
judge whether the *prose* is understandable to a newcomer. A third technique
closes that gap conceptually: give a fresh LLM agent **only** the received
bundles and this document — no repository, no chat history, no author — and ask
it to complete the handoff.

Design sketch (not built here; this is a non-CI supplement):

- **Inputs:** the same hermetic sandbox contents (`*.arena`, `match.yaml`,
  `docs/clean-room.md`) mounted read-only. Nothing from the source repo.
- **Harness:** an agent loop with a restricted toolset — read files, run shell
  commands *inside the hermetic sandbox only*, observe output. The system prompt
  states only "you received these files; follow the included guide," giving the
  model no Arena-specific priors beyond the docs.
- **Success signal:** the agent, using its own reading of the prose (not a
  hard-coded command list), reaches the same success criteria — `arena check`
  exits 0 per role, `arena match run` reports `failures=0`, and a trajectory
  bundle with `run.yaml` + `episode_*.json` exists.
- **What it adds over techniques 1–2:** it exercises *comprehension*. If a step
  is ambiguous, out of order, or assumes unstated context, the agent stumbles
  where the scripted gate would not — surfacing exactly the "undocumented
  assumption" a human reviewer would catch.
- **Caveats:** nondeterminism (a passing run does not prove the docs are always
  clear; a failing run may be model error, so use best-of-N and treat it as a
  usability signal, not a hard gate); cost and latency (real model + compute per
  run); infra (sandboxed shell access for an autonomous agent is a security
  surface — keep it network-isolated and scoped to the sandbox); and prompt
  sensitivity (results shift with model and phrasing). For these reasons it
  belongs in a periodic, human-reviewed "doc usability" job, **not** in the
  blocking CI suite.

The residual after all three techniques is small but real: a human on genuinely
foreign hardware, judging clarity and friction in ways a deterministic gate and
even an LLM reader cannot fully certify.
