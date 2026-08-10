# Arena

**Local-first interoperability for RL artifacts.**

[![CI](https://github.com/almanzalex/Arena/actions/workflows/ci.yml/badge.svg)](https://github.com/almanzalex/Arena/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](pyproject.toml)

Arena is a Python CLI and SDK for **portable policies, populations, evaluations,
and release evidence**. It defines content-addressed artifacts, checks whether
they compose, runs seeded matches and evaluation suites, and preserves lineage
across native and qualified external runtimes—without replacing trainers,
environment servers, or artifact hosts.

That is useful inside one lab (train → eval → claim on a clean machine) and when
sharing artifacts with another party.

Current distribution: **Arena 1.0.0rc1**. The final `v1.0.0` tag stays blocked until
the release-commit evidence gates in [docs/1.0-readiness.md](docs/1.0-readiness.md)
are attached to that exact commit.

---

## Why it exists

Adjacent tools solve nearby problems; they usually do not own portable policy
identity, compose-checks before execution, or match/eval provenance binding:

| Tool class | What it owns | What it usually does not own |
|---|---|---|
| Trainers (RLlib, CleanRL, …) | Learning | Cross-repo policy handoff |
| Env APIs (Gymnasium, PettingZoo, OpenEnv) | Interaction | Artifact identity + eval lineage |
| Experiment stores (W&B, MLflow, HF Hub) | Bytes / runs | Semantic compose-check before execution |
| Datasets (Minari, …) | Offline data | Match/eval provenance binding |

Arena fills that gap: export, verify, and re-run policies, populations, and
evaluation claims with digests and lineage.

---

## Quickstart

```bash
python -m pip install 'arena[quickstart]==1.0.0rc1'
arena demo handoff --out ./arena-demo
arena inspect ./arena-demo/restored-policy.arena
```

After install, the demo is source-free and network-free. It exports a reference
policy, verifies it, mirrors through `file://`, pulls to a new path, proves the
digest is unchanged, and prints an evaluation-intent digest. The destination is
staged transactionally so interruption cannot look like a finished handoff.

More flows: [docs/1.0-user-flows.md](docs/1.0-user-flows.md).

---

## What you can do

- **Export / verify** portable policies (templates or BYO TorchScript) without
  the trainer import path.
- **Match & evaluate** with seeded reproducibility, failure accounting, and
  non-transitivity-aware reports.
- **Populate & cross-play** versioned policy sets with a sampling ledger.
- **Bind external systems** (OpenEnv, OpenSpiel, Gimitest, HF/OCI/W&B/MLflow)
  behind registries; unknown kinds fail with an extension recipe.
- **Qualify & doctor** support claims via machine-readable evidence
  (`arena doctor`, `arena adapter qualify`, packaged support matrix).

```bash
arena --version
arena doctor --json
arena schema list --json
```

Support truth (what is stable vs preview) lives in
[`arena/support-matrix.json`](arena/support-matrix.json) and is summarized below.

| Capability | RC status | Final 1.0 condition |
|---|---|---|
| Core identity/inspect, native runtime, `file://`, quickstart | stable | Claimed-platform release CI |
| OpenSpiel frozen qualified cases | stable | Claimed-platform release CI |
| OpenEnv | preview → target stable | Fresh separate-service qualification |
| Gimitest | preview → target stable | Non-no-op isolated-interpreter qualification |
| Hugging Face | preview → required stable | Fresh credentialed immutable-revision round trip |
| OCI, W&B, MLflow | preview | May remain preview; never simulated into a live claim |

---

## Install

```bash
# From PyPI (release candidate)
python -m pip install 'arena[quickstart]==1.0.0rc1'

# From a checkout
python -m pip install -e '.[dev]'

# Optional integrations
python -m pip install 'arena[openenv]'     # external task runtime
python -m pip install 'arena[openspiel]'   # frozen game adapter
python -m pip install 'arena[gimitest]'    # worker deps; install gimitest separately
python -m pip install 'arena[hf]'          # Hugging Face mirrors
python -m pip install 'arena[wandb]'
python -m pip install 'arena[mlflow]'
python -m pip install 'arena[completion]'  # optional richer argcomplete tab completion
# OCI uses the ORAS CLI and its normal login credentials.
```

Core stays small (`pyyaml`, `numpy`). Heavy dependencies are extras.
Static completion needs no extra: `eval "$(arena completion bash)"` (or zsh/fish).
`arena help` covers install, handoff, completion, and naming topics.

> **Name note (PyPI collision risk):** This project’s PyPI distribution is
> currently `arena`. That short name is easy to confuse with unrelated packages
> such as `diambra-arena` (fighting-game envs) and `rl-arena` (competitive RL
> envs)—different products, different APIs. Prefer a pinned install
> (`arena[quickstart]==1.0.0rc1`) and confirm with `arena --version` /
> `arena doctor`. If collisions cause real install harm, deferred rename
> candidates (CLI would stay `arena` unless a coordinated break is planned):
> `arena-rl`, `arena-interop`, `rlx-arena`, `portable-arena`. See
> [TODOS.md](TODOS.md) and `arena help naming`.

---

## Documentation

| Start here | |
|---|---|
| [1.0 user flows](docs/1.0-user-flows.md) | Executable producer/consumer journeys |
| [1.0 readiness](docs/1.0-readiness.md) | What blocks `v1.0.0` |
| [1.0 RC local evidence](docs/1.0-rc-local-evidence.md) | Latest local proof record |
| [Releasing](docs/releasing.md) | Signed release procedure |
| [Clean-room handoff](docs/clean-room.md) | Second-machine install guide |
| [Adapter qualification](docs/adapter-qualification.md) | Evidence required before “supported” |
| [Docs index](docs/README.md) | Full map of guides, milestones, RFCs |

Milestone records (0.2–0.5) and RFCs remain under [`docs/`](docs/) and
[`rfcs/`](rfcs/). They are the historical contract trail, not the landing page.

---

## Development

```bash
python -m pip install -e '.[dev]'
ruff check .
pytest -q                       # fast suite (slow/docker deselected)
pytest -m slow -q               # hermetic wheel + clean-room gates
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Security reports: [SECURITY.md](SECURITY.md).

---

## Deliberate non-goals

Arena is **not** a hosted service, trainer replacement, universal OpenSpiel
catalog, malware sandbox for untrusted Python, or silent Elo ranking system.
Live remote qualifications require the user’s own credentials and are never
conflated with `?simulate=` evidence. Deferred work is listed in
[TODOS.md](TODOS.md).

---

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Citation

If you use Arena in academic work, see [CITATION.cff](CITATION.cff).
