# Deferred Arena product work

These are deliberate post-1.0 expansions, not hidden 1.0 release gaps. Each needs
its own RFC and qualification owner before it enters a stable claim.

| Item | Why deferred | Revisit trigger |
|---|---|---|
| Windows support | Experimental CI scaffolding only (`windows-latest`, `continue-on-error`); no package-handoff or user-evidence owner; **not** a stable claim | A named owner can sustain green Windows CI, hermetic handoff, and the full capability × platform matrix without `allow-failure` |
| Linux arm64 and macOS x86_64 stable support | Experimental optional-arch CI scaffolding only; current evidence does not cover a stable claim | Native runners stay green without `allow-failure`, plus source-free handoff evidence |
| Hosted accounts/catalog/control plane | Changes Arena from local protocol/tooling into a service business; see [RFC 013](rfcs/013-hosted-control-plane.md) (deferred; local `arena catalog local` stub only) | Repeated user demand cannot be met by user-owned stores |
| Distributed/online RL training | Requires collection, replay, orchestration, and new failure semantics | A separate trainer/runtime RFC proves a bounded use case |
| Hosted accounts/catalog/control plane | Changes Arena from local protocol/tooling into a service business | Repeated user demand cannot be met by user-owned stores |
| Distributed/online RL training | Requires collection, replay, orchestration, and new failure semantics | RFC 011 + CPU spike prove **bounded** collect→bind→offline-train only; Ray/PPO/replay remain deferred |
| Arbitrary OpenSpiel catalog support | Semantic qualification is per game/family | A game has an owner and frozen trace/legality evidence |
| Streaming, sharded, compressed datasets | Current verified episode files are adequate for the 1.0 corpus | Corpus scale makes materialization the measured bottleneck |
| CA, revocation, transparency, and hardware keys | Detached signatures intentionally use user-supplied trust anchors | A real organizational trust-lifecycle requirement appears |
| Sandboxing untrusted Python/providers | Process isolation is not an OS security sandbox | Arena accepts a threat model requiring adversarial-code execution |
| Shell completion, manpages, GUI, editor integrations, telemetry | Not necessary for the first verified handoff | Shell completion + `arena help` topics shipped; revisit GUI/editor/telemetry only if measured DX studies show dominant friction |
| PyPI name ambiguity (`arena` vs `diambra-arena` / `rl-arena`) | Short name is easy to install by mistake; products differ | Documented in README + `arena help naming`. Prefer pinned install + `arena --version` / `arena doctor`. Deferred distribution renames (CLI stays `arena` unless coordinated): `arena-rl`, `arena-interop`, `rlx-arena`, `portable-arena`. Rename only if collisions cause real install harm |
