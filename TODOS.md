# Deferred Arena product work

These are deliberate post-1.0 expansions, not hidden 1.0 release gaps. Each needs
its own RFC and qualification owner before it enters a stable claim.

| Item | Why deferred | Revisit trigger |
|---|---|---|
| Windows support | No maintained CI, package-handoff, or user-evidence owner | A named owner can sustain the full capability × platform matrix |
| Linux arm64 and macOS x86_64 stable support | Current evidence does not cover them | Native runners and source-free handoff are available |
| Hosted accounts/catalog/control plane | Changes Arena from local protocol/tooling into a service business | Repeated user demand cannot be met by user-owned stores |
| Distributed/online RL training | Requires collection, replay, orchestration, and new failure semantics | A separate trainer/runtime RFC proves a bounded use case |
| Arbitrary OpenSpiel catalog support | Semantic qualification is per game/family | A game has an owner and frozen trace/legality evidence |
| Streaming, sharded, compressed datasets | Current verified episode files are adequate for the 1.0 corpus | Corpus scale makes materialization the measured bottleneck |
| CA, revocation, transparency, and hardware keys | Detached signatures intentionally use user-supplied trust anchors | A real organizational trust-lifecycle requirement appears |
| Sandboxing untrusted Python/providers | Process isolation is not an OS security sandbox | Arena accepts a threat model requiring adversarial-code execution |
| Shell completion, manpages, GUI, editor integrations, telemetry | Not necessary for the first verified handoff | Measured DX studies show one is a dominant friction source |
