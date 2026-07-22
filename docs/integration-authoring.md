# U-03 integration author template

An integration is supportable only when another user can add and qualify it without
editing RLX core dispatch.

1. Implement one registered interface: `TaskPackager`, `EvalProvider`, or
   `ExternalStoreAdapter`.
2. Keep optional imports inside the selected adapter; prove native operation with the
   external package blocked.
3. Define content identity: runtime/schema revision, provider config digest, or mirrored
   byte digests. Never use a mutable display name as identity.
4. Add a smallest frozen fixture, fail-loud unknown/incomplete tests, failure accounting,
   and a successful end-to-end path.
5. For tasks, add a trace suite and run `rlx task verify-equivalence`; declare tolerances.
6. Run `rlx adapter qualify ...` and retain its JSON report with release evidence.
7. Document install, one runnable command, security/trust boundary, scope, non-goals,
   overhead, and how to uninstall/disable the optional integration.

Copy [integration-usability-signoff.md](integration-usability-signoff.md) for a human
reader. The automated approximation is covered by the 0.3 task/provider/store acceptance
tests and the offline-core gate.
