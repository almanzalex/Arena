# RFC 002 — Portable Policy Generalization (Beyond Template Categorical)

**Status:** Proposed (design + prototype-backed)  
**Date:** 2026-07-19  
**Depends on:** RFC 000, RFC 001  
**Scope:** Post-0.1 design for bring-your-own PyTorch policies, serializable preprocessing, custom task handoff, and safe checkpoint ingestion.

> **Implemented-boundary update (2026-07-21):** RLX dispatches portable behavior
> through an **axes + case registry** (`rlx.core.registry` / `rlx.plugins`).
> Registered + qualified cases include Discrete templates, TorchScript BYO,
> complete MultiDiscrete / Dict / diagonal-Gaussian Box, preprocess ops, and
> SuperSuit wrappers. Opt-in (not default, not sandboxed) cases:
> `trusted_source` payload and `entrypoint_bundle` task packaging. Unknown kinds
> fail loud with an extension recipe; mixtures and unscriptable arbitrary Python
> remain non-goals until registered, tested, and qualified.
>
> **Implemented-boundary update (2026-07-20):** RLX implements the T1
> TorchScript path as the preferred BYO payload. It does not claim T2
> (`torch.export`) as a shipping runtime tier. The table below remains design
> analysis except where the registry capability matrix in README supersedes it.
>
> **0.1 release-hardening decision (2026-07-20):** messy scratch probes and
> clean-room regression tests were used to test each boundary. Incomplete
> MultiDiscrete/Dict/Gaussian claims, unscriptable Python without trust opt-in,
> and arbitrary environment source handoff without digests remain explicit
> fail-loud non-goals. `rlx adapter qualify` records the qualification evidence
> required before an adapter/case can be declared supported.

## Problem (what 0.1 actually is)

RLX 0.1’s `custom-pytorch` adapter is a **two-template declarative categorical contract**:

- Architectures: `mlp_categorical`, `gru_categorical` only (`rlx/adapters/policy_custom_torch`).
- Preprocessing: elementwise mean/std (+ optional clip) only.
- Weights load with `torch.load(..., weights_only=True)` into a **reconstructed template graph**.
- Match runner tasks: bundled pilot env or `pettingzoo.*` modules.

That satisfies the pilot pair and clean-room gates for policies that already fit those templates. It does **not** deliver the product slogan “bring your own PyTorch policy” for a lab’s real `nn.Module` (custom recurrence, in-graph masks, frame-stacking wrappers, non-PettingZoo envs).

**Status-quo lab friction today:** a second researcher gets a checkpoint + Slack instructions, then writes a load script that imports the trainer repo, guesses wrapper order, hard-codes role maps, and silently diverges when frame-stack / running-norm / mask conventions drift. RLX 0.1 removes some of that for the two templates, but for bespoke modules it pushes users back to the same hand-written handoff — or worse, into a false sense of portability when elementwise preprocess is declared while frame-stack was required.

### Prototype evidence of the silent-failure class

On a 20-step synthetic stream with a bespoke masked-GRU and a real `running_norm → frame_stack(k=4)` pipeline, an RLX-0.1-style elementwise pad+normalize path produced **10% wrong actions** while still running without error. The messy-trainer adversarial case elsewhere reported ~63% wrong under the same bug class; both show the failure mode is **silent behavioral drift**, not a hard crash.

---

## Design goals

1. **Trainer-free inference** for a wide class of discrete, recurrent, action-masked PyTorch actors — without importing the training repository.
2. **Express-or-fail preprocessing**: composable, serializable transform pipelines; never silently approximate frame-stack / concat / running stats as elementwise mean/std.
3. **Custom task packaging** for non-`pettingzoo.*` Parallel envs, with an explicit trust model.
4. **Safe checkpoint ingestion** as a first-class boundary (complements tactical `weights_only` fixes).
5. **Much easier than today’s handoff scripts** — fewer steps, fail-loud on inexpressible graphs, conformance as the acceptance gate.

Non-goals for this RFC: universal ONNX of every research graph; bit-identical numerics across all hardware; replacing OpenEnv; claiming ecosystem standardization.

### Current capability matrix

| Status | Boundary | Enforced semantics / rationale |
|---|---|---|
| Supported now | Scriptable custom PyTorch actor | T1 `torch.jit.script`, explicit tensor I/O, digest-authenticated payload, source-captured conformance. Trace is opt-in and only appropriate when representative cases prove no frozen control-flow divergence. |
| Supported now | Deterministic bounded `Box` actor | Exact declared shape/dtype/bounds; finite output only; no automatic clipping. |
| Supported only with explicit trust | None | No arbitrary bundled-source inference/task tier is present. TorchScript is trusted-lab code, not a sandbox. |
| Deliberately rejected | `MultiDiscrete` | No shipped factorized categorical manifest defining head slices, per-factor masks, dtype, and seeded draws. Flattening changes the action semantics. |
| Deliberately rejected | Generic `Dict` action | No recursively typed, canonical-key field adapter survives exporter, task, match, and trajectory serialization together. |
| Deliberately rejected | Stochastic `Box` | No portable mean/log-std, canonical RNG, transform/scaling order, or exact/tolerance conformance specification. |
| Deliberately rejected | Unscripable arbitrary Python | T2 is not a runtime tier; T4 would execute arbitrary code and is not a sandbox. Export fails with refactor guidance instead of shipping source code. |

---

## Recommended architecture (summary)

Ship a **tiered portable-policy runtime** and a **preprocess IR**, selected at export time by capability probing + conformance:

| Tier | Payload | When to use | Clean-room fidelity (prototype) |
|------|---------|-------------|-------------------------------|
| **T0** (keep) | Declarative templates + `state_dict` | Policies that already match `mlp/gru_categorical` | Existing 0.1 gates |
| **T1** (0.2 primary) | **TorchScript** (`script` preferred, `trace` fallback) + preprocess IR | Most custom discrete actors with fixed I/O | **20/20 actions match, logit max-abs diff 0.0**, recurrence+masks held |
| **T2** (0.2 secondary) | **`torch.export` ExportedProgram** + preprocess IR | Graphs that script poorly but export cleanly | Same perfect round-trip in proto |
| **T3** (later / optional) | ONNX + pinned ORT | Cross-language / non-PyTorch consumers | **Not round-tripped here** (export deps missing: `onnxscript`) |
| **T4** (escape hatch) | Allowlisted **bundled inference module** + `weights_only` `state_dict` | Research ops TorchScript/export cannot capture | Perfect fidelity; **arbitrary-code trust required** |

**Default export path for “BYO module” in 0.2:** try T1 script → T1 trace → T2 export → refuse with a structured “inexpressible” report (do not fall back to T0 template remapping). T4 only with `--trust-bundled-source` (or lab policy). Preprocessing is **never** optional vapor: either emit a pipeline IR that conformance exercises, or fail export.

```text
Trainer machine                         Clean-room machine
─────────────────                       ──────────────────
nn.Module + wrappers
    │
    ├─ capture preprocess IR ─────────► payloads/preprocess.json
    ├─ serialize actor (T1/T2/T4) ────► payloads/model.(pt|pt2) [/ inference/*.py]
    ├─ record reference cases ────────► payloads/reference_cases.json
    └─ rlx policy verify (source)       rlx policy verify (self)
                                        rlx match run (no trainer repo)
```

---

## 1. Bring-your-own `nn.Module` inference

### 1.1 Mechanisms evaluated

#### A. TorchScript (`torch.jit.script` / `trace`)

| Dimension | Assessment |
|-----------|------------|
| Faithfulness | **High** when script succeeds. Prototype: scripted + traced masked-GRU matched source actions/logits exactly over 20 recurrent steps with masks. |
| Recurrence + masks | Works if hidden + mask are **explicit tensor inputs/outputs** (not Python attributes mutated outside the graph). |
| Safety | Loads via `torch.jit.load` — **no trainer `sys.path`**, no `pickle` of arbitrary Python classes. Still native code; treat as trusted binary for the lab, not internet-untrusted malware sandboxing. |
| Dependency capture | Needs a **pinned torch** (and CUDA/CPU notes) on the manifest; no trainer package. |
| Friction vs handoff script | Removes: “clone trainer, fix PYTHONPATH, find Policy class, guess ctor args.” Adds: one export that freezes the inference graph. |

**Caveats (honest):** `script` fails on dynamic Python, data-dependent control flow, and some custom autograd. `trace` freezes control flow to the example path (mask-always-present worked; optional-mask branches are hazardous). Module classes must be real files (not notebook/`exec` cells) for `script`’s source introspection.

#### B. `torch.export` / ExportedProgram

| Dimension | Assessment |
|-----------|------------|
| Faithfulness | **High** on the same masked-GRU proto (0 mismatches, 0.0 logit diff). |
| Recurrence + masks | Same contract: explicit `(obs, hidden, mask) → (logits, hidden)`. |
| Safety | No trainer imports; PT2 archive load. Newer stack; pin torch major/minor. |
| Maturity | APIs still moving (warnings observed on load). Better long-term IR than TorchScript, slightly sharper export failures. |

**Recommendation:** Support as T2 in 0.2; prefer T1 if both pass conformance, until export packaging UX stabilizes across torch minors.

#### C. ONNX + pinned runtime

| Dimension | Assessment |
|-----------|------------|
| Faithfulness | Potentially high for static discrete actors; **not validated in this prototype** — `torch.onnx.export` failed with `ModuleNotFoundError: onnxscript` in the test environment. |
| Spec posture | Product MVP **explicitly deferred universal ONNX**; still a valid *optional* adapter later. |
| Tradeoffs | Pros: non-Python consumers, ORT pinning. Cons: custom ops, recurrence/mask conventions, dtype/bool gaps, dual-runtime conformance cost. |

**Recommendation:** Optional `onnx-runtime` adapter in **0.3+**, only when export+ORT round-trip conformance is green; never marketed as universal.

#### D. Bundled minimal inference module (source) + allowlist

| Dimension | Assessment |
|-----------|------------|
| Faithfulness | **Highest** for exotic Python (perfect proto match with `weights_only=True` state_dict). |
| Safety | **Executes arbitrary Python** shipped in the bundle. Must be an explicit trust tier (see §4). |
| Dependency capture | Manifest `dependencies.allowlist` + hashed source tree; refuse imports outside allowlist at load (best-effort; not a seccomp sandbox). |

**Recommendation:** Escape hatch for graphs that cannot script/export — not the default.

### 1.2 Inference I/O contract (all tiers)

Freeze a small tensor contract so match runtime stays tier-agnostic:

```text
encode(raw_obs) -> FloatTensor[B, obs_dim]     # preprocess IR
forward(obs, hidden, action_mask?) -> (logits, hidden')
initial_hidden(batch) -> Tensor | None
```

- Discrete MVP: logits → argmax / categorical sample (existing RNG contract).
- Masks: `none | optional | required` unchanged; if the serialized graph expects a mask tensor, missing mask is a **pre-step error** (not silently all-ones unless declared).
- Stochastic sampling stays **outside** the serialized graph (numpy Generator), matching RFC 001 — keeps script/export graphs simpler.

### 1.3 Prototype results (trainer-free subprocess)

Environment: PyTorch **2.9.1**, clean-room `PYTHONPATH` = artifact dir only, trainer module import blocked.

| Mechanism | Export | Clean-room load | Action match | Logit max-abs diff | Recurrence | Masks |
|-----------|--------|-----------------|--------------|--------------------|------------|-------|
| TorchScript `script` | OK | OK | **100% (20/20)** | **0.0** | held | held |
| TorchScript `trace` | OK | OK | **100%** | **0.0** | held | held |
| `torch.export` | OK | OK | **100%** | **0.0** | held | held |
| Bundled source + `weights_only` | OK | OK | **100%** | **0.0** | held | held |
| ONNX | **FAIL** (`onnxscript` missing) | n/a | n/a | n/a | n/a | n/a |
| E2E scripted + preprocess IR from **raw** obs | — | OK | **100%** | — | held | held |

**What broke / limits observed:**

- Defining modules only in `exec`/stdin broke `torch.jit.script` (`OSError: could not get source code`) — export must use importable source files.
- ONNX not exercised end-to-end in this environment.
- Full-module `torch.save(nn.Module)` pickle remains a hostile ingestion path; see §4.

### 1.4 Friction reduction vs lab handoff scripts

| Today (typical lab) | With T1/T2 + preprocess IR |
|---------------------|----------------------------|
| Share checkpoint path + branch SHA | Share one `.rlx` bundle digest |
| Clone trainer + create venv + CUDA dance | Install `rlx[torch]` (+ pinned torch) |
| Write `load_opponent.py` importing `my_proj.models` | `rlx policy verify` / `rlx match run` |
| Manually copy RunningMeanStd / FrameStack order | Pipeline IR embedded; verify fails if omitted |
| Silent wrong actions if wrapper missed | Conformance cases + shape checks fail loud |
| “Works on my machine” recurrence resets | Declared `reset_on` + reference trajectories |

**Estimated steps removed for a successful opponent handoff:** ~5–8 manual steps (clone, path hacks, class discovery, wrapper archaeology, ad-hoc verify) → ~2 (`verify`, `match run`), **when** the policy is expressible. Inexpressible graphs still need T4 or a trainer-side refactor to explicit I/O — that limit should be documented, not papered over.

---

## 2. Preprocessing / wrapper pipeline as a serializable spec

### 2.1 Why elementwise-only is insufficient

RLX 0.1 `Preprocessing` is `(x - mean) / std` with optional clip, applied after pad/truncate to `observation_dim`. Frame-stacking, concat of heterogeneous fields, and running-stat vectors **cannot** be represented. Export that “succeeds” with the wrong preprocess id produces **silent policy drift** (prototype: 10% wrong actions; adversarial messy case higher).

### 2.2 Proposed IR: `rlx.preprocess/v1`

```yaml
preprocessing:
  included: true
  id: "norm_framestack_v1"          # content hash of steps+params preferred
  pipeline:
    version: rlx.preprocess/v1
    steps:
      - op: running_norm
        mean: [...]                  # broadcastable to current tensor
        std: [...]
        eps: 1.0e-8
      - op: frame_stack
        k: 4
        feat_dim: 2                  # required; mismatch => error
        pad: zeros                   # zeros | repeat_first
      # future: clip, concat, one_hot, dict_select, role_map, ...
```

**Execution rules:**

- Ordered list; each op has declared input rank/shape constraints.
- Stateful ops (`frame_stack`, running stats if online) expose `reset()` aligned with policy `reset_on`.
- Unknown `op` → **export/load error** (fail loud).
- Shape mismatch → **error** (prototype: feeding 8-d into `feat_dim: 2` raised `ValueError` / broadcast error).
- Registry lives in RLX; adapters may register extra ops behind plugin namespacing (`lab.foo.bar`) with allowlisted code (same trust tier as T4).

### 2.3 Capture from trainer wrappers (minimal user effort)

Provide a **capture helper** used on the trainer machine (imports trainer — that is fine at export time only):

```python
# trainer-side, once
pipe = rlx.preprocess.capture([
    env.get_wrapper_by_id("NormalizeObservation"),  # or duck-typed
    env.get_wrapper_by_id("FrameStack"),
])
# or explicit:
pipe = rlx.preprocess.from_steps([...])
```

Heuristics (best-effort, never silent):

| Wrapper pattern | Capture |
|-----------------|---------|
| Running mean/std with `.mean`/`.var`/`.std` | `running_norm` |
| Frame stack with `.frames`/`.k` | `frame_stack` |
| `gymnasium.wrappers.ClipObservation` | `clip` |
| Unknown wrapper | **Refuse export** unless user supplies IR or `--allow-unverified-preprocess` (marked `conformance: unverified`, match warns) |

**Export gate:** generate reference cases from **raw task observations** through the captured pipeline + module; clean-room verify must replay the same. If the user only supplies elementwise mean/std while the capture probe detects a FrameStack-like wrapper, export **fails** with a repair message (“declare frame_stack or unwrap”).

### 2.4 Prototype results

- `running_norm → frame_stack` JSON pipeline matched the trainer `FrameStackNorm` exactly (`max_abs_diff: 0.0`).
- Wrong input width failed loud.
- E2E: pipeline + TorchScript module from raw obs → **0/20** action mismatches.

---

## 3. Custom environment / task handoff

### 3.1 Gap

`make_env` today: pilot aliases, a few classic ids, or `importlib` of `pettingzoo.*`. Lab Parallel envs outside that namespace are not portable.

### 3.2 Contract sketch (`rlx.task/v1`)

```yaml
schema: rlx.task/v1
adapter: python-entrypoint          # or pettingzoo-parallel | openenv (later)
name: lab/pursuit_custom
interaction: parallel
entrypoint: package.env:parallel_env   # factory(**config) -> Parallel API
config: { max_cycles: 100 }
payloads:
  source_tree:
    path: payloads/task_src/
    digest: sha256:...
dependencies:
  python: "3.12"
  pip_allowlist: ["numpy==1.26.*", "gymnasium==0.29.*"]
safety:
  trust_level: lab_explicit          # required for arbitrary code
  allows_arbitrary_code: true
roles:  # optional cache; still discovered via describe_task
  ...
```

**Packaging options (prefer in order):**

1. **Entrypoint + source tree** in the task bundle (prototype sketch: `env_handoff_sketch:parallel_env`) — works offline; requires trust grant.
2. **Pinned pip package** name/version on allowlist (no source tree) — cleaner when the env is already published.
3. **OpenEnv / container** (roadmap 0.3) — best isolation; higher overhead; equivalence suite required.

**Safety:** default deny. `rlx match run` refuses `python-entrypoint` unless workspace trust includes the task digest (or `--trust-task-code`). No network during load by default.

**API surface:** factory must expose PettingZoo Parallel semantics used by the match runner (`reset`, `step`, `agents`, spaces, optional `action_mask` in obs dict). AEC deferred to existing 0.2 roadmap.

### 3.3 Friction reduction

Today: “install my private env package from this git URL + this commit + these wrappers.”  
Target: one task artifact + trust prompt + `rlx check` space discovery — still not magic for proprietary simulators with native binaries, but removes path/import archaeology when the env is pure Python.

---

## 4. Safe checkpoint ingestion (trust model)

Ingestion is a **boundary**, not a convenience flag.

| Input kind | Allowed by default? | Mechanism |
|------------|---------------------|-----------|
| `state_dict` / tensor mappings | Yes | `torch.load(..., weights_only=True)` |
| TorchScript archive | Yes (lab trust) | `torch.jit.load` |
| ExportedProgram | Yes (lab trust) | `torch.export.load` |
| Full `nn.Module` pickle | **No** | Refuse; instruct re-export to T1/T2/T4 |
| Bundled `.py` inference / task code | **No** until trust grant | Explicit CLI/workspace ACL + digest pin |
| ONNX | Opt-in extra | Pinned ORT; no pickle |

**Principles:**

1. **Never** `weights_only=False` on untrusted paths in library code.
2. Export-time may use broader loads **only on the trainer machine** under user-supplied `--source`, then re-serialize to a safe payload.
3. Manifest records `payloads.*.digest`, `runtime.tier`, `runtime.torch`, and trust requirements.
4. Tamper check: existing digest verification remains mandatory before act().
5. Conformance status is part of trust UX: `unverified` bundles can inspect but should not silently enter shared eval suites.

This complements (does not replace) tactical hardening of 0.1 loaders.

---

## 5. Manifest / adapter changes (additive)

Extend `rlx.policy/v0alpha1` → `v0alpha2` (or additive fields with forward-compatible readers):

```yaml
runtime:
  adapter: custom-pytorch
  tier: torchscript          # template | torchscript | torch_export | onnx | bundled_source
  python: "3.12"
  torch: "2.9.*"
architecture:
  type: serialized_module    # or mlp_categorical | gru_categorical
  io:
    obs_dim: 8
    action_n: 5
    recurrent: true
    mask_in_graph: true
preprocessing:
  included: true
  id: sha256:...             # of canonical pipeline JSON
  pipeline: { ... }          # or payload ref
payloads:
  model: { path: payloads/model_scripted.pt, digest: sha256:... }
  preprocess: { path: payloads/preprocess.json, digest: sha256:... }
  reference_cases: { ... }
  # tier bundled_source only:
  # inference_src: { path: payloads/inference/, digest: ... }
```

`validate_policy_manifest`: require `architecture` **or** serialized payload per tier; templates keep current required fields.

CLI sketch:

```bash
rlx policy export --adapter custom-pytorch \
  --source ./ckpt.pt \
  --module my_proj.policy:Actor \
  --capture-preprocess env:make_env \
  --role evader \
  --out ./artifacts/evader.rlx

# refuses template remapping if module is not a template
# writes tier chosen by probe + embedded reference cases from raw obs
```

---

## 6. Prioritized implementation plan

### 0.2 (near-term — maximize “much easier” coverage)

1. **Preprocess IR v1** + registry: `running_norm`, `frame_stack`, `clip`, `identity`; stateful reset; fail-loud shapes.
2. **Export probe** for BYO modules: script → trace → export; write tier + digests; **refuse** silent template coercion.
3. **Runtime loader** multiplex on `runtime.tier`; match runner calls encode+forward only.
4. **Conformance**: raw-obs reference cases (P-01…P-04) must pass in hermetic clean-room without trainer repo.
5. **Capture helpers** for common Gymnasium/PZ wrappers; export error when unknown wrappers present.
6. **Ingestion policy** module: centralize load rules; document refuse paths for module pickle.
7. Keep T0 templates for back-compat and the pilot pair.

### 0.2 / early 0.3

8. **Task `python-entrypoint`** adapter + trust ACL + source-tree digests.
9. T4 bundled source allowlist loader (opt-in).

### 0.3+

10. OpenEnv task adapter (existing roadmap) as preferred isolation for heavy/native envs.
11. Optional ONNX tier **after** pinned ORT conformance suite exists.
12. Richer preprocess ops (`concat`, `dict_select`, per-role maps) as real lab needs appear — driven by failed captures, not speculation.

### Explicitly later / never promise

- Universal conversion of every research `nn.Module`.
- Sandboxing untrusted internet policies as safe.
- Bit-identical cross-GPU reproducibility without declared tolerances.

---

## 7. Honest limits

- **Expressibility ceiling:** dynamic Python, custom CUDA ops, and trainer-side masking outside the module may force T4 or code changes.
- **Trace fragility:** control-flow and optional mask paths can look green on the traced example and diverge later — mitigate with diverse reference cases at export.
- **Trust ≠ sandbox:** TorchScript/export/bundled source all execute native/Python code; RLX provides **integrity + consent**, not a malware VM.
- **ONNX:** attractive for portability marketing; **unproven here**; deps and op coverage are real costs.
- **Env handoff:** pure-Python Parallel factories package cleanly; complex simulators still need OpenEnv/containers.
- **Adoption/performance at scale:** not measured; do not claim RLX becomes “the standard.”

---

## 8. Acceptance criteria for this design (when implemented)

| ID | Gate |
|----|------|
| G-01 | Bespoke recurrent+masked module exports via T1 or T2 and passes hermetic verify without trainer repo |
| G-02 | Frame-stack pipeline round-trips; elementwise-only export **fails** when capture detects stack wrappers |
| G-03 | Missing/unknown preprocess op or shape mismatch errors before match |
| G-04 | Module pickle checkpoints refused by default with repair guidance |
| G-05 | Non-PZ Parallel env runs via entrypoint task under explicit trust |
| G-06 | Documented friction: opponent handoff without clone/load script for ≥1 real lab policy beyond templates |

---

## 9. Appendix — prototype session notes

Throwaway prototypes lived under `/tmp/rlx_byo_proto/` (trainer_repo export + clean_room verify subprocess) and were **deleted after RFC authorship**. No changes were made to `rlx/` or `tests/` for this design work.

Key clean-room JSON fields observed:

```text
mechanisms.torchscript_script.action_match_pct: 100.0
mechanisms.torchscript_trace.action_match_pct: 100.0
mechanisms.torch_export.action_match_pct: 100.0
mechanisms.bundled_source.action_match_pct: 100.0
mechanisms.onnx.error: model.onnx not present (export deps missing)
preprocess_pipeline.roundtrip_ok: true
e2e_scripted_plus_pipeline.ok: true
status_quo_elementwise_wrong_pct: 10.0
```
