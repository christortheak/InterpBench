# Review: `codex/probe-policy-evidence` @ `4996b4c` (P3–P6 stack) merged with main `adf8b3f`

Reviewer: the maintainer's integration agent, 2026-09-13. Read against the
phase-1 plan, `docs/PROBE-STUDY-MEASUREMENTS.md` (P3),
`docs/PROBE-INTERVENTION-RUNTIME.md` (P4), `docs/INTERVENTION-POLICIES.md`
(P5), the P5 handoff, and `docs/PROBES-INTERVENTIONS-P1-P6-REVIEW-HANDOFF.md`
(P6). Eight commits on the stack, 106 files, +5,671/−343 against main. The
stack was cut from `3457d89`; main has since gained the managed lens-closure
fix (`37279e4`) and two documents, so this is **not a fast-forward**. The
landing tree is a merge of main into the stack; its one conflict was the
generated Python identity, resolved by regeneration.

## 1. Verdict

**Landable by merge with no landing fix.** The stack does what the four
contracts say and stays within the boundaries the plan set. P3 adds study
measurements that read named residual sites inside the existing generation
hook session without adding a forward pass or consuming sampling RNG. P4
introduces one named runtime that orders pre-action readings, the unchanged
legacy action chain, and post-action readings explicitly, replacing hook
registration order, on both engines. P5 adds immutable policies whose
decisions come from a snapshot of the pre-action tensor, bounded fixed,
threshold, and affine rules or a trusted provider, a separate policy action
phase (joint-subspace removal then ordered additions), and token constraints
that intersect and never unmask. P6 makes the evidence transportable and
honest: requirements outside the hashed study document, admission on every
submission and chat path including the controller's proxy to its worker, a
guarded queued entry point, version-2 records that distinguish requested
from acknowledged actions, streaming validation at import, and one offline
descriptive summary shared by every surface.

The audits hold. Every existing scientific owner is unchanged apart from
the declared, audited extensions (the agent schema and scenario executor
for P3/P5, one descriptive-report block in the analysis workflow for P6).
Legacy steering arithmetic is untouched on both engines: `apply_legacy` and
`applyLegacy` are the same loops the hooks ran before, called from the same
place, and the hot path with no subscriptions differs by one attribute test.

## 2. Verified independently

| Check | Result |
|---|---|
| Ancestry and landing tree | stack tip `4996b4c` contains `3457d89`; main `adf8b3f` is not an ancestor; merge commit `6556a35` = stack + main, one conflict in `PythonClientIdentity.swift` resolved by `check-generated.py --write`; the lens-closure fix and its test are present in the merged tree |
| Unified gates and audits on the merged tree | every generator matches; all audits pass, including the extended lazy-import audit (proves the original migration at `b1190f7`, then compares every body outside `run_scenario`, `ModelVariant`, and the two declared policy-admission substitutions) and the managed-owner audit (strips exactly one P6 block from `analysis_workflow`); bridge gates pass |
| Public scan, whitespace, vocabulary in the diff and all eight commit messages | clean |
| Legacy chain unchanged | Python `HookedModel` dispatcher calls `apply_legacy(hidden, interventions, index, offset)` = the previous `for intervention: hidden = intervention.apply(...)`; with no subscriptions the only added cost is `runtime.subscriptions` truthiness; Swift `applyLegacy` is the same loop and the model forward calls it when no runtime is installed |
| Ordering | `Runtime.apply`: preAction readings → decisions from `h.detach().clone()` → legacy chain → policy `residual` under `apply_with_evidence` → postAction readings; matches P4/P5 text; native runtime: preAction → legacy → postAction |
| Policy residual arithmetic | joint orthonormal basis `q` over all ablation vectors at a site, `x − s·(x qᵀ) q` with a per-position fraction `s` that every ablation at the site must share exactly, then additions `+ s·v` in declaration order, float32 math cast back to the activation dtype, non-finite refused |
| Token constraints | biases add first; allowed sets and forced tokens intersect; empty intersection refuses; out-of-vocabulary ids refuse; result checked for NaN, +inf, and all-−inf rows; installed after the finite-logit guard and before Transformers' warpers |
| Acknowledgement | `apply_with_evidence` calls every action's `on_result(True, …)` only after the whole site operation returns, and `on_result(False, reason, …)` if it raises; `appliedStrengths` are copied from `strengths` only on success; row status is `failed` if any action failed, `applied` only when every requested action is acknowledged, else `requested` |
| Decision inputs | same-site residual policies score the snapshot tensor; logits policies read a residual probe captured on the current forward and refuse if the cached reading's position is not the consumed token; provider RNG is `random.Random(SHA256(digest + sorted context))`; provider state is per policy per response and cleared on close |
| Admission | `instrumentation_contract.requirements` scans raw JSON recursively including embedded panel agents; `require` compares with the engine's `capabilities.instrumentation`; enforced on Mac study submit, bundle submit, continuation, chat and battery; on the controller's proxy to its worker (reads the stored agent through the safe path resolver); in `_bundle_execute_command` (swaps the queued module to the guarded `instrumented_bundle`, which refuses before importing the CLI); and in `bundle execute` itself |
| Import validation | evidence bundles' `generations.jsonl` / `turns.jsonl` are validated row by row in staging before the commit; a row over 64 MiB is an explicit error; token alignment, position arithmetic, declaration/hash agreement, bounds, and acknowledgement consistency are all checked; legacy version-1 decisions are accepted and counted, never inferred applied |
| Refusals for unsupported paths | direct-choice scoring and capability batteries refuse a policy agent; measurements refuse a study without sampled text; `variant_injections` refuses policies unless the caller declares support |
| Immutability | new agent versions embed exact policy text and hash; frozen studies and completed runs are never rewritten; failure sidecars use `open('x')` |
| Summary arithmetic | Welford moments with sample standard deviation, nulls for empty groups; groups keyed by condition, agent, kind, label, artifact hash, site, stage; omitted events counted separately |
| Full Python suite on the merged tree (`HF_HUB_OFFLINE=1`) | see §5 |
| Full serial Xcode beta suite on the merged tree | see §5 |

## 3. What the stack does

**P3** (`probe_measurements.py`, `probe_observation.py`, `ProbeMeasurements
.swift`, `StudyMeasurementsView`). An optional `probeMeasurements` study
field (at most 32 probes, bounded readings and activation bytes, explicit
error policy) that enters study identity and freeze; readings scored on CPU
float64 with the same arithmetic as fitting; results embedded in the
response record with absolute positions and token ids; omitted and
unexecuted readings counted separately; failures written as sidecars.

**P4** (`steering/runtime.py`, `hooks.py`, `ResidualRuntime.swift`, the two
vendored native models). One named runtime per arming handle; readings
subscribe under the response owner; block-input hooks are installed only at
requested layers and reference-counted; a superseded abandonable stream
closes its runtime immediately.

**P5** (`policy_artifacts.py`, `policy_authoring.py`, `policy_execution.py`,
`steering/policy_actions.py`, six CLI verbs, `InterventionPoliciesView`).
Self-contained policy documents; a new agent version per attachment; the
execution owner above; Playground chat executes policies and returns
decision records.

**P6** (`instrumentation_contract.py`, `instrumented_bundle.py`,
`instrumentation_evidence.py`, `InstrumentationSupport.swift`,
`InstrumentationEvidence.swift`, `InstrumentationEvidenceView`, bundle and
import changes). Described in §1 and §2.

## 4. Findings

**No landing fixes.**

**N1 — the stack reverts nothing, but it predates the lens-closure fix.**
The merged tree carries both; the stack's own diff against main shows the
fix as removed only because the stack was cut before it landed.

**N2 — an unmarked bundle is refused on an engine that lacks any of the
three capabilities.** `requireBundleInstrumentation` treats a bundle
without `runtimeRequirements` as possibly carrying an additive field an old
engine would ignore, so it demands full support. Conservative and correct
for mixed-version windows; after this deploy every engine offers all three.

**N3 — `_add_files` reads every JSON member into memory to discover
requirements.** Bounded in practice (study bundles' JSON members are small;
activation datasets are capped at 64 MiB) but worth a size guard if bundles
ever carry larger JSON.

**N4 — expert providers are `exec`'d trusted code.** Stated everywhere it
matters; the RNG and state conventions are recorded in the evidence. Not a
sandbox, and not claimed to be.

**N5 — P7 remains.** No live checkpoint, remote submission, cancellation,
continuation, or Results interaction has been exercised; overhead of
readings and policies on a research model is unmeasured; native MLX policy
execution, padded batching, direct-choice and battery execution of policy
agents are explicitly out of scope and refuse.

## 5. Landing shape

Fast-forward main to the merge commit `6556a35` (it has main as a parent),
then one commit carrying this review. Shipped Python and the compiled
identity change, so the app and its payload are rebuilt together and the
engine is pushed to the cluster; the controller is stopped, so the next
start picks the new build up.

Suite results on `6556a35`:

- Python: 6,580 passed, 9 skipped, 8 warnings (the handoff's 6,579 plus the lens-closure test from main).
- Swift: `TEST SUCCEEDED`, 4,939 passed and 5 skipped of 4,944 (295 SteeringKit and 4,649 ExperimentKit).
