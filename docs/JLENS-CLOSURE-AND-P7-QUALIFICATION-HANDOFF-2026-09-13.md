# J-lens closure and probes/interventions P7 qualification

Date: 2026-09-13
Review baseline: main `168aee8`
Audience: coding, running, and independent reviewing/integration agents
Status: proposed execution handoff; no new qualification runs were performed in preparing it.

## 1. Outcome and order

Finish a bounded J-lens validation pass, then qualify the probes and interventions
implementation already landed through P6. Do not make P7 wait for a full production
Qwen-27B lens, an exact reproduction of an unavailable published fitting script,
or an open-ended investigation of every early-layer readout difference.

The researcher should be able to create an **agent** from a base model, vectors,
adapters, and optional intervention policies; attach **probes as study
measurements**; run that study locally or remotely; and inspect what was measured,
what actually changed, and what remains uncertain. The app, both CLIs, and HTTP
must expose consistent objects, decisions, and evidence. Equivalent access does
not imply identical arithmetic across execution backends.

| Order | Work | Primary owner | Completion evidence |
|---|---|---|---|
| J1 | Correct the numerical conclusions and inventory actual coverage | Coding + running | Claims mapped to reports, with unsupported claims narrowed |
| J2 | Close direct-comparison gaps | Running; coding supplies reusable diagnostic support | Actual tensor comparisons, identities, and negative controls |
| J3 | Make readout qualification reproducible through managed assessment | Coding, then running | Float32 readout, matched logit-lens baseline, and a small managed acceptance run |
| J4 | Record a scoped J-lens qualification decision | Running + independent reviewer | Explicit supported configurations, limitations, and deferred questions |
| P7-A–F | Qualify the landed probe/intervention journeys | Running + coding for fixes | Live, surface, failure, and performance evidence described below |
| Close | Review and publish an accurate support statement | Independent reviewers through the user | Reviewed changes and a qualification report with no silent gaps |

P7 planning, fixture preparation, and read-only environment inventory can proceed
while J1–J3 run. Start its live acceptance after J4 records the decision. This
handoff describes work; it does not independently authorize new installations,
deployment, paid jobs, controller restarts, or cleanup. Use existing session
authorization where it applies. For additional resource commitments, present a
concrete plan with model, GPU type/count, runtime cap, and storage before asking.

## 2. Sources and established baseline

Read these alongside this handoff:

- [Original J-lens testing program](JLENS-NUMERICAL-REVIEW-AND-TEST-PROGRAM-2026-09-13.md).
- [Numerics decision](JLENS-NUMERICS-PROGRAM-DECISION-2026-09-13.md) and
  [results and work proposals](JLENS-NUMERICS-PROGRAM-RESULTS-AND-HANDOFF-2026-09-13.md).
- [Phase-1 plan](PROBES-AND-INTERVENTIONS-PHASE-1-PLAN.md),
  [P1–P6 handoff](PROBES-INTERVENTIONS-P1-P6-REVIEW-HANDOFF.md), and
  [independent P3–P6 review](REVIEW-PROBES-INTERVENTIONS-P3-P6-STACK-2026-09-13.md).
- [Probe training](PROBE-TRAINING-AND-EVALUATION.md),
  [study measurements](PROBE-STUDY-MEASUREMENTS.md),
  [runtime](PROBE-INTERVENTION-RUNTIME.md), and
  [policies](INTERVENTION-POLICIES.md).

The J-lens tests provide substantial positive evidence: independent tiny-hybrid
Jacobian and finite-difference controls, repeat/order checks, forward-pass
localization of batch-shape differences, native-logit reconstruction checks, and
a completed 1,000-prompt Gemma-3-4B managed fit and assessment. Preserve those
results. The corrections below concern the strength and scope of the conclusions.

P1–P6 landed with the existing lens-closure fix in merge `6556a35`. The independent
review recorded 6,580 Python tests passed, 9 skipped, and 8 warnings; Swift reported
4,939 passed and 5 skipped. Those are historical checks on that tree, not fresh
results on whichever commit is qualified next. The review found no landing fix.
Its outstanding P7 items are real-checkpoint execution, remote lifecycle, manual
Results interaction, and overhead. Its note about JSON dependency scanning is a
bounded-size concern to exercise, not a mandate for an unrelated rewrite.

## 3. J1 — correct the interpretation before extending it

Update the decision and results documents together. Add a dated correction with
links to new evidence; do not modify old run outputs or silently rewrite the
history of what was tested.

1. **Engine versus reference:** the supplied `jlens_numerics_t4b.py` computes
   batch-B-versus-batch-1 distances separately inside the bare and engine paths.
   Matching distance summaries do not establish that the two paths produced
   bitwise-identical matrices. The supplied Qwen T4b report is a forward scan,
   not this engine comparison. State the actual coverage until J2 closes it.
2. **Batch-to-batch equality:** identical agreement records against batch 1 do
   not establish batch 2 equals batch 4, or batch 4 equals batch 64. Require a
   direct comparison before calling those tensors identical.
3. **Logit-lens comparison:** the results table at layer 8 favors our J-lens on
   both divergence and top-10 overlap for both corpora. Remove “does not beat it
   at any depth.” The baseline table samples five layers, not every layer.
   “At chance” also requires a defined chance model; use measured values instead.
4. **Rounding interpretation:** forward differences on dense and hybrid models
   support a hardware/batch-shape numerical explanation. Exact kernel selection
   was not captured, and backward arithmetic may contribute too. Deterministic
   differences are not proven unbiased random draws. Do not infer that averaging
   arbitrary GPU configurations cancels them, or label all bf16 gaps harmless.
5. **Reproduction:** report a successful managed fit and a comparison with the
   published artifact under documented recipe differences. Lower final-prediction
   divergence at all assessed layers does not prove that the larger sample count
   caused the improvement or that the population Jacobian was estimated better.
   Prompt selection, hardware, batch size, compilation, and storage precision
   are not held constant. Published stopping-window semantics remain unknown.
6. **Coverage:** the supplied Qwen T1 report names eight application-text windows,
   not sixteen WikiText rows. The inspected repeatability reports cover Gemma on
   H100 and A100, and Qwen on H100. Add any missing supporting report or narrow
   “both models, both cards.” Identify component subsets as subsets.

Do not describe a tiny-model derivative check as a proof for every checkpoint,
precision, or kernel. Do not treat the final-residual bf16/fp32 readout discrepancy
as a universal error bound on transported intermediate residuals.

## 4. J2 — direct numerical controls

### Coding and diagnostic support

Retain selected comparison tensors, or compute exact hashes over a documented
serialization including dtype, shape, layer, and component order. Persist tensors
when distances must also be computed. Make retention explicit and bounded: a
full lens per benchmark case can be expensive. Reuse retained evidence before
requesting a rerun; missing tensors cannot be reconstructed from summary distances.

For bare versus engine, compare the **same dimension batch** directly, on exactly
the same tokens, positions, source layers, and output components. Match checkpoint
revision, dtype, attention implementation, compile mode, device, TF32 settings,
and reference-kernel identity. Compare batch 1 and one practical batched setting.
For batch-size claims, directly compare the claimed pair, not only their separate
distances to batch 1.

Report exact equality, relative Frobenius, maximum absolute error, cosine where
defined, and zero-denominator behavior. Preserve actual tensors or their hashes.
A mutation that changes one coordinate, and a deliberately permuted component
order, must fail the relevant identity check. Equality of scalar summaries must
not be sufficient to pass the control.

### Running plan

- Start with cached Gemma-3-4B on H100 using a 128-token row and one shorter row
  already known to expose the discrepancy. Repeat the small bare/engine comparison
  on A100 where that two-card claim is retained.
- Add a bounded Qwen/Qwen3.8-27B H100 comparison over declared representative
  layers/components. Do not call a subset a full-lens check. Use the approved
  memory budget; batch 4 previously approached the 80 GB ceiling.
- Compare selected batch pairs directly using saved tensors where possible. Add
  a repeat of one configuration to separate reproducibility from cross-setting
  agreement. No full 27B fit is required for this control.
- Retain the existing float32 Gemma check as useful reference evidence. If the
  corrected controls disagree unexpectedly, investigate the affected case before
  broadening the model or hardware grid.

For a cross-GPU merge recommendation, compare matched small fits across the
specific proposed configurations and a mixed result against a uniform-config
result on the same fixed corpus budget. Preserve corpus weights and deterministic
merge order. If that comparison is deferred, mark mixed-GPU scientific equivalence
unqualified; provenance and successful file merging alone do not establish it.
This limitation need not block exploration or P7.

**Exit:** the direct-comparison claims have actual evidence, or are explicitly
withdrawn; unexpected discrepancies have an explanation or a scoped open issue.

## 5. J3 — managed assessment improvements

### Implement now

1. Add an explicit **readout precision** option without silently changing existing
   requests. Specify whether it covers transport multiplication, final norm, and
   unembedding, and report the actual precision of each. Reuse the same captured
   activations and lens bytes for the paired comparison. Higher-precision readout
   does not turn a bf16 forward pass or fitted lens into a float32 one.
2. Add the **identity-transport/logit-lens baseline at every assessed layer** on
   the same positions and token populations as the candidate and reference lenses.
   Keep between-lens comparison separate from each lens versus final predictions.
   Keep the model's native final distribution as the identified target, and retain
   native-logit/self-comparison controls.
3. Expose these through the managed owner, operation declaration, interview,
   app controls, both clients, API, report, and agent guide. Validate a drafted
   request through the real owner. Preserve the old default encoding and hashes
   where absence of a new option is intended to preserve old behavior; prove this
   with fixtures rather than assuming an additive field is harmless.
4. Present relative matrix differences and readout differences side by side.
   Do not make `JS / relative Frobenius` a scientific sensitivity or condition
   number without a defined perturbation experiment. Remove or contextualize a
   universal tolerance flag rather than replacing it with automatic bf16 approval.

### Run a bounded acceptance

Use the existing local and published Gemma lenses and pinned held-out inputs.
Run through plan, submit, collect, register/import as applicable, and assessment;
verify the enhanced report in the app and programmatically. Start with a small
position budget that includes layers 3–6 and the reported middle-layer comparisons.
Then reuse the existing evaluation population if the approved budget allows it.

For the layers 4–5 discrepancy, inspect transported-vector norms, final-norm
behavior, and fp32 readout on matched positions before attributing it to particular
coordinates. Per-coordinate claims after normalization need an explicit analysis;
the downstream mapping is not simply additive. A remaining use-specific research
question is a limitation, not automatic evidence of an implementation failure.

### Useful follow-ups that do not hold P7 open

- Immutable retained checkpoints at selected row budgets, with row count,
  averaging semantics, hashes, and normal artifact import. A 546/1,000-row nested
  comparison would isolate budget more cleanly, but still is not the unknown
  published prompt set. A rolling final checkpoint cannot recover earlier means.
- A baseline-only assessment if needed as an independent research workflow.
- Stopping-statistic diagnostics. Do not add a new stopping rule merely to imitate
  an unknown published implementation; distinguish convergence of an estimator
  from usefulness for a research outcome.
- Additional corpus windows belong in corpus preparation, with pinned output
  bytes and documented sampling weights. Do not duplicate window selection in
  fitting config as a second, potentially inconsistent owner.
- Direction-transport validation, further early-layer investigations, and full
  Gemma-27B or Qwen-27B fitting remain separately budgeted research work. Predicting
  final tokens well is not equivalent to accurate local directional transport.

## 6. J4 — decision to proceed

The J-lens closure report should say which estimator, model, precision, batch,
hardware, and readout combinations were tested, and distinguish:

- derivative correctness controls;
- reproducibility within a configuration;
- numerical agreement across configurations;
- successful managed execution and custody;
- usefulness for readout or direction-transport research.

Proceed to P7 when J1 is corrected, J2's direct controls are accounted for, and
J3's bounded managed assessment is verified and reviewed. Any unresolved finding
must have a stated impact. An unexplained estimator or shared-runtime discrepancy
needs resolution; an unmeasured full-size lens or mixed-GPU equivalence does not
prevent qualification of the independent probe/intervention workflow.

## 7. P7 — qualify the existing product, without expanding its scope

### Execution boundary

Portable probe capture, study measurements, and dynamic policies execute through
**Python Compute**, locally or remotely. App and native-CLI access to those owners
is part of parity. Native MLX portable study measurements and dynamic policies are
not implemented. Existing native legacy steering and the native runtime must
retain their behavior, but P7 must not advertise new MLX feature support.

Also outside this slice: padded/multi-sequence policy execution, direct-choice
scoring or capability batteries for policy agents, auxiliary models, execution-flow
changes, and resuming arbitrary provider state mid-generation. Unsupported routes
must offer the supported path and must not silently run an unmodified agent.
Missing hardware qualification alone remains a disclosed limitation, not a refusal.

### P7-A — exact builds and a small end-to-end probe journey

Record checkout, installed app/payload, client environment, controller, queued
child/worker, model revision, tokenizer/template, dependencies, and hardware. A
push does not replace a running controller's code. Coordinate authorized deployment
and restart with running work, then verify the worker actually advertises the
capabilities required by the bundle. Do not deliberately disrupt a shared service
to test mixed versions; use an isolated fixture or test runner.

Use an existing cached, supported small research checkpoint, preferably Gemma-3-4B
where available, with separate local Python and CUDA bindings. Do not assume MPS
and CUDA activation artifacts have interchangeable precision or identity. A second
architecture is needed only for a broader claim, or to investigate an observed issue.

Take a small, reviewed binary dataset with group-separated fit, selection, and
final-test roles. Use existing data or approved fixtures; offer authoring help
instead of automatically delegating dataset generation. Execute managed capture,
linear fitting, nonlinear fitting, evaluation, collection, and library discovery.
Compare with constant-class and shuffled-label controls. Preserve preprocessing
and score orientation. A nonlinear probe need not outperform on every real task;
its nonlinear capacity already has an independent fixture and should not be claimed
from a chance accuracy improvement on a small pilot.

**Evidence:** request and input hashes, actual model/site/position bindings,
reports, group counts, artifact round trips, independent score spot checks, and
no training or threshold selection on final-test data.

### P7-B — readings, policies, and independent outcomes

Run matched conditions with no instrumentation, read-only probes, an enabled
zero-action policy, a nonzero fixed policy, and a threshold or bounded adaptive
policy. Use the same actual sampling seed, tokens, and generation settings for
noninterference tests: changing study identity can change derived seeds. Verify
RNG state and actual tensor non-mutation in the controlled diagnostic; matching
output text alone is insufficient. Record any nondeterministic baseline floor.

Exercise block input/output sites, pre-action/post-action readings, prompt-end and
decode schedules, and multiple probes. Check consumed versus predicted positions
and token IDs by hand on a short example. The final sampled token is not observed
unless naturally consumed in a subsequent forward pass. Do not add a pass to fill
that gap or interpret pre-action readings as globally unsteered activations.

For a few retained, bounded activations, independently calculate expected addition
and subspace removal, including composition with legacy actions. Check recorded
requested strengths against acknowledged applied strengths and actual deltas.
Exercise a logit bias or token constraint and its conflict path. Cover a pinned,
trusted custom provider with local RNG and state, without claiming a code sandbox.

Include two panel seats and successive responses with different settings; verify
state isolation and reset after success, failure, and cancellation. Use independent
behavioral outcomes to compare fixed and conditional agents. Lowering the very
probe score used to control an intervention is not evidence of behavioral benefit.

**Evidence:** alignable responses, readings, decision/action records, independent
arithmetic checks, and correctly labeled exploratory behavioral results.

### P7-C — remote execution, cancellation, continuation, and custody

From an authoring client, package every declared probe, policy, provider, and asset;
plan and submit to an isolated runner or approved cluster; monitor the queue; fetch
and import evidence; and resolve it locally without hidden checkout dependencies.
Exercise the per-request GPU selection path if the approved cluster supports it.

Cancel a bounded multi-response run after at least one completed response and
while another is in progress. Inspect partial evidence and then use the supported
continuation/retry path. Completed responses must retain their original evidence
and not be duplicated or remeasured. Failed partial responses must not masquerade
as completed resume keys. Verify a fresh response has fresh policy state.
Probe capture/training has no optimizer or activation checkpoint resume: retry is
a new run. Study continuation likewise does not promise mid-response provider-state
restoration or exact recovery of interrupted token generation.

Test worker capability mismatch and changed-input rejection on isolated copies.
Verify required instrumentation is never silently dropped. Test conflicts, nonfinite
scores, selected error handling, bounded recording omissions, and failure sidecars.
Verify archive hashes and structural evidence validation before import publication;
a valid archive hash alone is not enough. Keep original artifacts untouched.

**Evidence:** plans, job IDs, deployment identities, status transitions, partial
and resumed run references, custody receipts, and evidence-count reconciliation.
Cleanup uses existing reviewed plans and cluster policy; this adds no blanket
permission to delete scratch, checkpoints, or runs.

### P7-D — surface parity and researcher accessibility

Maintain an operation matrix with **app / native CLI / portable CLI / HTTP** columns.
For each row record the real owner, input identity, output identity, evidence link,
and status: passed, failed, pending, or unsupported. Cover capture/train/evaluate,
probe library/inspection, measurement review/save, policy review/publication,
new-agent attachment, submission/monitoring, collection/import, and evidence analysis.

Use a shared pilot artifact across the surfaces; expensive computation need not be
repeated four times merely to demonstrate access. Do exercise each claimed route
against its actual owner, including HTTP service-role boundaries. Use current help,
operation declarations, and packaged guides for commands rather than inventing new
verbs. Existing entry points include `science measurements-review/save`, the
`science policy-*` family, and `science evidence-analyze`; training is managed
`probe-capture`, `probe-train`, and `probe-evaluate` authoring/execution.

Complete one app-driven journey and one app-free, agent-driven journey. In the app,
inspect model selection, file selection, validation guidance, progress, cancellation,
library refresh, study measurements, policy attachment, and Results filters. Read
JSONL in the app. Compare the complete evidence summary with the limited response
preview and verify neither conceals omissions or failures. A command-line check
does not substitute for clicking through these paths.

The researcher should understand what a probe measures, what data is needed, where
it executes, and whether a policy actually changed anything. Explain training,
selection, and final-test roles; signed scores are not calibrated probabilities.
Avoid unexpected training, downloads, or dataset-authoring agents. Use the Oxford
comma and the established terms agent, probe, measurement, policy, and template.

**Evidence:** completed matrix, screenshots or interaction notes, exact repairs
for unavailable paths, and matching artifact identities and analysis results.

### P7-E — measured cost on representative hardware

Benchmark no instrumentation, read-only scores, zero-action policy, fixed policy,
conditional policy, and a small opt-in activation-retention case. State the number
of probes, sites, tokens, evidence budgets, and recording schedules. Warm up, use
several repeats, vary order, and report dispersion, not just a single timing.

Separate prompt processing, decode throughput, end-to-end wall time, peak device
and host memory, evidence bytes, and transfer/import time. Compare a fixed-workload
or replay diagnostic for implementation cost and ordinary generation for user-visible
cost; interventions can change output length, so report both tokens and duration.
Synchronize GPU timing where required and distinguish it from host event timestamps.
Include CPU scoring/transfers, which can dominate per-token probe overhead.

Set practical latency/memory targets with the maintainer before interpreting the
measurements; do not invent a universal acceptable percentage afterward. If overhead
is high, identify and fix the actual cause or expose honest scheduling/budget
guidance. Do not silently change scoring, precision, or observation frequency.
Local MPS and CUDA results qualify their recorded settings, not every model or GPU.

### P7-F — regression, review, and release statement

Address observed defects in small commits, each with a meaningful regression test.
Preserve the old legacy action arithmetic and historical artifact bytes. Run the
full Python and native suites serially on the final code, required generated-resource
and public-content checks, and the existing audits/negative controls. Read the full
diff. Use AST audits for claimed mechanical moves; scientific changes need independent
numerical tests instead of an unchanged-body claim.

Use Xcode beta, the installed Metal toolchain identifier, scratch under `/private/tmp`,
`CLANG_COVERAGE_MAPPING=NO`, and the real test Python environment as described in the
P1–P6 handoff. Do not run Python and Swift suites concurrently or change the bundled
Python sources during native verification. Coordinate app closure for suite execution
and reopen the matching build for manual acceptance. Record failures and reruns.

The independent reviewing/integration agents, working through the user, review and
land the changes. Neither a coding agent's self-review nor a green suite establishes
independent approval or live scientific qualification.

## 8. Deliverables and completion criteria

Produce one J-lens closure report and one P7 qualification report. Each needs:

- exact code/build identities, configuration, hardware, and input/artifact hashes;
- planned versus executed cases, with pass/fail/pending/unsupported distinguished;
- a compact evidence index pointing into the research workspace;
- deviations, resource use, failures, repairs, and remaining limitations;
- researcher-facing statements of what works and through which execution route.

Keep site names, credentials, study-case vocabulary, and research data out of public
repository files and commit messages. Store detailed private evidence in the workspace;
public reports can use neutral case IDs. Secrets remain in existing credential stores.
Frozen studies, completed runs, and prior evidence bytes remain immutable. Start any
implementation branch from current main and incorporate intervening reviewed fixes.

P7 is qualified when representative local Python and remote CUDA journeys pass,
the four surfaces reach the declared operations, evidence survives cancellation and
collection, actual actions and alignment are verified, and measured cost and current
limits are visible to researchers. If a required environment is unavailable, issue
a partial qualification report and name what is pending; do not call it a full pass.

No requirement here to demonstrate that every probe predicts well, every policy
improves behavior, or every fitted J-lens is scientifically useful. The requirement
is a reliable instrument that lets an honest researcher find out, with clear
guidance and evidence they can inspect.
