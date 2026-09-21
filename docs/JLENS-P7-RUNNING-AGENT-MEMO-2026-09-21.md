# Memo: finishing J-lens acceptance and P7

Date: 2026-09-21
Review baseline: main `800d1e5`
Audience: running agents, coding agents supporting acceptance, and independent reviewers
Status: recommendations following the running agents' status report; not a new execution or deployment authorization.

## 1. Main recommendation

Continue the pending mixed-corpus lens assessments, but proceed with the remaining
P7 acceptance independently. J2's direct comparisons and J3's live acceptance
resolve the specific implementation questions that previously held up P7. A new
positive lens result is not a prerequisite for testing probe cancellation,
continuation, panel isolation, the app journey, or overhead.

The report and underlying notes provide useful positive evidence: completed probe
capture/training/evaluation, independent score recomputation, matching token streams
for uninstrumented/read-only/zero-action conditions within each tested backend,
and measured score changes consistent with applied interventions on MPS and CUDA.
The MPS scoring defect found and fixed during live testing demonstrates the value
of this program. Preserve the original failed run and the successful rerun.

This memo is based on the committed report and local evidence notes. The live
scheduler was not polled in preparing it. “Pending” below means pending according
to those records; reconcile actual jobs and collected artifacts before submitting
anything new.

Companions:

- [Current status report](JLENS-AND-P7-STATUS-HANDOFF-2026-09-21.md).
- [Original closure and P7 handoff](JLENS-CLOSURE-AND-P7-QUALIFICATION-HANDOFF-2026-09-13.md).
- [J4 decision](JLENS-QUALIFICATION-DECISION-J4-2026-09-13.md).
- [J3 independent review](REVIEW-JLENS-ASSESSMENT-READOUT-BRANCH-2026-09-13.md).
- [Mixed-corpus merge review](REVIEW-JLENS-MIXED-CORPUS-MERGE-BRANCH-2026-09-17.md).

## 2. Work to do now

| Priority | Running agents | Coding support | Required result |
|---|---|---|---|
| First | Reconcile pending mixed-merge/assessment jobs and receipts; collect finished results | Address a concrete execution blocker if found | Actual pending list; no duplicate submissions |
| First | Run the bounded P7-C cancellation/continuation test and panel-seat isolation leg | Inspect cancellation/state handling and fix demonstrated defects | Completed evidence preserved, incomplete responses distinguished, fresh state verified |
| Next | Assemble the P7-D surface matrix and perform the app walkthrough | Inspect the policy-study freeze limitation and Results presentation | Honest, usable paths across app, both clients, and HTTP |
| Next | Measure P7-E workloads with repeats and dispersion | Provide a focused replay/timing diagnostic if current tools cannot measure them | Reported costs, without invented performance targets |
| Close | Update qualification coverage and limitations | Regression checks for landed fixes | Independent review of updated J4 and P7 statements |

Use the researcher's existing authorizations and approved resource budgets.
Do not infer new permission to increase a fitting budget, raise steering strengths,
generate datasets with other agents, restart occupied services, or remove artifacts.
Bring any additional commitment as a concrete, bounded proposal. Ordinary read-only
reconciliation and already-authorized acceptance should continue without repeatedly
asking for the same permission.

## 3. Pending J-lens research: finish it with a precise interpretation

The status report describes the mixed merge and assessment on three held-outs as
in progress. For each job, record the actual state, input and lens hashes, code
version, GPU, result location, and custody status. A payload push does not prove
that a running controller or worker is using that version.

Compare the existing general-text lens with the mixed lens on the same positions,
using the J3 float32 readout and plain-residual baseline. If the approved assessment
budget permits, include the already-fitted domain-only lens. Reusing that artifact
is useful; another full fit is not required for this comparison.

Interpretation requirements:

- Report the actual contribution counts and weights. With 828 general-text and
  300 domain rows, equal row weighting is approximately 73.4% and 26.6%, not an
  equal-corpus mixture. Verify counts of successfully fitted rows in the report.
- Adding domain text changes both corpus composition and total fitting budget.
  An improvement does not by itself isolate domain matching as the cause. Mark
  this as an exploratory comparison; a controlled budget comparison can be a
  later research question, not a new requirement to spend GPU time now.
- Report absolute divergence and overlap, as well as differences and crossover
  layers. A relative gain when both methods remain near the divergence ceiling
  may have little practical readout value. Do not reduce the result to one
  “first winning layer,” especially if the advantage is small or non-monotonic.
- Keep general-text results alongside domain results to expose tradeoffs. Retain
  negative and inconclusive outcomes; do not tune the mixture repeatedly against
  the same held-out set while continuing to call it untouched final evaluation.
- Lens-to-lens readout agreement does not establish that they recover the same
  subspace. That claim needs a matrix/subspace analysis. Readout prediction also
  does not establish accurate causal direction transport.

Add a dated J4 supplement reflecting the full 27B fit and these later assessments.
Retain the original decision as history. The supplement should distinguish
successful full-size execution from agreement with an independent full-size
derivative reference, which has not been established merely by fitting more rows.
Mixed-GPU scientific equivalence and direction-transport usefulness remain separate
qualification questions. Have the independent reviewer assess the updated scope.

## 4. P7-B: complete coverage without requiring a favorable outcome

Finish the panel-seat leg with at least two deliberately different policies or
measurement selections. Check seat and turn attribution, reset between responses,
and absence of score/state leakage. Verify evidence against the declarations,
including omitted events and failed actions.

Change the completion criterion from “a behavioral outcome at a dose that shows
one” to **a declared behavioral assessment with interpretable results**. No detected
target behavior at strength 8 is a legitimate observation. It does not establish
that the dose is below a known effective threshold. Do not escalate until a
positive result appears and then present that selection as confirmatory evidence.

A separately specified positive-control leg may be useful to verify sensitivity.
If proposed, state its bounded settings, outcome measure, and stopping rule before
running it, and label dose selection as exploratory. Use an independent behavioral
outcome rather than the same probe score that drives the policy. Qualification
requires correct operation and honest evidence, not a policy that improves behavior.

Two interpretation corrections belong in the evidence notes:

1. Different MPS and CUDA generated texts do not establish a defect, but the
   runs do not isolate bf16 kernel rounding as their cause. Hardware and runtime
   versions differ, and a shared integer seed does not demonstrate a shared
   sampling stream across devices. State the demonstrated result: within each
   tested environment, baseline/read-only/zero-action streams match.
2. Identical generated tokens are useful noninterference evidence, but do not
   alone prove tensor non-mutation or RNG-state preservation. Link the existing
   controlled runtime tests and retain the scope distinction. After policy paths
   diverge, a threshold arm need not reproduce the separate fixed or plain arm's
   text: its activation history is different. Check its rule on its own recorded
   inputs, and use same-input controls for action arithmetic.

The small final-test probe result is promising pilot evidence, not general
construct validity. A single shuffled-label fit below chance is not a robust null
distribution. Keep it as one recorded control; repeated predeclared permutations
are needed only if making a stronger claim about performance relative to that null.
Do not expand dataset work automatically to satisfy a product acceptance test.

## 5. P7-C: cancellation, continuation, and evidence custody

Use a bounded multi-response run. Ensure one response is complete while another
is in progress, cancel, then follow the documented continuation or retry route.

Record and verify:

- The completed response and its measurement/decision bytes remain unchanged and
  are not duplicated or remeasured on continuation.
- Incomplete or failed responses remain identifiable and do not become completed
  resume keys; partial failure evidence stays inspectable.
- A restarted response begins with fresh response-local provider state. Phase 1
  does not promise arbitrary provider-state restoration midway through generation.
- A subsequent response or panel seat does not inherit canceled state or hooks.
- Collection verifies archive custody and evidence structure before publication;
  imported counts reconcile with completed, partial, failed, and omitted events.
- Retries after an uncertain submission reconcile the durable job first, rather
  than creating an unobserved duplicate.

Probe capture/training retries create new runs; no optimizer checkpoint resume is
claimed. Test capability mismatch and changed-input handling on isolated copies,
not by disrupting a shared controller. Keep the original outputs immutable.

Do not preemptively implement a new cancellation system. Inspect the existing
owners, exercise the supported path, and fix a concrete missing invariant with a
regression test if the evidence exposes one.

## 6. P7-D: the freeze limitation needs a product decision

The current qualification study used an explicitly approved forced freeze because
the variant battery-evidence gate requires evidence that policy-agent battery
execution does not support. Its resulting non-citable stamp is correctly recorded.
That is acceptable as transparent instrument-testing evidence; it is not a smooth
normal workflow for a researcher conducting policy studies.

Ask the coding/reviewing agents to trace this route across both freeze owners and
the battery execution owner, then propose a narrow, reviewed resolution:

- Distinguish concrete integrity/execution failures from missing capability
  qualification. Missing evidence should have a clear, scoped interpretation.
- Define what a sampled-response policy study can legitimately claim and how
  users see its missing battery evidence before and after execution.
- Never present a policy-free baseline battery as a test of the policy-bearing
  agent, manufacture a successful battery result, or suppress a genuine mismatch.
- Preserve historical manifests, force stamps, and completed runs. A new policy
  for future studies must not silently reclassify the old qualification run.

This is a design recommendation, not permission to remove the gate or weaken
integrity checks. Do not require full policy-aware battery implementation merely
to investigate a more honest, usable qualification status. The product principle
is guidance for honest research, with refusal reserved for concrete problems.

Complete the four-surface matrix using actual operations, not endpoint counts.
Test Results inspection of the J3 report, mixed-corpus contribution provenance,
and probe/policy summaries. Check that response preview limits do not conceal the
complete evidence summary or omissions. Use the running agent's UI access where
available; involve the researcher for usability feedback and choices, without
treating every mechanical UI check as something only the researcher can perform.

## 7. P7-E: descriptive performance measurements

Honor the researcher's decision to report measurements without acceptance targets.
The current unsynchronized per-decision host timings are contextual measurements;
they do not establish GPU execution latency or end-to-end speed ratios.

Measure the six configurations from the closure handoff: no instrumentation,
read-only scores, zero-action policy, fixed policy, conditional policy, and bounded
opt-in activation retention. Record warm-up, repeats, order, workload, and dispersion.
Report prompt processing, decode throughput, wall time, device and host memory,
evidence bytes, and transfer/import cost where measurable. Explain unavailable
metrics rather than substituting a different quantity under the same label.

Use a fixed-workload/replay comparison to isolate implementation cost, alongside
ordinary generation for user-visible cost. Interventions can change output length;
report tokens and elapsed time. Use device synchronization for timing where needed,
and account for CPU transfers/scoring. Optimize only a demonstrated bottleneck,
without silently changing measurement precision, schedule, or semantics.

## 8. Coding work: distinguish immediate support from larger follow-ups

**Good small next patch:** bounded hash reuse inside one fitting-round action.
The current action invokes planning and submission while the hash-cache context
is managed by inner operations. Inspect the smallest correct shared scope; prove
one read of an unchanged large input within that action, detection of mutation or
replacement, and a fresh verification scope for the queued worker. Scope-exit
checking alone is not sufficient if a job has already been submitted: review the
ordering around publication/submission and test changes before that boundary.
Do not extend this into persistent receipt-based trust or cross-request caching.

**Immediate product follow-up:** the freeze/qualification distinction above,
plus a targeted Results inspection. These affect the researcher experience more
directly than a general infrastructure expansion.

**Separate slices if needed:** durable asynchronous planning/staging, and bulk
cleanup. They are worthwhile but not uniformly small changes. Planning jobs need
durable identity, retry/idempotency behavior, drift checks, cancellation semantics,
and progress/error reporting through every client. Moving blocking work off one
thread does not by itself solve submission-lock contention or lost results.

Cleanup must distinguish disposable execution copies and exports from original
research artifacts; protect shared dependencies, active jobs, and resumable state;
and verify the exact custody relationship before applying a reviewed plan. A
receipt for one output does not by itself prove all related staged inputs are
unused. The reported storage cost justifies this work, but not manual deletion as
an implicit part of P7 acceptance.

For evidence from a controller that no longer exists, record the exact blocked
import/reconciliation case. A design should distinguish verified artifact custody
from unverifiable scheduler/job status. Do not invent live reconciliation or drop
hash/structure checks. Give this its own reviewed resolution if it blocks P7-C.

## 9. What to return

Return an updated status table and compact evidence index after each substantive
acceptance leg. Use **passed, failed, pending, and unsupported**, and preserve failed
runs alongside fixes and reruns. Include exact build/runtime identities, input and
artifact hashes, actual resource use, custody receipts, and deviations from the plan.

The final report should make it possible to answer: can a researcher author the
agent and measurements, run them, recover from interruption, retrieve the evidence,
and understand what actually executed and what remains unqualified?

Independent review of the updated J4 record and P7 support statement remains owed.
Code fixes follow the established process: isolated branch from current main,
small commits, full diff review, both suites serially, generated-resource checks,
and applicable scientific audits. Research data and detailed private evidence stay
in the workspace; public repository text must remain free of site/host identities,
credentials, and study-specific material. No existing authorization is expanded by
this memo.
