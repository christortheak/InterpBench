# J-lens assessment reuse and scaling-review follow-up

- Date: 2026-09-11
- Branch: `codex/jlens-assessment-reuse`
- Base: main `224de64`
- Code commits: `41da0ce`, `5f71ffa`, and `a8d080d`
- Review addressed: [scaling branch review](REVIEW-JLENS-SCALING-BRANCH-2026-09-11.md), N1–N3.

## What changes for the researcher

Assessment uses the same request, lenses, text, token positions, and readout
metrics. It now reads and places each lens layer once per assessment, instead
of once per row and again on the device for every token chunk. The input review
explains temporary storage and lens-matrix size before submission. This avoids
repeated multi-gigabyte lens reads without requiring two entire lenses in memory.
Actual large-model speed and peak memory still need measurement on the runner.

Two smaller changes make existing safeguards easier to understand. A merge
whose checkpoint sums disagree with its published mean identifies the source
run and layer and explains how to select matching output. A fitting-round
review identifies the scientific jobs using capacity, including unrelated jobs,
and lists uncertain shard submissions separately. The concurrency rule and the
exact-value check are retained.

## N1: bounded matrix residency

The assessment has two phases:

1. Forward each usable corpus row once. Hooks copy only the selected source
   and final-layer token activations to CPU, preserving their dtype. Save one
   safetensors file per usable row in a private temporary directory under
   `.steerlab/jlens-assessment-state/` in the execution root. Short rows retain
   their skipped status. Remove the hooks before comparison.
2. Read and place the two matrices for one source layer. Compare every usable
   row in its original order, retaining the existing eight-position chunks,
   matrix orientation, final normalization, unembedding, and accumulation.
   Release that pair before loading the next source layer.

`jlens_assessment_inputs.py` owns selected-activation capture and payload
estimates. `jlens_assessment.py` owns comparison and reporting. No full lens
cache, new configuration field, fitting estimator change, or qualification claim
is introduced.

Let `R` be the considered row cap, `P` the smaller of the selected-position cap
and `maxSeqLen - skipFirst - 1`, `L` the number of distinct captured layers
(sources plus the final target), and `D` the hidden width. The storage review
budgets float32 activations even with half-precision weights, since residuals
may retain higher precision. Temporary tensor payload is budgeted at
`R × P × L × D × 4` bytes; one row at `P × L × D × 4`. The float32 matrix pair
is `2 × D² × 4` bytes. Actual activation dtypes are preserved, and their bytes
are reported after capture.
These are payload bounds, **not peak host/GPU memory or total disk requirements**.
Model weights, ordinary forward activations, vocabulary logits, conversion and
transfer copies, allocator caches, and file headers also consume resources.

For usable inputs, matrix reads and placements are `2 × source-layer count`,
independent of row count. An all-short corpus reads no matrices. Completed
reports add actual staged tensor bytes, staged row count, and logical layer read
and placement counts. A placement on CPU may be a no-op; these are not hardware
transfer measurements. Temporary activations are removed on success and ordinary
exceptions. A killed process or machine restart may leave scratch files; managed
recovery and cleanup are outside this change. Completed run evidence is retained.

The tradeoff is selected-activation files, CPU/device copies, and a capture
phase before readout comparison. There is no claim of a measured throughput gain
or of numerical equality on every GPU kernel from the CPU fixture alone.

## N2 and N3: keep the contracts, explain them

**Merge consistency.** Keep the existing `torch.equal(sum / fittedRows, mean)`
predicate. The regression changes one float32 value by one ULP and updates its
fixture file hash: ordinary tolerance-based equality would accept it, but merge
still rejects it before creating a run. This is exact numeric tensor equality,
not a new byte-identity check. Any future change to stored precision or division
semantics needs a deliberate migration. The source comment and shared guide also
clarify that CPU merge runs in a child within the controller's allocation, sharing
its memory budget, rather than in the controller's Python process.

**Round capacity.** The same all-active-scientific-jobs calculation and uncertain
submission reservations remain. Reviews add a stable, job-ID-sorted `capacity`
object containing the limit, occupied slots, active job IDs/kinds/statuses, round
membership, uncertain shard indices, and a plain summary. This explanation sits
outside the existing review hash: an unrelated job moving from submitted to
running, without changing available slots, adds no new reason to refuse. The
existing shard state, available slots, and child plans still guard submission.
No automatic retry, concurrency increase, scheduler override, or job cancellation is added.

## Surfaces and documentation

- The engine owner returns assessment resource estimates through the existing
  managed `operationReview`, and round actions return the capacity explanation.
  Both CLI clients and HTTP use those existing routes and envelopes.
- `FittingReviewSummary` displays assessment storage and its limits in the Mac
  method review. `DiagnosticLifecycleSheet` shows the capacity summary and job
  list beside the round controls; the complete JSON remains available.
- The maintained method guide under `WorkspaceSeed/prompts/method-guides/jlens.md`
  is regenerated into the Python package and compiled Swift resources. Agent
  instructions therefore describe the same storage and queue behavior.
- The shipped Python identity changes. Rebuild the app and bundled payload
  together; update the engine and restart its controller at an agreed safe point.
  This branch does not install the app, modify environments, or deploy a runner.

## Validation and review

The focused fitting, scaling, assessment, and round tests pass. The runtime
baseline is a byte-pinned source fixture, verified against the reviewed commit by
the AST audit, so the numerical tests also work in a shallow or exported clone.
New coverage includes exact report comparison with the landed `224de64` owner using different
lenses, multiple rows and token chunks, skipped rows, and a nontrivial unembedding;
one load/placement per lens layer; release of each pair before the next; float16,
bfloat16, and float32 capture; cleanup after capture and readout failures; exact
merge mismatch rejection; unrelated-job capacity; and stale-plan refusal.

`scripts/ci/audit-jlens-assessment.py` compares the unchanged distance body and
token-chunk arithmetic with `224de64`, allowing only explicit substitutions for
the relocated placement, per-layer totals, and function arguments. Negative
controls change divergence scaling and the accumulation operator. This audit is
part of `check-generated.py --audits`; it does **not** claim that the new capture
and storage orchestration is a mechanically unchanged body.

Final validation on code commit `a8d080d` (followed only by this handoff):

| Check | Result |
|---|---|
| Full Python suite | 6,426 passed, 9 skipped, 8 warnings; 243.36 seconds |
| Full serial Xcode beta suite | `TEST SUCCEEDED`: 290 SteeringKit and 4,625 ExperimentKit tests |
| Generated declarations, packaged guides, and compiled Python identity | Pass |
| Built CLI reference | All 17 generated regions match |
| Historical AST audits and new assessment arithmetic/origin audit | Pass, including negative controls |
| Swift bridge gates, normal and release | Pass |
| Public scan and `git diff --check` | Clean |

The suites ran serially. Swift used Xcode beta, the installed Metal toolchain,
external derived data, `CLANG_COVERAGE_MAPPING=NO`, and
`TEST_RUNNER_STEERLAB_TEST_PYTHON` pointing to the existing test environment.
Python ran with this worktree's `Server/` on `PYTHONPATH`; no environment was
installed or modified. The nine Python skips and eight warnings match main.
The aggregate gate was `check-generated.py --audits --cli <built-helper>`.

## Live acceptance still belongs to the running agents

After review and the agreed deployment window, compare assessment results on a
fixed small held-out set with the previous implementation on the intended CUDA
model. Record elapsed time, peak host/GPU memory, and storage use. Verify capture,
readout, report collection, and scratch removal, then run the intended assessment
budget. Keep CPU numerical fixtures distinct from large-model qualification.

The standing scaling handoff still governs kernel/dimension-batch measurements,
the resolved CUDA environment, sharding/continuation and merge, stopping, and
held-out interpretation. This branch does not resolve review N4 through a pin
list alone, and review N5 remains closed by the landed provenance work.

Landing remains the maintainer's independent diff review, both suites, and the
AST gates, coordinated through the user. Main and running cluster jobs have not
been changed by this branch.
