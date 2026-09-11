# Review: `codex/jlens-scaling` @ `0ffb9dd` (on main `68e9b58`)

Reviewer: the maintainer's integration agent, 2026-09-11. Read against
`docs/JLENS-SCALING-IMPLEMENTATION-HANDOFF.md`, §5 and §8 of
`docs/JLENS-FITTING-PILOT-AND-SCALING-HANDOFF-2026-09-11.md`, and §4 of
`docs/REVIEW-JLENS-PILOT-OPERATIONS-BRANCH-2026-09-11.md`. Five commits,
62 files, +3,815/−76. Main has not moved, so a fast-forward is available.
The handoff asks for a scientific review of the new numerical control flow,
and this is one.

## 1. Verdict

**Landable by fast-forward with one landing fix (F1).** The slice
implements the agreed order faithfully: a benchmark that measures speed and
numerical agreement together in isolated subprocesses; explicit global row
selection with deterministic disjoint partitions bound into the checkpoint
identity; fixed-budget rounds materialized as immutable child requests and
topped up through reviewed durable jobs; a merge that adds raw float32 sums
in a deterministic order, refuses overlap by global row coverage, and stamps
partial results; a stopping rule specified exactly as §8.4 asked, with the
window carried through checkpoints and never applied per shard; a held-out
assessment that compares two registered lenses' readouts against each other
and against the final residual; and the registered-provenance repair from
the previous review's N4. The reference accumulator loop is proven unchanged
after removing exactly the three declared stopping hooks. What the branch
got wrong is operational, not scientific: the merge is a cpu-class
operation, cpu-class operations run inside the controller process, and the
merge as written loads each shard's sums and means whole. On this site the
controller is a 16 GB, one-core job, and a 27B merge needs about 20 GB. The
fix is to stream one layer at a time and take the mean in place; it lands
with this review.

## 2. Verified independently

| Check | Result |
|---|---|
| Ancestry | `main` (`68e9b58`) is an ancestor of `0ffb9dd` |
| Unified gates and audits (`check-generated.py --audits`) | every generator matches and every audit passes; the fitting-loop audit now strips exactly `control.before`, `control.observe`, `if control.reached: break`, the third checkpoint-trigger operand and the `stopping_state` keyword before comparing to `878bca2`, with an accumulator mutation control and a changed-hook control |
| Public scan, whitespace, vocabulary in the diff and all five commit messages | clean |
| Merge estimator | raw sums added in ascending (first global row, checkpoint hash) order, count summed over fitted rows only, mean = sums / count; the fixture compares against serial fitting and the reference `merge` within 1e-6 as §8.2 asked, and bitwise equality is not claimed |
| Overlap and lineage | coverage is by global row set, so a shard plus its own continuation, or a source plus an earlier merge containing it, refuses; the fixture covers both |
| Stopping semantics | N consecutive available updates strictly below threshold; skipped rows do not advance the window; an unavailable statistic (zero-norm running mean) clears it and fitting continues to the cap; the window travels in `state.json` and a continuation without it refuses; per-shard stopping is refused at config time |
| Placement of cpu-class operations | `compute: cpu` selects the local executor, which runs in the controller process (`scientific_execution.plan`); the site's controller job is 16 GB, one core |
| Full Python suite on the tip (main venv, cwd = worktree/Server) | see §5 |
| Full serial Xcode beta suite on the tip (`TEST_RUNNER_STEERLAB_TEST_PYTHON` set, external derived data, coverage mapping off) | see §5 |

## 3. What the branch does

**Selection and rounds** (`jlens_fit_selection.py`, `jlens_round.py`,
`api/jlens_rounds.py`). A fit may name explicit ascending global row
indices and a shard stamp (index, count, start row, row count, layout, and
a plan hash over the fitting settings plus partition); both join the
checkpoint identity. A round takes a published fitting request, a shard
count, a concurrency cap, a start row and a layout, and materializes N
immutable child requests over one global budget that counts skipped rows.
Top-ups are reviewed actions on the round job: status, plan, submit, cancel,
merge-plan, merge-submit; capacity is measured on this controller; an
uncertain submission reserves a slot and is never retried; the round's
children execute against the original staged capsule.

**Merge** (`jlens_merge.py`). Each input must be a completed run with a
report, final checkpoint and lens whose hashes agree; identities must match
on everything but row selection, shard, stopping and driver provenance;
coverage must be disjoint. The merged run carries raw sums and count in a
checkpoint, the mean lens, a report listing every source's run, hashes,
rows and count, the expected and missing rows, and a partial flag, plus an
artifact description so registration is unchanged.

**Stopping** (`jlens_stopping.py`). Rule `{threshold, window, minPrompts}`;
statistic named `max-layer-relative-frobenius-running-mean-change`; the
report says which rule ended the fit and the final value.

**Benchmark** (`jlens_benchmark.py`). Cases over dimension batches, kernel
policies and compilation, each in a fresh subprocess with private compiler
caches; per-prompt and mean matrices compared to the batch-1 baseline
within researcher tolerances; rows per hour, telemetry, kernel dispatch and
hardware recorded; a pinned benchmark report lets a fit's cost review show
a measured extrapolation.

**Kernel policy** (`jlens_kernel_policy.py`). `current` or `torch`; the
latter rebinds the model module's fused-kernel names to their inspectable
torch fallbacks for the fit and restores them after. Bindings are recorded
as configuration; dispatch is still an observation. A `[jlens-kernels]`
extra pins candidate versions on Linux only, outside every lock.

**Assessment** (`jlens_assessment.py`). Two registered lenses, a held-out
corpus, the first eligible positions per row up to a cap; per layer, the
Jensen–Shannon divergence and top-k overlap of readouts between the lenses
and of each against the final residual, through the model's own final norm
and unembedding; the report says whether the corpus hash equals a lens's
fitting corpus.

**Provenance** (`fit_artifact_provenance.py`). A registered lens whose
config file is a matching SteerLab fit or merge report carries the
reference commit, kernel hash, driver hash and report hash; anything else
stays unknown.

**Bounded hash reuse** (`input_hashes.py`). Within one plan, submit or
worker verification, a file's hash is computed once and its inode, size,
times and mode are rechecked on reuse and at scope exit; nothing persists
across operations.

## 4. Findings

**F1 — the merge cannot run where cpu-class operations run.** `jlens-fit-
merge` is bound with `compute: cpu`, which selects the local executor, and
the local executor runs inside the controller process. The merge loaded
each source's `sums.safetensors` and `jacobians.safetensors` whole (6.6 GB
each for the 27B) on top of a 6.6 GB accumulator, then built a second full
mean set for saving: about 20 GB at peak, against a 16 GB controller job.
The first live merge would have killed the controller. Landed with this
review: the merge opens each file with `safe_open` and reads one layer at a
time, keeps only the accumulator resident, and divides in place after the
raw-sum checkpoint is written, so peak memory is one matrix set plus one
layer. The mutation-during-read fixture is adapted to hook the per-layer
reader. The guide should still tell the researcher that a merge needs one
matrix set of host memory on the controller (about 6.6 GB for the 27B).

**N1 — the assessment reloads every lens matrix for every row.**
`lens_store.load_layer` is called inside the row loop, so a 16-row
assessment over 63 layers reads each 100 MB layer 16 times per lens. It is
correct and bounded, but on Lustre that is roughly 200 GB of reads for the
default settings. Hoisting the loads out of the row loop costs two matrix
sets of host memory on a GPU node, which is available.

**N2 — the checkpoint-versus-mean check is exact.** The merge refuses a
source whose published mean is not bit-identical to its checkpoint sums
divided by its count. True today because the fit computes the mean the
same way; a future dtype or accumulation change to the fit must keep it so
or relax this check deliberately.

**N3 — concurrency counts every active scientific job.** The round's free
slots subtract all non-terminal scientific jobs on the controller, not only
this round's children. Conservative and stated in the plan's scope text.

**N4 — `[jlens-kernels]` is a pin list, not an environment.** The handoff
says so. The candidate versions are plausible for the transformers series
in the runner environment but unresolved against its lock; a resolution and
a numerical comparison on the target belong to live acceptance, and the
kernel policy stays `current` until then.

**N5 — the reference-kernel commit is now carried through registration.**
The previous review's N4 is closed; a lens registered from a matching fit
report records `referenceCommit`, `kernelSHA256`, `driverSHA256` and
`fitReportSHA256`, and the pilot's lens can be re-registered to gain them.

## 5. Landing shape

Fast-forward to `0ffb9dd`, then one landing commit carrying F1, its test
adaptation and this review, with both suites re-run on that commit. Shipped
Python and the compiled identity change, so the app and its payload are
rebuilt together. The cluster push still waits for the running continuation
job, and the controller restart after it.

Suite results on `0ffb9dd`:

- Python: 6,414 passed, 9 skipped, 8 warnings (matching the handoff's claim).
- Swift: `TEST SUCCEEDED`, 290 SteeringKit and 4,623 ExperimentKit tests.

Suite results on the landing commit:

- Python: 6,414 passed, 9 skipped, 8 warnings.
- Swift: `TEST SUCCEEDED`, 290 SteeringKit and 4,623 ExperimentKit tests.
