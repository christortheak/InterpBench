# J-lens scaling implementation handoff

Date: 2026-09-11. Branch: `codex/jlens-scaling`, based on main `68e9b58`.
Review and integration remain with the maintainer's auditing agents, through
the user. This branch does not deploy an engine, restart a controller, install
kernel packages, run cluster jobs, or change the installed app.

## What changes for the researcher

The new path is: prepare text, measure a small pilot, approve a fixed total row
budget, execute independent shards, merge their contributions, and compare
readouts on held-out text. These operations share the same owners through the
app, both command lines, and HTTP. The shipped J-lens guide explains the
choices and gives the round-coordination request shapes.

A researcher chooses a corpus, a budget, and what evidence would change their
conclusions. The product partitions rows, pins inputs, coordinates durable
jobs, verifies merge coverage, and records the numerical runtime. A benchmark
report records the GPU name, CUDA build, and compute capability where available,
and isolates compiler caches for each subprocess. It does not change defaults. A stable running mean or agreement between
lenses does not confer qualification or establish behavioral validity.

Read alongside the standing order in
[the scaling handoff §8](JLENS-FITTING-PILOT-AND-SCALING-HANDOFF-2026-09-11.md)
and [the pilot operations review §4](REVIEW-JLENS-PILOT-OPERATIONS-BRANCH-2026-09-11.md).
Those documents describe the live observations and the original priorities;
this document states what this implementation does and what remains untested.

## Commits and review scope

- `27b4728`: preserve registered fitting provenance and repair staging guidance.
- `4c41cae`: benchmarking, bounded hash reuse, fixed-budget rounds, merging,
  stopping, assessment, surface declarations, UI controls, and regression tests.
- `66f85e5`: expose benchmark budgets and merge coverage before execution,
  record benchmark hardware, and isolate compiler caches per case.
- `c2a7156`: render the reviewed budget, pilot estimate, and partial coverage
  in plain language in the Mac authoring screen, with presentation tests.
- The final handoff commit records completed verification below.

Read the whole diff before landing. The new numerical control flow deserves a
scientific review; this is not presented as an entirely mechanical refactor.
The original accumulator and reference-kernel loop remain checked against the
historical baseline after removing only the explicitly enumerated stopping
hooks. The mutation controls change the actual accumulator and a stopping call.

## Owners and invariants

| Area | Owner | Contract |
| --- | --- | --- |
| Registered provenance | `experiment/fit_artifact_provenance.py` | Carry reference commit, kernel hash, driver hash, and report hash only from captured matching fit/merge reports; unknown external provenance remains unknown |
| Input verification | `experiment/input_hashes.py` | Reuse hashes only for unchanged regular files within one bounded operation; check inode, size, times, and mode on reuse and exit; no cross-request trust |
| Benchmark | `experiment/jlens_benchmark.py` | Same small corpus and reference estimator, fresh subprocess per case, per-prompt and mean comparisons, visible failures, no fitted artifact published |
| Kernel choice | `experiment/jlens_kernel_policy.py` | Explicit reversible model-owned fallback selection; current policy preserves bindings; no installation or unproved dispatch claim |
| Row selection | `experiment/jlens_fit_selection.py` | Sorted global indices, deterministic partition, reviewed shard stamp, selection bound in checkpoint identity |
| Round materialization | `experiment/jlens_round.py` | One global budget including skipped rows, immutable child requests, explicit per-controller concurrency cap |
| Round coordination | `api/jlens_rounds.py` | Reviewed top-ups through ordinary durable scientific jobs, no automatic retry, uncertain submissions reserve capacity |
| Merge | `experiment/jlens_merge.py` | Disjoint global coverage under identical model/corpus/numerical identity, raw float32 sums, fitted counts, deterministic order, source rechecks, explicit missing rows |
| Stopping | `experiment/jlens_stopping.py` | Strict consecutive valid updates below threshold, fitted-row minimum, persisted window, no per-shard stopping |
| Held-out assessment | `experiment/jlens_assessment.py` | Same captured activations, actual final norm/unembedding, per-layer distributional comparisons, no causal or qualification claim |

The Mac authoring screen displays a **Compute and coverage** section, including
measured pilot extrapolations and explicit missing-coverage notices.
Drafts and execution plans carry an `operationReview` with the actual benchmark
row budget, round partition, merge coverage, or assessment position/layer scope,
so these choices are visible before submitting computation.

The four new managed operations are `jlens-fit-benchmark`, `jlens-fit-round`,
`jlens-fit-merge`, and `jlens-fit-assess`. Their descriptors live under
`docs/techniques/operations/`; generated owners, input roles, catalogs,
interviews, and Swift resources derive from these. The guide's maintained
source is [WorkspaceSeed/prompts/method-guides/jlens.md](../WorkspaceSeed/prompts/method-guides/jlens.md).

`jlens-fit` adds explicit compilation and kernel policy, optional global row
selection, optional stopping, and an optional pinned benchmark report. Old
identities normalize to compilation off, current kernel policy, no declared
optional packages, and no subset or stopping. Source driver hashes remain
provenance; numerical settings and dependency versions remain equality-bound.
There is no automatic checkpoint migration after a benchmark comparison.

The round helper uses the original verified capsule for child execution and
merge output, so users do not re-upload shard outputs to merge them. Only
internal owner calls may supply this execution-root binding; public requests
cannot supply an arbitrary root override. The queued worker re-verifies its
capsule and actual request inputs.

## Deliberate boundaries and corrections

1. **Hashing:** this narrows redundant reads within an operation. It does not
   remove all three plan, submit, and worker verification passes. Subprocess
   validation and archive verification can still reread bytes. A persistent
   capsule-receipt cache has not been introduced.
2. **Kernel environments:** `[jlens-kernels]` is an opt-in Linux extra with
   direct candidate pins, separate from `[all]` and the existing platform locks.
   It is not a resolved or qualified CUDA environment. The running agents must
   review resolution/build prerequisites on the target environment, produce
   its lock, and measure numerical agreement before adoption. The existing
   `bootstrap --with-jlens` is unchanged because it also acquires curated lens
   artifacts; enabling kernels must not cause unrelated downloads.
3. **Round scheduling:** explicit top-ups, not background autonomous refill.
   Capacity is measured on this controller, not across unrelated controllers.
   Site scheduler policy remains authoritative. A lost submission response
   does not authorize a retry.
4. **Continuation:** individual shards use the existing explicit checkpoint
   continuation path. The round helper knows its original child jobs. For
   continued shards, select their latest completed runs in an explicit merge
   request; automatic adoption of continuation jobs into the original round
   has not been added.
5. **Merge order:** ascending first global row, then checkpoint hash. For one
   partition this is shard-index order. Grouped raw float32 addition is compared
   to serial fitting and the reference weighted-mean merge within `1e-6` on
   analytic fixtures; no claim of bitwise equality on production workloads.
   Including a source and its continuation, or a source and an earlier merge
   containing it, is rejected by overlapping global contribution coverage.
6. **Zero denominator:** contrary to the earlier handoff's suggested abort,
   a zero-norm prior layer mean makes the stopping statistic unavailable,
   clears the window, and continues to the cap. It never counts as convergence.
   This retains exploration while keeping the measurement truthful. Invalid
   or non-finite fitting matrices still refuse under the original contract.
7. **Assessment:** the first eligible token positions up to the explicit cap
   receive equal weight. Jensen–Shannon divergence and top-k set overlap measure
   readout agreement, including each lens versus the final residual. Different
   corpus hashes do not prove independence. Domain conclusions and instrument
   qualification still require separate assessment.
8. **Storage:** no broad fitting-run or checkpoint cleanup. Benchmark scratch
   retains baseline and current case matrices for comparison, and normal exit
   removes them; interruption can leave scratch. Completed fit runs, imported
   evidence, and frozen studies are never rewritten.

Kernel candidate versions were selected from the official package releases:
[flash-linear-attention 0.5.2](https://pypi.org/project/flash-linear-attention/0.5.2/)
and [causal-conv1d 1.7.0](https://pypi.org/project/causal-conv1d/1.7.0/).
Their inclusion is not evidence that either supports this model's backward
path on the deployed CUDA stack. No packages were installed for this branch.

## Verification

Final Python suite on `66f85e5`: **6,414 passed, 9 skipped, 8 warnings**
(235.94 seconds). The subsequent code commit changes only Swift presentation
and its tests; Python sources are unchanged.

Final Xcode beta suite on `c2a7156`: **TEST SUCCEEDED**, with **290 SteeringKit
and 4,623 ExperimentKit tests**. The new presentation suite ran, along with the
real Python-client parity checks. The build used the supplied Metal toolchain
and `/private/tmp` DerivedData. No app was installed.

Generated-resource checks (including the built CLI), historical AST audits,
both bridge gates, the public scan, and `git diff --check` pass. Numerical
checks here use analytic CPU fixtures; there is no new CUDA qualification.

Targeted tests exercise:

- Real reference gradients and an independently defined analytic causal model.
- Prompt-by-prompt and final-mean benchmark comparisons across dimension batches.
- Sharded versus serial fitting, reference merge comparison, skipped-row
  weighting, partial coverage, cross-shard continuation refusal, ancestor
  overlap, and source mutation during merge reads.
- Stopping across continuation, strict window admission, and zero-mean behavior.
- Identical-lens assessment, differing-logit metrics, and explicit same-corpus
  reporting without qualification.
- Durable top-ups, uncertainty, capacity, stale reviews, tampered round plans,
  and HTTP request-body admission.
- Every managed interview drafting through its actual owner, including all four
  new operations; generated catalog and service-role completeness.
- Hash-cache reuse, invalidation, replacement, and scope boundaries.
- Matching fit provenance preserved through Python registration and Swift
  decoding/re-encoding; unknown provenance remains absent.
- Python and Swift scientific action routing and long verification timeouts.

The first full Python pass found only two coverage omissions in the expanded
interview census and the literal endpoint scanner. Both were repaired without
removing the completeness checks. The final run follows those corrections.

## Live acceptance after review and deployment

Keep the standing operational order: finish the active continuation before
any reviewed cluster push or controller restart. A pushed engine is not the
running controller until it is restarted. The build-drift advisory itself
requires a controller running a version that knows how to report it.

1. Rebuild the app and update the managed client from the landed source. Deploy
   the reviewed engine and restart the controller after checking active work.
2. Repeat plan and submit from the Mac against a staged multi-gigabyte
   continuation. This closes review N5; a login-node-only success is insufficient.
3. Benchmark the same approved two-row corpus and exact checkpoint with batches
   1, 8, and 16. Record free/capacity memory, peaks, timing, kernel observations,
   and matrix agreement. Add compilation only as an explicit comparison.
4. After a separate environment review, compare the Torch fallback and optional
   kernels. Preserve the environment resolution and benchmark report. Do not
   translate a speed result into a checkpoint-compatibility exception.
5. Materialize a small two-shard round. Exercise reviewed submission, capacity,
   cancellation, and a fresh continuation from a completed shard checkpoint.
   Merge only the latest disjoint contributions. Inspect coverage and lineage.
6. Fetch, verify custody, and register a completed or merged lens. Check the
   registered reference commit, kernel hash, driver hash, and fit-report hash.
   Inspect the partial flag before using a partial merge.
7. Register successive lens versions and assess them on the declared held-out
   text. Inspect per-layer metrics and the research conclusions; then run the
   existing qualification for the intended runtime.
8. Only then fund a larger round and choose its dimension batch, global row
   budget, shard count, concurrency, walltime, and storage allowance from the
   measured evidence. No row count is an adequacy threshold.

These live items are intentionally not reported as passing from CPU fixtures.
The branch leaves main and the active cluster jobs untouched for the auditing
and running agents to integrate through the user.

## Reproduce the local checks

Use the review worktree as the working directory and the already provisioned
Python environment. Do not install packages as part of these checks. Run the
Python and Xcode suites serially; concurrent suites have exposed an unrelated
cancellation timing flake in prior reviews. Set `PYTHON` to the absolute path of
the existing test interpreter:

```sh
cd Server
PYTHONPATH=. "$PYTHON" -m pytest -q
cd ..
"$PYTHON" scripts/ci/check-generated.py --audits
"$PYTHON" scripts/ci/public_scan.py
git diff --check
```

On the Mac, use Xcode beta and the installed Metal toolchain, with DerivedData
outside the synchronized checkout:

```sh
DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer \
TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.920.1 \
TEST_RUNNER_STEERLAB_TEST_PYTHON="$PYTHON" \
xcodebuild test -skipMacroValidation -scheme SteerLab-Package \
  -destination 'platform=macOS' -parallel-testing-enabled NO \
  -derivedDataPath /private/tmp/interpbench-corpus-build \
  CLANG_COVERAGE_MAPPING=NO
```

Finally run `scripts/ci/check-generated.py --audits --cli <built-steerlab-cli>`
with the same interpreter. Leave any deployment, app installation, merge to
main, or cluster environment change to the maintainer's reviewed handoff.
