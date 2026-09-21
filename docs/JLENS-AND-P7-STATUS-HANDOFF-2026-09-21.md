# J-lens qualification and P7 status handoff

Date: 2026-09-21
Review baseline: main `cfe1d2a`
Audience: coding agents (fixes and follow-ups), running agents (remaining
acceptance), independent reviewers
Status: status report and work list. It updates the
[closure handoff of 2026-09-13](JLENS-CLOSURE-AND-P7-QUALIFICATION-HANDOFF-2026-09-13.md)
and the [J4 decision](JLENS-QUALIFICATION-DECISION-J4-2026-09-13.md); it changes
neither. Live evidence lives in the study workspace (paths below are
workspace-relative); repository documents are linked.

## 1. Where things stand

| Item | Status | Evidence |
|---|---|---|
| J1 corrections | Done | Dated correction sections in the numerics decision and results documents |
| J2 direct controls | Done | Engine equals bare kernel bitwise; batch-pair identity measured directly; corpus content hashes |
| J3 managed readout assessment | Landed and live-accepted | [review](REVIEW-JLENS-ASSESSMENT-READOUT-BRANCH-2026-09-13.md); float32 readout and logit-lens baseline used in every assessment since |
| J4 decision | Recorded; independent review pending | [decision](JLENS-QUALIFICATION-DECISION-J4-2026-09-13.md) |
| Full hybrid 27B lens (828 rows) | Fitted, merged, registered, assessed on two held-outs | §2 |
| Mixed-corpus lens (case prose added) | Case fit merged on the cluster; mixed merge and assessment in progress today | §2, [design note](JLENS-MIXED-CORPUS-MERGE-2026-09-17.md) |
| P7-A exact builds and probe journey | Done | workspace `diagnostics/probes-p7-2026-09-13/P7-A.md` |
| P7-B readings, policies, outcomes | Local MPS and remote CUDA legs done; panel seats and behavioural dose open | workspace `diagnostics/probes-p7-2026-09-13/P7-B.md` |
| P7-C remote cancel, continue, custody | Not started | |
| P7-D surface matrix and app journey | Not started (app journey needs the researcher) | |
| P7-E cost | Two data points only; no targets by the researcher's decision | P7-B.md cost lines |
| P7-F regression, review, release statement | Not started | |

Engine and app are both at `cfe1d2a`; the cluster payload is stamped the same.
Suite counts on that tree: Python 6635 passed, 9 skipped; Swift 4954 passed,
5 skipped.

## 2. J-lens results since the closure handoff

**Full 27B lens.** 828 WikiText-103 rows, bf16, dimension batch 4, fitted as
eight interleaved H100 shards of 103 to 104 rows (about 19 hours each), merged
deterministically, fetched with custody, registered as
`custom-lens-d163093927494fa2862225f21a7a48c1` (63 source layers, hidden 5120,
float32 converted, 6.6 GB; cumulative sums retained). Assessed with the J3
operation (float32 readout, logit-lens baseline, 64 positions per row) against
the eight-row pilot lens on two held-outs. Workspace records:
`diagnostics/jlens-full-27b-2026-09-15/INTERPRETATION-FULL-LENS.md` with
per-layer tables.

- Against the pilot: better at 58 of 63 layers on both corpora; the gain grows
  with depth (layers 45 to 52: top-10 overlap about 1.5 times the pilot's).
- Against the plain residual: better from layer 21 upward on WikiText
  validation (42 of 63 layers), but only from layer 32 upward on the study's
  case passages (30 of 63). Below those layers the logit lens reads the final
  prediction at least as well.
- Absolute readability is low below about layer 45 to 50 under any readout
  (JSD to the final prediction above 0.6 of a 0.693 maximum). The lens's
  advantage in the middle of the stack is relative.
- The two lenses agree with each other more than either agrees with the final
  prediction below layer 40; the extra rows sharpen a subspace the pilot
  already found.

This supersedes the J4 table row "Full-lens hybrid 27B agreement: bounded
subset only" as a matter of evidence; the J4 decision text itself is not
edited, and its independent review is still owed.

**Two engine limits found by that merge, both fixed and reviewed the same
day** ([note](MERGE-INPUT-BOUNDS-2026-09-15.md),
[review](REVIEW-MERGE-INPUT-BOUNDS-BRANCH-2026-09-15.md)): the 16 GiB
transport bound was being applied to the in-place inventory of merge inputs
(about 106 GB), and the merge preflight hashed every shard tensor inside the
120-second validation subprocess. In-place inventories are now bounded
separately (`MAX_PINNED_BYTES`); preflight is structural and trusts recorded
tensor hashes; every byte is still hashed before anything is combined.

**Mixed-corpus program** ([design](JLENS-MIXED-CORPUS-MERGE-2026-09-17.md),
[review](REVIEW-JLENS-MIXED-CORPUS-MERGE-BRANCH-2026-09-17.md)). Because the
lens is a linearization taken at its corpus, the crossover moving from layer
21 to 32 on legal prose motivated adding case text. Three hundred appellate
opinions (federal courts of appeals and state courts of last resort, 1970
onward, opinion bodies only, stimulus line excluded) were drawn from the
public Caselaw Access Project with a documented, seeded, stratified script
and published as `prompts/fitting/jlens-cap-appellate-300`, with a 64-opinion
held-out from a different shard and seed. The 300-row fit ran as three H100
shards and merged on the cluster (300 of 300 rows). The mixed merge with the
828 WikiText rows uses the new `allowMixedCorpora` opt-in and runs on the
authoring machine over the two merged runs; assessment follows on all three
held-outs (WikiText validation, case passages, case held-out). The result to
read: whether the case-text crossover moves back toward layer 21 without the
WikiText crossover moving later.

**J-lens numerics in one line.** Estimator exact; the batch-shape gap is
forward-pass bf16 kernel selection (dense paths too); engine equals bare
kernel bitwise; the layer 4 to 5 sensitivity peak on Gemma-3-4B persists in
float32 and remains an open observation, not a defect.

## 3. P7 results so far

**P7-A** (done): Gemma-3-4B layer-17 capture, linear/MLP/mean-difference
probes on the rule-legitimacy set (72 fit, 24 selection, 40 final test);
linear 97.5% final-test, shuffled control at chance or below; independent
score recomputation to 1e-14; probe library lists all four.

**P7-B** (two legs done): one force-frozen study (`batteryEvidence` skipped
with the researcher's written acceptance; non-citable by stamp) with five
arms: baseline, plain, zero-strength policy, fixed strength 8, threshold on
the probe score. Executed on local MPS through the loopback runner and on a
cluster A100 from the same frozen bundle.

- Noninterference: baseline, plain, and zero arms byte-identical (12 of 12) on
  both substrates.
- Actions: fixed arm diverges at generated token 3; threshold arm diverges from
  fixed at tokens 13 to 17; 1152 decisions per arm all applied; requested
  equals applied; rule adherence exact; alignment 96 of 96.
- Readings: 6912 of 6912 recorded on both substrates; post minus pre equals
  the predicted probe shift to 0.4% median; policy float32 and observer
  float64 scores agree to 1e-6.
- Cross-substrate: texts differ on every prompt between MPS and CUDA under the
  shared seed (bf16 kernel selection), as expected; within each substrate the
  arms compare exactly as designed.
- Cost data: 8.7 ms per decision on MPS, 0.7 ms on the A100, unsynchronized
  host time. The researcher chose to report measurements without targets.
- Defect found and fixed: study-measurement scoring lost every reading on MPS
  (combined device-and-dtype cast); fix is transfer then cast, with stage-named
  non-finite reasons ([review](REVIEW-PROBE-OBSERVATION-MPS-SCORING-BRANCH-2026-09-14.md)).
- Open: panel seats, and a behavioural outcome at a dose that shows one
  (strength 8 on the golden-gate direction produced no Golden Gate content in
  96 tokens; a higher dose is a research choice).

**Operational incidents during the program, both fixed:** a controller
auto-resubmitted a months-old checkpointed record
([review](REVIEW-AUTO-RESUBMIT-STALENESS-GUARD-BRANCH-2026-09-13.md)); an app
build's launch check with a tunnel open triggered a bulk auto-import
([review](REVIEW-LAUNCH-CHECK-AND-TRANSFER-ROBUSTNESS-BRANCH-2026-09-14.md)).

## 4. Work for coding agents

Ordered by what unblocks the remaining acceptance. Each item is small; land
each with a regression test and a dated note, and do not widen scope.

1. **Fitting-round actions as one hashing session.** `merge-submit` plans
   inside the round action and then `scientific_execution.submit` re-plans in
   its own `input_hashes` session, so a large round hashes its closure twice
   in one request (about five minutes each for 106 GB). Wrap
   `jlens_rounds.action` in `input_hashes.session()`; fingerprints are still
   rechecked at use and at scope exit, so pinning is not weakened.
2. **Long science requests off the request threadpool.** The fitting-round
   route and `science-stage` run synchronously; a multi-minute plan holds the
   submitting lock and a client timeout abandons a plan that completes
   unobserved. Make plan and stage durable jobs polled by ID (the same pattern
   as `model-install`), keeping the request or plan hashes as the identity.
3. **Bulk cleanup of staged inputs and exports.** `cleanup-plan`/`cleanup-apply`
   remove one job's output directory and deliberately retain staged inputs,
   export archives, and records; after a large program those retained classes
   were about 320 GB and had to be removed by hand. Add a reviewed, receipt-
   gated cleanup for staged workspaces and export archives whose outputs have
   verified local custody, with the same explicit-confirmation shape.
4. **P7-C support.** Confirm the cancel path on a multi-response study run
   with mixed completed and in-progress responses: completed responses keep
   their evidence, partial failures do not become resume keys, and a fresh
   response has fresh policy state. Where the engine cannot prove one of these
   today, add the check and its test before the running agents exercise it.
5. **P7-D readiness.** The app's Results view of the J3 assessment report and
   of a mixed-corpus lens record (`fit.corpora`) has not been inspected; make
   sure both render, and that the evidence summary and the response preview
   do not conceal omissions or failures. Keep the `JLensCorpusContribution`
   field surfaced wherever a lens's corpus is shown.
6. **P7-E instrument.** A fixed-workload replay diagnostic that reports prompt
   processing, decode throughput, wall time, peak device and host memory, and
   evidence bytes with warm-up, repeats, and dispersion, for the six
   configurations named in the closure handoff. No targets; the researcher
   reads the numbers.
7. **Assess several lenses from one upload.** Every `jlens-fit-assess`
   request carries both lenses it compares (about 12 GB for a pair of 27B
   lenses), so comparing two candidates against one reference on three
   held-outs meant six uploads of the same two pairs, six stagings that each
   hash 12 GB and hold the submission lock, and several hours of transfer
   for a few minutes of GPU work. Either let one assessment take several
   candidate lenses and several corpora (one upload, one job, a report per
   pair and corpus, activations captured once per corpus), or let a request
   reference a lens the runner already holds by its converted-tensor hash so
   a lens uploads once per runner. Keep content pinning: the plan still binds
   every lens by hash. *Status:* the first option is implemented; see
   `docs/JLENS-MULTI-CANDIDATE-ASSESSMENT-2026-09-21.md`. Cross-request lens
   reuse remains open.
8. **Walltime for managed science jobs.** Every science job requests the
   site's default walltime (24 hours from the cluster environment), so a
   19-minute assessment queues behind multi-day jobs instead of backfilling
   into gaps; on a busy day the scheduler estimated a five-hour wait for a
   job that runs in twenty minutes. Add a per-request walltime (as the GPU
   type already is: reviewed, bound into the plan hash, refused above the
   site cap) and a per-operation default sized from the plan's own estimate
   with margin, so short assessments, benchmarks, and CPU merges backfill.
   The fitting shards keep the long default. *Status:* implemented; see
   `docs/SCIENCE-WALLTIME-2026-09-21.md`. The estimated default covers
   `jlens-fit-assess`; benchmarks take an explicit `--walltime` (no
   calibrated estimate precedes the measurement itself), and CPU merges run
   on the controller's local executor, where no walltime applies.
9. **Cold-ledger import policy** (carried from the results handoff): decide
   and document how imported evidence from a controller that no longer exists
   is adopted without a live reconcile.

Constraints that still apply: public repository hygiene (no site, host,
person, or username words; no home-directory paths); frozen artifacts and
`runs/` immutable; secrets by path only; both suites serial on the final
tree; the reviewer reads the whole diff before landing.

## 5. Work for running agents and the researcher

- Today: mixed merge on the authoring machine, import, three assessments,
  interpretation note beside the full-lens note.
- P7-B remainder: panel seats; one behavioural-dose leg.
- P7-C, P7-D (app journey with the researcher), P7-E (report only), P7-F.
- Independent review of the J4 decision, then the support statement.

## 6. Evidence index (workspace)

- `diagnostics/jlens-numerics-2026-09-13/` (T-series, J2, J3 records)
- `diagnostics/jlens-full-27b-2026-09-15/` (full-lens assessments)
- `diagnostics/probes-p7-2026-09-13/P7-A.md`, `P7-B.md`
- `prompts/fitting/jlens-cap-appellate-300`, `jlens-heldout-cap-appellate-64`,
  `prompts/fitting/sources/cap-appellate-1970plus/` (script, shard hashes,
  selection reports)
- `runs/jlens-lenses/custom-lens-d163093927494fa2862225f21a7a48c1` (full lens)
- `runs/jlens-assessment-4e68bacb…` (WikiText), `…b570c5b4…` (case passages)
