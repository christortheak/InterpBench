# J-lens fitting: pilot results and the scaling slice

Handoff from the maintainer's integration agent to the refactoring agents,
2026-09-11. Main is `9a9f528`. This document records what the first live
fitting pilot on the cluster measured, the defect it exposed and fixed, and
the work needed before a paper-grade lens can be fitted on a shared batch
cluster: dimension-batch guidance, sharded fitting with a merge, a stopping
rule, and a handful of smaller repairs the pilot surfaced.

## 1. What the pilot was

The managed `jlens-fit` operation, exactly as landed (`878bca2`, follow-up
`abcb42a`), run end to end for the first time on GPU hardware:

- Model: the hybrid-attention 27B checkpoint (48 linear-attention layers,
  16 full-attention layers, width 5,120, 64 blocks), pinned to its
  40-character revision, prepared in the runner's offline cache.
- Corpus: eight rows of the study's neutral maintenance-prose corpus,
  300+ characters each, prepared through `science corpus-preview` and
  `corpus-publish` with a receipt; `maxPrompts 4`, so four rows fitted.
- Settings: the interview defaults. `dimBatch 1`, `maxSeqLen 128`,
  `skipFirst 16`, `checkpointEvery 2`, bfloat16, `device cuda`, tier
  testing, all 63 source layers.
- Path: draft → publish → input-plan → package → archive copied to the
  runner's run root → `remote science-stage` → `remote science-plan` on
  `{inputBundleSHA256}` → `remote science-submit` → one Slurm job on a
  single 80 GB A100 with 80 GB of host memory and four cores.

## 2. What it measured

| Measure | Result |
|---|---|
| Outcome | job completed, exit 0; `COMPLETED` marker, `fit-report.json`, `artifact-description.json`, final checkpoint present |
| Rows | 4 fitted, 0 skipped; 115 / 120 / 120 / 128 tokens; 98 / 103 / 103 / 111 valid positions |
| Seconds per row | 4,206 / 4,206 / 4,191 / 4,457 (about 70 minutes) |
| Wall time | 4 h 48 min, of which about 3 minutes was model loading |
| Peak host memory | 61.8 GB of the 80 GB requested (the Python step's MaxRSS) |
| Peak device memory | not recorded (see §6, R7) |
| Output | 63 float32 matrices of 5,120 × 5,120, 6.6 GB; every entry finite; file hash equals the report's `tensorSHA256` |
| Checkpoint | `sums.safetensors` 6.6 GB plus `state.json` (`nDone 4`, `nextIndex 4`); scratch snapshots pruned to the final one |
| Convergence series (`meanRelativeChangeMax`) | row 2: 0.78, row 3: 0.79, row 4: 0.42 |
| Node scratch after exit | empty (verified by a list-only job on the same node): the EXIT-trap cleanup works on a normal exit |
| Cost review vs reality | the review said 5,120 backward passes per row and 6.15 GiB per matrix set; both exact |

Two runtime facts from the job's stderr:

- The 48 linear-attention layers ran on transformers' torch fallback:
  "The fast path is not available because one of the required library is
  not installed" (flash-linear-attention and causal-conv1d are absent from
  the runner environment). The backward therefore ran, which was the
  question, but through the slow path.
- One benign warning at the first backward: "Attempting to run cuBLAS, but
  there was no current CUDA context".

Sanity of the matrices, read directly from the tensor file: diagonal means
rise from about 0.13 at layer 0 to about 1.0 at layers 60–62, and mean
off-diagonal magnitude falls from 0.049 to 0.0035 over the same span. That
is the shape a Jacobian through a residual network should have. It is a
pipeline check, not a claim about readout quality.

## 3. The defect the pilot exposed, already fixed

`diagnostic_archives.publish_directory` used the kernel's no-replace rename
flag for every create-only directory publication. The cluster's Lustre
scratch and its NFS home both answer `EINVAL` to that flag; only node-local
disk supports it. Every publication path refused on the cluster with
"[Errno 22] Invalid argument": diagnostic input staging, fitting checkpoint
snapshots, corpus and request publication, artifact import, cleanup, and
workspace bootstrap. None had run there before this pilot.

Fixed in `9a9f528`: where the flag is unsupported the engine claims the
target with `mkdir` (atomic; `EEXIST` if anyone holds it) and renames the
staged directory over the empty directory it just created; a claim another
process wrote into refuses with `ENOTEMPTY` and is left alone; every other
errno keeps its meaning. Four regression tests; verified live on both
cluster filesystems before the redeploy. Nothing further is needed on this,
but any new create-only publication must go through `publish_directory`.

## 4. What the reference fits look like

The published lenses carry their own fitting record. For the reference
27B-class fits: WikiText-103 training records of at most 2,000 characters,
capped at 1,000 prompts, 128 tokens each, bfloat16, `torch.compile`,
dimension batch 64 on a 179 GB card, with early stopping once the mean
relative change of the running average stayed below 0.002 over a
10-prompt window after at least 100 prompts. Where they stopped:
gemma-3-27b-it 828 prompts, gemma-2-27b 780, gemma-3-12b-it 844, gemma-2-9b
462, gemma-2-2b 454. So a paper-grade lens for a 27B is on the order of
800 prompts, set by the stopping rule, not by the cap.

At the pilot's rate that is about 39 days on one A100. Two levers exist,
and neither is measured yet: dimension batch (the reference used 64; the
80 GB card holds 54 GB of weights, so 8 or 16 is the plausible range) and
the linear-attention kernels. The reference also says, in
`jlens.fitting`'s own docstring, to shard across machines on disjoint
prompt slices and merge with `JacobianLens.merge`, a prompt-count-weighted
mean, which is exact because the lens is a plain average.

## 5. The scaling slice

Requested, in priority order. Numerical fitting stays inside the pinned
reference kernel; the historical loop audit stays in force.

### R1 — Dimension-batch pilot support and guidance

- Add measured throughput to the run record: per-row seconds already exist
  in `progress.jsonl`; add per-row and peak device memory
  (`torch.cuda.max_memory_allocated`, reset per row) and the effective
  dimension batch. The cost review should be able to cite a measured rows
  per hour for a model once one exists, the way walltime preflight uses
  throughput history for generation.
- Guide text: state that the reference used 64 on a 179 GB card, that on
  an 80 GB card the weights leave roughly 26 GB for activations, and that
  the right first move on a new card is a two-row pilot at 8 and 16.
- Refuse early, not late: if a chosen dimension batch cannot fit, the
  failure should arrive at the first backward with a failure record naming
  the batch and the measured free memory, not after a checkpoint interval.

### R2 — Sharded fitting

Design constraints, in the project's terms:

- **Authoring.** A fit request gains an optional `shards` count, or a
  reviewed shard plan derived from one corpus and one receipt: shard k
  fits rows `k, k+N, k+2N, …` (interleaved) or contiguous blocks, chosen
  explicitly and recorded. Every shard shares the corpus hash, receipt,
  model, revision, and every numerical setting; only the row subset
  differs. The row subset belongs in the checkpoint identity so a
  continuation of shard 3 cannot be confused with shard 4.
- **Submission.** One reviewed plan, N Slurm jobs, the same durable-job
  machinery as the study fan-out, but this is a scientific submission and
  must not be relabelled as a study stage. Respect the site's running-job
  limit (eight here) with staggered submission; `planSHA256` must bind the
  shard plan.
- **Per-shard continuation.** Each shard is its own run with its own
  checkpoint and can be continued independently under the existing
  contract. A shard of 100 rows at 70 minutes exceeds a 24-hour walltime,
  so continuation is the normal path, not the exception, until the batch
  size lever is measured. Consider a managed resubmit from the last shard
  checkpoint with the same explicit-review gate as today.
- **Merge.** A new managed operation, `jlens-fit-merge`, reads N completed
  fit runs, checks that their identities agree on everything except the
  row subset and the driver provenance, and publishes one lens: summed
  sums, summed counts, mean = sums / count, in float32, with the reference's
  own `merge` as the numerical fixture (n-weighted mean of per-shard means
  must equal sums-over-counts to the bit for the fixture). The merged run
  records every shard's run id, tensor hash, row subset and count, and
  writes its own `artifact-description.json` so registration is unchanged.
  Partial merges (some shards missing) are allowed and stamped as partial.
- **Storage.** N shards each hold a 6.6 GB checkpoint and a 6.6 GB lens for
  this model; the merge reads them one at a time. Say so in the cost review.

### R3 — Stopping rule or fixed row count, the researcher's choice

Today `maxPrompts` is the only control. Offer both, recorded in the
identity and the report:

- **Fixed rows**: the current behaviour, `maxPrompts`.
- **Stopping rule**: the reference's, made explicit as three fields:
  `stopAtDelta` (mean relative change of the running average, the
  reference used 0.002), `stopWindow` (rows over which it must hold, 10),
  and `minPrompts` (never stop before, 100); `maxPrompts` remains the cap.
  The per-row `meanRelativeChangeMax` already computed in
  `progress.jsonl` is the series; define the statistic precisely (max
  over layers of the mean relative change, or the mean over layers, and
  say which; the reference reports `final_mean_rel_change`).
- **Under sharding**, the rule has two honest forms: per shard (each shard
  stops on its own series; cheap, but shards of different lengths), or
  merged (fit fixed-size shards, merge, and evaluate the rule on the
  merged series by merging cumulatively in row order). Offer per-shard
  stopping as the default and document that the merged rule is what the
  reference's single-process run computed.
- The report must say which rule ended the fit and the final value of the
  statistic, so a lens can be compared with a reference fit on equal terms.

### R4 — Kernel package as a reviewed environment extra

Add a `[jlens-kernels]` extra (flash-linear-attention and causal-conv1d
pinned in the platform lock) that `bootstrap.sh --with-jlens` can install
on request. The fitting runtime identity already records the transformers
version; add whether the fast path was available, so a checkpoint fitted
on the fallback is not continued on the fast path without a numerical
review. Whether the kernels change the numbers is exactly the kind of
question the compatibility contract exists to ask.

### R5 — Export and fetch of large outputs

`remote science-export` on this run timed out client-side after 60 seconds
of silence while the controller compressed 13 GB; the server carried on
and the export owner is idempotent, so a retry returns the finished bundle.
Make the client wait properly: either an asynchronous export job the
client polls, or a streaming response, or a documented longer idle timeout
for this verb. The same applies to `science-fetch` for multi-gigabyte
bundles. Also consider excluding the checkpoint from the default evidence
bundle, with an explicit flag to include it: the lens is 6.6 GB and the
checkpoint doubles the transfer for a researcher who only wants to
register the fit.

### R6 — Two rough edges in the remote chain

- `remote science-stage` requires the archive to sit under the runner's
  run root and refuses any other path with "is not in the subpath of".
  Either accept a path anywhere the served workspace can read and copy it,
  or say in the guide and the refusal that the researcher must place the
  archive under `<runRoot>` first.
- `remote science-plan` on the *published* `request.json` fails with
  "Transport and custody refuse symlinks or non-directory ancestors:
  prompts/fitting/…/corpus.jsonl", because the runner cannot see the Mac's
  workspace paths. The verb needs the staged form,
  `{"inputBundleSHA256": "…"}`, which nothing in the guide or the
  `science-stage` result tells the researcher to write. Have `science-stage`
  write that document beside the request (it already returns the object),
  and have the plan refusal name it when it sees a `parameters` request
  from a client that staged a bundle.

### R7 — Record what a pilot needs to record

The handoff asked the pilot to record peak memory, per-row timing,
effective precision, finite outputs and skipped rows. The run records
timing, precision and skips; finiteness and host memory had to be checked
by hand, and device memory is not recorded anywhere. Add to
`fit-report.json`: per-layer finiteness (already asserted in the loop, so
it is a boolean), peak device memory, peak host RSS if cheaply available,
and the fast-path availability from R4.

## 6. Acceptance for the slice

- Unit: merge fixture equals the reference `merge` bit for bit on a toy
  model; stopping rule fires on a synthetic converging series and never
  before `minPrompts`; shard identity refuses cross-shard continuation;
  partial merge stamps partial.
- Live, after deploy: a two-row pilot at dimension batch 8 and 16 on one
  A100 and one H100, recording rows per hour and peak device memory; then
  an eight-shard fit of the reference-style corpus (WikiText-103 training
  records, 600+ characters, first-N selection, a publicly documented
  choice) with the stopping rule, merged, registered, and qualified for its
  runtime.
- The continuation test from the handoff (row-4 checkpoint → row 8 in a
  new job) still stands and is being run on the pilot's checkpoint.

## 7. Things not to change

- The estimator, the position mask, the float32 accumulation, and the
  per-row loop (the `878bca2` AST audit).
- The checkpoint contract `jlens-fit-v1`, except to add the row subset and
  fast-path fields under a reviewed migration.
- Registration through `artifact-plan` / `artifact-import`; a merged lens
  is one more description file.
