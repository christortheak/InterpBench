# Per-request walltime for managed science jobs (2026-09-21)

Closes item 8 of `docs/JLENS-AND-P7-STATUS-HANDOFF-2026-09-21.md` §4. The
shape follows the per-request GPU type
(`docs/REVIEW-SCIENCE-GPU-PLACEMENT-BRANCH-2026-09-12.md`) exactly: a walltime
is execution shape, reviewed at plan time, bound into the plan hash, and never
written into the published request.

## 1. The problem

Every managed science job (`scientific_execution.plan`/`submit`) requested the
site's default walltime from the cluster environment
(`STEERLAB_SLURM_WALLTIME`, a day at the researcher's site). A twenty-minute
`jlens-fit-assess` therefore asked for a day of GPU time and queued behind
multi-day jobs instead of backfilling into the scheduler's gaps; on a busy day
the scheduler estimated a five-hour wait for it.

## 2. The request field

| Surface | Field |
|---|---|
| `scientific_execution.plan(request, profile, gpu_type=…, walltime=…)` and `submit(…)` | keyword `walltime` |
| `POST /api/science/plan`, `POST /api/science/submit` | `{request, gpuType?, walltime?}` (a bare request document is unchanged); submit also carries `planSHA256` |
| `steerlab-cli remote science-plan` / `science-submit` | `--walltime <hh:mm:ss>` |
| `steerlab runner science-plan` / `science-submit` (Python client) | `--walltime <value>` |
| Fitting-round `plan`, `submit`, `cancel` (`/api/science/fitting-round/{job}/{action}`) | `walltime` for every shard in this top-up, `shardWalltimes` `{"<index>": "<walltime>"}` for individual pending shards in `submitIndices`; `status` and the CPU merge actions refuse both |

Accepted spellings are `HH:MM:SS` (hours unbounded, as Slurm accepts) and
`D-HH:MM:SS`; both normalize to `HH:MM:SS`, so `0-02:30:00` and `02:30:00`
review to the same plan. Anything else (`90`, `30:00`, `2h`, `01:60:00`, an
empty string, a number) refuses with `scientificExecutionRefused` and a
`repairAction` naming the two shapes. Minutes and `mm:ss`, which `remote
resubmit` accepts for a continuation, are deliberately not accepted here: a
science job's limit should be unambiguous at a glance.

**Cap.** A walltime above the site cap refuses (`Walltime 36:00:00 exceeds the
site cap of 24:00:00`). The cap is the configured `STEERLAB_SLURM_WALLTIME`
value (the engine's `04:00:00` fallback when a site declares none), which is
also the site default that applies when no walltime is given.

**Hash binding.** The validated walltime is written into the plan's
`resources.walltime`, the block `planSHA256` already covers, and rendered into
`schedulerPreview` and the sbatch script as `#SBATCH --time=…`. A plan reviewed
with one walltime cannot be submitted with another, or without one: submit
re-plans with the submitted arguments and refuses on `planSHA256` mismatch
before any scheduler call, exactly as it does for a changed GPU type. The plan
also records `requestedWalltime` (canonical form) when one was asked for.

**Recorded where.** The plan's `resources`, `walltimeBasis`, and
`walltimeReview`; the job's `scientificPlan` and `requestedResources`; and,
for rounds, each shard row (`walltime`, `walltimeBasis`) plus the round state
file's `walltimes` map, written before each attempt so an uncertain submission
keeps what it asked for. A shard that has been attempted cannot be given a
different walltime through an override.

**Local executor.** A walltime is validated for shape and recorded
(`requestedWalltime`, `walltimeBasis: requested`, a review saying it is
recorded only); nothing is enforced and no cap applies.

## 3. The default rule

`walltimeBasis` names which rule chose the walltime:

- `requested`: the researcher asked for it.
- `estimated`: no walltime was given and the operation's plan carries a
  calibrated workload estimate. The default is
  `3 × estimate + 30 min` (model load and staging), rounded up to the next
  15 minutes, at least 1 hour, and never above the site cap. If that value
  reaches the cap, the site default applies and the basis says so.
- `siteDefault`: everything else.

Only `jlens-fit-assess` is estimated today. Its work is a bounded number of
readouts with no backward passes, and the plan's `operationReview` already
states the rows per corpus, the source layers, and the candidates. One unit of
work is one held-out row read through one source layer for one lens
comparison (`rows × sourceLayers × candidates`, doubled when
`readoutDtype: float32` adds the paired readout), and the allowance is one
second per unit. The calibration is a recorded acceptance run
(`docs/JLENS-DIMBATCH-BENCHMARK-RESULTS-AND-HANDOFF-2026-09-12.md`): 16 rows
× 63 source layers × 1 comparison took five minutes on an H100 for a 27B model
including model load, under 0.3 s per unit, so the allowance is already more
than three times the observed rate before the rule's own margin. Worked
examples at a 24-hour cap: that acceptance run reviews to `01:30:00`; a
64-row single comparison to `04:00:00`; three candidates on three 64-row
corpora (36,288 units) to `24:00:00` with `walltimeBasis: siteDefault`. The
review records the estimate's inputs, the allowance, the calibration, and its
limitation (it is an allowance, not a measurement of this workload).

What stays long, and why:

- `jlens-fit` shards keep the site default by the researcher's rule; a pilot
  extrapolation on the plan is context, and the checkpoint protocol resumes
  across a walltime kill anyway.
- `jlens-fit-benchmark` keeps the site default: it is the throughput
  measurement itself, so no calibrated estimate precedes it. A researcher who
  knows the model is small passes `--walltime` explicitly.
- `battery` and `stability` carry no workload review, and every other managed
  operation likewise keeps the site default.
- `jlens-fit-merge` and round materialization are CPU operations and run on
  the controller's local executor, where no walltime applies.

The scheduler kills a job at its limit. When in doubt the engine keeps the
site default and says so in `walltimeReview.summary`; the researcher can
always request more, up to the cap.

## 4. Compatibility

Slurm plans now carry `walltimeBasis` and `walltimeReview`, so a plan reviewed
before this change hashes differently: review again before submitting.
Round reviews likewise (their shard rows gained `walltime` and
`walltimeBasis`), as they did when `gpuType` joined the rows. Local plans
without a walltime are byte-identical to before. No published request,
checkpoint identity, or scientific owner changed.

## 5. Tests

`Server/tests/test_science_walltime.py`: rendering and hash binding, the
submitted bundle and job record, the cap, malformed strings, the local
executor, the assessment default (below the site default, above the estimate,
`estimated`), the rule's margins and floor and rounding and cap, `jlens-fit`
shards at the site default with per-shard overrides bound into the round
hash and recorded per shard, the HTTP wrapper, the round route, and the
Python client. `Tests/ExperimentKitTests/ClusterClientTests.swift` covers the
Mac client's request bodies and the parser flags.
