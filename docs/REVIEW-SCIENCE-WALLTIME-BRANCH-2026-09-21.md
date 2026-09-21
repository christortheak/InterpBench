# Review: per-request science walltime branch (2026-09-21)

Branch `claude/science-walltime` at 4ed6b98, one commit on main c467ea8,
landed by fast-forward. Companion note: `docs/SCIENCE-WALLTIME-2026-09-21.md`.

## What prompted it

Every managed science job requested the site's default walltime (24 hours),
so a twenty-minute assessment could not backfill and the scheduler estimated
a five-hour wait on a busy afternoon. Lowering a pending job's limit by hand
worked once; the instrument should size its own requests.

## What the branch does

- A `walltime` (`HH:MM:SS` or `D-HH:MM:SS`, canonicalized) rides beside the
  request on `/api/science/plan` and `/api/science/submit`, as `--walltime` on
  both clients' science plan and submit verbs, and as a round-level or
  per-shard field on fitting-round actions. It is validated, refused above the
  site cap (the configured default), written into the resources block so it
  is inside `planSHA256`, and rendered as the scheduler's time limit. A plan
  reviewed with one walltime cannot be submitted with another.
- Without a request, `jlens-fit-assess` gets a default sized from its own
  plan: three times a calibrated per-unit allowance plus thirty minutes for
  model load, rounded up to fifteen minutes, at least one hour, never above
  the cap. The allowance is itself about three times the recorded rate, so
  the default is roughly ten times the expected runtime. Every other
  operation, including fitting shards, keeps the site default; `walltimeBasis`
  and `walltimeReview` record which rule applied and why.
- The local executor validates and records a requested walltime and enforces
  nothing. Twenty-four new tests; generated CLI reference and resource text
  regenerated.

## Review

I read the owner and the planner threading in full. The shape mirrors the
GPU-type feature exactly, and the default rule errs on the long side, which
is the right failure direction for a limit that kills the job. The agent
deliberately kept benchmarks on the site default because no calibrated
estimate precedes a throughput measurement; I agree. Plans reviewed before
this change hash differently, as with the GPU-type landing.

Suites on the exact branch tree, run by me:

| suite | result |
|---|---|
| Python (`HF_HUB_OFFLINE=1 pytest`) | 6684 passed, 9 skipped, 0 failed |
| Swift (`xcodebuild test`, serial) | 4956 passed, 5 skipped, 0 failed |

Gates: `check-generated.py --audits` PASS, `public_scan.py` clean,
`git diff --check` clean, vocabulary scan clean.

## Deployment

The client identity moved. Because a rebuilt client refuses to stage against
an older running engine, this landing is followed by the app rebuild, the
payload push, and a controller restart together, before any further science
staging.
