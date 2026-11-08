# Remote and cluster execution

When this workspace's study runs on a cluster, the same contract holds — and
one rule outranks convenience: **never write a bare sbatch script.** Submit
through the engine's rendered path — this client's submission verbs below, or
`steerlab-server study submit …` on the engine itself — which requests
node-scratch via the site's gres and arms the cleanup
trap; a hand-rolled script silently gets neither, and stale node-scratch is
how clusters come to email their operators about you. If a submission need
seems to force a hand-roll (a dependency chain, a resume), that is a missing
verb to report, not a reason to bypass the renderer.

**What a submission refuses before anything is queued.** A study that its
settings or pinned task prompts alone make unrunnable is refused when it is
submitted, with the run's own sentence and repair, rather than after the
queue wait and the model load: temperature 0 with several seeds, or several
samples per item at temperature 0, on a study that generates sampled text, an
ordinal scale with no declared aggregation, answer-token instruments over
items they cannot read, an attention-check rule with no checked item, a
`run` of a concept study with no arm, and a `pipeline` with no readable
`pipeline` block (declare one with `experiment set-pipeline`, or submit the
one step as its own verb). A chain whose run is still ahead meets the same
settings refusals at its start, before any stage; one resumed after its run
completed is not refused for them. A readout-only study at temperature 0 with
several seeds or samples per item runs on an engine from this release, and
the client refuses to send it to an older engine, which would refuse it after
the model load.

**A large study can take minutes to submit.** The server measures every task
prompt before it answers, so several hundred long transcripts can take ten
minutes or more; both clients wait up to an hour. A submission that was sent
and never answered is `submitOutcomeUnknown`, state `pending` (exit 12), and
not a failure: the server is probably still preparing the job. Look first,
in this client's list of the server's jobs (below); the job usually appears
within a few minutes. Every submission carries a submission key derived
from what the study contains and from every setting of the command, never
from when it was packaged. A server that advertises `cluster.submissionKey`
(`result.repeatIsSafe` is true) answers the same command with the same
flags, from the same client, with the job the first one created, while that
job is being prepared or has not ended, and says so (`alreadySubmitted`,
with its parts as they are now). So if no job appears once the server has
finished measuring, run the same command again with the same flags. A
command with any setting changed, or the same study sent from the other
client, is a new submission. On an older server, submit again only if no job
appears.
A resume (resubmitting a checkpointed or cancelled job) waits as long, and
one that was sent and never answered is `resumeOutcomeUnknown` (exit 12):
check the job's status before resuming again.

**The walltime check is advice, and never refuses.** The submission
estimates the time its job needs from the cluster's own history for this
kind of study: the past jobs whose items were nearest this study's length,
each timed from its own records where known (without the model load). When
the estimate exceeds the requested walltime, the check warns in plain words
(`walltimeAdvice`, echoed as `result.walltimeNotice` and printed on stderr by
both clients): the estimate and its basis, the walltime to ask for (the
estimate rounded up to the hour), and splitting the study as the
alternative; past twice the request it says the job is very likely to run
out of time. A past job can be slow or fast for reasons that are not this
study's, so no history-based figure refuses a submission. What happens at
the walltime depends on the step (`atWalltime`): a `run` keeps the records it
finished and can be resumed; a pipeline can be resumed only when the walltime
finds it in its run stage; `extract`, `validate`, `evaluate`, and a `sweep`,
and a pipeline in any other stage, stop and must be submitted again (a sweep
on the cluster is not checkpointed). When no past job's items were near this length,
or the history does not record how long its items were and these are long,
the estimate says so too (`promptLengthNotice`). Tell the researcher, and ask for more time than the
estimate or split the study; a short pilot of a few items measures the real
pace.

## Runners

This client reaches compute through a **runner**: an engine that executes what
it is handed. Every verb that addresses one takes `--runner <url>`. The token
comes from `--token-file <path>` or `STEERLAB_RUNNER_TOKEN`; there is no
`--token` flag, because a command line is readable by every process on a
shared machine. Ask the researcher for the runner's URL and token file. Never
invent one, and never write a token into a file in this workspace.

### The whole round trip: `run`

```bash
steerlab run <experiment> --runner <url>                 # the whole round trip
steerlab run <experiment> --runner <url> --verb sweep --executor slurm
steerlab run <experiment> --runner <url> --no-wait       # submit and detach
```

A frozen study in this workspace becomes verified evidence in this workspace.
`result.stages` always carries nine rows — `load`, `package`, `capabilities`,
`upload`, `submit`, `wait`, `evidence`, `import`, `provenance` — each with a
state, and a failure names `result.failedStage`. `--verb` takes `analyze`,
`evaluate`, `extract`, `pipeline`, `run`, `sweep`, `validate`, or `verify`
(default `run`). A **draft** is accepted for `--verb validate` and `--verb
extract`, the steps that come before freeze (`workspace guide lifecycle`,
step 7). Every other verb refuses a draft with `experimentNotFrozen`, and the
refusal names the route to a frozen study.

`--timeout` is the WAIT deadline here (default 24 hours);
`--request-timeout` is the per-request budget (absent, a minute per request
and an hour for the submit). `--no-wait`, ctrl-c during the
wait, and a wait timeout all **detach**. None of them cancels the remote job.
`--no-wait` and ctrl-c answer `pending` (exit 12); a wait timeout answers
`waitDeadlineExceeded` (70) and says the job is still running. All three list
the commands that finish the job by hand in `result.followUps`. A job that
fails after producing partial output still has its bundle fetched and
imported, and the envelope says `remoteJobFailed` (70): a partial is evidence
about a failure, never a result.

### The steps, one at a time

```bash
steerlab bundle package <experiment>
steerlab runner capabilities --runner <url>
steerlab runner upload <bundle.tar.gz> --runner <url>
steerlab runner submit --runner <url> --bundle-path <printed-path> \
  --bundle-sha <printed-digest> --verb run [--executor local|slurm] [--dry-run]
steerlab runner jobs [<job-id>] [--cancel] --runner <url>
steerlab runner logs <job-id> [--follow] --runner <url>
steerlab runner evidence <job-id> --out <file.tar.gz> --runner <url>
steerlab bundle inspect <file.tar.gz>
steerlab bundle import <file.tar.gz> --sha256 <digest>
```

**`runner submit` is not idempotent — never retry it blindly.** It creates a
job and, on Slurm, spends an allocation. A `submitOutcomeUnknown` (exit 12)
from `runner submit` or `run` means "look with `runner jobs`", not "submit
again". If no job appears once the runner has finished measuring, run the
command the repair names (`result.sameCommand` for `run`): the same command
with the same flags, which a runner where `result.repeatIsSafe` is true
answers with the job it already made. Upload and evidence download are safe
to retry. On
`bundle package` and `runner evidence`, `--out` names the archive, not the
envelope's file. `runner jobs --cancel` needs a job id: this client will not
cancel a runner's whole queue.

### A local runner: `runner serve`

```bash
steerlab runner serve [--port <n>] [--runner-root <dir>]
```

Serves this machine as a managed local runner: the engine on loopback, in
token mode, under a runner-owned root. It prints the URL and the token file's
path (never the token's value). It runs in the foreground until stopped; with
`--json` it emits one startup envelope and then streams diagnostics on
stderr. It needs the client's `runner` extra installed and refuses with
`runnerExtraMissing` otherwise; it refuses on Windows
(`runnerPlatformUnsupported`). The runner root is never this workspace
(`runnerRootIsWorkspace`): local and remote execution use the identical bundle
round trip, and the runner's files are a cache. Starting the runner loads no
model, but every job submitted to it does: ask the researcher first.

### Sharding a long run

`--parallel <n>` on `run` or `runner submit` asks a Slurm runner to fan a
measured run out across sibling jobs whose partial results the runner merges
back into one ordinary run directory. Sharding is execution logistics and
never enters the manifest or its content hash, so a sharded run and a
single-job run of the same frozen study are the same measurement. Two rules
are not optional. **Read what the submit says about its parts:** when the
cluster's job limit is reached part-way through a fan-out, the runner holds
the remaining parts and submits them, in order, as slots free. The submit
succeeds, carries `shardsHeld`, and prints a note; `steerlab runner status
<job-id> --runner <url>` shows each held part. A part refused for any other
reason fails the submit with a nonzero exit. An older runner answers that
case as a success, so on one, count the jobs with `steerlab runner jobs
--runner <url>`. **And stagger submissions where the site caps queued jobs
per user**; ask the researcher for the limit rather than guess. The merge is
performed by the running runner service, not by the submitting process.

### Resuming a checkpointed job

**A shard that hits its walltime checkpoints — resume it with the managed
verb, never a hand-rolled scheduler command.** The checkpoint trap flushes
what ran, writes a resume pointer, and exits with the checkpoint code (85);
the record reads `checkpointed`. Continue it with:

```bash
steerlab runner resubmit <job-id> --runner <url> --walltime 08:00:00 --json
```

The server re-submits the job's **own rendered sbatch script byte-for-byte**
(the resume pointer beside its child record is what makes the continuation
pick up where the shard parked), with `--walltime` applied on the scheduler's
command line so a shard that parked AT its limit gets a longer one. The verb
refuses any record that is not resumable — running, succeeded, hard-failed,
or already resubmitted — and **read the `walltime` echo in the
response**: an older server ignores the override silently, and the echo is
the proof it was applied. The sharded parent needs nothing from you; it
merges when its shards finish.

**Auto-resume (`autoResubmit`) is bounded, and the bounds are deliberate.**
When a submission carries `autoResubmit: true`, the running controller
re-submits a checkpoint it *witnessed* itself, up to `autoResubmitLimit`
times (default 5). It never auto-resumes a record whose Slurm job the
scheduler recorded as `cancelled` (cancelled beats checkpointed), a record
inactive for longer than `autoResubmitMaxAgeSeconds` (default 48 hours;
`STEERLAB_AUTO_RESUBMIT_MAX_AGE` on the controller sets the site default),
or a checkpointed record a freshly started controller merely *adopted* from
its store — those three are logged once on the job (`auto-resubmit skipped:
…`) and parked for a person. In every parked case the managed resubmit verb
above is the way to continue; it is explicit consent and is not bounded by
the automatic guards.

### Clusters

This client has no cluster-lifecycle verbs: it does not deploy an engine, open
a tunnel, or authenticate to a login node. Whoever operates the runner does
that. `--executor slurm` asks a runner that is configured for a scheduler to
submit there; `steerlab runner capabilities --runner <url>` says what the
runner can do before anything is uploaded.

### Checking on a running job

One call answers "is it waiting, and why? running, and how far along? ended,
and how?", and returns:

```bash
steerlab runner status <job-id> --runner <url> --json
steerlab runner logs <job-id> --tail 50 --runner <url> --json
```

**`status`** reads the scheduler's own view of the job, through the site's
configured scheduler commands, in fixed fields under `result.response`:

- `scheduler`: for a job that waits, the scheduler's `reason` (`Priority`,
  `Resources`, a limit) and `expectedStartAt`; for a job that runs or ran,
  `startedAt`, `nodeList`, `elapsed`, `timeLimit`, `walltimeRemainingSeconds`,
  `endedAt`, and `exitCode`. Times are epoch seconds.
- `progress`: the records written so far, counted the way a resume counts
  them (`recordsWritten`, `lastRecordAt`), out of `recordsExpected`. The total
  is exact for a part of a split run (`recordsExpectedSource: "shardStamp"`)
  and the submission's planning estimate for a whole run
  (`"preflightEstimate"`). Say which when you report it.
- `schedulerOutcome`: the scheduler's own word for an ending, or an
  interruption, that came from outside the job's code: a time limit
  (`TIMEOUT`), memory (`OUT_OF_MEMORY`), a preemption (`PREEMPTED`), a node
  failure (`NODE_FAIL`), or a cancel by a person, SteerLab, or an
  administrator (`CANCELLED`). `schedulerOutcomeNote` says it in a sentence,
  and says whether SteerLab asked for a cancel. Report it as "the scheduler
  recorded …". A preempted job that the scheduler queued again is waiting,
  not ended. The job's `status` keeps its usual words (a job that hit its
  time limit is `failed`), and `deathDetail` is the scheduler's own sentence
  on how a job that ended, ended.
- For a split run, ask about the parent: each part answers under `shards`
  (the job carrying it now, if it was resumed), and `progress` sums them.
  The read answers within about 30 seconds: a part whose scheduler did not
  answer in time says so (`scheduler.timedOut`), so ask again rather than
  concluding anything from it.
- `held` and `heldNote`: set on a part SteerLab holds because the cluster's
  job limit is reached (`held.reason` is `clusterJobLimit`, `held.since` when
  it began to wait). Its status is `pending`, and it has no scheduler view
  until the controller submits it. Report it as held by SteerLab, not as
  waiting in the scheduler's queue.

A job's `startedAt` is the scheduler's start time, and stays empty while it
waits; `createdAt` is when it was submitted. This holds for study runs,
their parts, and standalone diagnostics. A split run's parent takes the start
of its first part. A job the scheduler requeued keeps its previous start
while it waits, and takes the new one when it runs again.

`status` and `startedAt` are SteerLab's record, which the controller updates
when its own check of the scheduler sees a change; `scheduler` is read live.
So the record can lag: it can still say `submitted` of a job the scheduler
runs, or `running` of one it has ended or queued again. That usually lasts one
check, and longer when the scheduler's accounting is slow. The summary says so,
for each part of a split run. Go by `scheduler` for what the job is doing now.

**`logs --tail N`** prints the last N lines (at most 1000) of the controller's
log and of the job's own scheduler output (`slurm-<id>.out` and `.err`), then
exits; without `--tail` the verb follows the log until the job ends. A file
that does not exist yet says so.

Both cover only jobs this server submitted, named by their SteerLab job id,
never the scheduler's number. **There is no reason to `ssh` in and run
`squeue` or `sacct`.** If a question needs more than these fields, tell the
researcher what is missing rather than reaching around the instrument. An
engine older than these reads refuses with `jobStatusUnsupported`: update the
engine, and meanwhile `steerlab runner jobs` still lists the job.

### Reconnect and recover remote jobs

Retain the submission's endpoint, serving root and job ID. Poll
`steerlab runner jobs <id> --runner <url> --json`, or `runner status <id>`
(above); streaming or client timeout stops observation, not execution. Use `steerlab runner jobs <id> --cancel`
when cancellation is intended. `steerlab runner resubmit <job-id> --runner
<url> [--walltime <hh:mm:ss>] --json` continues eligible checkpointed or
cancelled jobs.

**A cancelled run can be resumed, by a person.** Cancelling a managed run
keeps every response it completed. To continue it, resume the cancelled job:
`steerlab runner resubmit <job-id> --runner <url> --json`. If the engine says to wait, wait a minute and retry; do not submit the
study again meanwhile. A new job continues the run and is named in the
cancelled record's `result.resubmittedAs`; import evidence from it. A run
cancelled in the middle of a long response is stopped before it can save its
place, and is still resumed, from the response records it had completed: the
answer carries `resumedFromRecords`, says how many records were kept
(`completedRecords`), and says that the response in progress when the run
stopped is generated again. Nothing of that unfinished response is kept or
counted. A run that had completed no response, or whose run folder cannot be
matched to the study, cannot be resumed; the refusal says why, and then you
submit the study again. For a run split across jobs, resume the parent.
Nothing resumes a cancelled job automatically.

**The Mac app sees the jobs you submit.** With a workspace named (`--root` or
`$STEERLAB_WORKSPACE`), `run`, `runner submit`, `runner resubmit`, and
`runner science-submit` record where each new job went in the workspace's
`.steerlab/job-origins/origins.json`. The Mac app reads that record, so the
researcher can import the job's evidence there without reconnecting. Never
edit or delete that file; it holds no token. A warning that the origin could
not be recorded means only that the app will ask for a reconnect by job ID:
the job was submitted.

Batteries and stability diagnostics have no checkpoint resume; never resubmit
an uncertain scheduler launch without inspecting its `schedulerSubmissionName`.

For stranded controller-owned work, inspect `runner recovery <job-id>`
against the original endpoint. Confirm the recorded controller exited before
`recover`, using `--review-token`, `--reason` and `--confirm-owner-exited`.
This invokes the existing ownership gate and records the operator's assertion;
an expired review or live owner refuses. It does not restart a computation.
Use `runner reconcile --runner <url>` to fold all known child records and run
the existing merge pass on that endpoint. No recovery command grants cleanup
permission. Keep local custody verification and site retention policy separate
from execution status.
