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
`--request-timeout` is the per-request budget. `--no-wait`, ctrl-c during the
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
job and, on Slurm, spends an allocation. A timeout there, or a
`submitOutcomeUnknown` from `run`, means "look with `runner jobs`", not
"submit again". Upload and evidence download are safe to retry. On
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
are not optional. **Verify the shard jobs actually landed:** a fan-out can
partially fail while the submit still succeeds, so count the jobs with
`steerlab runner jobs --runner <url>` and never report a sharded submit as
successful on the exit code alone. **And stagger submissions where the site
caps queued jobs per user**; ask the researcher for the limit rather than
guess. The merge is performed by the running runner service, not by the
submitting process.

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

### Reconnect and recover remote jobs

Retain the submission's endpoint, serving root and job ID. Poll
`steerlab runner jobs <id> --runner <url> --json`; streaming or client timeout
stops observation, not execution. Use `steerlab runner jobs <id> --cancel`
when cancellation is intended. `steerlab runner resubmit <job-id> --runner
<url> [--walltime <hh:mm:ss>] --json` continues eligible checkpointed or
cancelled jobs.

**A cancelled run can be resumed, by a person.** Cancelling a managed run
keeps every response it completed. To continue it, resume the cancelled job:
`steerlab runner resubmit <job-id> --runner <url> --json`. If the engine says to wait, wait a minute and retry; do not submit the
study again meanwhile. A new job continues the run and is named in the
cancelled record's `result.resubmittedAs`; import evidence from it. A run that
was stopped before it could save its place cannot be resumed: the cancel
answer's `cancelResume` field says which it is, and then you submit the study
again. For a run split across jobs, resume the parent. Nothing resumes a
cancelled job automatically.

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
