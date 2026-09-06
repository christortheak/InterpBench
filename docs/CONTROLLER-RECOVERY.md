# Controller recovery

Job observation is read-only: listing jobs never fails or cancels them. A
controller restart can recover local worker jobs and incomplete submission
fan-out parents only when the former controller's exit is established. Normal
scheduler-executed jobs remain the poller's responsibility.

## Automatic recovery

Ownership metadata lives in the job database, outside manifests, freezes and
historical runs. It records host, PID, a unique ownership instance and, when
available, the **controller's** Slurm allocation. This is separate from the
allocation of any compute job the controller submits.

On this host, an absent PID proves exit. A live PID prevents recovery, including
when a PID has been reused. For a controller on another node, accounting must
identify the recorded cluster, raw job ID, database index, submission time and
start time, and report an unambiguously terminal allocation. Capture happens
once per controller process while its allocation is running. Missing accounting
at capture leaves the identity unknown; it is not filled in later by guessing.

Queries use the configured `STEERLAB_SLURM_SACCT` wrapper and the existing
scheduler timeout. They request allocation rows, including duplicate IDs, and
normalize timestamps to UTC. Slurm documents that IDs can recur and that
accounting may hide records; neither an empty result nor a failed query is
proof of exit. See the [official accounting reference](https://slurm.schedmd.com/sacct.html).
Requeued, preempted, incomplete, mismatched or ambiguous results stay uncertain.
Scheduler queries hold no SQLite write lock. Before claiming recovery, a single
transaction rechecks the complete job and owner snapshot, transfers ownership,
and records the evidence. A competing change invalidates the claim.

## Legacy or uncertain ownership

Startup prints `jobRecoveryRequired` with the number of eligible jobs whose
owner cannot be established and leaves those jobs unchanged. This is a request
for investigation, never a signal to automatically confirm every record.

1. `steerlab-server jobs list --json` locates the job.
2. `steerlab-server jobs recovery <job-id> --json` returns `ownerState`,
   eligibility, owner details, `reviewToken` and repair instructions. It does
   not change the job or its owner.
3. Establish independently that the original controller exited. If its status
   is still uncertain, stop here. For old ownerless rows, this may require
   checking the old service's shutdown and scheduler history with an operator.
4. Recover that reviewed record:

   ```sh
   steerlab-server jobs recover <job-id> --review-token <reviewed-token> \
     --confirm-owner-exited --reason "Controller shutdown independently verified" --json
   ```

Use the same engine metadata database for review and recovery (the deployment's
configured metadata root or `STEERLAB_JOBS_DB`). Do not include credentials or
private study content in the reason. The reason describes the exit evidence.

The command refuses a changed snapshot, an ineligible job, or a known live
controller. It records timestamp, actor host/PID, previous owner, review token,
reason and `operator-attested-exit`, which is explicitly different from
`proven-exit`. Explicit recovery uses the same finalization and shard cleanup
as automatic recovery. Failed shard cancellation retains `cleanupIncomplete`;
it never reports unconfirmed cancellation as success. Recovery changes execution
metadata, not frozen scientific artifacts, and never resubmits the study.

`jobs reconcile <records-dir>` reconciles executor records and is a different
operation. Do not use it as a substitute for controller recovery.

## Qualification

Regression fixtures cover process death, cross-node accounting, ID reuse,
requeues, uncertainty, concurrent owner/job changes, legacy migration/reporting,
explicit attestation and observation. These use disposable databases and mocked
scheduler boundaries. They do not qualify a live cluster configuration.
