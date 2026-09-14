# Review: claude/auto-resubmit-staleness-guard

Date: 2026-09-13. Reviewed at branch head a3e11de (three commits over main
2aea1b9), merged with main f7b83db for verification (no conflicts; main's
later commits are the UI follow-up and review documents). Origin: the live
controller incident of the same evening, recorded in
`docs/AUTO-RESUBMIT-STALENESS-GUARD-2026-09-13.md` on the branch.

## 1. What the branch does

- **Scheduler observations carry the end time.** `SlurmExecutor.poll_observation`
  reads `State,ExitCode,End` from sacct and returns a `SchedulerObservation`
  (mapped state, raw state, exit code, epoch end time); `poll_state` and
  `poll_state_detailed` delegate to it unchanged. `sacct -j` needs no
  explicit start time (the manual widens the default window to epoch 0 when
  `-j` is given).
- **Three guards on the automatic resubmit path**, applied after the enabled
  check and the missing-stamp repair, each a deduplicated note that leaves the
  record untouched: the observed scheduler state (this tick's or the stamped
  `result.lastSchedulerState`) is `cancelled`; the record's last activity is
  older than the bound (`autoResubmitMaxAgeSeconds` on the request, else
  `STEERLAB_AUTO_RESUBMIT_MAX_AGE`, else 48 hours; malformed values refuse
  loudly); or this controller instance did not witness the checkpoint (it
  adopted the record already checkpointed). Last activity is the newest of the
  record's lifecycle stamps, the child record's own checkpoint time, sacct's
  end time, and the newest log line written before this instance started; the
  poller's own lines do not count. Witnessing is in-memory: a status that was
  not `checkpointed` at the start of a tick and is `checkpointed` at its end,
  or a record this instance created checkpointed.
- **Quiescent poll loop.** The handled scheduler state is stamped on the record
  after the fold and preserved across folds (with `checkpointedAt`); the same
  observed state over a fold-restored `checkpointed` is not re-logged or
  re-folded on later ticks; a different state is handled afresh.
- **Content-idempotent folds.** `reconcile` skips a record file whose bytes
  this process already folded (sha256 by path) and no longer re-appends child
  log lines; the operator's explicit reconcile verbs pass `force=True` and
  behave as before. This closes the agent's explanation of the 21:28 timing:
  an unrelated job's terminal transition re-folded the whole shared records
  directory and resurrected the July record's `checkpointed` status.
- Request key `autoResubmitMaxAgeSeconds` on submissions; changelog entry;
  guide text on the auto-resume bounds (agent guide and its generated
  copies); the incident and design note.

## 2. What I checked

- Read the full diff: executors (+95), jobs (+276), routes (+4),
  submissions (+8), fake sacct and its test, 16 new tests (399 lines),
  changelog, docs, regenerated identity and bootstrap text.
- The incident's exact shape against the new code: the July record would
  have been refused on the first guard (scheduler state `cancelled`), and
  independently on the second (inactive seven weeks) and the third
  (adopted); the poll loop would have logged the transition once.
- The legitimate paths that must keep working: a job running under this
  controller that exits 85 is witnessed and resubmits; a job that checkpoints
  while no controller is up is loaded as `running` and its checkpoint is
  witnessed by the next controller's first observation, so it resubmits; the
  masked-85 correction (a `failed` observation over a checkpointed fold)
  still resubmits; manual Resume keeps its own gate and works on stale or
  adopted records.
- Time handling: sacct's `End` is parsed in the scheduler host's local time,
  which is where the controller runs.
- Gates on the merged tree: `check-generated.py --audits` PASS,
  `public_scan.py` clean, `git diff --check` clean, identifying-vocabulary
  grep clean (the incident note names no site, login, or case).
- Suites on the merged tree: see §4.

## 3. Findings

No landing fix. Notes:

- **N1 Adopted checkpoints need a manual Resume.** A record that is already
  `checkpointed` when a controller starts (for example a shard parked during a
  maintenance deferral across a controller restart) is no longer resumed
  automatically. This is the intended trade; operators running long rounds
  should keep one controller up for the round's duration, and the note on the
  record says what to press.
- **N2 Witness state is per process.** Two controllers would each treat the
  other's checkpoints as adopted; the single-controller discipline already in
  force makes this moot.
- **N3 Records directory digests are per process.** After a restart the
  first fold of every record happens again once, by design.
- **N4 Live acceptance owed.** The running controller must be restarted to
  load the guard; the first restart is the acceptance: the two cancelled
  incident records must stay quiet, and a fresh checkpoint under the new
  controller must still resume. Recorded below when done.

## 4. Suite results on the merged tree

Python (`Server/`, main venv, `HF_HUB_OFFLINE=1`): 6,604 passed, 9 skipped,
8 warnings in 217 s (16 new tests; the merged tree also carries the UI
branch's tests).

Swift (Xcode beta, Metal toolchain 32023.920.1, serial, coverage mapping off,
test Python = main venv): TEST SUCCEEDED, 4,951 tests, 4,946 passed, 5 skipped,
0 failed (identity gate satisfied by the regenerated constant).

## 5. Landing

Fast-forward of main to the merge commit, this review on top, app rebuilt
and installed (generated identity and bootstrap text changed), engine pushed,
controller restarted.
