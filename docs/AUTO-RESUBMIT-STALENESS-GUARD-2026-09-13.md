# Auto-resubmit staleness and provenance guards (2026-09-13)

Branch `claude/auto-resubmit-staleness-guard`. Offline engine work with unit
tests only; nothing here touched a cluster, a running controller, or a
workspace.

## 1. The incident

A live controller on 2026-09-13 auto-resubmitted a **seven-week-old test
record** and an A100 job ran for 24 minutes before a person cancelled it.
What the controller's job store showed:

- A `study-submit-bundle` record created on 2026-07-22, status
  `checkpointed`, `cancellation_requested = 0`, with
  `auto_resubmit: true, auto_resubmit_limit: 5` in its requested resources.
  Its Slurm job had been **cancelled on 2026-07-22** (the job log carries the
  `→ cancelled` transition from that day), yet the record read
  `checkpointed`: the transition's child-record fold restored that status from
  the July child record, which honestly said "checkpointed" — it had parked a
  resumable run before the cancel.
- The controller started at 18:40 that evening. At 21:28:09 its poll loop
  logged the same `→ cancelled` transition again for the record, folded the
  child record (restoring `checkpointed`), passed the auto-resubmit gate, and
  submitted the July `run.sbatch` verbatim as a new Slurm job (`auto-resubmission
  … (1/5)`).
- From 21:28 onward the loop logged `→ cancelled` for that record **every
  tick** (~18 s): each tick saw sacct's `CANCELLED` ≠ the stored
  `checkpointed`, wrote the transition, folded, and the fold restored
  `checkpointed`.

Three independent defects lined up:

1. **Cancelled did not beat checkpointed on the observed state.** The gate
   consulted only the record's own `cancellation_requested` flag, which was
   0 (the July cancel came from the scheduler side or a hand `scancel`, not
   through the API). The scheduler's own answer — the one the loop had just
   logged — was never consulted.
2. **No notion of age or provenance.** A record inactive for seven weeks, and
   one the controller had never seen checkpoint, was treated exactly like a
   shard that parked thirty seconds ago under this controller's watch.
3. **The fold-restore loop was not quiescent.** Nothing recorded that the
   `cancelled` observation had already been handled, so the identical
   transition was re-logged and re-folded on every tick (and the child's own
   log lines were re-appended on every fold).

## 2. Why 21:28 and not 18:40 — the sacct question

The open question was why the sacct lookup for the old job only "started
returning a state" at 21:28 rather than at controller start. Two hypotheses
were examined.

**Time window (rejected).** `poll_state_detailed` ran
`sacct -j <id> -n -o State,ExitCode -P` with no explicit `--starttime`. sacct's
default window is 00:00 of the current day — *unless* `-j` (or `-s`) is given,
in which case the manual widens the default to epoch 0. A job that ended in
July therefore answers a `-j` query at any time of day. No `-S` was added; an
operator can confirm on their site with
`sacct -j <old-id> -n -o State,ExitCode,End -P`.

**Shared records directory re-fold (the plausible mechanism).** A records
directory is shared by every submission of a workspace, and `poll_slurm` calls
`reconcile(records_dir)` — which folds **every** `.json` in the directory —
whenever any one job reaches a terminal state. The sequence that fits the
evidence: the record was terminal (`cancelled`) in the store when the
controller started, so it was not in the poll set at all from 18:40; at about
21:28 some *other* job in the same records directory reached a terminal state,
its fold re-read the July child record and restored `checkpointed` on the old
record; the next tick found a non-terminal Slurm record, asked sacct, got
`CANCELLED`, logged the transition, folded (restoring `checkpointed` again),
and auto-resubmitted. This explains the exact log shape — nothing on the record
between July and 21:28, then a transition every tick — without requiring sacct
to have been silent for three hours. It cannot be proven from the job store
alone (the sibling transition would be on a different record), so the branch
closes **both** vectors rather than betting on one.

The re-fold is now structurally impossible: `reconcile` skips a record file
whose bytes this process already folded (SHA-256 by path), so an unrelated
job's terminal transition no longer re-folds months-old siblings, and a fold
no longer re-appends the child's log lines. The operator's explicit reconcile
verbs (`POST /api/jobs/reconcile`, `steerlab-server jobs reconcile`) pass
`force=True` and behave exactly as before. A rewritten record file (new
bytes) folds again.

## 3. What the automatic path now refuses

Applied in `_maybe_auto_resubmit` **after** the enabled check and the
missing-stamp repair, in this order; every refusal is a deduplicated
`_resubmit_note` (logged once per distinct situation, not per tick) and leaves
the record exactly as it was:

| Guard | Condition | Note on the job |
|---|---|---|
| Scheduler cancelled | this tick's observed scheduler state, else the stamped `result.lastSchedulerState.state`, is `cancelled` | `not auto-resubmitting: the scheduler recorded Slurm job N as cancelled (cancelled beats checkpointed) — resubmit manually if it should continue` |
| Stale | `now − lastActivity > bound`; bound = `autoResubmitMaxAgeSeconds` / `auto_resubmit_max_age_seconds` on the request, else `STEERLAB_AUTO_RESUBMIT_MAX_AGE` (seconds), else **48 hours** | `auto-resubmit skipped: record inactive for N hours (bound M hours) — resubmit manually if it should continue` |
| Adopted | this controller instance did not witness the record becoming `checkpointed` | `auto-resubmit skipped: this controller adopted the record already checkpointed (the checkpoint happened before it started) — press Resume in the app or run the remote resubmit verb (POST /api/jobs/{id}/resubmit) to continue it` |

Only a checkpoint exit (85) — seen by sacct or corrected by a fold — means
"resumable"; a `failed` observation over a checkpointed fold (the wrapper
that masks the 85) still resubmits, as before.

**Last activity** is the newest of: `created_at`, `started_at`,
`finished_at`, `result.checkpointedAt` (the child record's own `finishedAt`
when it wrote one, else sacct's recorded `End`, else the observation time),
sacct's `End` (this tick or stamped), and the newest job-log line written
**before this controller instance started**. Lines this instance wrote are
excluded on purpose: the tick evaluating the guard has just written its own
transition notes, and a controller observing a months-old record late is not
the record becoming active.

**Witnessed** means one of: this instance's poll loop held a non-checkpointed
status for the record at the start of a tick and the tick ended with it
`checkpointed` (sacct 85, or a fold correcting a masked 85); or this instance
created the record already checkpointed (`record_external`). A fold that
restores `checkpointed` over a terminal state the same tick observed is not a
witness (the pre-tick status was already `checkpointed`). The set is
in-memory by design — it records what *this process* saw.

`sacct` polls now read `State,ExitCode,End`; `SlurmExecutor.poll_observation`
returns the mapped state, the raw state, the exit code, and the epoch end
time, and `poll_state_detailed` / `poll_state` delegate to it unchanged.
Executors that only offer `poll_state` (test doubles, older wrappers) still
work — the loop reads state alone from them.

## 4. Quiescence

`poll_slurm` stamps `result.lastSchedulerState = {state, observedAt,
endedAt?, rawState?}` after handling a transition (after the fold, which
replaces the result; `reconcile` preserves `lastSchedulerState` and
`checkpointedAt` across folds alongside `resubmittedAs`). A later tick that
observes the same state over a fold-restored `checkpointed` record does
nothing; a *different* observed state is handled afresh. The auto gate still
runs each tick for checkpointed records (a maintenance-window deferral must
still fire later), but its notes deduplicate.

## 5. Deliberately unchanged

- **Manual resume** (`POST /api/jobs/{id}/resubmit`, the app's Resume, the
  `remote resubmit` / `runner resubmit` verbs) keeps its own gate untouched:
  a stale or adopted checkpointed record is resumable by a person; a record
  that reads terminal is still refused. The gate was not widened.
- **The fold's precedence.** A child record that says `checkpointed` still
  wins over the poller's terminal transition when it folds, as before — that
  is the masked-85 correction contract. The guards make the consequence
  harmless instead of changing which side is believed.
- **Chain cap, maintenance-window deferral, claim/token crash-safety** — all
  as before; the new guards sit in front of them.
- **The env default for the bound is read at reconcile time** rather than
  stamped at submission (unlike `auto_resubmit_limit`), so tightening or
  widening `STEERLAB_AUTO_RESUBMIT_MAX_AGE` on the controller applies to
  every parked record without resubmitting anything. A request that pins
  `autoResubmitMaxAgeSeconds` is stamped and wins.

## 6. Tests

`Server/tests/test_auto_resubmit_staleness.py` (16 tests): the incident's
shape on a restarted controller (no sbatch, cancelled note, stamp durable);
the same veto within one instance; a cancelled job without a child record
simply reads cancelled; a witnessed-but-72-hour-old checkpoint is skipped
with the 48 h note and stays quiet on later ticks; env tightens and the
per-request key wins; malformed bounds refuse loudly; the request key pins the
resource; a fresh witnessed checkpoint still resubmits; an adopted record is
surfaced for manual Resume; adoption still repairs a missing `resubmittedAs`
stamp; the fold-restored transition and the child's log lines appear once
across six ticks; a changed scheduler state is handled afresh; `reconcile`
skips unchanged files and `force` re-folds; manual resubmit works on a stale
adopted record; the executor reads sacct's `End`. The fakebin `sacct` gained
an optional `end` column; the wrapper-name assertion in
`test_fake_scheduler.py` follows the new column list.
