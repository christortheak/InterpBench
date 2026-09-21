# Fitting-round actions as one hashing scope (2026-09-21)

The bounded hash reuse proposed in `MERGE-INPUT-BOUNDS-2026-09-15.md` ("Round
trip timing, noted and left alone") and item 1 of
`JLENS-AND-P7-STATUS-HANDOFF-2026-09-21.md`, specified in §8 of
`JLENS-P7-RUNNING-AGENT-MEMO-2026-09-21.md`. Engine only; no schema, wire, or
client change.

## What was hashed more than once

`input_hashes.file_hash` reuses a digest only inside one `session()`, and
every `@input_hashes.operation` (`scientific_execution.plan`, `input_plan`,
`managed_inputs.plan`, `diagnostic_transport.verify_inputs`) opens its own
session when none is active. `jlens_rounds.action` was not a session, so one
`merge-submit` request over an eight-shard 27B round read its inputs as
follows, each pass about five minutes for roughly 106 GB:

1. `context()` resolves the execution capsule: `verify_inputs` hashes the
   staged archive and every staged member (fitting request, corpus).
2. The round plan calls `scientific_execution.plan` for the merge request:
   the capsule is verified again, then `managed_inputs.plan` pins the merge
   closure, which is every file of every completed shard's run directory
   (the fitted tensors, checkpoints, and reports).
3. `scientific_execution.submit` re-plans in its own session: the capsule a
   third time and the merge closure a second time.

The shard `submit` action had the same shape: the capsule in `context()`, then
once per pending shard in the round plan and once more in each submit re-plan,
with the staged corpus and fitting request hashed for every shard plan.

## The chosen scope

`jlens_rounds.action` now runs its whole body inside
`with workspace_lock.submitting(), input_hashes.session():`. The inner
operations join that scope instead of opening their own, so within one request
each input is read once and every later use compares the file's fingerprint
(device, inode, size, mtime, ctime, mode) against the stamp taken when its
digest was computed. The whole body is the smallest scope that covers both the
round plan and the re-plan inside `scientific_execution.submit`; the
read-only `status`, `plan`, and `merge-plan` actions gain the same reuse
between `context()` and the plan, which is free.

The scope's own exit check runs only when the action returns, which is after
the state write and the submission. That is why the action also calls
`input_hashes.recheck()` (new, stat-only, no bytes read) through
`reviewed_inputs_unchanged()` immediately before every state write: before
appending to `mergeAttempts` in `merge-submit`, and before appending each shard
index to `attempted` in `submit`. Ordering inside `merge-submit` is therefore:

1. plan (reads bytes once), compare `planSHA256` with the reviewed one;
2. `recheck()` every reviewed input; a change refuses here, and nothing has
   been recorded or submitted;
3. write `mergeAttempts`;
4. `scientific_execution.submit` re-plans, which rechecks every input again at
   use before any submission directory, job record, or `sbatch` exists;
5. return; the scope's exit check runs last.

A change in the window between steps 3 and 4 is still refused before anything
queues, but the attempt stays recorded, as for any post-record failure (the
existing "already attempted a merge; inspect its jobs" rule). A change after
step 4 is detected by the exit check and surfaces as a refusal on a request
whose job was nevertheless submitted; the durable job and the recorded attempt
remain, and the child refuses on its own. Neither window is new; both are
narrower than before.

## The three guarantees and the tests that prove them

All in `Server/tests/test_jlens_rounds.py`, over a two-shard round whose
shard 0 is a real (tiny, CPU) fit under the capsule root. `byte_reads` counts
calls of `hashlib.file_digest`, the one primitive `input_hashes` reads with,
per full path.

1. **One read of an unchanged large input within the action.**
   `test_merge_submit_reads_each_input_once_across_plan_and_submit_replan`
   observes `scientific_execution.plan` being called twice for the merge
   request inside the action, both inside the shared scope, while the fitted
   tensor, the staged archive, and every other input are read exactly once.
   `test_shard_submit_reads_each_input_once_and_rechecks_before_each_attempt`
   does the same for two pending shards submitted in one action.
2. **Detection of mutation or replacement before the boundary.**
   `test_merge_submit_refuses_a_changed_input_before_recording_an_attempt`
   appends a byte to the fitted tensor after the merge closure was hashed and
   before the action seals its plan; the action refuses with "changed before
   review finished ... Nothing was recorded or submitted", `submit` is never
   called, no job appears, the round state holds no attempt, and a fresh
   `merge-plan` carries a new hash.
   `test_merge_submit_replan_refuses_a_change_after_the_intention_record`
   mutates from inside a patched `submit` before its re-plan: the re-plan
   refuses at use, no job appears, and the attempt stays recorded.
   `test_shard_submit_stops_before_recording_a_later_attempt_when_an_input_changes`
   mutates the staged corpus after shard 0 is submitted: shard 1 is neither
   recorded nor submitted. `test_recheck_is_stat_only_and_scoped` pins the
   primitive: 0 outside a scope, one stat per reviewed input inside it, and
   a mutated file fails both `recheck()` and the next `file_hash` without
   being re-read. Replacement (a new inode under the same path) is covered by
   the existing fingerprint comparison, which includes device and inode.
3. **A fresh verification scope for the queued worker, and none across
   requests.** `test_queued_child_verifies_in_its_own_scope` runs
   `execute_packet` on the packet the action submitted, after the action
   returned with no scope active: the child's `verify_inputs` and
   `input_plan` read the tensor and the archive again.
   `test_sequential_requests_do_not_share_digests` runs `merge-plan` twice
   and shows every path's read count doubling, with no scope active between.

## Deliberately not done

- No persistent receipt cache, no capsule receipt trust, no cross-request or
  cross-process reuse: the scope is a `ContextVar` set on entry and reset on
  exit of one `action` call, and the child is a separate process that opens
  its own scopes.
- No change to what the child verifies, and no deduplication across the
  child's own operations (its `verify_inputs` and `input_plan` still each
  open a scope).
- No change to `jlens_merge.source`, which hashes each fit's two tensors again
  at execution; that is the executing owner's own custody check.
- No move of the long hashing off the request threadpool (item 2 of the
  status handoff); this change halves the work but does not change where it
  runs.
