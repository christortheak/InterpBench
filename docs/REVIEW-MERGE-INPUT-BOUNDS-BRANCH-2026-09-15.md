# Review: merge-input bounds branch (2026-09-15)

Branch `claude/merge-input-bounds` at 32c8d9c, one commit on main c196213,
landed by fast-forward. Companion note: `docs/MERGE-INPUT-BOUNDS-2026-09-15.md`.

## What prompted it

The first full-scale J-lens fitting round (eight shards of a 27B model, each
run directory about 13 GB) completed, and the round's `merge-plan` action was
refused after roughly five minutes of hashing with the transport refusal
"The diagnostic archive is empty or exceeds transport bounds." The merge
planner pinned the whole `fits` trees (about 106 GB) through
`diagnostic_archives.snapshot`, whose 16 GiB bound was written for archives
that travel. Nothing in that path transports bytes: the fits stay on the
controller's filesystem and the merge child reads them there.

## What the branch does

- `diagnostic_archives.pin(root, paths)`: same entry shape as `snapshot`, bounded
  by `MAX_FILES` and a separate `MAX_PINNED_BYTES` (1 TiB, a sanity bound
  against pinning a whole workspace by accident), own refusal text.
- `managed_inputs.inventory` returns `pin(...)`; every other `snapshot` caller
  (packaging, staging, verify, export, cleanup, import, campaign static files)
  is unchanged, so what may travel is still decided by the 16 GiB bound and
  the same refusal text. A pinned closure asked to travel is refused at
  `package`.
- Both inventories refuse on declared sizes before hashing, so an over-bound
  inventory no longer spends minutes reading bytes it will never accept; the
  hashed entries are bounded again afterwards.
- Seven new tests (`Server/tests/test_managed_input_bounds.py`) cover the
  over-bound merge plan through `managed_inputs.plan`, `diagnostic_inputs.plan`
  and a real `scientific_execution.plan`, the unchanged transport refusals at
  package, export, staged-copy verification and battery plan, the pin bounds,
  and the refuse-before-read behaviour.
- `PythonClientIdentity.swift` regenerated (it hashes the Python sources).

## Review

I read the whole diff. The split is the right one: transport bounds belong
where bytes are packaged, staged, exported, imported or verified, and the
in-place inventory only pins content. Plan entries keep their shape, so plan
hashes of unchanged inputs are unchanged. The pre-hash size check is a
behaviour improvement with no contract change (an empty or over-bound set was
refused before too, only later).

Suites on the exact branch tree, run by me:

| suite | result |
|---|---|
| Python (`HF_HUB_OFFLINE=1 pytest`) | 6623 passed, 9 skipped, 0 failed |
| Swift (`xcodebuild test`, serial) | 4954 passed, 5 skipped, 0 failed |

Gates: `check-generated.py --audits` PASS, `public_scan.py` clean,
`git diff --check` clean, vocabulary scan clean.

## Left for follow-up (recorded in the note)

- `merge-submit` hashes the closure twice in one request because the round
  action is not itself an `input_hashes` session; wrapping it would halve the
  cost without weakening pinning. Proposed, not done.
- The fitting-round route is a synchronous handler, so a multi-minute plan
  holds the request threadpool and the submission lock; a client timeout can
  abandon a plan that completes unobserved. Known follow-up.

## Deployment

The compiled client identity moved, so the app is rebuilt after this landing
and the cluster payload pushed after the rebuild; the merge then needs a fresh
controller running the new engine.
