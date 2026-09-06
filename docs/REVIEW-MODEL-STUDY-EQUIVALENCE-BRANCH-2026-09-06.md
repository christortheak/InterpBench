# Review — `codex/model-study-equivalence` at ae6d928 (2026-09-06)

Reviewer: Claude (maintainer's reviewing/integration agent). Range reviewed:
`55606d9` (current main) to `ae6d928`, one commit, 60 files,
+1,794 / −88. Main is an ancestor; the branch fast-forwards. Worktree
`/private/tmp/interpbench-workflow-handoff` clean at the tip, no venv. No
edits were made to the branch.

## 1. Verdict

**Landable by fast-forward.** Both suites are green on ae6d928 (§2). The model
preparation path is a read-only plan plus a reviewed install over the
existing durable job, with the egress policy enforced at the service and
the new Mac verbs resolving credentials only from the Keychain. The
authoring owners on both sides publish immutable inputs by hard link and
edit only reviewed drafts, and the previous review's coverage gap (a real
Python-cast agent decoded by the Mac) is closed. Two notes (§4), and one
deployment consequence worth stating (§5).

## 2. Verified independently

| Check | Result |
|---|---|
| Lazy-import audit, interview gate, client reference check, task-prompt parser audit, bridge gates normal and `--release` | all pass on the tip |
| Route census | `GET /api/models/plan` is censused RUNNER; the "runner reads without a token gate" test is generalized from "the expected three" to an explicit list that now includes the plan route; the runner-reachability test lists both model routes |
| Vocabulary in the diff (including the three regenerated golden env fixtures) and commit message; `git diff --check` | clean |
| Full Python suite on the worktree (main venv, cwd = worktree/Server) | 6,036 passed, 9 skipped, 8 warnings, matching the branch's claim |
| Full Xcode beta suite from the venv-less worktree, `TEST_RUNNER_STEERLAB_TEST_PYTHON` in the shell environment before `xcodebuild` | `TEST SUCCEEDED`: 277 SteeringKit + 4,581 ExperimentKit, matching the branch's claim; the parity tests ran the real Python client |

## 3. What the code does

**Model preparation, engine side (`api/model_preparation.py`).** `plan`
validates the model id through the hub's own validator, refuses MLX repos
and malformed revisions, and returns a material record (cache root, profile,
topology, workspace root, `computeEgress` from `STEERLAB_COMPUTE_EGRESS` or
`unknown`) with its SHA-256 as `planSHA256`. Cache observation is
conservative: `config.json` located through the hub cache API, tokenizer
files present, and either a non-empty single weights file or an index
whose every shard exists non-empty with no path escape; any `.incomplete`
blob makes the set not present. It never contacts the hub or loads
weights. `admit_install` refuses with 409 when egress is declared `no`
and, when a `planSHA256` is supplied, requires it to match the current
plan (412 otherwise). The existing install route keeps its durable job
and its MLX refusal; the precondition is optional for existing callers.

**Egress declaration.** Both site environment renderers emit
`STEERLAB_COMPUTE_EGRESS` only when the profile declares a known value;
the three golden fixtures gain one line each and carry no site
vocabulary. An undeclared value stays `unknown`, which preserves today's
explicit-install policy. Delivery to a live deployment needs the profile
rematerialized; the branch edits no configuration.

**Model preparation, clients.** Python `model plan/install/status/cancel`
require `--runner`; install requires `--plan-sha256`, posts exactly once,
and treats a response without a job id as uncertain rather than retrying;
status and cancel refuse a job whose kind is not `model:install`. The Mac
`remote model-plan/install/status/cancel` verbs take only `--site` or
`--url` for connection; with `--url` the token comes from the Keychain
under the URL's key and the `--token` flag is not accepted. The app's
Plan action and its install both run on a captured connection profile
and drop late replies when the selection changes.

**Study authorship, Python.** `agent list/inspect` walk `runs/` without
following links and decode through the existing artifact model;
`attach-agent` runs inside the reviewed-draft lock and embeds the exact
reviewed artifact. `panel check/import/inspect/list` validate a semantic
panel by binding throwaway model settings and running the engine's
validator, then publish by content hash with `os.link`; `panel compile`
requires both digests, casts every seat, pins the compiled panel, and sets
the study to multi-agent. `set-pipeline` validates through the engine's
`resolve_pipeline` and replaces or clears the block under the review.
`design expand` is read-only, bounded to 64 seats and 4,096 rows, and
emits multiset permutations or the baseline / each-solo / all-treated
composition.

**Study authorship, Swift.** `StudyPanelAuthoring` mirrors the above:
ordinary-path admission under `prompts/panels/`, validation through a
rehearsal scenario and `MultiAgentRunner.validate`, publication by
temp-file plus `linkItem`, and `compile`/`saveAssignment` that require
the reviewed draft and panel digests and then delegate to the existing
`SeatCasting.compile`. The app's seat save and the Mac `panel compile
--seat` form now both go through `saveAssignment`; a changed semantic
panel saved in the app becomes a new immutable version instead of
rewriting a file an older study may pin, and an unchanged save reuses the
selected record. `StudyPipelineAuthoring.saveBlock` validates with the
store's existing violation check. `StudyDesignExpansion` produces the same
rows as Python, and the parity test asserts the batches are equal for
both modes. Agent catalog paths are canonicalized so a `/var` enumeration
does not leak into a pin.

**HTTP.** Six new workbench routes with closed field sets per operation;
unknown fields refuse; missing preconditions 428, stale 412.

## 4. Notes, not blockers

### N1 — `panel compile` changes the study kind as a side effect

Compiling a panel into a study sets `studyKind` and `studyType` to
multi-agent on both clients. That is the existing Mac behaviour carried
over, and it is what compiling a panel means, but an agent that compiles
into a concept study by mistake will find the study's type changed. The
draft review and the refusal on a frozen study bound the damage. Worth a
sentence in the workflow document.

### N2 — Plan material includes absolute deployment paths

`planSHA256` covers `cacheRoot` and `workspaceRoot` as absolute paths, so
the same request against a differently mounted view of the same cache
yields a different digest. That is the intent (the digest binds the
deployment), and the paths are already exposed by the capabilities route,
so nothing new leaks. Recorded so nobody expects the digest to be portable.

## 5. Landing instructions and deployment consequence

1. Both suites are green on ae6d928 (§2).
2. Fast-forward main to the tip.
3. Deployment: the app rebuild already owed now also carries the Plan
   action and the shared panel owners; the engine deploy already owed
   carries the plan route and the egress check; and the site profile's
   rendered environment must be regenerated for the egress declaration to
   reach the engine. None of this changes the order already in force
   (app first, engine after the running ladder job) and none of it is
   performed by the branch.
