# Review — `codex/researcher-workflow-authoring` at 2662dc8 (2026-09-06)

Reviewer: Claude (maintainer's reviewing/integration agent). Range reviewed:
`95d14aa` (the approved implementation tip) to `2662dc8`, two commits:
`64102bf` (management bridge retirement, claimed mechanical) and `2662dc8`
(freeze and remote coordination retirement, declared semantic). Worktree
`/private/tmp/interpbench-workflow-handoff`, clean. 48 files,
+1,289 / −557; no Python source or test changed. Main `bfd13a5` and
`95d14aa` are both ancestors, so the branch fast-forwards. No edits were
made to the branch.

## 1. Verdict

**Landable.** Both slices do what the handoff says, the mechanical claim
reproduces, the semantic slice is a genuine tightening with tests that
exercise the retargeting paths, and no substitute forwarding layer was
introduced. All four bridge files are gone and the release-mode bridge gate
passes. Two notes in §4; neither blocks.

## 2. Verified independently

| Check | Result |
|---|---|
| Bridge files on the tip | `StudyFreezeBindings`, `StudyManagementBindings`, `StudyPanelBindings`, `StudyRemoteCoordinationBindings`: all absent |
| `scripts/ci/check-swift-bridge-retirement.py` normal and `--release` | both PASS |
| Substitute forwarding layer | none. `ExperimentPanel.swift` gains two computed readers (`studyCreationContext`, relocated with its original body per the audit, and `operationEnvironment`, a context reader plus a credential resolver closure). Every other panel change is `selected` → `management.selected` and `selectedName` → `management.selectedName`, plus calls into the four controllers. Zero get/set forwarding accessors before and after. |
| Management syntax audit, recompiled from `64102bf`'s script, run on `git archive` trees of `64102bf` vs `95d14aa` | "Audited 6 properties, 16 commands and 30 changed files; syntax-tree differences: 0" |
| Vocabulary in the diff and both commit messages | none |
| `git diff --check` | clean |
| Full Python suite on the worktree (main venv, cwd = worktree/Server) | 5,928 passed, 9 skipped, 8 warnings; no Python files changed, run anyway per the landing method |
| Full Xcode beta suite on the worktree (serial, external DerivedData, app not running) | `TEST SUCCEEDED`: 277 SteeringKit + 4,539 ExperimentKit, matching the handoff's count |

## 3. The semantic slice, read against the handoff's five claims

1. **Admission before and after credential resolution.** `StudyOperationContext.matches` compares workspace root, server URL, server flag, pairing, the client's profile, and the evidence-import origin (server identity plus serving root); optionally the selection. Every new entry point (`freezeOnServer(in:)`, `pushManifest(in:)`, `StudyServerJobCoordinator.run(…in:)`, `StudyBundleSubmissionController.submit(…in:)`) checks `isCurrent` before `connect()`, then again after, and additionally requires the returned client's profile to equal the captured one. `isCurrent` also returns false when the task is cancelled. Confirmed in source; `connectionResolverCannotReturnAnUnreviewedProfile` and the four-way parametrized `freezeRefusesContextChangedDuringCredentials` cover it.
2. **Selection retained where it should be.** Freeze, draft sync, and the delayed pipeline action pass `selection: true`; named server runs and bundle submits do not. Correct: a named target must not follow the UI selection.
3. **Delayed pipelines.** `pipelineSubmissionAction` captures context and the manifest bytes when the warning is presented; the action refuses if the capture was nil (a missing file), if the context moved, or if the bytes changed, and the same admission closure runs through credential resolution and every submission stage. Compute options are the captured request. The parametrized test covers before-selection, during-selection, during-bytes and missing-file.
4. **Draft sync.** Refuses unless `isServer`; rechecks the local bytes after `connect()`; the reviewed server digest now carries `serverOrigin`, and `pushManifest` refuses when the serving identity or root differs even on the same endpoint. `reviewedServerVersionIsScopedToTheServingRoot` (parametrized) proves the write count is 0 on a changed root and 1 otherwise. This was the regression their first full run caught; the fix is present.
5. **Captured roots.** `LocalPipelineCatalog.summaries` takes an explicit `workspaceRoot` (the default parameter still reads the global, which is fine for other callers); the pipeline controller passes the captured root; the three transports take `workspaceRoot` explicitly for `RemoteJobOrigin`. Bundle packaging still guards on the process-global store but compares standardized URLs before capturing the source, as the handoff says. Not a complete store-adapter migration, and honestly labelled.

The web server's `/api/experiment/freeze`, `create`, `select`, `duplicate`
and the state DTO now reach `management` and `freezeCoordinator` directly;
behaviour is unchanged for those routes.

## 4. Notes, not blockers

1. The semantic commit `2662dc8` has a subject line and no body. The
   handoff document carries the rationale, but the commit is what survives
   a doc rewrite. Worth a body on landing if the maintainer amends; not
   worth a rebase on its own.
2. `freezeOnServer(in:)` and `pushManifest(in:)` read the manifest bytes
   twice (once to build the request, once in the post-connect guard). That
   is the intended recheck, but the first read happens before the first
   `isCurrent`, so a study deleted between the two reads produces a
   "changed" note rather than "select a study first". Cosmetic.

## 5. Landing instructions

1. Both suites are green on `2662dc8` (§2).
2. Fast-forward main to `2662dc8`. That lands `95d14aa` and both bridge
   slices together; the previous review's deploy rule still applies
   (app and Swift CLI before the engine).
3. Bridge retirement is complete; WP-2's remaining draft-writer and adapter
   migration, WP-3 to WP-7, and interactive qualification of the
   freeze/sync, delayed-warning, server-run and pipeline-history paths are
   the next work, as the handoff states.
