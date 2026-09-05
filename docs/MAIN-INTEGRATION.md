# Main integration before further refactoring

Integrated main: `5309b5f2efc8f70dfb7b1a444feb5312b27c7704`

Refactor tip before integration: `509df98`

Date: 2026-09-05

## Intent

The refactor branch must retain main's landed scientific and operational fixes.
This merge brings the complete local main history into `codex/maintainability-refactor`
before further restructuring. It does not merge the refactor into main, update
the installed application, or change the active coding agent's checkout.

A merge preserves the three existing refactor commits and gives the eventual
reviewer an explicit integration point. The final review should compare against
the then-current main, not against the original pre-fix baseline.

## Fixes included

- J-lens full-vocabulary readout uses the final normalization gain; qualification
  checks both vocabulary paths against the reference.
- LoRA accumulation uses the token-average objective, including partial groups.
- MLX instruction training renders completed answers without an added generation
  prompt and verifies prompt-prefix compatibility.
- Dose-response checks reject flat responses and require distinct doses; SAE
  qualification claims are checked against the retained rows.
- Template capabilities are probed and recorded per pinned model, used by the
  declaration/rendering paths, and stamped with run metadata.
- Bundle-derived names and destinations are validated, and explicit digest pins
  that cannot be checked are refused.
- Local cancellation stops the child process; failed children return partial
  evidence with the failure.
- Shard merges publish atomically, interrupted merges can be completed, and run
  imports hold back unfinished stages using the appropriate summary key.
- Viewer type generation and CI type checking are included.

This inventory summarizes the landed commits; it does not replace their tests
or assert that no other scientific issues remain.

## Conflict resolutions

### Run metadata

Main changed `_write_config_snapshot` and added `_model_capabilities_for_run`,
`_advise_capabilities`, and `_RENDERING_RUN_TYPES` in `tasks.py`. The refactor had
already moved snapshot writing into `run_artifacts.py`.

The updated writer, both helper bodies and the constant now live in
`run_artifacts.py`. Their function bodies match main exactly, including the new
`root`/`log` arguments, capability stamps and advisory behavior. `tasks.py`
re-exports the names for compatibility. Main's updated call sites remain intact.
There is no second snapshot implementation in the task facade.

### Shard assembly

Main's complete staged-assembly and recovery implementation is retained.
Only its support imports and calls were redirected to `run_artifacts`,
`run_reporting`, and `pipeline_ledger`. The staging/publish order, cleanup,
reconciliation and completeness checks match main.

One new interrupted-merge regression test patched `tasks._write_report`.
Its failure injection and restoration now patch `run_reporting.write_report`,
where the merge actually writes its report. Every assertion about visibility,
partial evidence, cleanup, recovery and final run identity was preserved.

### Automatic merges

The Swift store/task changes and Python CLI/J-lens changes merged automatically.
Their differences from main were inspected to verify that the remaining changes
are the refactor's contract moves, repository delegation and helper imports.
No incoming scientific fix was discarded to resolve a conflict.

## Preservation checks

- Files outside the original refactor footprint match the integrated main
  byte-for-byte, apart from the explicitly adapted shard test and this record.
- 159 remaining Python task definitions match main's parsed ASTs. The one
  intentional difference is the existing battery compatibility adapter.
- All three relocated metadata/capability function ASTs match main exactly.
- Shared battery, multi-agent, qualification and shard consumers still avoid
  imports of the task orchestrator.
- Validation uses the isolated worktree for Python and a byte-verified temporary
  source copy for Swift, with Xcode beta, the installed Metal toolchain and serial
  tests. Build outputs are separate from the active checkout.

## Validation results

- Complete Python suite: **5,603 passed, 9 skipped**.
- Serial Xcode beta suite: **277 SteeringKit and 4,347 ExperimentKit tests
  passed** (`TEST SUCCEEDED`), including the new training and capability tests.
- Viewer: **261 tests passed**; runtime-type generation and TypeScript checking
  passed; lint reported **0 errors and 11 warnings**.
- Offline-import and shared-consumer architecture checks passed as part of the
  Python suite. All incoming regression tests were retained.

Python integration tests and viewer runtime-type generation ran with loopback
access. The initial sandbox-only type-generation attempt could not open its
local listener; the permitted rerun passed without changing the viewer code.

## Review and merge procedure

Before the eventual main-agent review, check whether main has advanced. Integrate
any new commits, resolve fixes in their current owning modules, and rerun affected
regressions. Then verify `git merge-base --is-ancestor main HEAD` succeeds and
inspect `git diff main...HEAD`: it should show the refactor and its integration
adaptations, not reversals of main's fixes.

The main coding agent will review and perform the final merge, as requested.
