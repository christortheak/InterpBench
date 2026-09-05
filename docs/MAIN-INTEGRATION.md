# Main integration record

Latest integrated main: `b6949ffdef0e48bc32813247f231feead2e9ed44` (SteerLab 0.9.5).

Refactor tip before this integration: `28fd78d`.

Previous integrated main: `5309b5f2efc8f70dfb7b1a444feb5312b27c7704`.

Date: 2026-09-05

## Current integration

The user requested bringing all newly landed main fixes into
`codex/maintainability-refactor` before continuing. This merge includes the
complete local main history through `b6949ff`, preserving the subsequent Python
execution, Swift analysis and study UI refactors. The other coding agents still
own the independent review and eventual merge into main.

The merge was performed only in the isolated refactor worktree. No installed
application, active checkout, runtime workspace or scientific artifact was
updated. The incoming diff covers 95 files. Only five of those files overlap
this branch's refactor footprint.

## Incoming fixes and capabilities

- **J-lens reference agreement and normalization:** float32 reference comparison
  by default; observed final-norm gain convention; explicit tier resolution for
  published lenses beyond the curated models. Freeze, qualification and run
  start retain the same tier decision.
- **J-space interpretation:** energy cross term, reproducible seeded null draws,
  precision floor and clearer separation of the questions each table answers.
- **Intervention scope and provenance:** path-specific token-position, dose and
  control descriptions; a once-written `intervention-scope.json` run sidecar;
  ablation mode preserved in ordinary and variant record stamps.
- **Condition authoring:** stored slots retain ablation mode, conditions retain
  control type, and unknown/incompatible declarations refuse explicitly.
- **Reader split discipline:** selection provenance, reserved final-test rows
  and rejection of held-out rows that repeat training rows.
- **Fine-tuning scale:** explicit adapter scale convention and PEFT translation;
  the Swift panel sends its adapter scale under the matching wire field.
- **Extraction diagnostics:** seeded resampling/order stability analysis over
  the captured extraction contrast, plus recipe documentation.
- **Operational fixes and evidence review:** optvec exit codes follow the
  shared 64/65/66/70 vocabulary; the new impact ledger inventories potentially
  affected artifacts; version declarations advance to 0.9.5.
- **Scientific documentation:** extraction/training recipes, intervention scope,
  dose-gate policy and a predeclared validation matrix accompany the code.

This inventory describes what was integrated. Passing software tests does not
complete J-lens recomputation, validate the production model matrix, or repair
historical evidence. Those remain separate science/operations workstreams.

## Conflict resolution and test adaptation

The sole textual conflict was `Server/steerlab_server/experiment/tasks.py`.
Main changed five function bodies in that former monolith; the refactor had
already moved all five. The compatibility facade stays at 348 lines, and the
incoming fixes now live in their current owners:

| Owner under `Server/steerlab_server/experiment/` | Incoming implementation |
|---|---|
| `condition_execution.py` | `_condition_injections` gains the advisory-only `preflight` switch; `_intervention_state` and `_variant_intervention_state` stamp ablation mode |
| `run_workflow.py` | `_run_impl` writes the intervention-scope sidecar before generation on a new run, supplies explicit condition resolvers, and preserves the resume guard |
| `run_readouts.py` | `_open_jlens_trace` uses the shared imported/curated tier resolver |

The automatically merged store retains main's condition projection and J-lens
freeze changes. API routes, CLI and J-lens qualification retain their new main
behavior alongside the existing direct imports from focused support modules.
No incoming function was discarded to keep the facade small.

Two incoming tests in `test_intervention_scope.py` mocked extraction and
generation through `tasks`. Their patch targets now use `vector_materialization`
and the generation backend, where production code resolves those dependencies.
Their public `tasks.run` calls, fixtures and every assertion are unchanged.

## Preservation checks

- **89 incoming files** outside the shared refactor footprint and the adapted
  intervention-scope test match main byte-for-byte, including every incoming
  Swift change and its tests.
- All **five changed task functions** match main's parsed ASTs after normalizing
  only the refactor's dependency names. Signatures, guards, provenance content,
  ordering and resume behavior are retained.
- The other overlapping files retain all incoming changed/new definitions:
  two in API routes, 21 in CLI, two in the store and six in J-lens qualification.
  The checks normalize only preexisting import/call migrations where necessary.
- The incoming scope tests match main after reversing just the patch-owner
  imports and targets. No assertion was weakened.
- The study UI remains decomposed into its state/controller owners and focused
  view components. Main introduced no conflict with that UI extraction.

## Validation

- Full Python suite: **5,858 passed, 9 skipped** (8 warnings, no failures).
- Full serial Swift suite: **277 SteeringKit + 4,372 ExperimentKit = 4,649
  passing tests**, including the updated fine-tuning wire contract tests and the
  compiled SwiftUI app target (`TEST SUCCEEDED`).
- Python log: `/private/tmp/interpbench-main-integration-python.log`.
- Swift log: `/private/tmp/interpbench-main-integration-swift.log`.

Python resolves this worktree's package, using the existing Python 3.12
interpreter; integration tests have loopback access. Swift uses the byte-checked
source/fixture copy under `/private/tmp/interpbench-refactor-validation`, Xcode
beta, `TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.920.1`, external DerivedData,
serial execution and `CLANG_COVERAGE_MAPPING=NO`. No dependencies were installed.

## Before handing back to the main coding agents

Recheck whether main has advanced. Merge any newer commits into this branch,
apply moved-code fixes to their actual owners and preserve their regression
assertions. Verify `git merge-base --is-ancestor main HEAD` succeeds and inspect
`git diff main...HEAD` for the refactor and necessary integration adaptations.
A clean textual merge alone does not prove a scientific fix survived relocation.

The earlier integration record follows for historical validation and provenance.

## Earlier integration: scientific baseline `5309b5f`

Integrated main: `5309b5f2efc8f70dfb7b1a444feb5312b27c7704`

Refactor tip before integration: `509df98`

Date: 2026-09-05

### Intent

The refactor branch must retain main's landed scientific and operational fixes.
This merge brings the complete local main history into `codex/maintainability-refactor`
before further restructuring. It does not merge the refactor into main, update
the installed application, or change the active coding agent's checkout.

A merge preserves the three existing refactor commits and gives the eventual
reviewer an explicit integration point. The final review should compare against
the then-current main, not against the original pre-fix baseline.

### Fixes included

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

### Conflict resolutions

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

### Preservation checks

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

### Validation results

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

### Review and merge procedure

Before the eventual main-agent review, check whether main has advanced. Integrate
any new commits, resolve fixes in their current owning modules, and rerun affected
regressions. Then verify `git merge-base --is-ancestor main HEAD` succeeds and
inspect `git diff main...HEAD`: it should show the refactor and its integration
adaptations, not reversals of main's fixes.

The main coding agent will review and perform the final merge, as requested.
