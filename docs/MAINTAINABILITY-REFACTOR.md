# Maintainability refactor: ownership and migration

Branch: `codex/maintainability-refactor`

Base: `cd299e2d0fcf6fe2fa93090631c3a2713e1f9047`

Prepared: 2026-09-05

This change establishes focused owners for reusable Python task support,
offline analysis, Swift contracts, and workspace file access. It is a staged
migration of the maintainability proposal, not a claim that the remaining large
workflows or presentation state have already been decomposed.

## Isolation

Development uses a separate Git worktree. A branch checked out in the active
agent's directory would still change that agent's files; the separate worktree
is what provides filesystem isolation. The original checkout, its running
installation, and the correctness-fix agent's worktree are not migration targets.

Nothing in this change updates an installed CLI or application, switches the
active checkout, deploys a runner, or rewrites study artifacts. Merge and rollout
remain separate actions after review and coordination with the correctness work.

## Owners introduced

All Python paths below are under `Server/steerlab_server/experiment/`.

| Owner | Responsibility | Boundary |
|---|---|---|
| `sampling.py` | Seed identity and scoped sampling state | Torch loads only when non-greedy generation enters its RNG scope |
| `task_inputs.py` | Prompt loading, pin checks, response-format and transcript admission | No model execution |
| `study_admission.py` | Existing manifest verification and source-epoch admission | Uses the existing `run_epoch` policy; does not redefine identity |
| `run_artifacts.py` | Manifest/config snapshots, actual dtype stamps, latest-run discovery | Does not acquire a model |
| `run_reporting.py` | Run reports, CSVs and reasoning-style summaries | Consumes retained records; no orchestration dependency |
| `pipeline_ledger.py` | Durable ledger encoding and schema constant | Shared by pipeline execution and shard continuation |
| `analysis_endpoints.py` | Endpoint reduction, transcript pairing, stratification and promotion summaries | Retains existing statistical definitions |
| `analysis_workflow.py` | Analyze and rescore-style workflows | Reads prior evidence and writes new output; no generation runtime import |
| `runtime_backends.py` | Battery generation/scoring capabilities bound to an intervention | Explicit callable injection, with lazy runtime defaults |

`battery_run.py`, `multi_agent.py`, `jlens/qualification.py`, and `sharding.py`
now use the focused owners. They no longer import the task orchestrator. The
CLI's choice-prompt preview uses `task_inputs` directly.

All Swift paths below are under `Sources/ExperimentKit/`.

| Owner | Responsibility | Boundary |
|---|---|---|
| `ExperimentManifest.swift` | Manifest schema, experiment errors, shared capability result | Declarations moved without changing encoding |
| `StudyRecordContracts.swift` | Task records, events and derived summaries | Existing `ExperimentTasks` nested names retained |
| `ExperimentRepository.swift` | Manifest filesystem operations | Constructed with an explicit workspace root |
| `StudyResultContracts.swift` | Result and preview value types | Independent of panel state |
| `StudyResultRepository.swift` | Run discovery, artifact parsing and result details | Constructed with an explicit workspace root |
| `StudyResultStore.swift` | Existing static result API | Compatibility adapter for the currently selected workspace |

`ExperimentStore` still admits lifecycle mutations before calling the repository's
internal persistence primitive. `persistAdmitted` is intentionally not a public
alternative to `save`. Frozen/complete immutability, arm-clearing refusals and
repair messages stay at their existing admission sites.

`StudyRecordContracts` uses an extension to preserve source compatibility. That
move improves contract ownership and navigation; by itself it is not an
independent execution layer. The actual independent owners in this change are
the Python modules and the Swift repository instances.

## Dependency rules for subsequent changes

1. Shared support modules must not import `tasks.py`, directly or indirectly.
   Workflows consume support modules, not the reverse.
2. Model-independent readers and calculations must remain importable without
   Torch, Transformers or PEFT. Import runtime libraries at the execution boundary.
3. A new caller with a known workspace should retain a repository constructed
   with that root. Do not add another ambient-root lookup inside a repository.
4. Do not expose raw manifest persistence as an alternative public authoring API.
   Admission must remain in the mutation path.
5. Reuse `run_epoch`, `execution_plan`, `judging_custody`, `run_status`,
   `response_format` and the existing numerical kernels. Do not introduce parallel
   versions of their policies.
6. Use small callable boundaries when execution needs to be faked. Do not pass the
   entire task module or panel into a new object as its dependency container.

## Compatibility and scientific contracts

Existing public Python task functions and signatures remain available through
`tasks.py`. Moved private spellings are re-exported there for the migration.
Some shared helpers retain their original underscore names; new production
consumers import them from the actual owner rather than from the facade.

`tasks._battery_backends` remains a small adapter that explicitly supplies
`tasks.generate`. Standalone battery and qualification execution bind directly
through `runtime_backends.battery_backends`. The two injected callables preserve
the battery's own rendering, token budget and latent/steering arguments.

Tests that injected the old task-module RNG or battery helpers now inject their
new owners. Existing behavioral assertions remain. Patching a private facade
name is not a promise that it will replace an independent owner's implementation.

Swift public names, nested record names and Codable keys are retained. Static
store entry points resolve the selected workspace and delegate; repository
instances support independently addressed workspaces without setting a global.

The intended invariants are unchanged:

- Manifest hashes, serialized fields, artifact filenames and ledger schema.
- Prompt admission and draft/frozen verification behavior, including existing
  cross-engine differences documented in the source.
- Seed derivation, draw order, intervention arithmetic and prompt rendering.
- Endpoint definitions, pairing, exclusions, corrections and promotion rules.
- Source-run epoch checks, measurement-drift stamps and refusal messages.
- Result ordering, preview limits and handling of legacy or unreadable files.

This is not the scientific-correctness repair. In particular it does not change
J-lens gain, fine-tuning loss normalization, chat template behavior or promotion
decisions. Passing characterization tests preserves current behavior; it does
not resolve the separate scientific findings.

## Verification

Validated on 2026-09-05:

- Python: **5,460 passed, 9 skipped** in the complete suite, with loopback
  networking available for managed-runner integration tests.
- Swift: **277 SteeringKit tests and 4,300 ExperimentKit tests passed** in a
  serial Xcode beta run (`TEST SUCCEEDED`).
- Nineteen functions moved into the artifact, admission, endpoint and analysis
  owners have identical parsed Python ASTs to the baseline.
- The original checkout remained clean on `main` at the baseline commit.

The initial Swift validation exposed a test-only comparison of equivalent
`/var` and `/private/var` temporary paths, followed by a URL directory-marker
difference. The new repository test now compares resolved filesystem paths;
production path behavior was not changed to satisfy the test.

`Server/tests/test_refactor_boundaries.py` enforces independent offline imports,
checks the four shared consumers for reverse imports, and verifies the injected
battery capabilities receive the correct rendering and intervention arguments.

`Tests/ExperimentKitTests/WorkspaceRepositoryTests.swift` exercises two repositories
with the same experiment name in different workspaces, plus isolated result
discovery and empty-artifact handling. Existing manifest, lifecycle, scientific,
report, pipeline, shard, battery and result suites supply regression coverage.

Use the repository's Python environment with the working directory set to this
checkout's `Server/`. An editable environment from another checkout can silently
resolve that other package when invoked from the repository root. Confirm
`steerlab_server.__file__` before accepting results.

For this machine's Xcode 27 beta, every Swift build/test invocation needs:

```sh
export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
export TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.920.1
xcodebuild test -skipMacroValidation -scheme SteerLab-Package \
  -destination 'platform=macOS' -parallel-testing-enabled NO \
  -derivedDataPath /private/tmp/interpbench-refactor-xcode/DerivedData \
  -clonedSourcePackagesDirPath /private/tmp/interpbench-refactor-xcode/SourcePackages \
  CLANG_COVERAGE_MAPPING=NO
```

The Metal identifier is installation-specific; recheck the installed component
if Xcode changes. Keep scratch outside iCloud. On this machine, the test process
also refused fixture reads from `Documents`, so validation uses a temporary
source copy outside `Documents`, with source/fixture bytes checked against the
worktree. Build outputs are never shared with the active agent's checkout.

## Remaining migration

The central files are smaller, but the following responsibilities remain and
should be migrated as separately reviewed slices:

1. Turn extraction, validation, sweeps, run and judging into workflows with
   explicit resource lifetimes and cancellation. Preserve existing model,
   adapter and hook cleanup behavior before redesigning their APIs.
2. Move pipeline coordination behind a small stage-execution interface. The
   ledger already has an independent owner; a stage failure, checkpoint, resume
   or scientific abort must retain its current disposition and artifacts.
3. Extract Swift and Python draft/freeze policies from filesystem fact gathering.
   The repository introduced here is only the filesystem boundary, not a full
   replacement for `ExperimentStore`.
4. Migrate Swift offline analysis through explicit evidence and workspace inputs;
   avoid giving it an `ExperimentTasks` or `ExperimentPanel` dependency.
5. Separate panel draft state, job execution/polling, and selection/result state.
   Result parsing has moved out; `UnifiedStudyRunner` still delegates through
   `ExperimentPanel`, and that coupling remains to be addressed.
6. Split the feature views after their state owners are independent. Remove
   compatibility bridges only when production callers and tests have migrated.

When integrating the separate correctness branch, apply each scientific fix in
its new owning module and carry its regression tests with it. Do not restore a
deleted implementation into `tasks.py` or a schema declaration into the store
just to resolve a merge conflict.
