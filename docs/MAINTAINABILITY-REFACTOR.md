# Maintainability refactor: ownership and migration

Branch: `codex/maintainability-refactor`

Base: `cd299e2d0fcf6fe2fa93090631c3a2713e1f9047`

Prepared: 2026-09-05

The base above identifies the original refactor. Main's subsequent scientific
and operational fixes through `b6949ffdef0e48bc32813247f231feead2e9ed44` have
now been integrated. See [main integration record](MAIN-INTEGRATION.md) for
the merge resolutions, preservation checks and updated validation.

The subsequent execution-stage and draft/freeze policy decomposition is recorded
in [stage and lifecycle handoff](REFACTOR-STAGES-AND-LIFECYCLE.md). That record
contains its ownership map, validation and integration instructions. The next
slice is recorded in the [Swift offline analysis handoff](REFACTOR-SWIFT-ANALYSIS.md).
The [study UI handoff](REFACTOR-STUDY-UI.md) records the subsequent state/controller
and component separation, including the subsequent authoring editor extraction
(main study view: 2,720 → 1,300 lines). The subsequent
[study/design management handoff](REFACTOR-STUDY-MANAGEMENT.md) records the
command/library owners and management UI extraction (main study view: 875 lines).
The [freeze coordination handoff](REFACTOR-FREEZE-COORDINATION.md) records the
next controller/transport/UI boundary (main study view: 699 lines).
The [remote submission/pipeline handoff](REFACTOR-REMOTE-COORDINATION.md) closes
this study UI sequence with request/job/ledger owners and dedicated remote
result components (main study view: 481 lines).

This change establishes focused owners for reusable Python task support,
offline analysis, pipeline orchestration, Swift contracts, manifest-save policy,
and workspace file access. It is a staged
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
| `cancellation.py` | Cancellation observation, checkpoint and exception contract | No model or workflow imports |
| `judge_dispatch.py` | Judge roster, remote preflight and fanout request files | No task orchestrator import or model acquisition |
| `judgment_evidence.py` | Judgment completion verification and canonical-run discovery | Retains existing hash, epoch and fail-closed checks |
| `pipeline_evidence.py` | Pipeline discovery, revision/drift evidence and promoted-agent identity | Reads existing evidence and uses the shared ledger writer |
| `pipeline_policy.py` | Remaining-stage model need and inline-judging admission | Existing decision rules, independent of stage execution |
| `pipeline_workflow.py` | Pipeline coordination, resume, gates and held-model lifetime | Receives seven stage callables and two model capabilities |

`battery_run.py`, `multi_agent.py`, `jlens/qualification.py`, and `sharding.py`
now use the focused owners. They no longer import the task orchestrator. The
CLI's choice-prompt preview uses `task_inputs` directly.

Pipeline listing routes use `pipeline_evidence` directly. Submission and orphan
recovery classification use `judge_dispatch` and `pipeline_policy`; only actual
execution enters the task facade. The pipeline owns the held-model context,
while the stage implementations retain their existing local cleanup behavior.

All Swift paths below are under `Sources/ExperimentKit/`.

| Owner | Responsibility | Boundary |
|---|---|---|
| `ExperimentManifest.swift` | Manifest schema, experiment errors, shared capability result | Declarations moved without changing encoding |
| `StudyRecordContracts.swift` | Task records, events and derived summaries | Existing `ExperimentTasks` nested names retained |
| `ExperimentRepository.swift` | Manifest filesystem operations | Constructed with an explicit workspace root |
| `StudyResultContracts.swift` | Result and preview value types | Independent of panel state |
| `StudyResultRepository.swift` | Run discovery, artifact parsing and result details | Constructed with an explicit workspace root |
| `StudyResultStore.swift` | Existing static result API | Compatibility adapter for the currently selected workspace |
| `ManifestMutationPolicy.swift` | Save admission, immutability and arm-clearing decisions | Operates only on supplied manifest values; no workspace or filesystem access |

`ExperimentStore` still admits lifecycle mutations before calling the repository's
internal persistence primitive. `persistAdmitted` is intentionally not a public
alternative to `save`. The store reads the existing manifest, delegates save
admission to `ManifestMutationPolicy`, then persists through the same repository
instance. Frozen/complete immutability, arm-clearing refusals and repair messages
are preserved. Other draft/freeze operations still need decomposition.

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

`tasks._battery_backends` now re-exports the adapter in `choice_scoring`, which
explicitly supplies `generate.generate`. Standalone battery and qualification execution bind directly
through `runtime_backends.battery_backends`. The two injected callables preserve
the battery's own rendering, token budget and latent/steering arguments.

Tests that injected the old task-module RNG or battery helpers now inject their
new owners. Existing behavioral assertions remain. Patching a private facade
name is not a promise that it will replace an independent owner's implementation.

`tasks.pipeline` retains its public signature and supplies `PipelineStages`
(extract, validate, sweep, run, evaluate, analyze, promote) and `PipelineModels`
(acquire, pin revision). It binds the existing entry points at invocation time,
preserving stage injection through the facade. The workflow can also be exercised
directly without task globals or a generation runtime. Promotion still uses its
existing domain types, identity rules and refusal contract.

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

The original refactor commits did not repair scientific behavior. The later
main integration brings in the landed J-lens gain, fine-tuning loss normalization,
chat-template and promotion fixes, with their regression tests. Further refactoring
must preserve this corrected baseline. Passing tests is evidence for the covered
contracts, not a general certification of every scientific claim.

## Verification

The latest stage/lifecycle validation is in the linked handoff. The results
below record the preceding pipeline-only continuation on 2026-09-05:

- Python complete suite: **5,605 passed, 9 skipped**. After the final promotion
  callback was added to the explicit stage interface, all **69 focused pipeline,
  fanout and boundary tests passed** again.
- Swift: **277 SteeringKit tests and 4,350 ExperimentKit tests passed** in a
  serial Xcode beta run (`TEST SUCCEEDED`).
- All **23 definitions** moved into the five new pipeline support owners have
  identical parsed Python ASTs to the integrated baseline. The pipeline function
  also matches after normalizing only its new capability parameters and calls.
- Two new orchestration tests cover CPU-only continuation without model
  acquisition and release of a held model after stage failure without recording
  completion. Three value-only Swift policy tests cover creation intent,
  status-only completion and the limits of explicit arm-clearing permission.
- Offline-import enforcement includes all six new Python owners. Existing
  lifecycle, pipeline, judgment, recovery and scientific assertions remain.

Original refactor validation on 2026-09-05, before integrating main's fixes
(updated results are in the integration record):

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

Execution stages now have workflow/support owners, and central draft/freeze
decisions operate on supplied values. The stores retain filesystem operations,
artifact authoring and freeze transaction sequencing. Remaining larger work:

Swift offline analysis now has explicit evidence, calculation and publication
owners; see the linked handoff for compatibility boundaries and validation.

1. Decompose remaining protocol/condition/prompt authoring transactions and
   legacy evidence import, sweep-judgment and remote-discovery paths.
   Study/design management, freeze coordination and remote submission/pipeline
   coordination now have focused owners and UI components; see their handoffs. Draft, job and result owners and the main authoring/run/result
   UI components have already moved.
2. Extend explicit request/workspace binding through remaining legacy paths,
   apply the ownership pattern to other large feature views, and retire
   compatibility bridges after their callers migrate.

Main fixes through `b6949ff` (SteerLab 0.9.5) are now integrated; see the
[updated integration record](MAIN-INTEGRATION.md) for the moved-code resolutions
and full-suite validation. J-lens recomputation and outstanding science work
proceed independently. Before review and merge, integrate the then-current
main into this branch, resolve moved-code conflicts in the owners listed above,
and rerun the affected regression tests and full suites. A clean textual merge
alone does not establish that a fix survived a move.

When integrating later main changes, apply each scientific fix in
its new owning module and carry its regression tests with it. Do not restore a
deleted implementation into `tasks.py` or a schema declaration into the store
just to resolve a merge conflict.
