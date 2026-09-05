# Remote submission and pipeline coordination handoff

Branch: `codex/maintainability-refactor`
Slice baseline: `27543ae`
Integrated main: `b6949ffdef0e48bc32813247f231feead2e9ed44` (0.9.5)
Date: 2026-09-05

This completes the remote submission/pipeline slice following study/design
management and freeze coordination. The branch is intended for review by the
main coding agents, not automatic rollout. The original checkout and installed
application remain untouched.

## Ownership

Paths below are relative to the repository root.

| Owner | Responsibility |
|---|---|
| `Sources/ExperimentKit/StudyBundleSubmissionController.swift` | Legacy, named batch and pipeline bundle sequencing: frozen-server/stochastic-agent/scope-drift checks, package, upload, submit, transcript and recent-job recording |
| `Sources/ExperimentKit/StudyBundleTransport.swift` | Package/network boundary with injectable operations; production delegates to the existing packager and client |
| `Sources/ExperimentKit/StudyServerJobCoordinator.swift` | Direct server-resident verbs and bundle display following; receives a captured client/transport and explicit presentation capabilities |
| `Sources/ExperimentKit/StudyRemoteJobController.swift` + `StudyRemoteJobActions.swift` | Durable job identities, bounded logs, recent jobs, resume and cancellation; shared across submission entry points |
| `Sources/ExperimentKit/StudyPipelineController.swift` | Draft pipeline declaration writes and local/server pipeline ledger listings, including superseded-response handling |
| `Sources/ExperimentKit/StudyRemoteCoordinationBindings.swift` | Compatibility commands, connection preparation, request snapshots and weak wiring into the panel |
| `Sources/ExperimentKit/StudySubmissionOptions.swift` | Editable options, immutable request values, resource filtering and Slurm-only resume-policy selection |
| `Sources/SteerLabApp/StudyServerRunsView.swift` | Server run-directory rows over supplied records and a refresh action |
| `Sources/SteerLabApp/StudyPipelinesView.swift` | Pipeline ledger rows, gate determinations, promoted-agent provenance and the duplicate-study action |

The owners do not retain `ExperimentPanel` or `ChatService`. Composition code
supplies narrow callbacks; the UI components receive an owner or data and
explicit actions. Existing callers can continue using panel methods while
migration proceeds. The primary `UnifiedStudyRunner` keeps its distinct
preflight/forced-override presentation and now shares execution-option rules.
Its existing explicit acceptance requirement for forced submission is unchanged.

`ExperimentPanel.swift` shrinks from 4,252 to 3,871 lines. The main
`ExperimentsPanelView.swift` shrinks from 699 to 481 lines. Line counts indicate
where code moved; the important boundary is that request sequencing and ledger
state now have their own dependencies and lifetimes.

## Preserved behavior

- Bundle submission retains the existing frozen-server, stochastic-agent
  capability and outcome-scope-drift refusal calls, before package/upload.
- Packager, client encoding, study store, scientific task implementation,
  routing rules and pipeline ledger interpretation are unchanged.
- No new force path or relaxation of scientific gates is introduced.
- Resources omit whitespace-only values. Resume policy travels only for Slurm;
  parallel jobs still travel through the client's existing sharding rules.
  Transcript sharding comes from returned child IDs, not the requested count.
- Single submissions follow logs; batch submissions record jobs without taking
  over the shared follower. Headless bundle following retains the plain log
  fallback. Successful validate bundles retain the evidence-import callback.
- Direct jobs refuse known-missing studies; failed residency discovery remains
  unknown, allowing the server to enforce its own checks. Timed-out follows
  retain cancellation slots; terminal jobs retire their own slot.
- Resume retains the server's verbatim refusal detail. Cancellation targets the
  corresponding tracked job, using the connection supplied by the caller.
- Pipeline writes remain draft-only. Older servers without the listing route
  still show no server pipelines while keeping local/imported ledger results.
- UI labels, gate details, measured/threshold values, duplicate-and-adjust
  action, and list limits remain unchanged (40 server runs, 20 server pipelines,
  10 local/imported pipelines).

## Deliberate coordination changes

1. Legacy/batch/pipeline submission now captures all execution options,
   capabilities and substrate label before its first await. An edit during
   frozen-server preflight, packaging or upload cannot alter the submitted
   executor, resources, dry-run mode, resume policy, fan-out or verb.
2. The pipeline GPU-dialog action captures the original manifest/options and
   workspace/server identity. A changed workspace/server or changed manifest
   file while the dialog is open refuses the deferred action. Switching the
   selected row alone leaves the captured action targeting its original study.
3. Bundle submission checks workspace/server identity before packaging, after
   packaging and after upload, and stops before creating a job if that context
   changed. Direct execution checks context after residency discovery. A job
   already accepted by the server is still recorded and reconnectable.
4. Bundle following uses the original client rather than resolving the panel's
   possibly changed connection. Selection-dependent completion callbacks are
   guarded by context, and residency callbacks additionally check study name.
5. A terminal response for an older direct job cannot clear a newer job's
   cancellation slots. Each slot is cleared only for the matching job ID.
6. Pipeline listing generation tokens reject superseded success and failure
   responses, including overlapping refreshes for the same study. Bindings
   also check selected study, workspace root, server URL, server/local mode
   and pairing; selection reset invalidates the outstanding generation.

These are coordination changes, not statistical or model-execution changes.
They are tested explicitly rather than described as byte-identical moves.

## Validation

Full serial Swift suite passed: **277 SteeringKit + 4,400 ExperimentKit =
4,677 tests**, including the 13 new coordination tests. `xcodebuild` reported
`TEST SUCCEEDED`; log: `/private/tmp/interpbench-remote-swift-final2.log`.

`Tests/ExperimentKitTests/StudyRemoteCoordinationTests.swift` adds 13 tests for:

- Captured options during preflight and returned job/transcript recording.
- Frozen-server refusal stopping before packaging.
- Changed context after upload preventing job submission.
- Batch follower behavior and upload failures not inventing job records.
- Accepted jobs remaining reconnectable after context changes.
- Execution-only resource/resume rules and verb override snapshots.
- Known-missing versus unknown server residency and timeout cancellation slots.
- Older terminal jobs preserving newer cancellation slots.
- Changed context stopping direct submission.
- Superseded pipeline responses, context changes and older-server failures.
- Deferred pipeline actions refusing a changed manifest.
- Draft-only pipeline declarations.

Tests use supplied transports and controlled async callbacks; they do not
submit production jobs. Source-preservation checks verified 136 unrelated panel
method bodies and four pipeline row/rendering helpers after whitespace and
explicit dependency substitution. Core routing, client, ledger, store and task
files remain byte-identical to the slice baseline.

The app target compiles as part of the scheme. Validation uses Xcode beta,
`TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.920.1`, serial testing,
`CLANG_COVERAGE_MAPPING=NO`, and a byte-checked source/fixture copy with build
scratch under `/private/tmp`. Initial compile attempts exposed a Swift closure
inference failure and incomplete/new test fixture fields; those were corrected.
No Python source changed. The latest full Python result remains the main
integration run: 5,858 passed, 9 skipped, 8 warnings.

Interactive visual QA and a real remote round trip remain review tasks; no
installed application was launched or replaced for this refactor.

## Review and remaining work

Local main was rechecked at the end of this slice and still points to
`b6949ff`, already an ancestor of this branch. Before eventual merge, check the
then-current main again, integrate any newer fixes into their owning modules,
and retain their tests. See [MAIN-INTEGRATION.md](MAIN-INTEGRATION.md) for the
scientific fixes already integrated and the earlier moved-code resolutions.

This closes the planned study/design → freeze → remote/pipeline sequence.
It does not make the entire application fully decomposed. Remaining work is
explicit and suitable for separate changes:

- Protocol, condition and prompt-authoring transactions still occupy much of
  the panel. Other large feature views also need the same ownership treatment.
- Evidence import/revision adoption, remote sweep judgment and optimization
  actions, server residency and general server-run discovery still use legacy
  panel/global-workspace paths.
- Packager/importer filesystem operations still consult the global workspace.
  Submission context checks stop later remote submission after a switch but
  do not provide a transactionally pinned filesystem snapshot. An evidence
  import that has already started is not made workspace-atomic by this slice.
- Job records/cancellation still use the existing connection model; per-job
  server identity and cancellation after switching servers are not redesigned.
- `UnifiedStudyRunner` and the legacy bundle owner retain distinct orchestration
  and presentation. Consolidating their complete submission engine should be a
  separate change preserving primary-run preflight/override semantics.
- Compatibility bridges should be retired after all callers migrate, rather
  than becoming a permanent alternative state owner.

J-lens recomputation, scientific validation matrices and historical evidence
repair remain separate science-agent work. Software tests and this refactor do
not complete those tasks.
