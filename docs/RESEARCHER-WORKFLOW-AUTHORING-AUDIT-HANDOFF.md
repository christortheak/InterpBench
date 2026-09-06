# Authoring and coordination: consolidated audit handoff

Date: 2026-09-06. Branch: `codex/researcher-workflow-authoring`.
Review the complete diff from **`95d14aa` to this branch tip**, including
`64102bf` (management bridge retirement) and the subsequent freeze/remote
coordination retirement. The earlier agent's acceptance of `95d14aa` does not
cover either new slice. The preceding implementation branch remains at that
approved tip. This branch has not been merged or deployed.

## What changes for the product

All four transitional Swift compatibility bridges are removed. App controls,
local HTTP handlers and tests use the owners that implement the operation.
The panel still composes runtime inputs and presents results, but no longer
exports the removed state aliases or coordination commands. This makes the
same operations easier to reach from another surface without copying panel
internals, and gives future agents a clear place to change each concern.

For researchers, the visible workflows remain familiar. Selection, workspace,
connection and serving-directory changes now refuse a stale operation rather
than allowing it to target a newly selected destination. A pipeline submitted
after a warning retains the reviewed file and compute options. Reading pipeline
history does not prompt for credentials and reads local evidence from the
captured workspace. Frozen artifacts and run directories stay immutable.

This completes **bridge retirement**, not all of WP-2 or the product vision.
The [implementation status](RESEARCHER-WORKFLOW-IMPLEMENTATION-STATUS.md) and
[operation matrix](RESEARCH-OPERATION-MATRIX.md) retain the outstanding writer,
surface-equivalence, cluster-lifecycle and qualification work.

## Ownership map

| Concern | Owner / entry point |
|---|---|
| Study inventory, selection, lifecycle management | `StudyManagementController` |
| Reusable design selection and lineage | `StudyDesignLibrary` |
| Creation's workspace/model inputs | Explicit `StudyCreationContext` from the surface |
| Local freeze | `StudyFreezeController.freeze(name:runSubstrate:)` |
| Server freeze and reviewed draft sync | `StudyFreezeController.freezeOnServer(in:)` / `pushManifest(in:)` |
| Server-resident named verb | `StudyServerJobCoordinator.run(experimentName:verb:in:)` |
| Single, batch and pipeline bundle dispatch | `StudyBundleSubmissionController.submit` |
| Delayed pipeline warning action | `StudyBundleSubmissionController.pipelineSubmissionAction` |
| Pipeline history | `StudyPipelineController.refresh(in:)` |
| Runtime selection/connection inputs | `StudyOperationContext` + `StudyOperationEnvironment` |

The environment is a context reader and explicit connection resolver, not a
command facade. Its production closures capture the panel weakly. Reading
context uses in-memory connection state; only user actions call the resolver.
No scientific revision field or secret value is added to persistent documents.

## What must be reviewed as behavior changes

The management slice is a mechanical caller migration with separate AST proof.
The coordination slice is **semantic** and must be reviewed as such:

1. Admission captures context before credential resolution and checks it again
   before transport work. Comparison includes connection profile, serving
   identity/root, local workspace and pairing. Cancellation also invalidates
   admission. The returned client's profile must match the reviewed profile.
2. Freeze, draft-sync and delayed pipeline operations additionally retain the
   selection. Explicitly named server jobs and batch submissions retain their
   named targets and do not follow UI selection.
3. Delayed pipelines require readable, unchanged file bytes; two missing reads
   no longer count as an unchanged document. Their additional admission check
   continues through credential resolution and the submission stages. Compute
   options are the captured request, not later picker values.
4. Draft sync now refuses in a local workspace. The local file is rechecked
   after credential resolution. A previously reviewed server digest is scoped
   to the serving identity/root as well as connection and local destination.
5. Local pipeline history receives an explicit workspace root. Transport
   origins also receive the captured root. Bundle packaging retains the
   existing process-store adapter and refuses a changed process workspace
   before capturing its source, comparing standardized URLs; this is not a
   complete store-adapter migration.

Review the existing freeze/firewall, server-version precondition, preserved-pin
adoption, durable job-recording and late-response paths together with these new
entry points. Do not infer scientific qualification from bridge removal.

## Validation evidence

Verified on 2026-09-06:

- Full serial Xcode beta suite: **277 SteeringKit + 4,539 ExperimentKit passed**
  (`TEST SUCCEEDED`; ExperimentKit completed in 43.412 seconds). Includes the
  ten added test functions and their parameterized cases for context admission,
  credential timing, delayed reviews, observation and serving-root scope.
- Full Python suite: **5,928 passed, 9 skipped, 8 warnings**, 161.23 seconds.
  No Python source or test changes were made in this slice.
- Freshly compiled management AST audit at its historical checkpoint:
  **six properties, sixteen commands, 30 changed files, zero differences**.
- Normal bridge ratchet, release-mode bridge gate and `git diff --check`: passed.
  The source/app compiled with Xcode beta and the specified Metal toolchain;
  scratch and derived data remained outside the checkout.

Local logs use the `/private/tmp/interpbench-coordination-` prefix:
`xcode.log`, `python.log`, `management-ast.log`. They are disposable local
evidence, not tracked artifacts; the reviewer should reproduce the checks.

The first focused `swift test` attempt exposed a Swift concurrency error in the
new URLProtocol test harness; that harness was corrected. The next attempt
compiled but stopped producing output before test results and was interrupted.
The full serial Xcode run is the required Swift gate; neither earlier attempt
counts as a passing suite. The first full Xcode run caught the new serving-root
regression: the saved identity had been added but its write-time comparison was
missing. The comparison was corrected; the full Xcode suite then passed. The
failed run is retained locally as `interpbench-coordination-xcode-first.log`.

Management proof is pinned to **`95d14aa` → `64102bf`**. It checks six properties,
sixteen commands, the creation-context relocation, the file census and complete
normalized syntax trees. Its three negative controls reject a changed owner
body, missing context and wrong owner. Reproduction commands are in
[bridge retirement](BRIDGE-RETIREMENT.md). Do not run that audit against semantic
coordination changes and then weaken it to make the comparison pass.

Run both normal and `--release` modes of
`scripts/ci/check-swift-bridge-retirement.py`; both must pass. The historical
inventory has not been expanded or regenerated. Read the actual source diff
and verify that no substitute panel forwarding layer was introduced.

## Limits and next work

- No installed-app interaction, live cluster run, live scheduler accounting,
  GPU scientific qualification or cleanup was performed by this slice. New
  transport regressions use disposable files and URLProtocol fixtures.
- Review and qualify the interactive freeze/sync, delayed GPU warning,
  server-run, cancellation and local/imported pipeline history paths before
  treating this as release-qualified.
- Continue remaining shared draft writers and adapters, plus context coverage
  for auxiliary jobs and other delayed actions. Then complete public-surface
  parity and practical method/dataset/coworker guidance.
- WP-5 and WP-6 remain in scope: documentation-assisted cluster profiles and
  managed remote lifecycle with policy-bounded cleanup. No estimate request or
  implicit deferral is introduced here.
- Imported submit receipts with still-growing logs remain the separate
  responsible agent's task. Do not alter imported historical runs to repair it.
- First controller-allocation capture logging and live accounting qualification
  remain non-blocking follow-up notes from the approved review.

## Landing authority and operating rules

The maintainer, **through the designated reviewing/integration agent**, reads
the diff, checks both suites and reproduces AST proof for mechanical claims.
That process authorizes landing; this implementation agent does not self-merge.
Recheck main ancestry immediately before landing. If main advanced, integrate
its fixes and rerun the affected checks. Fast-forward remains the intended
landing method while ancestry permits it.

Keep public source, fixtures, docs and commits generic. Secrets follow the
Keychain/path-indirection contract. Runs and frozen artifacts are immutable.
File preconditions stay outside content-hashed scientific JSON. Do not change
private profiles or live services as part of review. The earlier deployment
rule still applies: updated app/Swift CLI before the engine requiring manifest
write preconditions.
