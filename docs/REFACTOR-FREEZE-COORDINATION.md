# Freeze coordination refactor handoff

Branch: `codex/maintainability-refactor`

Starting commit: `b2c8683`.

Integrated main baseline: `b6949ffdef0e48bc32813247f231feead2e9ed44` (0.9.5).

Date: 2026-09-05

## Result

`StudyFreezeController` owns freeze readiness, remote identity/gate/advisory
presentation, local freeze coordination and the server freeze/draft-sync
sequences. The controller does not retain an `ExperimentPanel`, chat service or
connection store. Existing public panel entry points remain delegates with the
same selection/connection preconditions.

`ExperimentPanel.swift` decreases from **4,653 to 4,252 lines**. The new controller
is 451 lines; its transport/request and compatibility boundaries are separate.
`ExperimentsPanelView.swift` decreases from **875 to 699 lines**. Freeze controls,
readiness presentation and the confirmation dialog now belong to
`StudyFreezeControlsView`.

## Ownership

| File | Responsibility |
|---|---|
| `Sources/ExperimentKit/StudyFreezeController.swift` | Readiness refresh, local freeze, remote residency/identity preflight and submission, draft push/recheck, preserved-pin adoption and operation-scoped display state |
| `Sources/ExperimentKit/StudyFreezeRequest.swift` | Captured study/server/pairing inputs, injected status/body/freeze/replace transport and notice/refresh/residency callbacks |
| `Sources/ExperimentKit/StudyFreezeBindings.swift` | Existing panel API, connection/token preparation and typed study/workspace/server context comparison |
| `Sources/SteerLabApp/StudyFreezeControlsView.swift` | Existing routing policy inputs, freeze confirmation, gate/identity/advisory display and sync action |

`FreezeRouting`, `ExperimentStore` and the existing freeze policy remain the
authorities for admission, same-engine manifest comparison, evidence substrate,
server-sync eligibility and filesystem mutation. Those implementations were not
rewritten. The controller's remote transport exposes no force parameter; both
UI routes retain the ordinary gated freeze paths.

Readiness still uses the local run-evidence perspective, including the server
substrate in known-unpaired Mac-authority mode. A paired server gets the existing
additional server-perspective advisory. Inventory refresh supplies the selected
manifest, violations and workspace perspective, then continues the existing
draft/result refresh sequence.

## Remote sequence and preserved behavior

Server freeze still performs these steps in order:

1. Read the server's study status. A missing study or non-draft status stops the
   attempt; other read failures retain the freeze endpoint as the backstop.
2. Read the current local manifest and fetch the server body. Compare them with
   `ExperimentStore.compareManifestDocuments`, then apply
   `FreezeRouting.remoteFreezePrecheck` with the captured pairing decision.
3. Offer draft sync only for the existing mismatch/local-draft case. An allowed
   attempt submits to the captured server client.
4. Preserve server advisories, refresh the local/shared-tree view and report the
   server freeze stamp. Gate refusals retain their verbatim server detail.

Draft sync still pushes the captured local draft, adopts only an omitted model
revision the server preserved, reports other preserved pins and re-fetches the
manifest body to verify equality. It never automatically freezes. Frozen local
studies refuse before submission; the server still decides whether its copy can
be replaced. The original pin-adoption body is unchanged.

Local freeze still calls `ExperimentStore.freeze(name:runSubstrate:)`, refreshes
on success and reports the existing gate/pin refusal on failure. No scientific
calculation, freeze hash, preregistration format or cross-engine schema changed.

## Explicit asynchronous protections

These are deliberate coordination changes in addition to moving code:

- A selected-study change invalidates the controller's operation generation and
  clears remote display state. Each suspension checks that generation and the
  supplied current-context predicate before another mutation or UI update.
- The context includes the study, workspace root, server URL, local/server
  workspace mode and pairing. A delayed sync response cannot adopt its pins
  after those inputs change. Captured transport/pairing inputs cannot silently
  change during preflight.
- A local manifest changed while the server identity body is being fetched
  requires another explicit Freeze click. The identity result for the previous
  document cannot authorize the newer one.
- A superseded attempt cannot clear a newer operation's busy state or replace
  its warning/advisory. Freeze controls are disabled while the owner is already
  freezing or syncing.

A request already submitted to the server may still finish there. Invalidating
its UI context suppresses stale callbacks; it does not cancel or roll back the
server's freeze. The store/task APIs continue to use their existing workspace
resolution. This slice does not claim general workspace isolation for all jobs.

## Verification

Full serial Swift validation passed: **277 SteeringKit + 4,387 ExperimentKit
= 4,664 tests**, with `TEST SUCCEEDED`. The build includes the SwiftUI app
target and the new freeze controls.

Final log: `/private/tmp/interpbench-freeze-coordination-swift-final.log`.

Ten tests in `Tests/ExperimentKitTests/StudyFreezeCoordinationTests.swift` cover:

- Mismatch/sync eligibility without a freeze submission.
- Missing and already-frozen server copies stopping before identity fetch.
- Verified submission, advisories, refresh and verbatim gate refusal.
- Draft sync preserving/adopting revision pins, rechecking identity and never
  auto-freezing.
- Superseded preflight, changed workspace context and local document edits
  stopping later submission.
- A response arriving after submission not overwriting a new selection's UI.
- A stale sync response not adopting revision pins into the current workspace.
- Local readiness/refusal parity and clearing readiness when selection is absent.

The remote tests use supplied transports and controlled streams; no server jobs
are submitted. Source-preservation checks match the local freeze, preserved-pin
adoption and manifest-comparison bodies after explicit input/transport
substitution, plus the advisory rendering and confirmation titles. Existing
routing/store/policy code is untouched. New context guards are reviewed as
coordination changes rather than claimed to be byte-identical moves.

The initial build found throwing expressions inside the new test assertions;
loading those values before asserting corrected the tests. Validation uses
Xcode beta, the supplied Metal toolchain, serial execution and byte-checked
source/fixtures with scratch under `/private/tmp`. No Python source changed.
No installed application was launched or replaced; interactive visual QA remains
outstanding.

## Next and integration

Next: remaining remote submission/pipeline coordination and presentation.
Other protocol/condition/prompt authoring commands and global workspace
boundaries remain separate work. Before the other coding agents' final review
and merge, integrate newer main commits if any, apply moved-code fixes to these
owners and retain their regression assertions. The integration record identifies
the current scientific baseline.
