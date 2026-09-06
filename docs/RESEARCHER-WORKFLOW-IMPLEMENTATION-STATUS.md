# Researcher workflow implementation status

Implementation branch: `codex/researcher-workflow-implementation`.
Baseline: `1954094` (implementation plan and companion audit), based on `bfd13a5`.
The main checkout is unchanged. Branch commits are review artifacts; landing remains
with the maintainer through the designated reviewing/integration agent.

The authoritative scope is [the implementation plan](RESEARCHER-WORKFLOW-IMPLEMENTATION-PLAN.md).
WP-5 and WP-6 are included. This ledger distinguishes work in progress from verified
completion; a passing focused test does not establish release or scientific qualification.

| Package | Status | Remaining gate |
|---|---|---|
| WP-0 operation inventory | [49-operation matrix](RESEARCH-OPERATION-MATRIX.md) and shared backlog | Expand advanced/per-field mappings and link executable journey evidence |
| WP-1 observation and context | In progress | Finish origin persistence/actions, import context, transport policy, explicit submission and offline results; both suites |
| WP-2 shared services | Runtime roles and draft preconditions implemented; writer/adapter migration in progress | External stale-write preconditions, authoring owners, service roles, bridge retirement |
| WP-3 public operations | Pending | Designs, models, assembly, advanced-method mappings and adapters |
| WP-4 research guidance | Extraction contract and control interpretation corrected; full guides pending | Shipped method guides, prompt resources, scientific wording and executable examples |
| WP-5 cluster coauthoring | Pending, in scope | Prompt, fact/question schema, validation and common preview |
| WP-6 remote lifecycle | Pending, in scope | Composed recovery/import, bounded cleanup plan/apply with custody and dependency checks |
| WP-7 qualification | Pending | Journey harness, UI checks, scientific matrix results or explicit unavailable outcomes |

## WP-1 working ledger

- OPS-01: CLI managers no longer run startup recovery when observing or submitting.
  Durable process ownership is separate from scientific manifests. Recovery only claims
  a controller whose process is proven absent on the same host; live, foreign-host,
  legacy and permission-uncertain records are preserved. Competing recovery claims
  serialize through SQLite. No age timeout is used as evidence of death.
- OPS-02: in progress. Persisted origins and action guards are implemented; audit all remaining adapters. Origin must include server and workspace through persistence,
  actions, monitoring and imports; unbound legacy records cannot borrow selection.
- OPS-03: in progress. Evidence destination is an explicit captured workspace;
  revision adoption uses the same repository. Audit remaining asynchronous callers.
- OPS-04: in progress. Server refuses prohibited HTTP streams; Swift and Python
  adapters preflight capabilities before transfer. Add policy-specific transport tests.
- OPS-05: in progress. Low-level API and engine CLI require an explicit operation.
  Swift server-resident requests now use the API's `experiment` body key.
- UI-01: imported pipelines are now rendered independently of compute selection; interactive QA remains.

## Validation recorded so far

All runs use disposable stores and workspaces. No production jobs or scheduler
allocations were used. The original OPS-01 regression failed before the fix, with
scheduler cancellation intercepted by a subprocess stub.

- CLI observation/submission subset: 118 passed before the ownership extension.
- Ownership, CLI, local lifecycle, parallel submission, fine-tune submission and
  housekeeping subset: 142 passed after the ownership extension.
- Cluster backend and sharding subset: 88 passed after adapting restart fixtures
  to simulate owner death explicitly.
- Transfer/submission subset: 157 passed, including real loopback-server coverage.
  The first TCP attempt failed because sandbox socket binding was prohibited;
  rerunning the disposable test outside that restriction passed.
- Runtime authority/census/submission-policy subset: 43 passed.
- Focused Swift suite: 123 passed across seven suites. Two transport fixtures were
  updated for capability preflight; a shared static URLProtocol fixture is now
  serialized to prevent concurrent tests replacing one another's handler.
- Census relocation AST audit against `1954094`: passed with only docstrings
  excluded. All declaration and helper bodies are unchanged.
- Full Python suite: 5,889 passed, 9 skipped, 8 warnings (174.82 seconds).
- Full Xcode beta suite: 277 SteeringKit + 4,405 ExperimentKit tests passed;
  `TEST SUCCEEDED`. Repeated after the final Swift/context/guidance changes.
- Bridge dependency ratchet: passed. The release retirement gate is not yet
  satisfied: retiring the four bridges remains WP-2 work.
- `git diff --check`: passed. Independent maintainer review remains pending.
- The route census relocation has the AST evidence above. Other changes intentionally
  alter behavior and are not claimed mechanical.

## Implementation constraints

No manifest revision field, canonical hash changes, frozen-evidence rewrites,
private site/study data, secret persistence, production installation or main merge.
Use Xcode beta and the installed Metal toolchain with scratch under `/private/tmp`.
A final handoff must include the actual diff, both suite results, applicable AST and
bridge audits, and limitations, for the maintainer's independent review process.

## Remaining context audit before closing WP-1

The completed changes are a first reviewable slice, not acceptance of the entire
package. Audit origin-aware evidence ledger keys, ambiguity recovery, all auxiliary
job/result panels, and asynchronous bundle-source capture. Add stronger network-level
context-switch tests, revision-adoption concurrency coverage under WP-2, and an
interactive offline pipeline check. Persisted job origins currently refuse a
connection mismatch; they do not open another server or recover a tunnel automatically.
Legacy and colliding IDs refuse automatic actions rather than borrowing selection.

Runtime authority uses `STEERLAB_SERVICE_ROLE=runner|workbench`; managed local
runners select runner. Workbench remains the existing engine default. Deployment
profile integration and UI capability presentation belong to the remaining public
operation/context work. Both suites passing does not establish production numerical
qualification or completion of journeys A–H.

## WP-2 authoring checkpoint (in progress)

See [Draft authoring preconditions](DRAFT-AUTHORING-PRECONDITIONS.md) for the
implemented read/review/apply protocol, HTTP status codes, and remaining migration.
Python raw saves carry external file digests; Swift has explicit snapshot and
replacement owners. Panel whole-document writes use their reviewed snapshots.
Fresh Swift field setters and both freeze owners use the shared lock. The HTTP
sync adapter requires the server version read during identity checking. A response
cannot adopt a model revision over a concurrent local edit. No concurrency key
is encoded in a manifest; frozen files are not rewritten as a migration.

The first full validation after these changes found only the CLI verb census's
old expected list/count; it was updated for the new public manifest read command.
That run passed 277 SteeringKit tests and exercised 4,411 ExperimentKit tests,
with two assertions failing in the single census test. The final rerun passed 277 SteeringKit and 4,411 ExperimentKit tests
(`TEST SUCCEEDED`), and 5,896 Python tests with 9 skipped and 8 warnings
(147.51 seconds). The actual diff was read locally; independent maintainer
review remains pending. The bridge dependency ratchet and `git diff --check`
passed. Python freeze body AST comparison against `75b14cb` passed inside
the new lock, with only its docstring excluded. A separate whitespace-normalized
Swift source check found only atomic publication changed inside the wrapped
freeze body; this is a source check, not a Swift AST audit. The transaction
changes are semantic and are not claimed as mechanical refactoring. A focused Swift run passed 94 tests across
three suites before the later local snapshot/freeze integration. The earlier
full Python run passed 5,896 tests, with 9 skipped.

The initial focused Swift test invocation stalled because workspace-override
fixtures held a semaphore across asynchronous work while sharing the main actor.
The affected suite is now serialized; full validation also uses the repository's
required nonparallel Xcode invocation. This observation does not establish the
cause of the separate installed-app-open hang reported in the audit.

## Property bridge retirement and lock interoperability

`a9545df` corrects the Swift lock key to use a POSIX real path, matching Python.
An additional cross-language check exposed the earlier Foundation `/tmp` alias
mismatch despite the existing suites passing. The regression starts Python's
actual transaction implementation while Swift holds the lock and verifies one
shared lock inode and serialized entry. A separate real Swift CLI edit was also
verified waiting for Python to release the lock before publication. The original
key comparison failed before the fix; both directions now pass.

The subsequent property migration removes all 109 forwarding properties from
`StudyPanelBindings.swift`, with 45 source/test callers retargeted to their actual
owners. The other three bridges remain. A parsed syntax-tree audit against
`a9545df` is supplied with its reproducible invocation in
[Bridge retirement](BRIDGE-RETIREMENT.md). It reports zero differences after its
explicit owner-access/binding normalizations. The actual changes and SwiftUI
binding adjustments were reviewed locally; independent review is still pending.

Latest validation: 277 SteeringKit and 4,412 ExperimentKit tests passed under
Xcode beta (`TEST SUCCEEDED`), and 5,896 Python tests passed with 9 skipped and
8 warnings (149.44 seconds). The new cross-language subset passed four tests.
The bridge dependency ratchet passes without a budget increase. These results
qualify this checkpoint, not all of WP-2 or the remaining work packages.

## Atomic protocol setup save

The next authoring check reproduced a partial-write defect: an invalid seed
policy was rejected only after the main protocol and earlier metadata setters
had already published. Setup now applies the existing pure field policies before
pinning or publication and writes the complete manifest once through the reviewed
snapshot. The regression fails against the old behavior and passes after the fix.
The full serialized Xcode suite passes: 277 SteeringKit and 4,413 ExperimentKit
tests. Python implementation is unchanged by this correction; its latest full
result remains the 5,896-pass run recorded above. These are branch checkpoints;
landing still requires the maintainer's independent review and final suite gates.

Combining workspace-mutating suites under SwiftPM's focused runner stalled again.
Use the nonparallel Xcode invocation for combined/full Swift qualification; do
not interrupt the installed production app to work around a test-runner stall.

## Protocol authoring owner

`StudyProtocolAuthoring` now owns setup validation, model-change invalidation,
judge declaration, input pinning, seat compilation and publication. It accepts
independent values plus a reviewed workspace snapshot. `ExperimentPanel` captures
those values, invokes the operation, then presents advisories and refreshes.
Input helpers now accept explicit roots, so a captured operation can complete in
its original workspace even after another workspace becomes current.

This is a semantic authoring migration, not a claimed mechanical body move:

- The manifest precondition and draft gate run before input pinning/compilation,
  under the shared lock retained through publication.
- Temperature, token count and sample count use the existing field policies;
  invalid values cannot publish, including a zero sample count.
- A scenario selection contains a decode and hash of the same input bytes.
  Compiled provenance retains that hash, or the previous casting's provenance,
  rather than re-identifying older semantic content with a later file read.
- Seat edits and warnings are presented after successful publication.

Five direct-service regressions pass, covering stale admission before input work,
workspace switches across pins/compilation, semantic-source drift, a scenario
from a different workspace, and invalid sampling. The first full Swift run exposed
one existing test borrowing the checkout's rubric instead of creating a fixture
in its temporary workspace. The test now authors its own rubric and asserts its
exact pinned hash. The final full Xcode beta run passes 277 SteeringKit and 4,418
ExperimentKit tests (`TEST SUCCEEDED`). The full Python suite passes 5,896 tests,
with 9 skipped and 8 warnings (151.82 seconds). The bridge ratchet and
`git diff --check` pass; the actual diff was read locally. Main remains `bfd13a5`.

Remaining boundaries are explicit: protocol HTTP requests still adapt through
panel state; prompt-file editing and other authoring operations need their own
captured requests; three compatibility bridges remain. The manifest publication
is atomic, but generated scenario creation and manifest publication are not a
crash-atomic multi-file transaction. Unreferenced generated inputs after a disk
failure remain a recovery/cleanup concern; no frozen or run files are rewritten.

## Operation inventory checkpoint

[The maintained matrix](RESEARCH-OPERATION-MATRIX.md) identifies 49 operations,
their owners, distinct CLI/HTTP/UI surfaces, explicit gaps/restrictions and next
verification gates. Source links resolve and operation IDs are unique. It records
the development CLI and installed-app metadata separately; no production app
or server was exercised. Remaining unverified cells are deliberate work items,
not assumptions of absence or parity. In particular, the selected-state Swift
protocol and prompt HTTP routes can answer ok after a swallowed panel refusal;
truthful explicit request/result adapters remain required work.

## Explicit Swift HTTP protocol authorship

The Swift protocol route now requires a named study, workspace and external file
digest. A new named manifest read supplies the complete JSON document and its
file precondition. The adapter builds fields from the reviewed persisted setup,
uses `StudyProtocolAuthoring` directly, and returns typed success/refusal results.
It neither selects a study nor updates unsaved native fields or their review tag.
Exclusion edits share the protocol's single publication, so a failed input pin
cannot leave separately published rules. The bundled web form sends its displayed
document's identity, reports failures, and provides explicit discard/reload.

Validation: five focused HTTP adapter tests pass. A real built-CLI loopback server
passed named read/save, stale 412, missing-precondition 428 and retention of the
saved edit after refusal, using disposable study data. The full Xcode beta suite passes
277 SteeringKit and 4,422 ExperimentKit tests (`TEST SUCCEEDED`); full Python
passes 5,896 with 9 skipped and 8 warnings (140.65 seconds). JavaScript syntax,
bridge ratchet and `git diff --check` pass. The actual diff was read locally.

The extracted exclusion policy has a reproducible parsed syntax-tree audit:
`scripts/ci/audit-exclusion-policy.swift` compares the old `updateDraft` closure
in `ExclusionRules+UI.swift` at `06a8149` with the new policy body in
`ManifestDraftEdits.swift`. It passes with only trivia excluded; no control flow,
literals or statements are normalized. Compile with the Xcode host SwiftParser
and SwiftSyntax libraries using the invocation pattern in BRIDGE-RETIREMENT.md,
then pass the current and archived baseline checkout paths to the executable.
The HTTP request/result migration itself intentionally changes behavior and is
not claimed mechanical.

**WP-7 browser blocker:** the real `/api/state` check stalled in the existing
`ExperimentPanel.judgeModelOffers` → `JudgeKeyStore.resolveKey` →
`SecItemCopyMatching` path. A process stack sample captured the main actor waiting
for Keychain access. The disposable server was stopped; no credential was exposed,
changed or approved, and the production app was not relaunched. Browser visual
and interaction qualification therefore remains incomplete. Rendering capability
choices should not require interactive secret retrieval. This concrete trace
is not yet evidence that it explains the separate installed-app-open suite hang.

Remaining WP-2 review includes native inventory refresh versus unsaved edit
snapshots: a refresh must not silently replace the authoring precondition while
retaining old form fields. The new HTTP route deliberately does not perform such
a refresh, but other native refresh paths still need their own regression and
migration. Prompt, condition, design and remaining HTTP operations also remain.

## Native editor review survives inventory refresh

The next regression reproduced the native refresh problem noted above: refresh
kept unsaved fields but replaced their authoring digest, allowing them to overwrite
a concurrent edit. `StudyManagementController` now retains a separate editor
review. Inventory refresh updates the catalog only. Selection/explicit reload
starts a review, and a successful operation from that editor advances it. The
setup view flags differing versions and offers **Discard edits and reload**.

The lost-update regression fails before the fix and passes after it. A second
test proves that explicit reload adopts the saved fields and that consecutive
successful saves advance the review normally. The focused run passes nine tests
across two suites; full Xcode beta passes 277 SteeringKit and 4,424 ExperimentKit
tests (`TEST SUCCEEDED`). Python implementation is unchanged; its latest complete
result remains 5,896 passed, 9 skipped and 8 warnings from the preceding HTTP
checkpoint. Final landing still requires both suites on the proposed landing
state. This is a semantic concurrency fix, not a mechanical move.

Prompt-file publication still needs its own early admission and file precondition:
refusing a manifest write after a prompt file was already replaced is too late.
Remaining asynchronous authoring callbacks must also retain the request's review
at operation start rather than look up a newer review when their work completes.

## Prompt authoring protects reviewed inputs and publishes new versions

Two regressions reproduced lost input updates: a stale study save changed its
prompt bytes before refusal, and an independently edited prompt file was silently
overwritten and repinned. `TaskPromptsAuthoring` now owns the explicit study/input
review, early draft admission, full-record edit, run-parser validation and pin.
The native panel captures values and presents the authoritative result.

Edited records are prepared as a content-addressed version in
`prompts/tasks/versions/`; the source remains unchanged for every existing consumer.
The draft receives the new path and exact hash. The UI explains this behavior.
A changed or unreviewed source refuses, as do workspace mismatches, differing
bytes at an existing version destination, path escapes and run-directory aliases.
Source-read and output-publication locks are acquired separately under the study
lock, avoiding a lock-order cycle when two studies edit different input versions.
The two-file crash boundary and possible unreferenced prepared version are stated
in DRAFT-AUTHORING-PRECONDITIONS.md; broad garbage collection is not implied.

Seven focused regressions pass; both lost-update cases fail on the preceding
implementation. Final Xcode beta passes 277 SteeringKit and 4,431 ExperimentKit
tests (`TEST SUCCEEDED`). Full Python passes 5,896 with 9 skipped and 8 warnings
(156.40 seconds). The bridge ratchet and diff whitespace checks pass; the actual
source/test diff was read. This intentionally changes publication semantics and
is not claimed as an unchanged-body mechanical move. Main remains at bfd13a5.

Remaining: explicit prompt HTTP/CLI adapters, raw JSONL/tabular imports and other
input writers, followed by the remaining authoring/bridge and workflow packages.
The legacy Swift HTTP prompt routes still use selected panel state and answer ok
after calling a method that can refuse. They are the next adapter migration.

## Named prompt HTTP operations and isolated web previews

The Swift prompt load/save routes now name the study, workspace and source file.
Saves require the previously reviewed study and source digests (or explicit source
absence) and invoke `TaskPromptsAuthoring`. Responses carry the authoritative study
and prompt version or a typed refusal. No selection-based fallback, native editor
mutation or unconditional success response remains on these two routes.

The web monitor retains a read-only preview using the named read. Its asynchronous
handler checks the originating render and path before updating visible content;
a late response can be cached for its own identity but cannot appear in another
study's preview. Four JavaScript handler cases exercise success, rerender, path
change and typed refusal against the actual bundled handler. They do not substitute
for interactive browser qualification, which still has the separate `/api/state`
Keychain blocker recorded above.

Five focused HTTP tests pass. A disposable built-CLI loopback server passed exact
source digest/read, versioned save with preserved instrument fields, stale 412,
missing-review 428 and refusal of an empty selected-state request. Its process was
stopped afterward. Full Xcode beta passes 277 SteeringKit and 4,436 ExperimentKit
tests (`TEST SUCCEEDED`). The unchanged Python implementation's most recent full
result is 5,896 passed, 9 skipped and 8 warnings at the preceding checkpoint.
JavaScript syntax/handler tests, bridge ratchet and diff whitespace checks pass;
the actual diff was read locally. Main is untouched.

Common HTTP response/document types and error classification now live in
`StudyAuthoringHTTP`, with intentional internal names and no compatibility aliases.
`scripts/ci/audit-authoring-http-results.swift` compares their parsed syntax trees
with `StudyProtocolHTTP.swift` at `89a75a3`: all member/error bodies match exactly
apart from trivia. Compile with the Xcode host SwiftParser/SwiftSyntax libraries
as documented in BRIDGE-RETIREMENT.md and pass current and archived baseline paths.
The new request semantics and browser handler are not claimed mechanical.

WP-2 remains open for other selected-state HTTP actions, asynchronous/whole-document
writers, imports and three compatibility bridges. WP-3 through WP-7 also remain in
scope; this checkpoint closes the named prompt-edit adapter, not the overall plan.

## Cluster-document coauthoring is available through shipped commands and the wizard

`cluster sites guide --json` now supplies author/reviewer prompts and an incomplete
companion example generated from the current profile types. `cluster sites review
<draft.json> --json` binds source evidence to exact profile values, names unresolved
questions, refuses unknown/missing critical policies and uses the existing profile
editor's validation rules plus `ClusterSitePreview`. Slurm and external-server
requirements differ so an external service does not demand irrelevant allocation
or installation facts. Invalid companion/profile input returns `invalidProfileDraft`;
unresolved declarations return `profileQuestions` with an actionable repair.

These two commands skip legacy registry migration, do not authenticate/connect,
and do not import anything. The wizard's **From documentation…** sheet copies the
same packet, reviews the companion, displays connection/source facts and the shared
preview, and uses the existing import path only after an explicit import action.
A review result describes consistency and admission; it does not verify citation
truth, authorize execution or qualify hardware. The private companion preserves
source/retention evidence separately from the profile and scientific artifacts.

The generated workspace contract discovers the guide; the CLI reference is updated
from the declarative table. CLUSTER-PROFILE-COAUTHORING.md documents the workflow,
format, evidence semantics and qualification limits. The pipeline control wording
also now describes the random-floor comparison without claiming causal proof.

Validation: ten focused coauthoring cases plus an offline CLI-runner regression
are in the full suite. Final Xcode beta passes 277 SteeringKit and 4,447 ExperimentKit
tests (`TEST SUCCEEDED`), including generated reference/contract checks. Full Python
passes 5,896 with 9 skipped and 8 warnings (150.84 seconds). Built CLI smoke checks
passed guide discovery, blocked incomplete draft, ready sourced external-service
fixture, changed/unknown policy refusal and malformed-input repair; no site was
imported or contacted. Bridge ratchet and diff whitespace checks pass. The actual
source/test diff was read; this is new functionality, not a mechanical body move.

WP-5's bounded Mac coauthoring path is implemented. A real-document agent interview,
interactive SwiftUI qualification and any additional cross-platform adapter remain
explicit matrix/journey work. WP-1/WP-2 residual ownership/authoring work, WP-3/WP-4
remaining operations/guides, WP-6 managed cleanup and WP-7 qualification remain in
scope. No merge, install, production-app relaunch or remote allocation was performed.


## Evidence reuse requires content verification

Auto-import and CLI chain import no longer treat directory presence or an
`overwrite` refusal message as a successful import. The shared archive importer
has an explicit verified-reuse policy: every declared file in an existing run
must be an ordinary local file with the expected SHA-256, and a carried portable
pipeline ledger must match exactly. Missing/different/link-backed evidence
refuses; matching files keep their bytes, inode and modification date. The default
manual importer continues to refuse an existing primary run unless its caller
explicitly requests verified reuse. Existing pipeline siblings use the same
content check. No existing run is overwritten to resolve a collision.

Auto-import and chain download operations use separate transfer directories so
concurrent servers cannot replace one another's archive basename. Legacy ledger
entries remain readable but lack the new external `contentsVerified` marker and
cannot suppress verification. The marker is not a cleanup receipt: origin-scoped
identity, durable verified archive/member receipts, late-response guards and
cleanup eligibility remain outstanding. No cleanup operation is enabled here.

Validation: the original overwrite-refusal regression failed before the fix.
Archive regressions cover repeated matching import, missing/different evidence,
file/run symlinks, and missing/different portable ledgers. Auto-import regressions
cover failed collisions, empty runs and legacy presence-only entries; chain
regressions require verification on retries and embedded stages. The focused
suites pass. Full Xcode beta passes 277 SteeringKit and 4,453 ExperimentKit tests
(`TEST SUCCEEDED`); full Python passes 5,896 with 9 skipped and 8 warnings
(155.42 seconds). CLI reference checks, bridge ratchet and whitespace checks pass.
The actual source/test diff was read. These are semantic fixes, not mechanical
moves. Main and existing researcher workspaces remain untouched.
