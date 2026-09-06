# Researcher workflow validation history through c490531

This appendix preserves the previous status ledger verbatim below. It is a
historical record, not the current completion claim. Some early limitations
were resolved by later checkpoints. Use the [current status](RESEARCHER-WORKFLOW-IMPLEMENTATION-STATUS.md)
for remaining scope and the latest review gates.

---

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
| WP-0 operation inventory | [52-operation matrix](RESEARCH-OPERATION-MATRIX.md) and shared backlog | Expand advanced/per-field mappings and link executable journey evidence |
| WP-1 observation and context | In progress | Finish origin persistence/actions, import context, transport policy, explicit submission and offline results; both suites |
| WP-2 shared services | Runtime roles and draft preconditions implemented; writer/adapter migration in progress | External stale-write preconditions, authoring owners, service roles, bridge retirement |
| WP-3 public operations | Design inspection/description/instantiation and reviewed agent attachment implemented; in progress | Designs, models, assembly, advanced-method mappings and adapters |
| WP-4 research guidance | Extraction contract and control interpretation corrected; full guides pending | Shipped method guides, prompt resources, scientific wording and executable examples |
| WP-5 cluster coauthoring | Mac guide, sourced-fact review and shared preview implemented | Real documentation-to-profile journey, interactive qualification and cross-platform adapter mapping |
| WP-6 remote lifecycle | Origin-aware import and durable archive custody implemented; in progress | Composed recovery, receipt adapters and bounded cleanup plan/apply with dependency checks |
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


## Evidence import records and displayed actions retain their origin

`EvidenceImportOrigin` identifies the remote endpoint/login, serving root and
canonical local workspace. It contains no secret values and does not use an SSH
forward's reusable local port as remote identity. Auto-import captures this before
listing or packaging; manual imports require the origin of the displayed list.
Changed contexts return `evidenceContextChanged` with a reconnection/refresh repair
before another package/download. A download already in progress keeps its captured
local destination and records its original origin on completion.

Ledger lookup and retry backoff now include origin and bundle version. Unscoped
legacy records remain readable but cannot authorize deduplication. Missing serving
roots or current bundle hashes also cannot establish unchanged remote identity.
Ledger publication reads and merges under a file lock; corrupt existing bytes
refuse rather than being replaced with partial history. Changing local workspaces
replaces the auto-import poller while an existing operation keeps its original
workspace. The pipeline triage no longer hides evidence merely because a local
run directory exists. Housekeeping's partial archive filenames retain failure
classification and the actual run ID.

Server Jobs and housekeeping retain their list's origin. Job import, cancellation,
resumption and retry check it before obtaining the action's client. Selection
changes invalidate old lists; late log callbacks from replaced streams are
ignored. Connection handshakes and state refreshes publish only into the context
that requested them. Connection metadata reads no longer require reading a token.

Validation: origin/backoff/restart tests cover identical paths on different
servers, changed bundle bytes, missing origin/root, stale manual actions after
server/root/workspace changes or disconnection, selection changes during listing
and import, concurrent-owner ledger merging and corrupt-ledger preservation.
Offline URLProtocol fixtures exercise late handshake/state replies; identity tests
cover tunnel-port reuse and workspace registration. Full Xcode beta passes 277
SteeringKit and 4,465 ExperimentKit tests (`TEST SUCCEEDED`); full Python passes
5,896 with 9 skipped and 8 warnings (155.56 seconds). Bridge ratchet and whitespace
checks pass, and the actual diff was read. No mechanical body move is claimed.

This closes the scoped auto-import identity and Compute job-action seams, not the
entire remote journey. A path-only housekeeping listing remains unverified even
after an earlier import; it needs a current bundle stamp to prove unchanged bytes.
Durable archive/member custody receipts, cleanup plans, remaining auxiliary
operations and interactive UI qualification remain work. Main is still `bfd13a5`;
there was no merge, install, allocation, or real remote cleanup.


## Durable local archive custody

The shared Swift importer now retains an exact content-addressed archive and
publishes a separate custody receipt after verifying local files. Receipt failure
rolls back new run publications, preserving verified existing runs. Unexpanded
logs remain in the archive. Completeness and failure status remain distinct from
byte verification, and absent completeness metadata stays unknown. Root comparison
uses canonical filesystem paths, accepting platform aliases without rebinding a
receipt to a different workspace.

Auto-import records receipt/archive digests; explicit manual import can refresh
a legacy record. CLI import returns the receipt digest, and the read-only
`data verify-custody` command calls the shared verifier with typed refusal and
repair. See [the custody contract](EVIDENCE-CUSTODY.md) for storage, semantics,
and remaining adapters. This does not authorize remote deletion.

Validation: full Xcode beta passes 277 SteeringKit and 4,471 ExperimentKit
tests (`TEST SUCCEEDED`); full Python passes 5,896 with 9 skipped and 8 warnings
(157.98 seconds). The compiled CLI passes a disposable loopback import → local
verification → missing-evidence refusal round trip, including exit 64 on a
missing argument and no network requests from verification. Tests cover complete archive
retention after transport download removal, corrupted/missing archive, receipt
or evidence, symlinks, another workspace, receipt-publication failure and CLI
verification/refusal. Generated CLI reference, shipped contract equality, bridge
ratchet and whitespace checks pass. The actual source/test/document diff was read.
No mechanical body move is claimed. Main remains untouched; this is an isolated
branch checkpoint, not a landing or release qualification.


## Custody discovery and workbench adapters

`data custody <run-id>` discovers receipts from individual or chain imports,
including pipeline archives carrying that run. Discovery validates receipt
identity but does not claim current file verification; damaged receipts surface
as inventory issues. Discovery of absent state writes nothing, and linked
storage or unsafe run IDs refuse. Receipt bytes are hashed and decoded from one
captured read.

The Swift workbench exposes explicit custody list/verify HTTP operations. Both
name the workspace, reject unknown/missing fields, and capture the served root
before background work. The local study Results view uses the same inventory and
verifier under “Evidence retained locally,” with its root captured from the run
path. View replacement cancels publication of late results; refresh clears the
previous verification message. See [the custody contract](EVIDENCE-CUSTODY.md) for
commands, endpoint fields and limitations.

Validation: full Xcode beta passes 277 SteeringKit and 4,474 ExperimentKit tests
(`TEST SUCCEEDED`); full Python passes 5,896 with 9 skipped and 8 warnings
(151.89 seconds). The focused suite passes 22 tests. A compiled CLI and
disposable Swift workbench wire test agree on discovery, verify without selecting
a study or connecting compute, and return typed refusals for wrong/unnamed
workspaces and later evidence loss. The SwiftUI path compiles but interactive
qualification is still outstanding. Python receipt parity and managed remote
cleanup remain work; no remote deletion is enabled. Generated reference, shipped
contract equality, bridge ratchet and whitespace checks pass. The actual source,
test and document diff was read. No mechanical move is claimed; main is unchanged.


## Pipeline declaration authorship

`StudyPipelineAuthoring` validates complete gates and stage dependencies for
headless callers as well as the composer, and saves/removes the declaration
through the reviewed manifest transaction. A concurrent edit or freeze refuses
without replacing saved bytes. The composer captures the file used to seed its
fields, preserves unsaved edits across inventory refresh, and advances the review
only on its own successful save or explicit reload. Context mismatch refuses a
UI save; the shared owner always targets its supplied workspace.

The old `savePipelineDeclaration` compatibility forwarding method is removed.
The three remaining bridge files still need migration; this does not satisfy
the release retirement gate. Focused tests pass for save/removal, stale edit or
freeze, incomplete gates/stage dependencies, and explicit workspace capture.
Full Xcode beta passes 277 SteeringKit and 4,478 ExperimentKit tests
(`TEST SUCCEEDED`); full Python passes 5,896 with 9 skipped and 8 warnings
(150.09 seconds). The bridge ratchet and whitespace checks pass; the actual
source/test/document diff was read. No mechanical body move is claimed.
Interactive composer qualification remains outstanding. Main is unchanged.


## Direct design-state ownership

Seven remaining design-state forwarding properties have been removed from the
management bridge. App and test consumers now read the design owner directly;
SwiftUI selection bindings use a directly bound design owner. The mechanical
change does not alter study/design behavior or file formats. Other members in
the three remaining bridges still require migration.

The shared SwiftSyntax auditor's `design` mode derives the seven accessor mappings
from baseline `4fea25f` and checks complete syntax trees, source census and exact
bridge declaration retirement. It passes with zero normalized differences for
seven changed consumer files. Its original mode still passes the historical
109-property retirement; a deliberately unrelated declaration in a disposable
copy fails the new mode. Full Xcode beta passes 277 SteeringKit and 4,478
ExperimentKit tests (`TEST SUCCEEDED`); full Python passes 5,896 with 9 skipped
and 8 warnings (146.47 seconds). The bridge ratchet and whitespace checks pass,
and the actual source/test/document diff was read. This is a mechanical owner
access migration; substantive design storage/authorship remains separate work.


## Reviewed design descriptions and public design inspection

The description editor now retains a `StudyDesignSnapshot` containing the exact
file version that supplied its text. The shared command checks that version
under the manifest-file lock, including unchanged-text requests, and returns an
updated snapshot only after success. Concurrent edits refuse with `designChanged`;
a missing design is not recreated. Traversal, linked paths and mismatched internal
names refuse. Template encoding, scientific content hashes and source studies
remain unchanged by metadata version tracking.

`design list/inspect/describe` and explicit Swift workbench HTTP operations expose
that owner without panel selection or a server connection. Listing reports
unreadable entries; inspection returns the document plus separate file/content
hashes; writes require the reviewed file digest. The native editor retains
failed edits and offers an explicit reload. Its unreviewed compatibility forwarding
method has been removed. See [the public contract](STUDY-DESIGN-AUTHORING.md).

Validation: six focused authoring tests pass, including a parameterized stale
edit case. A compiled CLI and disposable loopback Swift workbench exercise list,
inspect and describe, missing/stale preconditions, wrong workspace, unknown fields
and missing designs; file digests match the published bytes, scientific hashes
remain unchanged and the source study is untouched. Full Xcode tests pass 277
SteeringKit and 4,485 ExperimentKit tests (`TEST SUCCEEDED`, 42.245 seconds);
full Python passes 5,896 with 9 skipped and 8 warnings (145.47 seconds).
The first full Swift run identified one namespace-census expectation; it was
updated and the full suite rerun. The generated CLI reference, bridge ratchet
and whitespace checks pass, and the actual diff was read. This semantic
checkpoint does not extend the earlier mechanical AST claim.

Other design writers (creation, save-back, rename, delete and instantiation),
Python adapters and interactive UI qualification remain work.

## State observation does not resolve stored credentials

The Swift workbench's state DTO and both judge-option catalogs no longer call
credential resolvers. `CredentialObservation` reports environment presence or
`notChecked`; it performs no Keychain access or legacy-defaults migration.
Unknown stored credentials remain unknown. The concept DTO now includes
`apiKeyStatus`; its historical `hasAPIKey` field is omitted when presence is
unknown. Consumers must not turn an omitted value into a missing-key claim.
Judge options remain authorable with a deferred-check caption. Structural model
refusals retain their existing explanations. Execution preflights still resolve
the actual required credential.

A temporary workbench answers `/api/state` in 0.002 seconds, and a selected-study
state including judge options in 0.016 seconds, without environment API keys.
Three focused tests pass. Full Swift tests pass 277 SteeringKit and 4,488
ExperimentKit tests (`TEST SUCCEEDED`, 43.994 seconds). Full Python passes 5,896 with 9 skipped and 8 warnings (154.92 seconds).
The actual diff was read; the bridge ratchet and whitespace checks pass.
No mechanical-move claim is made. This is a bounded observation fix: other native readiness badges,
settings appearance handlers and execution preflights still retrieve credentials;
the whole app is not yet qualified as non-interactive during observation.

## Design instantiation derives scope before publishing

`OutcomeInstrumentScopeAuthoring` now owns scope vocabulary checks, clearing,
item selection and zero-item refusal for both the declaration command and design
instantiation. Instantiation checks and derives that scope before any casting
output or study publication. A bad scope no longer leaves a newly saved draft
behind. The prompt bytes checked against the design pin are reused for scope
selection, so a second read cannot silently select a different item set.

The destination manifest is held under the shared file lock, with an absent-file
precondition both before preparation and at publication. The fully derived draft
is saved once. Design inspection, prompt reads, semantic-panel reads, scenario
compilation and manifest publication use the captured workspace where those
owners accept it. This does not complete the migration of agent-library paths,
reviewed batch commands or all design writers. Nor does it claim crash-atomic
rollback of every compiled scenario file if a later filesystem write fails.

Three focused tests pass, including parameterized unknown-format/empty-selection
refusals, unchanged source/design bytes, successful scope and provenance, explicit
workspace reads and reuse of reviewed input bytes. Full Swift tests pass 277
SteeringKit and 4,491 ExperimentKit tests (`TEST SUCCEEDED`, 43.667 seconds).
Full Python passes 5,896 with 9 skipped and 8 warnings (151.70 seconds).
The actual source/test/document diff was read; the bridge ratchet and whitespace
checks pass. This is a semantic publication change; there is no claim of a
mechanical body move.


## Reviewed agent inspection and attachment across surfaces

`agent list/inspect` exposes the recognized local library, exact artifact document
and file digest. `experiment attach-agent` and the named Swift HTTP adapter require
both reviewed file digests. `StudyAgentAuthoring` publishes through the draft
transaction, with model compatibility and artifact-version checks before writing.
The embedded artifact and its condition hash come from the same read; changed
picker records, missing artifacts and malformed artifacts now carry a typed
`artifactPin` refusal. No artifact or frozen study is rewritten.

The native study picker captures its artifact review on selection and reloads
explicitly; the Agent Library validates its displayed record before attachment.
Both call the same owner with the management owner's retained study review.
The optional unsaved model choice remains an explicit native input, not a default
in CLI/HTTP. See [the public attachment contract](REVIEWED-AGENT-ATTACHMENT.md).

Validation: eight focused tests pass, including parameterized artifact failures,
plus compiled CLI and disposable HTTP checks for discovery, exact inspection,
attachment, stale/missing preconditions, changed artifacts and wrong contexts.
Full Xcode tests pass 277 SteeringKit and 4,499 ExperimentKit tests
(`TEST SUCCEEDED`, 41.951 seconds). Full Python passes 5,896 with 9 skipped and
8 warnings (156.32 seconds). The generated CLI reference, bridge ratchet and
whitespace checks pass; the actual diff was read. Broad checks first found the
new command's multiline contract span and a duplicate refusal-registry entry;
both were corrected, and the complete Swift suite rerun. A newly introduced
compatibility-property reference was replaced with direct owner access.

Generic artifact/reader attachment, Python parity, reviewed design batches and
interactive UI qualification remain. Discovery uses the existing store's
recognition rules and is not a complete malformed-file census. This semantic
change does not claim a mechanical body move.


## Reviewed design instantiation and casting-table checkpoint

- Added `StudyDesignInstantiation`: a retained design file review, captured
  workspace and explicit casting supply the command. It holds the shared design
  lock, checks the destination is unoccupied, validates prompt and agent pins,
  derives scope, and delegates arm attachment and seat compilation to their
  existing owners before publishing an ordinary draft. Existing stateless store
  calls capture one review; batch callers retain it across rows.
- `steerlab-cli design instantiate` and `POST /api/design/instantiate` call that
  command. The shared casting JSON accepts agent references or exact panel seat
  IDs, with explicit baseline seats. Unknown fields, missing/stale design
  preconditions, stale artifacts, wrong base models, workspace mismatch and
  unsafe destinations refuse. Results carry the actual collision-resolved name,
  exact saved manifest and its external file digest. No submission occurs.
- Design inspection includes ordered seat IDs when the pinned panel is readable.
  A missing/drifted panel becomes an inspection advisory, so inspection and an
  already-published description edit can still return their actual result;
  instantiation continues to refuse unusable panel inputs.
- The app's casting table captures design/agent reviews and its originating
  workspace, retains rows after refusals, and offers an explicit discard/reload.
  Presets use the displayed seat IDs. Workspace switches stop minting or later
  submissions and prevent completion callbacks selecting a study in another
  workspace. Partial batches preserve and report every earlier successful row.
- Existing absolute seat pins are normalized only inside the captured workspace,
  including macOS path aliases, then subjected to the same artifact hash and
  ordinary-file checks. The original study and artifact bytes remain unchanged.
  This also keeps preloaded permutation pickers aligned with their actual cast.
- Updated the shipped agent contract and mirror, generated CLI reference, design
  guide and operation matrix. No compatibility wrapper was added. This changes
  admission and behavior; it is not claimed as a mechanical move or AST-equivalent
  refactor.

Validation for this checkpoint:

- Focused instantiation command suite: 8 tests passed, covering stale/deleted
  designs, explicit-root naming/publication, CLI/HTTP preconditions, partial
  outcomes, scope preservation and destination redirects.
- Final casting-table/batch suite: 36 tests passed, including reviewed agents,
  exact seats, workspace switches before/during submission, retained absolute
  pins and inspection of a design whose panel input is unavailable.
- The first full Swift run exposed a real permutation regression: existing
  absolute seat pins were rejected as new relative input. The focused original
  regression reproduced it, passed after the bounded normalization fix, and
  remains in the suite with a more informative failure assertion.
- Final full Xcode beta suite: 277 SteeringKit + 4,510 ExperimentKit tests passed;
  `TEST SUCCEEDED` (41.468 seconds of ExperimentKit execution). Log:
  `/private/tmp/interpbench-reviewed-instantiation-xcode-qualified.log`.
- Full Python suite: 5,896 passed, 9 skipped, 8 warnings (155.80 seconds). Log:
  `/private/tmp/interpbench-reviewed-instantiation-python.log`. Subsequent edits
  affected only Swift, its tests and documentation; Python source was unchanged.
- Actual compiled CLI and disposable HTTP workbench: reviewed casting, artifact
  pins, lineage, collision names, byte digests and missing/stale/root/unknown-field
  refusals passed. Final log:
  `/private/tmp/interpbench-instantiation-wire-qualified.log`.
- Actual diff read, CLI reference regeneration, bridge ratchet and whitespace
  checks completed. Independent maintainer review remains the landing gate.

Limits remain explicit: public batch/expansion adapters and design creation,
save-back, rename/deletion, Python design parity and interactive UI qualification
are unfinished. The design lock cannot serialize external editors or legacy
writers that have not migrated. A compiled panel can remain unreferenced if later
manifest publication fails; this is not a crash-atomic multi-file transaction.
No scientific hashing schema, frozen study, immutable run, main checkout,
installed app or live scheduler job was changed by this checkpoint.

## Reviewed design saving across the app, CLI and HTTP

- Added `StudyDesignSourceReview` and `StudyDesignSaving`. A save retains the
  source manifest and panel bytes; an update additionally retains the destination
  design version and requires matching lineage. Shared file locks and external
  digests guard publication without adding schema fields. Frozen sources remain
  legitimate read-only inputs; source studies, prior studies and runs are unchanged.
- `design save` and `design update`, their named Swift HTTP operations, and the
  app's design actions delegate to this owner. Results distinguish creation,
  reuse and scientific change, carry the actual destination and file digest, and
  report derivation warnings. Missing, stale, mismatched-workspace and malformed
  requests have explicit refusals. Both reviewed files are required for updates.
- App confirmation retains the source and destination even after catalog refresh,
  names the captured source, and refuses after workspace changes. It explains
  that design saving uses saved settings. Two design-saving forwarding methods
  were removed from `StudyManagementBindings`; other bridge retirement remains.
- Fixed semantic-panel reuse: structural equivalence after stripping is insufficient
  when the stored file still contains a casting. Reuse now requires that the file
  itself equals the stripped semantic panel; otherwise immutable content-addressed
  semantic bytes are published separately. The original bound input is preserved.
- Moved study-file review to `DraftAuthoringSnapshot`, shared by attachment and
  design saving. This includes admission changes and is not an AST-equivalent
  mechanical move. No new compatibility bridge was introduced.

Validation:

- Seven focused saving tests passed, covering source preservation, both stale
  reviews, no-op reuse, frozen sources, lineage, panel drift, bound-panel reuse,
  transport refusals and retained app confirmation authority. The bound-panel
  regression was reproduced before the fix.
- Root-switch tests now change both test overrides and assert that the active root
  actually changed. The fixtures also restore the workspace override; this
  strengthens earlier instantiation tests whose higher-priority override had
  masked the intended switch.
- Actual compiled CLI and disposable HTTP workbench save/update checks passed:
  `/private/tmp/interpbench-design-saving-wire.log`.
- Both initial full suites caught the newly added `designDerivationWarning`
  missing from their copied parity lists. The lists were updated without weakening
  the closed-vocabulary assertions.
- Final full Xcode beta suite: 277 SteeringKit and 4,517 ExperimentKit tests passed,
  `TEST SUCCEEDED`; `/private/tmp/interpbench-reviewed-design-saving-xcode-qualified.log`.
- Final full Python suite: 5,896 passed, 9 skipped, 8 warnings (170.31 seconds);
  `/private/tmp/interpbench-reviewed-design-saving-python-qualified.log`.
- Actual diff read, generated CLI reference, contract mirror, bridge ratchet and
  whitespace checks passed. A comment-placement correction followed the suites;
  executable code was unchanged. Maintainer review remains the landing gate.

Public batch/expansion, reviewed rename/deletion, Python design parity and
interactive UI qualification remain unfinished. Legacy writers that do not take
shared locks are not serialized. A newly published semantic input may remain
unreferenced if later design publication fails; this is not a crash-atomic
multi-file transaction. No merge, installation or live remote action occurred.

## Public batch design creation

- Added `design batch` and `POST /api/design/batch`. Explicit rows use the same
  reviewed casting format as single instantiation. Batch shape errors and an
  initially stale design refuse before publication. One captured design version
  and workspace supply all rows; artifact reviews are resolved before publishing.
- Reused `StudyDesignInstantiation` and the app's per-row mint owner. Results keep
  every successful study name and failure, add typed row issues and repairs, and
  retain shared batch provenance. A later refusal neither rolls back nor hides
  earlier drafts. No batch creates jobs, freezes studies or submits work.
- Incomplete CLI batches return 65 for admission problems or 70 for operational
  failures, with the complete result attached. `ExperimentCLIStop` can now report
  durable changes explicitly; the batch envelope correctly says `changed: true`
  after partial publication. HTTP returns 207 with `ok: false`; complete batches
  return 200. The contract warns that replaying successful rows creates more drafts.
- Updated the shipped agent instructions and mirror, parser/help, generated CLI
  reference, refusal registry, design guide and operation matrix. No compatibility
  bridge or scientific-schema field was added. This is a new operation/result
  contract, not a claimed mechanical or AST-equivalent move.

Validation:

- Five focused batch tests passed: partial publication and collision names,
  source preservation, malformed shape before publication, retained design review
  between rows, HTTP preconditions and partial results, and the real CLI runner's
  mutation flag and typed repair.
- Actual compiled CLI and disposable HTTP workbench passed complete/partial
  batches and missing/stale/root/unknown-field/empty-batch refusals. Log:
  `/private/tmp/interpbench-design-batch-wire.log`.
- Full Xcode beta suite: 277 SteeringKit and 4,522 ExperimentKit tests passed,
  `TEST SUCCEEDED`; `/private/tmp/interpbench-design-batch-xcode.log`.
- Full Python suite: 5,896 passed, 9 skipped, 8 warnings (174.05 seconds);
  `/private/tmp/interpbench-design-batch-python.log`.
- Actual diff and new-file review, reference regeneration, bridge ratchet and
  whitespace checks passed. Independent maintainer review is still required.

Automatic expansion presets, reviewed design rename/deletion, Python design
parity and interactive app qualification remain open. Batch results are returned
synchronously and are not a durable command journal; an interrupted caller must
inspect the workspace before reconstructing a request. Main, installed software,
real research data and live remote jobs were unchanged.

## Local model preparation through the shared installer

- Added `model plan` and foreground `model install`, plus the Swift workbench's
  explicit `/api/local-model/plan`, `install`, `status` and `cancel` operations.
  Planning delegates to the loader's existing requested-revision cache checks;
  installation delegates to the app's existing `LocalModelInstaller`. No second
  downloader, weight loader or server submission path was introduced.
- Plan output distinguishes cached file presence from memory fit, credential
  readiness and scientific qualification. It reports the resolved cached revision
  when available and makes no download-size estimate. Planning and status perform
  no download, weight load or credential query.
- Each accepted installation records a fresh in-process request ID and optional
  revision. The revision is forwarded to the existing snapshot downloader. Busy
  admission preserves the current request. HTTP cancellation requires the observed
  ID and refuses a successor, including another install of the same model.
- Cancellation now invalidates the predecessor's epoch immediately, preventing a
  late completion from restoring status after cancellation and clearing. Existing
  installer tests retain their state-machine checks with the revision argument
  added to injected fetches.
- The CLI waits for its own installer and reports completion or failure. The
  workbench API observes the app's installer; it cannot observe a separate CLI
  process. These are local-cache operations, independent of the app's selected
  remote server. No source study or scientific revision pin is modified.
- Updated the shipped agent instructions and mirror, parser/help and generated
  reference, operation matrix and `LOCAL-MODEL-PREPARATION.md`. This is semantic
  operation/admission work, not a mechanical move or AST-equivalence claim.

Validation:

- Five focused tests passed, covering incomplete/exact-revision cache observation,
  invalid path/ref input, revision forwarding, successful/failed injected fetches,
  busy admission, status, stale cancellation and cancelled-tail suppression.
- Actual compiled CLI and disposable HTTP workbench passed read-only cache-plan
  parity, explicit revision observation and invalid/unknown-target/cancellation
  refusal checks. Log: `/private/tmp/interpbench-local-model-preparation-wire.log`.
  Valid installation was exercised through injected fetches only: no real model
  download, weight load or server installation occurred.
- Full Xcode beta suite: 277 SteeringKit and 4,527 ExperimentKit tests passed,
  `TEST SUCCEEDED`; `/private/tmp/interpbench-local-model-preparation-xcode.log`.
- Full Python suite: 5,896 passed, 9 skipped, 8 warnings (181.21 seconds);
  `/private/tmp/interpbench-local-model-preparation-python.log`.
- Actual diff/new-file review, generated reference, bridge ratchet and whitespace
  checks passed. Independent maintainer review remains required before landing.

Server preparation plans and CLI adapters, durable remote install observation,
server policy qualification, native revision-entry controls and interactive UI
qualification remain open. Local installation status is not a durable journal;
restarting its process requires inspecting the cache and reconstructing an
explicit request. Failed installs may leave partial files. Cache presence and
successful fetching do not prove memory fit or scientific correctness. Main,
installed software, researcher data and live remote jobs were unchanged.


## Study assembly continuation — 2026-09-06

Branch `codex/researcher-study-assembly`, based on main `a5db8c3` after importing
its two review documents. Executable review base remains approved `2662dc8`.
The [audit handoff](STUDY-ASSEMBLY-AUDIT-HANDOFF.md) records semantic scope and
remaining qualification. No new mechanical-move claim is made.

The initial compile/test passes exposed obsolete import callers, test isolation
and fixture errors, missing new-verb census/help/contract entries, a lost rename
repair and generated-reference drift. These were corrected before the full green
run. Final diff inspection then found that a missing vector-attachment positional
could be confused with a flag; strict arity and a positive/negative CLI regression
were added and the complete Swift suite rerun.

- Final serial Xcode beta: 277 SteeringKit + 4,551 ExperimentKit passed,
  TEST SUCCEEDED. `/private/tmp/interpbench-assembly-xcode-verified.log`.
- Python: 5,928 passed, 9 skipped, 8 warnings, 165.80 seconds.
  `/private/tmp/interpbench-assembly-python-final.log`.
- Normal/release bridge checks, generated reference/contract tests, public scan
  and `git diff --check` passed. The diff was read locally; independent review
  through the designated maintainer agent remains the landing gate.

No app installation, live runner/model/cluster execution or interactive scientific
qualification occurred. Runtime studies and imported historical evidence were
not edited. This continuation leaves the previous approved branch tips intact.


## Python study assembly — 2026-09-06

Branch `codex/python-authoring-equivalence`, based directly on main `ef3dec8`.
Main's `3ea802a` receipt gate and immutable sibling reimport are included.
The [workflow and audit handoff](PYTHON-STUDY-ASSEMBLY-WORKFLOW.md) describes
this bounded WP-3 slice and the remaining design/model/cluster work.

- Final Python suite: **5,971 passed, 9 skipped, 8 warnings**, 162.52 seconds;
  `/private/tmp/interpbench-python-parity-final.log`.
- Final serial Xcode beta suite: **277 SteeringKit + 4,572 ExperimentKit passed**,
  `TEST SUCCEEDED`; `/private/tmp/interpbench-python-parity-xcode-verified.log`.
  Xcode beta and the explicit Metal toolchain were selected; derived data was
  outside the checkout. `TEST_RUNNER_STEERLAB_TEST_PYTHON` selected the configured
  client environment for the real Python/Mac/Python interchange test.
- Parser AST audit: record admission matches `ef3dec8` after the explicit
  file-stream-to-StringIO seam; prior file/hash/frozen gates unchanged. The
  negative control rejected a changed generated prompt-ID rule.
- Client assembly reference, Swift generated CLI reference, shipped workspace
  contract, normal/release bridge gates, public scan and whitespace checks pass.
- The diff was read locally. Independent review through the maintainer's
  designated agent is still required; this is not approval to merge or deploy.

The first targeted pass caught missing required flags reaching the handler; the
adapter now rejects missing/repeated flags and wrong arity explicitly. Full
Python passed before the final atomic-publication improvement and passed again
after its two new failure/preservation tests. Swift interchange initially used
a Python without the client dependencies, then exposed an incomplete test
workspace override; both fixture configurations were corrected, and the entire
Swift suite passed. No production Swift behavior was changed to satisfy those
fixtures. The real round trip preserves prompt bytes, metadata and their pins.

No live GPU/model study, cluster/accounting exercise, app installation or
interactive UI qualification was performed. Runs and frozen studies remain
immutable. The previous branch tips and main are unchanged by this branch.
