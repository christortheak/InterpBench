# Audit handoff: retained authoring and study assembly

Branch: `codex/researcher-study-assembly`, based on current main `a5db8c3`.
Main now includes independently approved `2662dc8` and two review documents.
Review this implementation against `a5db8c3`. The approved authoring and earlier
implementation branches remain at `2662dc8` and `95d14aa` respectively. This is a
semantic continuation; it makes no mechanical-move claim and does not extend the
historical AST audit to changed authoring bodies.

## What changes for researchers and agents

The Mac app, CLI and Swift workbench can now use the same conceptual study
interview, preview a proposed study pack, create its draft and inputs, import
full prompt records, and inspect/attach an existing vector. They continue from
one saved workspace. Successful import reports remaining verification issues;
it does not claim research readiness or start execution.

The [researcher/agent workflow](STUDY-ASSEMBLY-WORKFLOW.md) gives the commands,
HTTP request fields, interface handover and readiness boundary. Existing shared
design and model-preparation operations fit into that sequence. The workspace
agent contract, parser/help and generated CLI reference describe the new verbs.

## Review slices

| Concern | Implementation and acceptance evidence |
|---|---|
| Retained draft commands | `DraftAuthoringTransaction.perform` admits exact reviewed manifest bytes under the shared lock and checks the active workspace for remaining store adapters. Native concept attachment, conditions, validation controls, instruments, baseline and promotion policy commands use it. The saved review is captured before releasing the lock; a catalog refresh cannot authorize old form fields. |
| Sweep declaration | Grid and selection validate and publish in one store update. The composer and editor provide a retained review. The editor receives the saved snapshot from the command instead of borrowing a later disk version. Stale and invalid declarations leave both halves unchanged. |
| Creation and lifecycle dialogs | Native creation publishes its complete initial settings once under an absent-file precondition. Rename/delete dialogs retain their named study; canonical rename locks source and destination and restores the directory if publication fails. Frozen names retain the display-label/duplicate repair. Rename-plus-label partial failure reports the already-completed rename. |
| Prompt inputs | JSONL and mapped-table imports share `TaskPromptsAuthoring` with the full-record editor. They publish immutable content-addressed versions and pin their real bytes. Removed the overwrite-oriented raw/table helpers instead of leaving compatibility shims. New CLI/HTTP raw imports require an external manifest digest and report unchanged re-imports truthfully. |
| Auxiliary server work | Residency, run/optimization listings, awaiting judgments and sweep-detail reads capture context; changed origins and superseded observations cannot publish late replies. Observation does not load Keychain credentials. Promotion rechecks context around connection and submission. Judgment dispatch requires the retained study/origin and complete reviewed awaiting record; subsequent work retains its admitted client. |
| Study packs | Read-only preview records exact pack/root and input observations. Apply rechecks under shared locks, refuses occupied names/differing files, writes only missing text inputs, strips freeze metadata, creates a draft and reports verification issues. Export includes text inputs and separately names external dependencies. |
| Vector attachment | `StudyArtifactAuthoring` inspects both tensor and sidecar bytes, then rechecks those digests and the reviewed draft before calling existing scientific admission. CLI, HTTP and the native sheet share it. Reader-derived norm refusal remains covered; no vector/run bytes are rewritten. |
| Discoverability and handover | New family/verb inventories, workspace contract and CLI reference are synchronized. The fixture journey asks for the shared interview, previews via CLI, applies through the HTTP adapter, observes the draft in the native panel, imports via HTTP and CLI, observes stale native edits, and exports again. |

These fixtures test the shared commands and HTTP adapters directly. They are not
a claim of interactive GUI or socket-level qualification for every new route.
The production route wiring is part of the required diff review.

## Boundaries that must remain explicit

- No manifest revision field or other concurrency metadata enters scientific
  content-hashed bytes. Existing freeze hashes, frozen studies and runs remain
  immutable. The experiment schema is unchanged.
- Canonical rename and draft writes are covered here. Display labels remain
  local sidecar metadata; this pass does not introduce separate label-file CAS.
- Synchronous store pin/verification adapters still use the active workspace.
  Their entry points guard that root, and the Swift workbench keeps these new
  commands on its actor. This is not a completed migration of every store to
  explicit dependencies or every writer in the repository.
- Pack apply rolls back files it successfully created and still owns if draft
  publication fails. A process crash or failed filesystem write is not a
  multi-file filesystem transaction. Prompt publication may leave an unreferenced
  prepared input after a later failure; it cannot overwrite previous input bytes.
- A pack is authored input, not an execution bundle. Exported text carries its
  manifest pins; missing or changed dependencies still owe verification. No
  imported draft is declared scientifically valid merely because JSON decoded.
- Python pack/design/model parity, remaining advanced-method guides and coworker
  templates, cluster coauthoring/managed cleanup, and complete scientific/GPU and
  interactive journey qualification remain in the implementation plan. WP-5 and
  WP-6 remain in scope. This checkpoint does not implement them.
- The separate upstream growing-log/evidence-receipt issue remains with the
  responsible agents. It is not repaired by rewriting historical runs here.

## Validation

Final validation on 2026-09-06:

- Serial Xcode beta: **277 SteeringKit + 4,551 ExperimentKit passed**;
  `TEST SUCCEEDED`, `/private/tmp/interpbench-assembly-xcode-verified.log`.
- Python: **5,928 passed, 9 skipped, 8 warnings**, 165.80 seconds;
  `/private/tmp/interpbench-assembly-python-final.log`. No Python source or tests
  changed in this continuation; main's subsequent additions were review docs only.
- Generated CLI reference, workspace contract, normal/release bridge checks,
  public-content scan and whitespace checks passed.
- Actual diff read locally; independent maintainer review remains required.

New regressions cover stale commands and dialogs, atomic sweep declarations,
immutable prompt versions and repeat imports, changed pack/input/workspace
reviews, vector-byte drift, existing norm admission, and late server-origin
responses. Existing import tests tied to overwrite semantics were replaced with
versioned-input regressions; parsing and baseline-import coverage remains.

The initial full Swift pass caught missing public discovery entries, test-fixture
errors and a lost rename repair. A subsequent complete run passed implementation
tests but found generated-reference drift. Keep those failures in the validation
history rather than citing only a filtered green run.

## Review and landing

The maintainer, through the designated reviewing/integration agent, must read the
actual diff and reproduce both full suites before authorizing landing. Historical
mechanical checkpoints retain their own AST evidence; this continuation changes
behavior deliberately. Normal and release bridge gates must continue to pass.
No substitute bridges or dependency-alias convention is added.

Recheck current main and ancestry before integration. Fast-forward is the intended
method while main remains an ancestor; integrate new main fixes and rerun the
required gates if it advances. No self-merge, deployment or live-service change
is authorized by this handoff. The existing rollout order remains app/Swift CLI
before the engine. Repository and commit vocabulary stays generic; secrets retain
the established Keychain/path-indirection rules.
