# Researcher workflow implementation status

Current branch: `codex/cluster-execution-workflow`, based directly on landed main
`73bfbd7`. It includes the reviewed scientific-workflow slice, its changelog
follow-up and the independent final-norm dtype fix. The main checkout and installed
applications remain unchanged by this implementation branch.

This checkpoint adds acceptance of sourced cluster-profile companions and reviewed
standalone battery/stability jobs across the app, remote CLIs and HTTP. Recovery
and reconciliation now have public client/app adapters over the existing owners.
See [the current handoff](CLUSTER-EXECUTION-WORKFLOW-HANDOFF.md) for exact interfaces,
review scope and the distinction between implemented controls and live qualification.
Earlier sections below retain their historical checkpoint and validation claims.

The [implementation plan](RESEARCHER-WORKFLOW-IMPLEMENTATION-PLAN.md) remains
authoritative; WP-5 and WP-6 remain in scope. This branch authorizes no live
execution, installation or self-merge. Independent review through the
maintainer's designated reviewing agent remains the landing gate.

Final suite results, build conditions and gate status are recorded in the
[validation history](RESEARCHER-WORKFLOW-VALIDATION-HISTORY.md).

## What researchers and agents gain

The app, Mac CLI and workbench HTTP routes now share owners for reviewed agent
attachment, reusable design inspection/instantiation/batching/save/update,
protocol edits, prompt versions, evidence custody, and local model preparation.
These operations reduce hand editing and make their decisions inspectable.
Draft writes use external file digests and shared locks: stale edits refuse
without changing scientific identities or historical evidence.

Remote observation preserves live jobs, actions retain their captured origin,
and imports retain their captured local destination. Transfer policy and runtime
runner/workbench authority are enforced rather than merely documented. Imported
pipelines remain visible independently of the selected compute target.

These are implemented foundations, not a claim that every journey or every
engine adapter is complete. The [operation matrix](RESEARCH-OPERATION-MATRIX.md)
tracks surface reachability separately from scientific qualification.

## Audit response before the next phase

The review of `c490531` identified four landing blockers. This follow-up addresses:

| Finding | Implemented response |
|---|---|
| F1 controller recovery | Records controller allocation identity; exact terminal accounting can prove exit across nodes. Unknown owners are reported at startup. `jobs recovery/recover` supplies a read-only review and explicit, audited recovery. Snapshot checks serialize claims; scheduler I/O stays outside the write lock. [Contract](CONTROLLER-RECOVERY.md). |
| F2 implicit submission | Mac `remote submit-bundle` now requires `--verb` before connection/credential resolution, matching the engine and HTTP submission routes. The explicit composite `steerlab run` keeps its intentional behavior. Help and reference updated. |
| F3 scientific wording | The study explanation now says matched-norm random controls help test effects against comparable random perturbations; they do not establish construct specificity. |
| F9 release documentation | [Unreleased changelog](../CHANGELOG.md) covers new surfaces, breaking preconditions, transfer policy, roles, recovery and rollout order. |

Other corrections: documented `Document.copy()` **and** `copy.deepcopy()` preserve
read metadata, whereas `dict()` and JSON round trips do not; RUNNER deployments
allow RUNNER **and BOTH** routes; transfer policy applies to every profile with a
non-HTTP transfer method. The SIGINT regression checks restoration of the
original handler, including a shell that inherited `SIG_IGN`.

## Validation and review gates

Verification on 2026-09-06:

- Full serial Xcode beta suite: **277 SteeringKit + 4,529 ExperimentKit passed**
  (`TEST SUCCEEDED`). Log: `/private/tmp/interpbench-audit-fixes-xcode.log`.
- Full Python suite: **5,928 passed, 9 skipped, 8 warnings** (158.34 seconds).
  Log: `/private/tmp/interpbench-audit-fixes-python-final.log`. The first full
  pass exposed one stale engine-verb count assertion; it was updated for the
  two new administrative verbs before this clean run.
- Built Swift CLI: omitted, empty and blank operations all refuse at 64 before
  site resolution; help marks the operation required. SIGINT regression also
  passed with `SIG_IGN` deliberately inherited.
- Freshly compiled syntax auditor: **0 differences** across 45 changed files
  at the 109-property retirement (`a9545df` → `22024f7`) and seven changed files
  at the design-property retirement (`4fea25f` → `8e8bc36`). Route census AST
  still matches main with docstrings excluded. Historical ledger preservation
  was verified byte-for-byte after its new appendix header.
- Bridge ratchet and `git diff --check`: passed. The release-mode bridge gate
  correctly refuses the three remaining bridges; this is not a 1.0 release.

No live cluster, real model download, GPU study or app installation is part of
these checks. The code changes in this follow-up are semantic fixes; no new
mechanical-move claim is made. The maintainer supplied an independent re-review of `95d14aa` on 2026-09-06:
all four blockers were closed and both full suites independently reproduced.
That tip is approved for the designated integration process. Approval does not
extend automatically to the subsequent management-retirement checkpoint below.

The full earlier checkpoint/test ledger is preserved in the
[validation history](RESEARCHER-WORKFLOW-VALIDATION-HISTORY.md). Its results are
historical evidence, not a substitute for testing the current branch.

## Management checkpoint: `64102bf`

Removed `StudyManagementBindings.swift` and redirected its six properties and
sixteen commands to their management/design owners across 27 caller files.
Creation now receives the same workspace model context explicitly, and lineage
queries receive the owner's inventory explicitly. No management implementation,
study semantics or artifact schema changed. No substitute forwarding bridge was
added. The [reproducible syntax audit](BRIDGE-RETIREMENT.md#management-bridge-retirement-follow-up-to-95d14aa)
checks those constraints against the approved tip.

Verification on 2026-09-06:

- Full serial Xcode beta suite: **277 SteeringKit + 4,529 ExperimentKit passed**
  (`TEST SUCCEEDED`); `/private/tmp/interpbench-management-xcode.log`.
- Full Python suite: **5,928 passed, 9 skipped, 8 warnings** (179.91 seconds);
  `/private/tmp/interpbench-management-python.log`.
- Syntax audit: six properties, sixteen commands, **30 changed source/test
  files, zero differences** after the allowed normalization. This includes
  27 caller migrations and three additional files whose comments were updated
  to name the surviving owner. Implementation bodies remain unchanged.
- Negative audit controls: rejected an altered management body, missing
  creation context and wrong target owner.
- Compiler, bridge ratchet and whitespace checks passed. The release gate
  still correctly refuses the two remaining bridges.

This new checkpoint awaits the designated agent's independent diff review and
validation; the approval of `95d14aa` does not cover it.

This completed the management bridge slice of WP-2. The next checkpoint below
also retires freeze and remote coordination; retained writer preconditions,
context coverage and public-surface parity remain separate acceptance work.
The earlier audit's non-blocking observation about logging first allocation
capture, and live accounting qualification, remain follow-up notes.

## Approved checkpoint: all four Swift bridges retired

Freeze, draft sync, named server execution, delayed pipeline submission and
pipeline ledger observation now enter their focused controllers directly.
The panel supplies explicit runtime context and credential resolution.
Workspace, connection and serving-root changes cannot redirect admitted work;
selection and reviewed bytes additionally guard freeze and delayed pipeline
operations. Observation does not read Keychain credentials. The server digest
review used by draft sync is bound to the serving root, not just the endpoint.

This is semantic coordination work; the management AST proof remains pinned to
`64102bf`. Both bridge checks now pass without changing their baseline inventory.
See the [consolidated audit handoff](RESEARCHER-WORKFLOW-AUTHORING-AUDIT-HANDOFF.md)
for current test evidence, the full review range and remaining qualification.

Current verification: **277 SteeringKit + 4,539 ExperimentKit passed** in the
full serial Xcode beta suite; **5,928 Python tests passed, 9 skipped**. The
management checkpoint AST audit reproduced with zero differences across
30 files. Both bridge gates and whitespace checks pass. The new serving-root
regression caught a missing comparison in the first full Swift run; after the
correction the entire Swift suite passed. The maintainer's designated reviewing agent independently approved `2662dc8`;
that approval does not cover the subsequent study-assembly changes.

## Current checkpoint: retained authoring and study assembly

The [audit handoff](STUDY-ASSEMBLY-AUDIT-HANDOFF.md) describes the semantic changes
since `2662dc8`. Native draft commands now retain reviewed versions, sweep grid
and selection publish together, and lifecycle/import dialogs retain their targets.
Auxiliary server observations discard changed-origin replies. Pack preview/apply,
full-record JSONL import and vector inspection/attachment are available through
shared Mac CLI, Swift HTTP and native owners, with the same saved workspace.
The [workflow guide](STUDY-ASSEMBLY-WORKFLOW.md) connects these operations to
existing designs, model preparation, verification and explicit execution.

Public import results distinguish saved drafts from verification/readiness.
Overwriting raw/table prompt imports were retired in favor of immutable versions.
This advances WP-2 and the Mac portion of WP-3; it does not close every writer,
Python parity or scientific/interactive qualification item. The prior approved
checkpoint and main remain unchanged. This continuation awaits independent review.

Validation on 2026-09-06: full serial Xcode beta **277 SteeringKit + 4,551
ExperimentKit passed** (`/private/tmp/interpbench-assembly-xcode-verified.log`);
full Python **5,928 passed, 9 skipped, 8 warnings**, 165.80 seconds
(`/private/tmp/interpbench-assembly-python-final.log`). Generated CLI-reference
and workspace-contract gates, both bridge checks, public scan and whitespace
checks passed. No live model, scientific/GPU study, cluster or interactive app
qualification is claimed. See the audit handoff for review boundaries.

## What remains to implement the vision

1. **Audit the diagnostic transport/cleanup slice.** The current branch adds
   isolated input copies, completed evidence import, shared offline diagnostic
   receipts, and policy-bound removal across both clients and the app. See the
   [handoff](REMOTE-CUSTODY-CLEANUP-HANDOFF.md). Only successful isolated diagnostic
   output copies are eligible; broad remote maintenance remains outside scope.
2. **Audit and qualify the managed method/authoring continuation.** Shared
   method-specific interviews/forms, request publication, SAE offline pinning,
   managed execution and explicit campaign coordination are now implemented on
   this review branch. J-space remains a separate method with the existing
   artifact-admission restriction. See the [handoff](MANAGED-METHOD-AUTHORING-HANDOFF.md)
   for tests, limitations and the complete unlanded range.
3. **Qualify complete researcher journeys (WP-5/7).** Exercise agent/CLI and app
   paths with actual documentation, model preparation, queueing, interruption,
   permitted external-only transport, receipt verification, policy refusal and
   offline results. Fixture suites do not establish live scientific/GPU,
   scheduler/accounting, Linux filesystem or interactive app qualification.

**Integrated upstream fix:** main's terminal-job receipt gate and immutable
sibling reimport are included in this branch. Live scheduler/accounting and
cleanup qualification remain outstanding; the code defect is no longer an
unintegrated upstream task. Never repair historical runs in place.

## Deployment and review

- The maintainer, **through the designated reviewing/integration agent**, reads
  the actual diff, checks both full suites, and reproduces AST audits for any
  claimed mechanical moves. Only that process authorizes landing. No self-merge.
- Fast-forward remains the intended landing method while main is an ancestor.
  Recheck ancestry and current main immediately before integration; if main has
  advanced, integrate its fixes and rerun the affected gates first.
- Install/deploy the updated **app and Swift CLI before the engine**, so manifest
  PUT callers send the required preconditions. The old engine ignores the added
  header; the old app receives 428 from the new engine. Coordinate any other
  manifest PUT clients as part of rollout.
- Decide service roles separately from this landing. The default stays
  `workbench`; a runner permits RUNNER/BOTH and refuses WORKBENCH authoring.
  No private launch configuration is changed here.
- Keep repository files and commits generic; secrets follow the established
  Keychain/path-indirection contract. Runs and frozen artifacts remain immutable.
  No revision field may be added to content-hashed manifest bytes.
