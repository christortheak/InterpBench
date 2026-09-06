# Researcher workflow implementation status

Branch: `codex/researcher-workflow-implementation`, based on main `bfd13a5`.
The branch is a review artifact. Main, the installed app, and live services
remain unchanged. The [implementation plan](RESEARCHER-WORKFLOW-IMPLEMENTATION-PLAN.md)
is authoritative; WP-5 and WP-6 remain in scope.

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
mechanical-move claim is made. Independent review of this follow-up is pending.

The full earlier checkpoint/test ledger is preserved in the
[validation history](RESEARCHER-WORKFLOW-VALIDATION-HISTORY.md). Its results are
historical evidence, not a substitute for testing the current branch.

## What remains to implement the vision

1. **Finish shared authoring and context ownership (WP-1/2).** Complete the
   remaining draft writer/adapter migration and retire the three remaining
   Swift bridges. Observation, auxiliary jobs and delayed actions still need
   end-to-end context coverage. The normal bridge ratchet passes; the 1.0
   release gate must refuse until all bridges are removed.
2. **Complete public surface equivalence (WP-3/4).** Fill cross-platform design,
   model preparation, study assembly and advanced-method gaps; ship practical
   method guides, dataset templates and coworker prompts. Every supported
   operation needs explicit inputs, truthful outcomes and actionable repairs.
3. **Complete cluster coauthoring and the managed remote lifecycle (WP-5/6).**
   Demonstrate documentation-to-profile authoring, resolve missing facts with
   the researcher, and finish composed monitoring, recovery, evidence transfer
   and bounded cleanup plan/apply with policy and dependency checks.
4. **Qualify the complete journeys (WP-7).** Exercise researcher/agent/UI paths,
   offline and failure recovery, and scientific/GPU checks. Record unavailable
   checks honestly; unit tests alone do not establish research validity.

**Separate upstream task:** imported submit receipts may contain still-growing
logs. Preserve that task with the responsible agents; this pass does not change
its classification. Custody/remote cleanup qualification depends on resolving
that evidence boundary. Never repair it by modifying imported historical runs.

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
