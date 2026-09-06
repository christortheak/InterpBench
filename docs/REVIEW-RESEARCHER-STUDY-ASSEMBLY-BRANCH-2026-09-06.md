# Review — `codex/researcher-study-assembly` at d674fcd (2026-09-06)

Reviewer: Claude (maintainer's reviewing/integration agent). Range reviewed:
`a5db8c3` (current main) to `d674fcd`, one commit, 49 files,
+2,176 / −985. No Python source or test changed. Main is an ancestor, so
the branch fast-forwards. Worktree `/private/tmp/interpbench-workflow-handoff`
clean at the tip. No edits were made to the branch.

## 1. Verdict

**Landable after one stale sentence is fixed** (F1). The slice does what
the handoff says: study-pack preview/apply/export, full-record prompt
import, and vector inspect/attach now have one owner each, reached from
the CLI, the Swift workbench HTTP routes, and the native sheets; native
draft commands, sweep declarations, rename and delete run against a
retained review under the shared lock; auxiliary server reads carry a
captured context and drop late replies. The immutability rules hold
everywhere I looked. Four notes follow (§4); none blocks. A parallel,
already-verified fix for the import-receipt drift lands immediately after
this branch and is described in §6 so you can stay clear of it; §7 is the
checklist before the merge.

## 2. Verified independently

| Check | Result |
|---|---|
| Bridge gates, normal and `--release` | both PASS; no bridge file reappears; zero forwarding accessors in the panel |
| Vocabulary in the diff and commit message | none |
| `git diff --check` | clean |
| Server tree | untouched (diff stat empty) |
| Full Python suite on the worktree (main venv, cwd = worktree/Server) | 5,928 passed, 9 skipped, 8 warnings; nothing Python changed, run anyway per the landing method |
| Full Xcode beta suite on the worktree (serial, external DerivedData, app not running) | `TEST SUCCEEDED`: 277 SteeringKit + 4,551 ExperimentKit, matching the handoff's count |
| Pack owner (`StudyPackAuthoring`) read in full | see §3 |
| Artifact owner (`StudyArtifactAuthoring`) and the three surface files read in full | see §3 |
| Store rename, sweep declaration, management controller diffs read in full | see §3 |
| Panel diff read in full | context capture with generation counters on residency, runs, optimizations, awaiting judgments and sweep detail; every store command routed through `editReviewed`; no forwarding layer |
| Test inventory | 4 new test files (assembly safety 11 cases, pack surfaces 7, auxiliary context 5×2 parametrized, reader-derived attach 9). Eleven overwrite-semantics tests removed from the two import suites and replaced by versioned-input regressions (preserve previous inputs, stale table import publishes nothing, garbage/empty refused). Coverage is replaced, not dropped. |

## 3. What the code actually does, against the handoff's slices

**Retained draft commands.** `DraftAuthoringTransaction.perform` refuses a
changed active workspace, takes the manifest lock, requires the reviewed
digest, admits draft-only edits, then runs the store command.
`StudyManagementController.editReviewed` wraps it and advances only that
editor's review. Concept attachment, conditions, validation controls,
instruments, human baseline and promotion rule all go through it. Correct.

**Sweep declaration.** `setSweepGrid` gained `selectionUpdate`; grid and
selection validate (`validateSelection`, `validateObjectiveRequirements`)
before the single `manifest.sweep = spec` publication. A stale review
throws before either half. Correct.

**Rename.** Locks source and destination in canonical-path order, requires
the reviewed digest on the source, refuses frozen and occupied targets,
counts stranded runs without touching them, moves the directory, and moves
it back if the manifest save fails, reporting a double failure explicitly.
The rename-then-label partial failure now names the completed rename.
Correct, and better than before.

**Prompt inputs.** `importJSONL` validates with the retained
`TaskPromptsImport.preview`, then publishes a content-addressed version
under `prompts/tasks/versions/` and pins it through the reviewed draft;
an identical existing version is reused and `changed` is reported
truthfully. The overwrite-oriented `importTaskPrompts` and
`importIntoStudy` helpers are gone rather than shimmed. Correct.

**Study packs.** Preview is read-only: refuses an existing study, restricts
pack files to relative paths under `prompts/` with no `..`, no symlink
destination, nothing under `runs/`; refuses a differing existing file
outright (packs never overwrite); reuses identical files; observes
referenced inputs the pack does not supply; returns a review digest over
all of it. Apply re-previews, requires the digest, locks every touched
path in sorted order, re-previews under the locks, writes only `create`
files with `.withoutOverwriting`, pins named inputs, saves the draft under
an absent-file precondition, and on failure removes only files whose bytes
still equal what it wrote. Freeze metadata is stripped on decode. Export
runs under the review (frozen sources allowed) and names anything outside
`prompts/` as an external dependency. Correct.

**Vector attachment.** `inspect` snapshots both files, refuses absolute or
escaping paths and anything outside the workspace. `attach` takes the
manifest lock, then both artifact locks, re-inspects, requires both
digests, and only then calls the existing `ExperimentStore.attachArtifact`
so residual-norm and substrate admission are unchanged
(`reviewedArtifactKeepsScientificNormAdmission`). Vector bytes are never
written. Correct.

**Auxiliary server work.** Residency, remote runs, remote optimizations,
awaiting judgments and sweep detail each capture the context and a
generation id, and discard a reply if either moved. Observation uses the
in-memory client; only promotion calls the resolver and rechecks the
profile afterwards. Judgment dispatch requires the retained awaiting record
byte-for-byte. Correct.

**Discoverability.** Seven new Swift verbs (`authoring study`,
`pack preview|apply|export`, `experiment inspect-artifact`,
`attach-artifact`, `import-prompts`), six new HTTP routes, the contract
and its workspace mirror, help, and the generated reference are in step;
`pin-prompts` is untouched.

## 4. Findings

### F1 — Stale reference sentence (fix before landing, one line)

`docs/CLI-REFERENCE.md:5364`: "There is still no Swift `attach-artifact`
CLI verb". There is one now, with a stricter contract than the engine's
(`--artifact-sha256`, `--sidecar-sha256`, `--manifest-sha256` required;
the engine verb takes only `--artifact`). The sentence should say that,
and the same paragraph's claim that extract/validate/sweep/run of an
artifact-pinned study remain server-only is still true and should stay.

### F2 — Residency indicator no longer loads the token (note)

The old residency check called `loadStoredRemoteToken()` before listing
server experiments; the new one reads `context.client` and does nothing
when the client is nil. That is the stated design ("observation does not
load Keychain credentials"), but on a fresh launch the "study is on the
server" indicator will stay unknown until some user action resolves the
credential. The run-path refusal still backstops it. Worth confirming in
the interactive pass, not a code change here.

### F3 — Same verb name, two contracts (note)

`experiment attach-artifact` now exists under both products with
different flags. That is consistent with the two-products rule (the names
are the product prefixes), but the reference's §4.3 engine section and the
new Swift entry should cross-reference each other so an agent does not
carry one contract to the other CLI.

### F4 — Pack apply's auto-pin swallows errors (note)

`autoPinNamedInputs` uses `try?` on each pin. A pin that fails leaves the
key unpinned and the draft is still created; `verify` then reports it as a
violation in `verificationIssues`. That is the documented behaviour
("reports remaining verification issues") and the tests cover it, but the
result document does not say *which* pin failed or why. A future
improvement, not a defect.

## 5. Landing instructions

1. Both suites are green on d674fcd (§2).
2. Fix F1 on the branch or in the landing commit.
3. Fast-forward main to the tip, then land the import-receipt gate as its
   own commit on top (§6). The deploy rule from the earlier reviews still
   applies: app and Swift CLI before the engine; the engine deploy still
   waits for the ladder job. The full pre-merge checklist is §7.

## 6. A parallel fix that lands right after this branch: the import-receipt gate

Your handoffs have correctly left the growing-log receipt problem to "the
separate responsible agent". That work exists, is reviewed, and is verified;
this section tells you what it is so you can finish your own slice without
colliding with it. **Do not merge it into your branch.** The maintainer's
reviewing agent lands it as its own commit on top of yours.

**Branch:** `claude/infallible-chandrasekhar-743df3` at `11dc783`, local
only (not on the remote), based on `bfd13a5`. One commit, 14 files,
+1,727 / −63. It merges cleanly onto current main `a5db8c3` and onto your
tip `d674fcd` (checked with `git merge-tree`, no conflicts).

**What it does.**

- `cluster import` now holds a submission receipt
  (`<stamp>-submit-bundle-<experiment>-<verb>` and the other `submit-…`
  shapes) while any Slurm job it names is still live, exactly as it
  already holds a stage directory without its completion artifact. A
  receipt names its jobs by the scheduler's own captures inside its
  bundle directories (`slurm/slurm-<jobid>.out`, `slurm-shard-<k>/…`).
  Reported under the existing `importSummary.skippedInProgress` key.
- "Ended" is proved first by content: the engine's rendered sbatch script
  gains an EXIT trap, installed after node-scratch cleanup and chaining to
  it, that writes `slurm-<jobid>.exit` (carrying the exit status) beside
  the capture as the last thing the script does. It prints nothing and
  never calls `exit`, so the job's own status is preserved, including the
  checkpoint code. Only under a real `$SLURM_JOB_ID`.
- Otherwise it asks the scheduler once per import pass: one
  `squeue -h -u $USER -o '%i|%T'` plus one `sacct -n -X -P -j <ids>`,
  through the site profile's declared commands over the shared SSH
  session. Never `steerlab-server jobs list`. A job squeue lists is live;
  a terminal sacct state is ended; an id neither knows after both
  answered is ended; a failed query is unknown, and unknown holds.
- `cluster import --site <id> --reimport-drifted` repairs the receipts
  that already drifted: the cluster's copy comes home beside the local one
  as `<name>-reimport` (then `-reimport2`, …), a new directory verified by
  content; the local original is never rewritten. The classifier strips
  the suffix so a copy classifies as its original. A later import reads a
  matching copy as `importSummary.driftResolved` instead of a violation.
  Without the flag, the refusal text and the envelope's `repairAction`
  name the exact command with the site id.

**Verified by the reviewing agent** on a throwaway merge of main plus the
branch: Xcode 277 + 4,559, Python 5,938 passed / 9 skipped (both counts
are main's plus the branch's new tests, so the generated reference and the
contract mirror survived the merge unchanged); vocabulary and whitespace
clean. Nine Python tests cover the trap (last EXIT trap installed, chains
cleanup, records the status, preserves the exit code, writes nothing
outside Slurm, survives a site with no cleanup function, quotes paths).
Ninety-one Swift tests cover the gate, the scheduler interpretation, the
hold texts, and the reimport round trip.

**Files it touches** (keep clear of these until it lands, or your next
merge will stop being clean):

- `Server/steerlab_server/api/executors.py` (`render_slurm_script`, new
  `job_end_marker_lines`, `JOB_END_MARKER_SUFFIX`)
- `Sources/ExperimentKit/WorkspaceImportPolicy.swift`,
  `WorkspaceRunImport.swift` (the gate, `resolveDrift`, the
  `remoteSchedulerStates` and three-argument `transfer` seams)
- `Sources/ExperimentKit/ClusterCLI.swift`, `ClusterCLIRunner.swift`,
  `ClusterCLIEnvelope.swift` (the `--reimport-drifted` flag; envelope keys
  `reimported`, `driftResolved`)
- `Sources/SteerLabApp/ServerJobsPanelView.swift` (status line counts
  reimports)
- One paragraph in `AgentContract.swift` and its mirror
  `docs/AGENTS-WORKSPACE-DRAFT.md`; the `cluster import` usage line and
  the hand-written `import` row in `docs/CLI-REFERENCE.md`; one gloss in
  `CLIHelp.swift`.

Your branch and this one already overlap on the last group (contract,
mirror, reference, help) and merge cleanly today. Editing those files
further is fine; editing the importer or the sbatch renderer is not.

**What you can do about it in your documents.** The operation matrix
(REMOTE-06 / REMOTE-10 evidence column), the implementation status
("separate upstream growing-log task"), and the validation history all
carry the receipt issue as open. You may close those rows now by naming
the hold (`skippedInProgress` for live receipts) and the repair
(`--reimport-drifted`, `importSummary.reimported` / `driftResolved`), and
attributing them to the branch above rather than to your slice. Do not
describe them as landed until the reviewing agent has landed them.

**Deployment note for the record.** The engine half only takes effect
after the next cluster deploy; until then receipts are decided by the
scheduler fallback. After deploy, one `cluster import --reimport-drifted`
settles the five drifted receipts in the study workspace. Neither is your
task.

## 7. Checklist before the reviewing agent merges your branch

1. Fix F1: `docs/CLI-REFERENCE.md:5364`, and cross-link the engine and
   Swift `attach-artifact` entries (F3) while you are there.
2. Optionally close the receipt rows in the matrix, status and history as
   described in §6, attributed to the parallel branch.
3. Do not touch the files listed in §6 under "Files it touches", other
   than the contract, mirror, reference and help you already share.
4. Rerun both full suites and both bridge gates on your tip; keep
   `a5db8c3` as an ancestor so the landing is a fast-forward.
5. Leave the SIGINT-restore test as it is (it checks restoration of the
   inherited handler, which is the contract), and keep the serial Xcode
   invocation as the Swift gate.

Landing order, for everyone's calendar: your tip by fast-forward; then the
receipt gate as its own commit on top; both suites on the result; push.
The deploy rule is unchanged: app and Swift CLI before the engine, and no
engine deploy until the running ladder job finishes.

## 8. Follow-up tip f6dae11 — verified, ready to land

One docs-only commit on top of d674fcd (`docs: correct reviewed vector
attachment guidance`, 2 files): F1's stale sentence is replaced with the
Swift verb's contract, and §3.3 / §4.3 of the reference now cross-reference
each other (F3). No source, test or generated region changed; the §6
files are untouched; the chip branch still merges clean; main is still an
ancestor. Reverified on f6dae11: Xcode `TEST SUCCEEDED` 277 + 4,551;
Python 5,928 passed / 9 skipped; both bridge gates PASS; vocabulary and
whitespace clean. **Landable by fast-forward.**
