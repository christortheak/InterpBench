# Review — `codex/researcher-workflow-implementation` at c490531 (2026-09-06)

Reviewer: Claude (maintainer's reviewing/integration agent). Baseline main
`bfd13a5`. Worktree read at `/private/tmp/interpbench-workflow-handoff`
(clean, no uncommitted work). 27 commits, 210 files, +13,537 / −3,169.
No edits were made to the branch. This document is untracked in the main
checkout; the maintainer decides whether it is committed.

## 1. Verdict

**Not yet landable as a whole; close to it.** The work honours every
constraint the plan named (no manifest revision field, no frozen-evidence
rewrite, no secret persistence, no site vocabulary, no main merge), the
defect fixes are real and each carries a regression that fails on the old
behaviour, and the two mechanical-move claims reproduce. Four things block a
landing, and none is large:

1. **OPS-01's new recovery rule cannot recover any job on this project's
   actual deployment** (§3, F1). It must gain a scheduler-aware or explicit
   recovery path before it replaces the sweep it removed.
2. **OPS-05 is now inconsistent across surfaces** (F2): the engine and API
   refuse a missing verb, the Swift `remote submit-bundle` still defaults to
   `run`, and the reference documents the old behaviour for both.
3. **SCI-01 is half done** (F3): the agent contract is corrected, the study
   authoring prompt still says controls "prove" the effect is the direction.
4. **No CHANGELOG entry** (F9) for 17 new CLI verbs, four new HTTP
   operation families, a 428-gated manifest write, and a service-role switch.

Everything else in §3 is a follow-up that can land after, or a note for the
deploy.

## 2. What was verified independently (not taken from the ledger)

| Check | Result |
|---|---|
| Full Xcode beta suite on the worktree (serial, external DerivedData) | `TEST SUCCEEDED`: 277 SteeringKit + 4,527 ExperimentKit (main: 277 + 4,400) |
| Full Python suite on the worktree (main venv, cwd = worktree/Server) | 5,895 passed, 9 skipped, **1 failed** — `test_an_interrupt_during_the_wait_detaches_rather_than_cancelling`. Reproduced as a harness artifact: it asserts the restored SIGINT handler is the default, and a backgrounded non-interactive shell inherits SIG_IGN. It passes in the foreground on the branch and fails under `&` on main too. The branch's own claim (5,896 pass) stands. |
| `scripts/ci/audit-panel-owner-access.swift` recompiled and rerun against `22024f7` vs `a9545df` per BRIDGE-RETIREMENT.md | "Audited 45 changed source files; normalized syntax-tree differences: 0" |
| Route census relocation `Server/tests/route_roles.py` (main) → `api/route_roles.py` (branch) | Python AST identical after docstring strip |
| Site/study vocabulary in the diff and in 27 commit messages | none |
| `git diff --check` | clean |
| Swift-side OPS-03 (captured workspace through auto-import and chain import) | `EvidenceAutoImport` captures `workspaceRoot` at construction and compares the origin's root on every ledger read; chain import takes an explicit root. Confirmed in source. |
| UI-01 | `ExperimentsPanelView.swift:290` now shows pipelines when local pipeline runs exist regardless of compute target. |
| Python precondition design (`manifest_files.py`) | Digest of exact bytes, sidecar `flock` under `.steerlab/manifest-locks/` keyed by sha256(realpath), reentrant per thread, `.gitignore` written once, no field in the document. Swift `ManifestFileTransaction` uses the same key and primitive (the `/tmp` alias fix in `a9545df` is correct). |
| All in-tree `save_raw` callers pass a `Document` from `load_raw` (store internals and `confirmation.py`) | yes |

Not done: a line-by-line read of the ~80 Swift authoring files (design,
agent, prompt, custody, coauthoring, model preparation). Those were read at
the level of the new owners, their HTTP/CLI adapters, the tests that
exercise them, and the generated reference. The suites and the ledger's
wire-test logs under `/private/tmp` are the evidence for their behaviour.

## 3. Findings

Ranked. "Blocks landing" means the maintainer should not merge until it is
resolved; the rest are follow-ups.

### F1 — OPS-01 recovery never fires on a scheduler-hosted controller (blocks landing)

`api/job_ownership.py` records `(hostname, pid)` per job and
`owner_has_exited` returns **False** whenever the recorded host is not the
current host. `_sweep_orphans` now only claims jobs whose owner is proven
dead on the same host; legacy rows with no owner are never claimed.

On this project's deployment the controller is a 24-hour Slurm job that
restarts on a different node every day. Today's job listing shows the
local-executor kinds that run inside that process: `model:install` (22),
`concept-extract`, `grandmean-extract`, `jlens-*`, `experiment:validate`,
`experiment:analyze`, `pipeline-orphan-reconcile`, `study-submit-bundle`
(local, 11). When a controller dies at walltime with any of those
`running`, the successor on the next node can never fail them. Before this
branch the startup sweep failed them; after it, nothing does, and there is
no manual verb (`jobs reconcile <records-dir>` is a different operation).
The 1,161-job store on the cluster has no `job_owners` rows yet, so any
record in flight at the moment of deploy (none today) would be permanently
unrecoverable; every job submitted after deploy is covered only while the
same controller node lives.

The intent is right (observation must not sweep; a second manager is not a
crash). The rule is too narrow. Two acceptable completions, either is fine:

- Record the controller's scheduler job id alongside host/pid when
  `SLURM_JOB_ID` (or the generic controller identity the serverd template
  already exports) is present, and treat a scheduler-reported terminal
  state for that id as proof of death. `sacct` is already wrapped.
- Or add `steerlab-server jobs recover <job-id> --owner-exited` as an
  explicit, typed admin operation that the successor controller (or a
  person) invokes, with the refusal naming why automatic recovery declined.

Tests to add: foreign-host owner with a scheduler id that `sacct` reports
COMPLETED/FAILED is recovered; the same id RUNNING is not; legacy rows on
the deployed store are reported (count) at startup rather than silently
kept.

### F2 — OPS-05 conflicting interpretations across wrappers (blocks landing)

- `steerlab-server study submit` and `POST /api/studies/submit`,
  `/submit-bundle` now refuse without a verb (correct).
- `steerlab-cli remote submit-bundle` still defaults:
  `ExperimentCLIRunner.swift:2239` `flag("--verb") ?? "run"`, and
  `CLIHelp.swift:343` documents "defaults to run".
- `docs/CLI-REFERENCE.md:437-439` still says both default to `run`; the
  first sentence is now false.

The plan's own text: "do not create conflicting interpretations across
wrappers", and "inspect composite entry points and explicitly decide which
defaults remain intentional". If the Swift default is intentional (the
composite `run` verb is the documented composite; `remote submit-bundle`
is the low-level one, so I would refuse there too), the decision must be
written down and the reference corrected either way. The Swift verb-census
test will need its expected list updated if the flag becomes required.

### F3 — SCI-01 unfinished (blocks landing, five-minute fix)

`Sources/ExperimentKit/StudyAuthoring.swift:94`: "controls (matched-norm
random) prove an effect is the DIRECTION, not the perturbation energy". The
agent contract's extraction paragraph (main `AgentContract.swift:524`) was
corrected and the same file now says controls "support interpretation; they
do not prove". The study-pack prompt was missed. Suggested wording: "a
matched-norm random control shows the effect is not explained by
perturbation energy alone; it does not establish that the direction
isolates the construct".

### F9 — No CHANGELOG entry (blocks landing)

`CHANGELOG.md` is untouched. The branch adds `experiment manifest`,
`experiment attach-agent`, `agent list/inspect`, `design
list/inspect/describe/instantiate/batch/save/update`, `data
custody/verify-custody`, `cluster sites guide/review`, `model plan/install`
(Swift); `--service-role` on `serve`, `STEERLAB_SERVICE_ROLE`, a 428/412
precondition on `PUT /api/experiment/{name}/manifest`, `ETag` on the GET,
403 `external_transfer_required` on upload/download, `operation_required`
on submit; and the `save_raw` semantic change in F6. All of it belongs
under `[Unreleased]`, with F5's deploy-ordering note.

### F4 — Service-role enforcement is opt-in and undocumented in the reference (follow-up)

`STEERLAB_SERVICE_ROLE` defaults to `workbench`, so nothing changes for
any existing deployment, including the cluster controller, until a launch
configuration sets it. That is the right default for landing. But:

- `docs/CLI-REFERENCE.md` §2.7 (environment table) has no row for it;
  PORTABILITY-CONTRACTS §11 is updated and is the only place it is named.
- The controller template that would set it lives in the private Sites
  repository. Deciding whether the cluster controller becomes a `runner`
  is a deployment decision with consequences: `model:install` is censused
  RUNNER (fine), but every `both`/WORKBENCH route the app uses through the
  tunnel would refuse. Record the decision; do not flip it in the same
  landing.
- `_ROUTES` sorts by `{` count so static paths win; an undeclared route
  refuses on a runner. `test_every_route_the_app_serves_is_censused`
  keeps that closed. Good.

### F5 — Manifest PUT is now a breaking API change; deploy the app before the server (deploy note)

`PUT /api/experiment/{name}/manifest` returns 428 without `If-Match` /
`If-None-Match: *`. The app at 98a9a4bf (installed) sends neither. Deploying
this branch's engine to the cluster while that app is installed breaks the
one-click server-draft sync until the app is rebuilt. The new app sends
`If-Match` (`ClusterClient.swift:2628`) and an old engine ignores the
header, so **app first, engine second** is the safe order. The Python
client has no caller of this route. Say so in the changelog.

### F6 — `save_raw` semantics changed for plain dicts (changelog + note)

`save_raw(dict)` on an existing manifest now refuses with
`staleManifest` because a plain dict carries no precondition and `None`
means "must not exist". Every in-tree caller is fine (they save the
`Document` that `load_raw` returned). Any out-of-tree script that did
`d = json.load(...); ...; save_raw(d)` breaks. Acceptable, and arguably the
point, but it has to be stated. `Document.copy()` preserves the digest;
`dict(d)` and `deepcopy` do not, which is a trap worth one sentence in the
docstring.

### F7 — Transfer policy: behaviour widened beyond `cluster` profiles (note)

`profile.py`: `externalTransferRequired` no longer requires
`profile == "cluster"`. Any profile that sets a non-HTTP `transfer_method`
now advertises and enforces it. Reasonable; a one-line changelog mention.
The client's `require_http_transfer()` adds one capabilities round trip per
upload/download; fine.

### F8 — Bridge retirement is 1 of 4 (expected, not a defect)

| File | main | branch |
|---|---|---|
| StudyPanelBindings.swift | 551 | removed |
| StudyManagementBindings.swift | 152 | 108 |
| StudyRemoteCoordinationBindings.swift | 124 | 120 |
| StudyFreezeBindings.swift | 88 | 88 |

The ratchet passes and the 1.0 gate in BRIDGE-RETIREMENT.md is still
correctly reported as unmet. The two audits (109-property retirement, seven
design properties) are honest about what is mechanical and what is not.

### F10 — The import-receipt drift this branch does not address (note)

Today's `cluster import` refused a fifth byte-drift violation, all five
being submit-bundle receipt directories imported while their Slurm logs
were still growing. The branch's "verified reuse" and custody work touch
the importer but not this classification. A separate task chip exists for
it; it should not be folded into this branch.

### F11 — Scale and landing shape (advice)

Twenty-seven topic-sliced commits are the right shape, but they are
interleaved across WP-1, WP-2, WP-3, WP-5 and WP-6 on one linear branch,
so they cannot be landed per package without rebasing. Given the suites
pass at the tip, the practical path is: fix F1–F3 and F9 on the branch,
rerun both suites, land the tip as one merge with this review attached.
The ledger (883 lines) should be trimmed to the per-package status table
plus limitations before it is committed as documentation; the
checkpoint-by-checkpoint narrative belongs in the commit messages it
already duplicates.

## 4. What is good and should be kept as is

- The precondition design on both engines: digest of exact bytes outside
  the document, one shared sidecar lock, reentrant, atomic replace inside
  the lock, HTTP mapped to 428/412 with the ETag from the GET. This is the
  design the plan asked for and it is done carefully.
- `EvidenceImportOrigin` uses endpoint + login, never the tunnel port, and
  captures the workspace at snapshot time; the ledger is origin- and
  bundle-version-scoped; corrupt ledger bytes refuse rather than being
  replaced.
- `service_authority` sits inside auth (registered before it, so auth is
  outer), fails closed on an invalid role, and never widens authentication.
- Every WP-1 defect has a regression that fails on the old code, using
  disposable stores and controlled transports.
- The operation matrix's five-valued vocabulary (present / gap / restricted
  / file / unverified) and its refusal to infer absence.
- Vocabulary, whitespace, secrets, and frozen-artifact hygiene are clean.

## 5. Suggested order

1. F3 (wording), F9 (changelog), F2 (decide and document the Swift default,
   correct the reference), F1 (scheduler-aware or explicit recovery with the
   three tests named above). Rerun both suites.
2. Maintainer diff read of the F1 change and landing of the tip.
3. Deploy order per F5; leave `STEERLAB_SERVICE_ROLE` unset on the cluster
   until the Sites decision in F4 is recorded.
4. Continue WP-2 (three bridges), WP-3 (Python design parity, model
   preparation on the engine), WP-6 cleanup plan/apply, WP-7 journeys.

---

# Re-review — follow-up commit 95d14aa (2026-09-06, later the same day)

One commit on top of c490531: `fix: close researcher workflow audit
blockers` (27 files, +1,703 / −972, most of it the ledger moving to
`RESEARCHER-WORKFLOW-VALIDATION-HISTORY.md`). Main is still an ancestor;
the branch fast-forwards. Worktree clean at 95d14aa.

## Verdict

**Landable.** Both suites are green on this tip (§R4). All four blockers
are closed in source with regressions, and none of the follow-ups (F4
through F8) has regressed. Two small notes remain (§R3); neither blocks.

## R1. Blocker by blocker

| Finding | What changed | Verified how |
|---|---|---|
| F1 controller recovery | `job_ownership.py` records host, pid, a per-record instance id and, when the process runs inside a Slurm allocation, that allocation's `(cluster, jobID, DBIndex, Submit, Start)` captured once via `sacct --duplicates` while RUNNING. `owner_state` proves exit across nodes only when exactly one accounting row matches every captured field and is in a terminal state; requeue, ID reuse, empty or failed queries stay `unknown`. Scheduler I/O runs before the `BEGIN IMMEDIATE` claim, and the claim rechecks a sha256 of the full job row plus owner. Startup prints `jobRecoveryRequired: N` for unknown owners and leaves them. New engine verbs `jobs recovery <id>` (read-only report with a review token) and `jobs recover <id> --review-token --confirm-owner-exited --reason` (audited `operator-attested-exit`, refused for a known-live owner or a changed snapshot). `docs/CONTROLLER-RECOVERY.md` written. | Read the module and the `jobs.py`/`cli.py` diffs in full. 13 tests in `test_job_recovery.py` cover exact allocation, reused/requeued ids, accounting absence, lock ordering, changed job, live owner, legacy migration and reporting, accounting failures, configured wrapper, CLI read-only vs explicit, controller-vs-compute allocation, and one query per allocation per pass. This closes the deployment case: legacy ownerless rows become an explicit operator step, and post-deploy jobs recover across nodes when accounting is unambiguous. |
| F2 implicit submission | `remote submit-bundle` refuses a missing, empty or blank `--verb` with exit 64 **before** site/credential resolution (`ExperimentCLIRunner.swift`, two guards), `requiredFlags: ["--verb"]` in the parser table, help text changed, usage string changed. Reference §1.3 traps rewritten; generated usage shows `--verb <verb>` as required; §remote table says "`--verb` is required". The composite client `steerlab run` keeps `default run`, which is the documented composite and the right place for a default. | Diff read; reference grepped. The Swift verb-census tests were updated for the new required flag. |
| F3 wording | `StudyAuthoring.swift:94-96` now: "Matched-norm random controls help test whether effects exceed comparable random perturbations; they do not establish construct specificity." | Diff read; a new test asserts the sentence and the absence of "prove an effect". |
| F9 changelog | `[Unreleased]` gains Added / Changed / Fixed / Deployment sections covering the 17 Swift verbs, the two engine verbs, the 428/412 manifest precondition, the `save_raw` semantics, service roles, transfer policy, explicit verbs, the recovery contract, and the app-before-engine rollout order. | Read in full. Accurate against the source I reviewed. |

## R2. Follow-ups from the first pass

- F4: `STEERLAB_SERVICE_ROLE` now has a row in the reference environment
  table. The cluster controller's role remains a separate Sites decision;
  the default stays `workbench`. Closed as documentation.
- F5: rollout order is in the changelog. Still applies at deploy time.
- F6: `Document.copy()` and `copy.deepcopy()` preserve the read digest
  (deepcopy works through the dict subclass's reduce protocol; a
  parametrized test now proves both); `dict()` and JSON round trips do
  not, and the docstrings say so. Closed.
- F7: profile widening is in the changelog. Closed.
- F8: bridges unchanged since c490531 (one of four retired). Expected.
- F10: the import-receipt drift is correctly left to its own task.
- F11: the ledger is now a 113-line status page plus an 892-line history
  appendix. Good enough to commit as documentation.
- The SIGINT test now asserts restoration of the handler it found, so it
  passes under a backgrounded shell as well as a terminal. Correct
  contract, and it removes the harness artifact I hit.

## R3. Notes, not blockers

1. `capture_allocation` runs `sacct` once per controller process at the
   first job insert. On the cluster that is a compute node; the poller
   already calls `sacct` from there, so this should work, but it is
   unexercised live. If accounting is slow at that moment the capture is
   `None` and every job of that controller becomes an operator-review
   case rather than an automatic recovery. Worth one line in the
   controller log at capture time saying which it was. Not required for
   landing.
2. The `jobs recovery` CLI prints the report document and returns an
   envelope; the `jobs list` path has the same shape and the test parses
   a single JSON document in `--json` mode, so the envelope layer must
   be suppressing the raw print there. Fine today; if that layer changes,
   both verbs break the one-envelope rule together.

## R4. Verification on this tip

| Check | Result |
|---|---|
| Vocabulary in the diff and commit message | none |
| `git diff --check` c490531...95d14aa | clean |
| Full Python suite on the worktree (main venv, cwd = worktree/Server, run from a backgrounded shell) | 5,928 passed, 9 skipped, 8 warnings, matching the branch's claim; the SIGINT test now passes with an inherited ignore handler |
| Full Xcode beta suite on the worktree (serial, external DerivedData, app not running) | `TEST SUCCEEDED`: 277 SteeringKit + 4,529 ExperimentKit, matching the branch's claim |
| SwiftSyntax and route-census audits | unchanged since c490531 (no Swift bridge or census files touched); the earlier reproductions stand |

## R5. Landing instructions

1. Both suites are green on 95d14aa (§R4).
2. Fast-forward main to 95d14aa. Commit this review and the branch's
   documents in the same landing or immediately after.
3. Rebuild and install the app before any engine deploy (changelog
   "Deployment" section). The cluster engine is still cd299e2d and must
   not be deployed until the ladder run 47923657 finishes.
4. Until the new engine is deployed, keep observing jobs through
   `steerlab-cli remote jobs --site …`; the old engine's `jobs list` still
   sweeps. After the first controller start on the new engine, read its log
   for a `jobRecoveryRequired` count and review each listed job with
   `steerlab-server jobs recovery <id> --json` before any `jobs recover`.
