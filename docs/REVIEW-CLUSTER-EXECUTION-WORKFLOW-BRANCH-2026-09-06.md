# Review — `codex/cluster-execution-workflow` at a66d55a (2026-09-06)

Reviewer: Claude (maintainer's reviewing/integration agent). Range reviewed:
`73bfbd7` (current main) to `a66d55a`, one commit, 53 files,
+1,911 / −116. Main is an ancestor; the branch fast-forwards. Worktree
`/private/tmp/interpbench-workflow-handoff` clean at the tip, no venv. No
edits were made to the branch. This branch carries its own CHANGELOG entry.

## 1. Verdict

**Landable by fast-forward.** Both suites are green on a66d55a (§2). The four places
where this slice could have gone wrong are sound: the recovery routes sit
behind the existing ownership gate with an explicit attestation and the
mutating one is token-gated; the Slurm submission records its intention
before the single scheduler call and parks rather than retries on an
uncertain reply; the stability preflight extraction and the battery
callback are proven unchanged by a new AST audit; and the site-companion
evidence lands in a private, owner-only archive inside the Sites registry,
outside every workspace and study hash. Two notes (§4), and one decision
for the maintainer about the Sites repository (N1).

## 2. Verified independently

| Check | Result |
|---|---|
| `scripts/ci/audit-stability-preflight.py` | passes: the admission block extracted into `preflight()` and the numerical tail left in `run()` match `73bfbd7` node for node; every other function in the module unchanged; two negative controls rejected. Battery `execute` is unchanged except the `on_run_created` parameter and its one guarded call after the existing directory mint. |
| Scientific owners otherwise untouched | no other extraction, battery, judgment, style or intake module in the diff |
| Route census | `/api/science/plan|submit`, `/api/jobs/{id}/recovery|recover`, plus the pre-existing `reconcile` and `resubmit`, added to the runner-reachability list; all four new routes censused `both` with rationale |
| Auth gating | `/api/science/plan` and `/submit` added to the privileged prefixes (token required); `POST …/recover` is mutating and not in the open-mutating list, so token-gated by default; `GET …/recovery` is an ungated runner read like the existing job reads, exposing host, pid and metadata root, the same class of fact the job record already exposes |
| All prior gates (science resources with the new engine-verb census, client reference, task-prompt parser, lazy imports, interviews, bridge normal and `--release`) | pass on the tip |
| Vocabulary in the diff (53 files) and commit message; `git diff --check` | clean |
| Full Python suite on the worktree (main venv, cwd = worktree/Server) | 6,076 passed, 9 skipped, 8 warnings, matching the branch's claim |
| Full Xcode beta suite from the venv-less worktree, `TEST_RUNNER_STEERLAB_TEST_PYTHON` in the shell environment before `xcodebuild` | `TEST SUCCEEDED`: 277 SteeringKit + 4,589 ExperimentKit, matching the branch's claim |

## 3. What the code does

**Diagnostic plan and submit (`api/scientific_execution.py`).** A request
is exactly `{operation, parameters}` with a closed parameter set per
operation; strings must be non-empty; stability names must be single
workspace components; the seed is canonicalized to decimal text and bounded
to UInt64 so a client whose JSON numbers are doubles cannot round it;
`fraction` is in (0, 1]. `input_plan` resolves every input under the runner
root with path escapes refused and, for batteries, also pins the vector
sidecars so a valid-but-different norm table changes the plan; every model
revision must be a 40-hex commit. `plan` refuses on a GPU-session worker
and on a controller not backed by Slurm, binds the controller host, pid and
metadata root, declares memory fit unchecked, renders the sbatch script
without writing so header and GRES gates apply before any directory exists,
and hashes the whole document into `planSHA256`. `submit` re-plans and
requires the digest to match, mints a unique submission directory, writes
the packet, and then either runs the child locally under the existing
`LocalExecutor` with cancellation, or for Slurm records the job as
`submitting` with a unique `schedulerSubmissionName` before the one
`sbatch`, flips to `submitted` on success and to `parked` with the error on
any exception, never retrying. The child (`execute_packet`) re-verifies the
packet's digest, re-plans the inputs and refuses if they moved, then runs
the existing battery or stability owner; a battery reports its output
directory the moment it is minted, so an interrupted child leaves a
findable location; every outcome is written to a child record for
reconciliation. Auto-resubmit is off: these have no checkpoint protocol.

**Recovery and reconciliation over HTTP.** `GET /api/jobs/{id}/recovery`
returns the existing ownership report and review token; `POST …/recover`
requires exactly `reviewToken`, `reason`, and `confirmOwnerExited: true`,
then calls the existing `recover_orphan`, which rejects a live owner or a
changed snapshot and audits the attestation. Both Python `runner` and Mac
`remote` verbs are thin over these; the Mac verbs take only `--site` or
`--url`, resolve the token from the Keychain, and refuse to borrow the
active target so a reviewed plan or job is always addressed to its own
endpoint. Python `runner resubmit`, `reconcile`, `recovery`, `recover`,
`science-plan`, `science-submit` are declared once and dispatched by name.

**Site companion acceptance (`ClusterProfileAcceptance.swift`).** Requires
the exact reviewed bytes by digest, re-runs the existing review and refuses
unresolved questions, takes a lock on the registry directory, refuses any
existing site with the same canonical identity (never replaces), writes the
companion bytes to `<Sites>/cluster-sites/.authoring/<sha256>.json` with
0700/0600 permissions by staging-and-move before the profile is published,
refuses if an archived copy differs, then imports through the existing
repository path with its login-safety finding. The CLI verb, the HTTP
operation and the wizard's "From documentation" sheet all call this one
owner. Acceptance connects to nothing.

**Catalog follow-up.** The two engine-only operations now name their
public paths; the science resource and client reference gates compare every
non-null engine command in the catalog against the engine's declared verb
tables and dispatch, with negative controls, closing the note from the
previous review.

## 4. Notes, not blockers

### N1 — The Sites registry is a git repository with no ignore rules (maintainer decision)

`~/SteerLab/Sites` is a git checkout and carries no `.gitignore`. After the
first acceptance it will show `cluster-sites/.authoring/<sha>.json` (the
companion with its citations) and `.steerlab/manifest-locks/` (which
ignores its own contents) as untracked. Committing the evidence into the
private Sites repository is a reasonable choice, since that is what the
repository is for, and the files are 0600; if you would rather the evidence
stay local to the machine, add `cluster-sites/.authoring/` to a Sites
`.gitignore`. Either way it never reaches this public repository or a
workspace freeze commit.

### N2 — The plan digest binds the controller process

`planSHA256` includes the controller's host and pid, so a plan taken from
one controller session cannot be submitted to its successor and must be
re-planned. That is the stated intent and it is cheap, but it interacts
with the on-demand controller: plan and submit in the same controller
session.

## 5. Landing instructions

1. Both suites are green on a66d55a (§2).
2. Fast-forward main to the tip.
3. Deploy consequence: none new beyond the standing order. The app rebuild
   already owed carries the Server Jobs diagnostic form, the recovery
   sheet and the wizard's acceptance; the engine deploy already owed
   carries the plan/submit and recovery routes.
