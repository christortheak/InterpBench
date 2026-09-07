# Review: `codex/remote-custody-cleanup` @ `4f407c0` (two slices on main `6a94a8f`)

Reviewer: the maintainer's integration agent, 2026-09-07. Landed the same day with F1 and F2 applied by the reviewer in the landing commit (renamed field, regression test, changelog). Read against the
two handoffs (`docs/REMOTE-CUSTODY-CLEANUP-HANDOFF.md` for `e74a34a`,
`docs/MANAGED-METHOD-AUTHORING-HANDOFF.md` for `4f407c0`). Nothing on the
branch was edited. Main has not moved since the branch was based
(`origin/main` = `6a94a8f`), so a fast-forward is available.

## 1. Verdict

**Landable by fast-forward after one small correction (F1).** The custody,
transport, cleanup, managed-execution and campaign code does what the
handoffs say, the scientific owners are provably untouched, every gate
passes on the tip, and both suites are green on the exact tree (one
load-only Python flake in a path this branch does not touch, see N5).
F1 is a one-line data defect in the shared interview source that makes
one of the thirteen managed operations unauthorable through the form; it is
not a safety or custody problem. F2 is the same omission as last slice: no
CHANGELOG entry for the second commit.

## 2. Verified independently

| Check | Result |
|---|---|
| Ancestry | `merge-base main codex/remote-custody-cleanup` = `6a94a8f` = `origin/main`; two commits, 84 files, +6,769/−184 |
| `scripts/ci/audit-managed-scientific-owners.py` | passes: eleven owner ASTs identical to `6a94a8f`; the three J-lens consumers differ only by the declared resolver substitution and import |
| `scripts/ci/audit-stability-preflight.py` | still passes (battery/stability owners unchanged since the previous slice) |
| `check-science-resources.py`, `check-client-assembly-reference.py`, task-prompt parser, lazy imports, study interviews, bridge gate normal and `--release`, `public_scan.py` | all pass on the tip |
| Seed copies | `WorkspaceSeed/…/workflows.json` and `catalog.json` byte-identical to the packaged Python copies; the gate also checks the compiled Swift copy |
| Route roles and auth | six new routes censused (`workspace/{action}` workbench-only, the rest `both`); all are `POST` under `/api/`, so token-gated by the mutating-by-default rule; none added to the open-mutating allowlist |
| Vocabulary in the diff and both commit messages; `git diff --check` | clean |
| Single JSON envelope on stdout | `steerlab science custody --root <empty> --json` emits exactly one document on stdout (the pretty print goes to stderr) |
| Interview → owner agreement (my probe, not in the branch's tests) | drafts for `optvec-train`, `optvec-eval`, `optvec-interpret`, `optvec-gradient-mint` are accepted by their owners' `from_dict`; `jspace` and `sae-family-report` key sets are accepted; **`optvec-gradient` is refused** (F1); geometry/fracture/family refused only for my one-artifact dummy, as they should |
| Full Python suite on the worktree (main venv, cwd = worktree/Server) | 6,121 passed, 1 failed, 9 skipped, 8 warnings while the Xcode suite ran concurrently; the one failure re-run 3/3 green in isolation (N5); the branch's own count is 6,122 |
| Full Xcode beta suite from the venv-less worktree (`TEST_RUNNER_STEERLAB_TEST_PYTHON` in the shell environment before `xcodebuild`) | `TEST SUCCEEDED`: 277 SteeringKit + 4,594 ExperimentKit, matching the branch's claim |

## 3. What the code does

**Archives and custody (`experiment/diagnostic_archives.py`).** One
portable owner with no GPU imports. Paths are relative ordinary components
(no `\`, NUL, `.`, `..`); every ancestor is checked for symlinks; trees are
walked without following links and any non-ordinary entry refuses.
Archives are gzip tar with a `steerlab-bundle.json` member declaring kind
(`diagnosticInput` or `diagnosticEvidence`), context and an exact
path/sha256/bytes inventory; `_inspect` verifies the whole-file digest
first, then bounds (20,000 members, 16 GiB, 8 MiB metadata), duplicate
names, ordinary-file-only members, closure equality between members and
the inventory, and streams each member through SHA-256 while optionally
extracting with `open(…, 'xb')`. Publication is create-only everywhere:
files by `os.link`, directories by `renamex_np(RENAME_EXCL)` on macOS and
`renameat2(RENAME_NOREPLACE)` on Linux, refusing on any other platform.
`import_evidence` locks `.steerlab/diagnostic-import`, copies the archive,
expands it into a temp dir, requires `runs/<x>` or `diagnostics/<x>` as the
single declared prefix with every member under it, reuses an identical
existing output and refuses a differing one, retains the archive under
`.steerlab/diagnostic-archives/<sha>.tar.gz`, publishes the directory, then
writes a content-addressed receipt whose file hash is its own name.
`verify` re-reads receipt bytes, the retained archive and the full expanded
tree (extra files refuse).

**Inputs (`diagnostic_inputs.py`, `managed_inputs.py`).** Battery closure
uses the real agent resolver (vector json+safetensors, variant artifact,
adapter directories and sidecars, neutral basis); stability closure uses
the real preflight. Managed closure walks the config by key role
(artifacts add both files; `path`+`sha256` dicts are hash-checked;
`gradients` pulls its sidecars; lens IDs pull the lens directory and the
converted tensor, which must lie inside the workspace and match the
imported hash). Style rescoring adds the manifest, taxonomy and prompt
tree; the family report adds qualification pointers and, by default,
`runs/model-variants`.

**Runner side (`api/diagnostic_transport.py`, `scientific_execution.py`).**
`stage` resolves the uploaded path strictly under the run root, inspects
the archive, expands it into `runs/diagnostic-input-<sha>/workspace`,
re-derives the closure inside the copy and requires it to equal the
inventory, runs the ordinary `input_plan` (for managed operations that
spawns the root-scoped validation subprocess), and publishes the capsule
create-only under a lock. `plan` accepts `{inputBundleSHA256}` and resolves
the isolated root; campaigns require a staged bundle; CPU operations pick
the local executor even on a Slurm controller and skip `registry.unload_all`;
GPU work on a controller still requires Slurm. The queued child re-verifies
the staged archive and closure before model loading, writes
`partial: true` plus `outputRoot` into the durable record before running,
and restores cwd/env afterwards. `output`/`export` accept only succeeded
`science:*` jobs whose output is a single `runs/<x>` or `diagnostics/<x>`,
hash the tree, and package it under `diagnostic-exports/<context-digest>`.

**Cleanup (`api/diagnostic_cleanup.py`).** Policy is an ordinary JSON file
named by `STEERLAB_DIAGNOSTIC_CLEANUP_POLICY` with exactly four keys and an
explicit `true`; absent or malformed refuses. `facts` accepts only battery
and stability jobs, re-derives the export, requires the client's custody
attestation to match context, archive hash and entries, refuses when the
execution root is the serving root, blocks on any non-terminal or parked
job, on any job result or workspace declaration that mentions the output
(basename match included, so it over-blocks rather than under-blocks), on
unreadable declarations, and on unexpired retention. `plan` and `apply` run
under the exclusive workspace lock plus the new `exclusive_snapshot`
(`BEGIN IMMEDIATE` on the durable store). `apply` re-derives facts, requires
the unchanged plan digest and eligibility, writes an `intended` audit
record, renames the target create-only into a quarantine sibling, re-hashes
it against the custody files, restores it on mismatch, removes it, and
records `removed` or `attention` with surviving paths.

**Managed methods (`experiment/managed_methods.py`, `api/managed_validation.py`).**
A fixed registry of nine owner functions plus four special cases; no module
or function name comes from a request. GPU operations require a 40-hex
revision. Validation runs `python -m …managed_validation` with the root
pinned in env and cwd, 120 s timeout, engine imports isolated from the
server and the portable client. Execution dispatches to the owner with
`root`/`log` only when its signature takes them.

**Authoring (`experiment/method_authoring.py`, `sae_authoring.py`).**
`interview` reads `workflows.json`; `draft` requires the six-key answer
document with non-empty decisions, converts text answers by kind
(`fileRef` and `documentFile` read the bytes and record their SHA-256),
refuses advanced JSON that would set or shadow a form key, builds the
request through the registry, runs the input closure and hashes the whole
review into `planSHA256`. `publish` re-drafts, requires the same digest,
requires `requests/<new…>`, and publishes `request.json` + `review.json`
create-only. SAE inspection is offline; the pin holds transactions on the
manifest and roster, re-plans, and only then calls the existing
`pin_sae_candidates`.

**Campaigns (`api/managed_campaign.py`, `managed_campaign_engine.py`).**
Materialization calls the unchanged owner, writes a packet binding the
source closure and every static file, and replaces each cell's
`submit.sbatch` with a bundle whose command is the root-scoped `cell`
child; that child re-verifies packet, static files and inputs, refuses a
cell with existing evidence, pins `STEERLAB_RUN_ROOT` to `cells/<id>/outputs`,
trains through `optvec_train.train`, writes `managed-result.json` with
`xb` and then the marker. `operate` runs under a per-campaign lock, binds
current state into `planSHA256`, and for `submit` calls the owner's one-cycle
top-up; `cancel` resolves unrecorded IDs by exact scheduler name before any
`scancel` and reports per-job outcomes. Export requires every cell complete
and the scheduler terminal for each.

**Clients and app.** Python: `science interview/draft/publish/sae-*/input-plan/package/import/custody/verify-custody`
(all local), `runner science-call/science-stage/science-export/science-fetch/cleanup-plan/cleanup-apply`.
Mac: the same verbs under `science` and `remote`, all delegated to the
checkout's venv Python through `DiagnosticWorkspace.perform` (stdin JSON,
no shell). The Mac web server serves `POST /api/science/workspace/{action}`
with a serving-root check. `ScientificActionRequest` routes catalogued HTTP
actions to their original owners with the body passed as raw bytes so
UInt64 seeds survive. `MethodAuthoringSheet` keeps every numeric answer as
text and splices the advanced JSON text verbatim into the answer document;
`DiagnosticLifecycleSheet` and `SAERosterSheet` capture root and client at
open.

## 4. Findings

**F1 — the `optvec-gradient` interview cannot produce a valid request.**
`workflows.json` declares the required field `targetTrain` (kind
`fileRef`), so `draft` writes `config.targetTrain`; the owner's
`OptVecGradientConfig.from_dict` accepts only `datasets.targetTrain` and
refuses with `unknown OptVec gradient config key(s): targetTrain`. Because
the field is required, neither the form nor an agent can author a gradient
survey through this path (advanced settings cannot supply `datasets` while
the form key is blank). Fix: rename the field id to `datasets.targetTrain`
in `WorkspaceSeed/prompts/method-guides/workflows.json`, regenerate the
packaged and compiled copies with `scripts/ci/check-science-resources.py --write`,
and add a test that drafts every managed operation with minimal valid
inputs and runs the result through `managed_methods.validate`; today only
`optvec-campaign`, `rescore-style` and `sae-qualification-record` reach an
owner from a drafted config, and the Swift authoring test stops at
`publish`.

**F2 — no CHANGELOG entry for the managed-method slice.** `e74a34a` added
three `[Unreleased]` bullets (transport, cleanup, `science-call`);
`4f407c0` added none for interviews, draft/publish, SAE pinning, managed
execution, the `jspace` operation or campaign coordination. I will write
the entry at landing unless the agents do it first.

**N1 — cleanup holds every lock while hashing.** `plan` and `apply` hold
the exclusive workspace lock, the store's RLock and the `BEGIN IMMEDIATE`
transaction for the duration of `facts`, which re-derives the export
(hashing the output tree and the archive). Same-controller job bookkeeping
blocks for that time and a foreign controller's writes fail after the 30 s
SQLite timeout. Battery and stability outputs are small, so this is a
documentation point, not a defect; keep it in mind before extending
cleanup to larger artifact classes.

**N2 — `science import` creates a missing root.** `import_evidence` calls
`root.mkdir(parents=True, exist_ok=True)`, so a mistyped `--root` silently
mints a new tree with `.steerlab/` and `runs/` instead of refusing. The HTTP
path is protected by the serving-root check. Suggest refusing when the
root does not already exist.

**N3 — interview defaults apply only when a key is absent.** An agent that
passes `""` for an optional field with an interview default (for example
`alphaNormFactor` `0.1`) gets the owner's default instead, which the review
displays but the interview text does not. The app pre-fills defaults so the
form path is unaffected. Also, "exact integer spelling" is Python's
`int()`: `1_000` and non-ASCII digits are accepted.

**N4 — Mac diagnostic verbs run the checkout's venv Python.** Same
dependency the stimulus screen already has; an app installed without a
checkout gets the setup repair. After landing, the checkout's `Server/`
must be at the landed commit for the app's `science`/`remote` diagnostic
verbs to behave as reviewed.

**N5 — one load-only Python flake, not this branch's.**
`test_local_job_lifecycle.py::test_cancelling_a_local_job_stops_it_and_keeps_its_evidence[bundle]`
failed once while the Xcode suite ran concurrently (durable result lacked
`runDirectory` after cancel); 3/3 green in isolation. The branch's only
change to `api/jobs.py` is the added `exclusive_snapshot` method, and the
test does not touch scientific execution. Pre-existing timing race in the
cancel-adoption path; worth a chip, not a landing condition.

**N6 — audit negative control is weak for the lens consumers.** The second
assertion only proves an appended statement changes the AST. The primary
assertion still fails if the substitution text is absent, so the gate is
sound; the control just does not prove much.

## 5. Deploy consequences (unchanged list plus two notes)

App rebuild and engine deploy are already owed from earlier landings; the
engine deploy waits for ladder job 47923657. New from this branch: cleanup
is opt-in and does nothing unless the controller's environment names a
policy file through `STEERLAB_DIAGNOSTIC_CLEANUP_POLICY`, and campaign
coordination needs the controller to be Slurm-backed. Nothing in the site
environment needs to change for this slice.
