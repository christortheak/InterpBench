# Review: `codex/workflow-review-hardening` @ `bbe4166` (on main `7301e6c`)

Reviewer: the maintainer's integration agent, 2026-09-07. Read against
`docs/WORKFLOW-REVIEW-HARDENING-HANDOFF.md`. Nothing on the branch was
edited. Main has not moved since the branch was based, so a fast-forward is
available. One commit, 26 files, +841/−63.

## 1. Verdict

**Landable by fast-forward.** Each of the six notes from the custody review
is addressed the way the handoff says, with a test that fails on the old
behaviour where a behaviour changed. No scientific owner, job, executor or
cancellation code is touched. The one thing to know before landing anything
else: the Python suite now carries a source-identity gate (N4), so every
future commit that changes a shipped Python file or seed must regenerate
`Sources/ExperimentKit/PythonClientIdentity.swift` with
`scripts/ci/check-python-client-identity.py --write`, or the suite fails.

## 2. Verified independently

| Check | Result |
|---|---|
| Ancestry | `merge-base` = `origin/main` = `7301e6c`; one commit `bbe4166` |
| `check-python-client-identity.py` | passes on the tip; the compiled constant equals the identity I recomputed from the tree |
| `audit-managed-scientific-owners.py` (rewritten) | passes: eleven owners unchanged; the three lens consumers pass the tightened gate and all three mutation controls (migration removed, resolver root changed, consumer body altered) are refused, with an AST-unparsed positive control |
| Stability preflight, science resources, client reference, task-prompt parser, lazy imports, study interviews, bridge normal and `--release`, `public_scan.py` | all pass on the tip |
| Vocabulary in the diff and commit message; `git diff --check` | clean |
| Full Python suite on the worktree (main venv, cwd = worktree/Server) | 6,151 passed, 9 skipped, 8 warnings, matching the branch's claim (run concurrently with the Xcode suite, no flake) |
| Full Xcode beta suite from the venv-less worktree (`TEST_RUNNER_STEERLAB_TEST_PYTHON` in the shell environment before `xcodebuild`) | `TEST SUCCEEDED`: 277 SteeringKit + 4,596 ExperimentKit, matching the branch's claim |

## 3. What changed, note by note

**N1 (cleanup lock duration).** No code change, by design. The handoff and
both earlier handoffs now state the exact critical section and set a scope
gate: measure on slow storage with competing controllers before admitting
larger artifact classes, and never hash outside the lock without a
revalidation design. That is the right call; the earlier note was a
documentation point.

**N2 (import creates a missing root).** `import_evidence` now refuses when
the root is not an existing directory, before any lock, cache or parent is
created. The test proves the parent is never created and that a file at
the root path is left byte-identical. The handoff corrects my attribution:
the Python CLI already refused a missing `--root` with exit 66; the hole
was in the shared owner, reachable through the Mac adapter and HTTP. The
round-trip fixtures now create their destination workspace explicitly.

**N3 (interview defaults).** A blank or whitespace optional answer now
resolves to the interview default exactly as an omitted one does, so the
value appears in both the request and `effectiveAnswers` and the owner's
fallback can no longer differ silently. The regression uses the fracture
interview's 0.8 against the owner's 0.9. Required blanks still refuse. The
contract wording drops "exact spelling" and says integers avoid float
conversion; underscores and Unicode digits are tested as accepted.

**N3 (authoring reaches every owner).** `test_interview_validation.py`
builds a fixture-backed draft for all thirteen managed operations, asserts
the case list equals the registry, and passes every drafted config through
`managed_methods.validate` with no swallowed exception. This supersedes the
key-only test I landed with F1 and is what I asked for.

**N4 (Mac adapter needs a checkout).** `DiagnosticWorkspace.perform` now
reads Python from the app's existing `ServerPayload` resource family (a
development build resolves that to the checkout's `Server/` as before), and
picks the interpreter by precedence: `STEERLAB_CLIENT_PYTHON` if absolute
(an invalid explicit choice refuses rather than falling back), then
`~/Library/Application Support/SteerLab/client-runtime/bin/python`, then
the checkout venv. The subprocess runs with `-B -s`, `PYTHONPATH` set to
the payload alone, `PYTHONHOME` cleared and a temporary cwd, so nothing is
written into the signed bundle. Both sides carry a SHA-256 over every
shipped `.py` and seed file; the Python entry point recomputes it and
refuses before importing any owner when it differs, and the Swift side
refuses a reply that does not echo it. Tests run a copied payload with no
checkout through both the Python and Mac paths and prove source drift
refuses and no `__pycache__` appears. `docs/PYTHON-CLIENT-RUNTIME.md`
gives the client-only setup and the release procedure. Not claimed:
clean-machine install, upgrade, GUI qualification, one-click provisioning.

**N5 (cancellation flake).** Diagnosed rather than guessed: the live `Job`
turns `cancelled` before the worker's final SQLite update, and the old test
helper waited on the live object then asserted on the durable row. A
deterministic test pauses that final update and shows `wait()` returned in
the gap on both submission paths. The helper now waits for a committed
terminal row and returns that same snapshot. No production code changed,
and the handoff is careful to say this explains the observed assertion
without ruling out other live cancellation races.

**N6 (audit controls).** The lens gate now requires the import once and the
new call exactly as many times as the old call appeared in the baseline,
forbids any surviving old call, and only then normalizes and compares. The
three mutation controls go through that same gate.

## 4. Landing-process consequence

The identity gate is the operationally important addition. After landing,
the checkout's `Server/` and the built app must come from the same tree,
and any Python or seed change must be followed by the identity regeneration
before the suites are run. Nothing changes for the cluster engine or the
site environment; the owed deploy list is unchanged.
