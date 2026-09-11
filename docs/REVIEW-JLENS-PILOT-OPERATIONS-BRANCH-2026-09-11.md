# Review: `codex/jlens-pilot-operations` @ `b9a688c` (on main `f0449d3`)

Reviewer: the maintainer's integration agent, 2026-09-11. Read against
`docs/JLENS-PILOT-OPERATIONS-HANDOFF.md` and §8 of
`docs/JLENS-FITTING-PILOT-AND-SCALING-HANDOFF-2026-09-11.md`. Four commits,
33 files, +795/−37. Main has not moved, so a fast-forward is available.

## 1. Verdict

**Landable by fast-forward, with one landing fix (F1) and one wording fix
(F2).** The slice does the first step of the agreed order and does it
carefully: the fitting run now records what a pilot needs (per-row timing,
device and host memory, finiteness by layer, compile status, kernel
dispatch as an observation), the reference call is wrapped rather than
touched, stage and export get an hour of patience in both clients, the
staged-request document is written for the researcher instead of by hand,
the abandoned-claim case has a message and tests, and the running
controller's build is reported beside the deployed one. What it missed is
that the same 60-second budget that broke export also breaks `science-plan`
and `science-submit` on a staged bundle that carries a multi-gigabyte
checkpoint, which is precisely the continuation case: the live plan for the
pilot's continuation timed out from the Mac today and had to be run from
the cluster's login node. That is a two-call fix and lands with this review.

## 2. Verified independently

| Check | Result |
|---|---|
| Ancestry | `main` (`f0449d3`) is an ancestor of `b9a688c` |
| Unified gates and audits (`check-generated.py --audits`) | every generator matches and every audit passes; the fitting-loop audit stays pinned to `878bca2` |
| Public scan, whitespace, vocabulary in the diff and all four commit messages | clean |
| Telemetry boundary | `Measurements.wrap` closes over the reference call and the loop is byte-identical; `KernelObservation` registers read-only pre-hooks on executed attention modules and removes them in `finally` |
| Driver provenance | `driverSHA256` now covers five files; the compatibility contract excludes it, so the pilot's row-4 checkpoint (four-file driver) stays continuable, which the live continuation job confirmed today |
| Live continuation (pilot checkpoint → rows 5–8, one A100) | preflight and execution both re-verified the 6.6 GB checkpoint, the model loaded, the first backward started, no failure record: the `jlens-fit-v1` contract accepted the checkpoint across the driver change |
| Full Python suite on the tip (main venv, cwd = worktree/Server) | see §5 |
| Full serial Xcode beta suite on the tip (`TEST_RUNNER_STEERLAB_TEST_PYTHON` set, external derived data, coverage mapping off) | see §5 |

## 3. What the branch does

**Telemetry** (`jlens_fit_telemetry.py`, new). `Measurements` wraps the
pinned per-prompt call: per row it records dimension batch, seconds,
tokens, valid positions, CUDA free/capacity/allocated/reserved before and
after with peaks reset per row, and the process's peak host RSS, appended
to `row-resources.jsonl`; a failing row is recorded as failed and the
original exception propagates. `KernelObservation` hooks executed
attention modules and reports whether the module's fast path is explicitly
disabled (fallback) or unverified, plus the bound implementation names.
`fit-report.json` gains `telemetry`, `execution` (compile off, attention
implementation, optional kernel package versions, dispatch observations)
and `finiteByLayer`; `fit-failure.json` gains the batch, telemetry and a
repair for out-of-memory at load or backward.

**Transfers.** Stage, export and download get an hour-long read budget in
both clients; ordinary calls keep theirs; an explicit Python `--timeout`
still wins. An export timeout tells the researcher to ask again and never
to resubmit the fit. Stage refuses an archive outside the run root with a
repair naming that root, and writes the staged request document to
`.steerlab/diagnostic-requests/<digest>.json` (create-only, identical
bytes repeatable), returning `localRequestPath`; the app and both CLIs do
this. A plan on an unstaged client request fails with a repair that names
the staged form, scoped to path failures only after the full suite caught
the first, broader version.

**Publication claims.** An empty directory at a publication target is
reported as a possibly active or abandoned claim with a hand-recovery
message; nothing is deleted by inference. Tests interleave a reader
between claim and rename and simulate an interruption there.

**Controller identity.** Capabilities carry `runningEngineVersion`
(captured at controller start) and `deployedBuildCommit` (read from disk
per call); cluster status and the app's health card show both with an
advisory on disagreement; push says it does not restart a controller.

## 4. Findings

**F1 — plan and submit kept the short budget.** `scientificPlan` and
`scientificSubmit` (Swift) and `scientific_plan` / `scientific_submit`
(Python) still post with the default budget. Planning a staged bundle
re-verifies every staged file, so for the continuation case it re-hashes
the 6.6 GB checkpoint on Lustre, which took longer than 60 seconds today
and the Mac client timed out with no hash to submit. Landed with this
review: the two calls use the same diagnostic budget as stage and export
in both clients, with a test.

**F2 — the guide names a flag that does not exist.** The new section says
to use the saved file "with `science-plan --request`"; both verbs take the
request file as their positional argument. Landed with this review: the
sentence names the positional form, and the generated resource copies are
regenerated.

**N1 — the observation hooks fire on every forward.** One dictionary write
per executed attention module per row; negligible next to 5,120 backward
passes, noted so nobody attributes a slowdown to it.

**N2 — plan re-verification cost is now visible but unchanged.** Both
plan and submit re-hash the staged inputs by design (the queued child does
so again). With the long budget the wait is honest; the cost is still
minutes per multi-gigabyte checkpoint, three times over. Worth a later
look at trusting the staged capsule's own verification receipt within a
controller session.

**N3 — the running-versus-deployed advisory needs the new engine on the
controller.** Older controllers report both fields unknown, as the branch
says; the first controller to show the advisory is the one started after
this deploy.

**N4 — a registered fit loses its reference-kernel commit.** Registering
the pilot's lens through `artifact-plan` and `artifact-import` produced a
record with `referenceCommit: null` and `referencePackage: null`, although
`fit-report.json`, the description's declared `configFile`, records the
pinned reference commit and kernel hash under `identity.runtime`. The
importer reads the report only for geometry. It should carry the reference
commit, kernel hash, and fit driver hash into the record, since a later
qualification and any merge (scaling R2) will want to compare them.

**N5 — the review path for the continuation went through the login node.**
Until this landing's F1 reaches the cluster, a Mac client cannot plan or
submit a staged bundle with a large checkpoint; the working path was the
Python client module invoked on the login node with the site's own token
file by path and a long explicit timeout. Recorded so the live walk can be
repeated from the Mac after the deploy and the difference noted.

## 5. Landing shape

Fast-forward to `b9a688c`, then one landing commit carrying F1, F2 and this
review, with both suites re-run on that commit. Shipped Python and the
compiled identity change, so the app and its payload are rebuilt together;
the cluster push waits until the running continuation job finishes, since
the controller restart it needs should not coincide with a live fit.

Suite results on `b9a688c`:

- Python: 6,387 passed, 9 skipped, 8 warnings (matching the handoff's claim).
- Swift: `TEST SUCCEEDED`, 290 SteeringKit and 4,617 ExperimentKit tests.

Suite results on the landing commit:

- Python: 6,388 passed, 9 skipped, 8 warnings.
- Swift: `TEST SUCCEEDED`, 290 SteeringKit and 4,618 ExperimentKit tests.
