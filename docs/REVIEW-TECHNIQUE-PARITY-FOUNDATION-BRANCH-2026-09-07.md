# Review: `codex/technique-parity-foundation` @ `b1d86ac` (on main `293888e`)

Reviewer: the maintainer's integration agent, 2026-09-07. Read against
`docs/TECHNIQUE-PARITY-FOUNDATION-HANDOFF.md` (foundation, `c9efe13` and
`d18df26`) and `docs/TECHNIQUE-PARITY-STEP3-HANDOFF.md` (Step 3, `9139e82`
through `b1d86ac`). Six commits, 48 files, +3,000/−91. Main has not moved,
so a fast-forward is available.

## 1. Verdict

**Landable by fast-forward with one small fix at landing (F1).** The
foundation delivers exactly the two documents I asked for in the technique
brief, plus a catalog-checked substrate inventory and an executable worked
example. Step 3 lands three of the four Swift twins the parity brief listed
as owed, and the one that matters most numerically, extraction stability,
is proven against the real Python engine by a subprocess fixture rather
than by hand-typed expected values. No Python scientific owner changes; the
AST audits and the compiled Python identity are untouched.

F1 is that the worked-example qualification script fails on the branch
tip on my machine, because Step 3 added test files as evidence links to
the inventory and the script's disposable source copy does not include
`Tests/`. The handoff's claim that the example was rerun holds for the
foundation commit, not the tip. One added path prefix fixes it; with that
change the example passes here (25 tests).

## 2. Verified independently

| Check | Result |
|---|---|
| Ancestry | `main` (`293888e`) is an ancestor of `b1d86ac` |
| New gates | `check-substrates.py` (24 operations, every profile cited, table regenerates identically) and `check-intervention-scopes.py` (Swift vocabulary equals the Python constants) pass; `check-science-resources.py` now runs both |
| All prior gates | science resources, workspace bootstrap, Python client identity (unchanged constant), client reference, study interviews, managed-owner audit with mutation controls, stability preflight, task-prompt parser, lazy imports, bridge normal and `--release`, `public_scan.py`: pass |
| Worked example (`scripts/ci/qualify-technique-example.py`) | **fails on the tip**: `Missing evidence/source: Tests/SteeringKitTests/ReaderEvidenceRoleTests.swift` inside the disposable copy; passes (25 tests) once `Tests/` is added to the copied prefixes (F1) |
| Stability port | read `DirectionStability.swift` beside `vector_math.direction_stability`: one SplitMix64 stream for draw then shuffle seeds, one-step derived seed for the unpaired class, identical partial Fisher–Yates, half-up subsample size, linear-interpolated percentiles on the sorted float64 cosines, same degenerate-draw accounting and the same two-draw floor |
| Cross-engine fixtures | `ScientificScopeStabilityTests` shells out to the Python engine and compares seeds exactly and cosines and statistics at 2e-5 for all three recipes, degenerate draws by seed and kind, and the scope inventory byte-for-byte as JSON |
| Vocabulary in the diff and six commit messages; `git diff --check` | clean |
| Full Python suite on the worktree (main venv, cwd = worktree/Server) | 6,175 passed, 9 skipped, 8 warnings, matching the branch's claim (run concurrently with the Xcode suite, no flake) |
| Full Xcode beta suite from the venv-less worktree (`TEST_RUNNER_STEERLAB_TEST_PYTHON` in the shell environment before `xcodebuild`) | `TEST SUCCEEDED`: 286 SteeringKit + 4,609 ExperimentKit, matching the Step 3 claim; the cross-engine stability and scope fixtures ran against the real Python engine |

## 3. What the branch does

**Foundation.** `docs/ADDING-A-TECHNIQUE.md` opens with your guidance-first
principle, classifies additions by scientific effect and cost, and walks
the managed-operation, training, intervention and recipe paths with the
real owners, registries, generators and audits named.
`docs/ADDING-A-TECHNIQUE-EXAMPLE.md` is a complete fictional CPU operation
whose code blocks the qualification script extracts and applies to a
disposable copy of the tree, regenerates resources there, and runs through
the real owners; the production tree never sees it. `docs/SUBSTRATES.md`
joins every catalog operation to one of ten execution profiles with CUDA,
MPS and MLX states from a closed vocabulary, cited sources and, where a
profile claims qualification, a required evidence link; the checker
regenerates the table and refuses unknown states, uncited profiles, missing
files and a `qualified` state with no evidence. It is documentation
tooling and says so: nothing reads it at runtime. `docs/TECHNIQUE-PARITY-IMPLEMENTATION.md`
refines my two briefs into six ordered steps with acceptance criteria and,
importantly, corrects the seed policy: the Python engine's ordinary and
multi-agent runs derive seeds differently, and the MLX work must preserve
each rather than invent one rule. Root `AGENTS.md` gains the principle
and the pointers.

**Reader final-test rows.** The Swift reader now recognizes three roles,
case-insensitively: train, held out (any other spelling except final test)
and final test. Capture order is train, held out, final test; only the
first two reach fitting, sign selection and layer recommendation; final
rows only score the fitted probe. Six optional schema-2 fields carry the
result and the roles; legacy artifacts decode with them absent and encode
them absent, and `resolvedEvidenceRoles` reads legacy stamps without
writing a measurement. Exact train-overlap refusal mirrors the Python
contract (whitespace and case folding, no Unicode canonical equivalence,
and deliberately not a near-duplicate audit). `readerSplitPreview` is the
single source for the app preview and both row encoders; final rows are
reserved first, held-out from the remaining tail, at least two rows stay
train, and the default final count is zero so existing request bytes are
unchanged. The Concept Lab gains the stepper, the preview line and the
per-layer final-test readout with a note that held-out scores are
selection statistics.

**Intervention scope.** `InterventionScopeVocabulary.swift` is generated
from the module-level constants in `steering/intervention.py`, so the
prose can never drift. `VectorInjector` and `SubspaceAblator` describe
their own configuration; `InterventionPlan.scopeInventory` describes the
armed chain in execution order, and the fixture proves the description
matches Python and matches what the chain actually does to a tensor at
mid-prefill, prompt-end and decode positions. Ordinary and saved-agent
measured runs write `intervention-scope.json` once per run directory,
never replacing an existing file; a baseline is an explicit empty chain,
centering travels as metadata from each resolved injection, and a resolver
failure becomes a named `unresolved` row while the run loop keeps
ownership of the execution error. Playground and native multi-agent runs
are explicitly out of scope, as are trainable and SAE-latent paths.

**Extraction stability.** `experiment extract-stability <study> <concept>`
with `--resamples`, `--fraction`, `--seed` and `--order-shuffles`.
Preflight refuses unsupported recipes, an unpinned designated reference
and bad draw settings before any weights load; the run captures each class
once and resamples in memory; the report goes under `diagnostics/` in a
UUID-named directory with the shared and per-layer key partition, exact
UInt64 seeds, live and pinned stimulus hashes, the recipe identity hash and
a flag that neutral projection was not applied. The manifest is never
written. The app's managed science sheet still runs the Python owner; this
is the native verb the parity brief asked for, not a reroute.

**Adapter provenance.** A completed MLX training records
`adapterScaleConvention: direct` with the effective and requested
multiplier taken from the training result rather than the panel's current
control. Historical artifacts decode without invented values.

## 4. Findings

**F1 — the worked-example script fails on the tip.**
`scripts/ci/qualify-technique-example.py` copies only `Server/`,
`WorkspaceSeed/`, `Sources/`, `scripts/ci/` and `docs/` into its scratch
tree, then runs `check-substrates.py --write` there, which asserts every
cited evidence file exists. Step 3 cited two files under `Tests/`. Fix:
add `Tests/` to the copied prefixes. Verified passing with that change. The
script is not in CI, which is why the branch's own validation did not see
it; adding it to the workflow would be reasonable once it runs in the CI
Python environment.

**N1 — the scope sidecar resolves vectors a second time.** Writing the
sidecar calls the same injection resolver the run loop calls, so every
condition's vectors are read twice per run. Cheap, and it guarantees the
description comes from the same resolution path, but worth knowing if a
study with many conditions ever feels slower to start.

**N2 — `splitOverlap` is a constant.** The artifact stamps
`exactDuplicatesAcrossSplits: 0`, which is honest only because the overlap
check throws on any duplicate before the fit; it is a stamp of "the check
ran", not a count. The field name will read as a measurement to a future
reader. Not wrong, worth a comment.

**N3 — the inventory is conservative by construction.** `qualified` needs
an evidence file to exist, not to be valid or to cover every operation in
a profile; the handoff says so. No profile is marked qualified today, so
the gate has not been exercised in the direction that matters. Revisit
when the first MPS or MLX measurement lands.

**N4 — three of four owed twins.** Stability, scope and adapter scale are
in; the reader final-test role is in; the evidence-role sidecar stamps
from the science follow-ups are covered by `readerEvidenceRoles` on
derived vectors. What remains owed from the parity brief's list is now
Step 4 (seeded MLX sampling) and Step 5 (MPS measurement), and the
implementation plan states both.

## 5. Landing shape

Fast-forward, then one landing commit with the F1 one-line fix and this
review. The compiled Python identity is unchanged because no shipped
Python or seed file changed; the CLI reference regenerated cleanly on the
branch (17 regions). The owed app rebuild covers this branch too.
