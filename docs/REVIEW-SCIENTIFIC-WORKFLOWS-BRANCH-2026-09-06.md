# Review — `codex/scientific-workflows` at ff84138 (2026-09-06)

Reviewer: Claude (maintainer's reviewing/integration agent). Range reviewed:
`b369693` (current main) to `ff84138`, one commit, 64 files,
+2,308 / −63. Main is an ancestor; the branch fast-forwards. Worktree
`/private/tmp/interpbench-workflow-handoff` clean at the tip, no venv. No
edits were made to the branch.

## 1. Verdict

**Landable after one small fix** (F1: no CHANGELOG entry). Both suites are green on the tip; the single Python failure in the first run is a pre-existing flake (§2a) that a second full run did not reproduce. The slice is
what it says: read-only discovery of twelve method guides and twenty
operation mappings, shipped from one maintained source into both clients
and both HTTP implementations with a drift gate; three CPU CLI adapters
moved out of the engine's command file with proper envelopes and exit
codes; one new engine verb over an existing intake owner. The scientific
owners themselves are untouched, which the diff confirms. Two notes (§4).

## 2. Verified independently

| Check | Result |
|---|---|
| Scientific owners unchanged | No extraction, battery, analysis, sweep-judgment, evaluation-judgment, style or intake module appears in the diff. The 45 lines removed from `cli.py` are the two adapters that moved to `scientific_commands.py`. |
| `scripts/ci/check-science-resources.py` (WorkspaceSeed guides and catalog equal the packaged Python copies and the generated Swift text) | "match both packaged clients" |
| Interview gate, client reference check, task-prompt parser audit, lazy-import audit, bridge gates normal and `--release` | all pass on the tip |
| Route census | the three `/api/science/*` routes are censused `both`, with rationale; the runner-role note in the handoff matches |
| Vocabulary in the twelve guides, the catalog, the whole diff and the commit message | none (site names, study-case names, authors) |
| Overclaim wording in the guides | every occurrence of "prove"/"proof" is a limitation statement ("do not prove", "not proof") |
| Exit-code change callers | the only Swift consumer of judgment completion uses the HTTP routes, not the engine CLI; no in-tree caller depended on the old exit 1 |
| `git diff --check` | clean |
| Full Python suite on the worktree (main venv, cwd = worktree/Server), first run | 1 failed, 6,042 passed, 9 skipped: `tests/test_jlens_run_start.py::test_a_matching_pinned_hash_is_accepted` — the pre-existing cross-thread flake analysed in §2a, not this branch |
| Full Python suite, second run on the same tip | 6,043 passed, 9 skipped, 8 warnings |
| Full Xcode beta suite from the venv-less worktree, `TEST_RUNNER_STEERLAB_TEST_PYTHON` in the shell environment before `xcodebuild` | `TEST SUCCEEDED`: 277 SteeringKit + 4,583 ExperimentKit; the catalog parity test ran the real Python client |

## 2a. The Python failure

`tests/test_jlens_run_start.py::test_a_matching_pinned_hash_is_accepted`
failed once in the full run with: the fake final norm "matches neither
RMSNorm fold (relative error offset 0.00136, direct 1.44e+06)". The
tolerance is 0.001.

- The test passes six of six in isolation on this tip; it had passed in
  every one of the ten full runs earlier today on other tips; this branch
  touches no J-lens file.
- The arithmetic points at a bf16-rounded output: the fake norm's forward
  ends in `type_as(x)`, and a half-ulp of bf16 on values of magnitude
  1.44 is 0.0019, which divided by the fold's scale is 0.00136. So the
  seeded probe vector `x` was created as bf16.
- `norm_convention.observe` builds that probe with `torch.randn(n,
  generator=…)`, which takes the process-wide default dtype. Nothing in
  the repository sets that default, but the model loader's library does,
  temporarily, for the duration of every `from_pretrained` call
  (`transformers/modeling_utils.py`, set and restore around the load), and
  twelve test modules start worker threads that load models. When such a
  load in another thread overlaps this test's observer, the probe is
  bf16 and the fold check misses.

Classification: a pre-existing, rare, cross-thread flake in code landed on
2026-09-05 (`9553c31`), exposed by test scheduling, not by this branch.
Fix, one line, in `Server/steerlab_server/jlens/norm_convention.py`: create
the probe with `dtype=torch.float32` (and cast `x` to float32 before
`probe(x)` for belt and braces), so the observation is independent of the
process default. That belongs in a separate small commit on main, not on
this branch; it should land before or right after this branch with the
Python suite rerun. The reproducibility rerun on this tip passed in full (6,043).

## 3. What the code does

**Catalog and guides.** One `catalog.json` (12 methods, 20 operations;
each operation names its engine CLI, Mac path, HTTP references, compute
class, outputs and restriction, with explicit nulls where no interface
exists) plus twelve Markdown guides under `WorkspaceSeed/prompts/method-guides/`.
The gate script copies them into the Python package data (declared in
`pyproject.toml`) and generates `ScienceResourceText.swift`; without
`--write` it is the drift check. Both `science_catalog.py` and
`ScienceCatalog.swift` are pure readers that return the SHA-256 of the
exact shipped bytes for the catalog and each guide, and the Swift parity
test asserts byte equality of catalog, guides and HTTP replies against the
real Python client. New workspaces are seeded with the files. Nothing here
executes anything; the catalog's own scope field says so.

**Surfaces.** `science list|guide|operation` on both clients with the same
three verb specs; `GET /api/science/catalog|guide/{method}|operation/{operation}`
on the Python service (censused `both`) and the Mac workbench; a Studies
panel sheet over the same owner.

**CPU adapters (`scientific_commands.py`).** `complete_judgment` handles
both the evaluation and the new sweep completion: strict flag parsing (64),
judgment loading through the existing loader (64 on malformed, 66 on a
missing file), a refusal that a sweep file carries an instructions digest
(64), then the existing `tasks.complete_evaluate_judgment` or
`tasks.complete_sweep_judgment` with lifecycle-gate mapping to 65 and
everything else to 70. `reused` is whether a judgment run already existed;
`changed` is that or a manifest digest change. `rescore_style` wraps the
existing `tasks.rescore_style` the same way. The engine's verb table gains
`complete-sweep-judgment`; the three verbs get `VerbSpec`s so the
generated reference covers them.

**Fixture race.** The local cancellation test used to `json.dump` straight
into the record path the parent polls for; the parent could cancel on a
half-written file. It now writes to a temp file and `os.replace`s it.
Production cancellation code is unchanged.

## 4. Findings

### F1 — No CHANGELOG entry (fix before landing)

`CHANGELOG.md` is untouched. The branch adds `science list|guide|operation`
to both clients, three HTTP routes to both implementations, a new engine
verb `complete-sweep-judgment`, exit-code semantics for three engine
verbs (64/65/66/70 replacing a generic 1 on `complete-judgment` and
`rescore-style`), and a Studies-panel sheet. The exit-code change in
particular belongs under a "Changed" heading so anyone scripting the
engine CLI sees it. Same standard applied to the implementation branch on
the 6th.

### N1 — Commit message has no body

Subject line only. The handoff document carries the rationale; a body
would survive a doc rewrite. Worth adding if the maintainer amends on
landing; not worth a rebase on its own.

### N2 — The catalog is a promise the matrix must keep

Twenty operation rows now state exact paths and restrictions in a shipped
artifact that agents will read as authoritative. The operation matrix and
the reference will drift from it unless the catalog is edited with them.
The drift gate protects the three copies of the catalog, not its agreement
with the verb tables. A small follow-up: have the reference check also
assert that every `engineCLI` string in the catalog names a censused verb.

## 5. Landing instructions

1. Fix F1; rerun the Python suite (the changelog is prose, but the
   standard is both suites on the landed tip).
2. The §2a failure is a pre-existing flake and does not block this branch;
   its one-line hardening in `norm_convention.py` lands separately on main.
3. Fast-forward main to the tip. Deploy consequence: none new. The app
   rebuild already owed carries the sheet; the engine deploy already owed
   carries the routes and the new verb.
