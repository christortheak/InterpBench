# Review: `codex/researcher-experience` @ `9d0f4fe`, landed at `a65d0d6` (on main `e4a06b3`)

Reviewer: the maintainer's integration agent, 2026-09-09. Read against
`docs/RESEARCHER-EXPERIENCE-HANDOFF.md` and the acceptance plan in
`docs/RESEARCHER-EXPERIENCE-FIX-PLAN.md`. Six commits, 72 files,
+2,891/−444 at `9d0f4fe`; §§1–5 review that tip. The refactoring agents
then answered every finding on the branch itself in `a65d0d6`, which §6
reviews and which is the commit landed. Main has not moved, so a
fast-forward is available.

## 1. Verdict

**Landable by fast-forward with one landing fix (F1) and a changelog entry
(F2).** The researcher-facing work is right and careful: the agent contract
and interviews now say that a model or concept choice is not permission to
generate data or spend money, the vector builder offers every source of a
vector in one place, the Playground's per-vector switches no longer depend on
a distant master toggle or reactivate an old mix, local projection builds run
off the UI actor and cancel, the Python engine's projection catalog and build
job are wired into the app with a mismatch guard, and Results can read a whole
JSONL file in bounded pages. What the branch got wrong is procedural, not
scientific: it edited the generated science declarations by hand instead of
the per-operation specifications they are generated from, so the repository's
one gate entry point fails on the tip and a regeneration would silently
revert the improved help text. The port into the specifications is
mechanical and reproduces the branch's files byte for byte; it lands with
this review.

## 2. Verified independently

| Check | Result |
|---|---|
| Ancestry | `main` (`e4a06b3`) is an ancestor of `9d0f4fe` |
| Unified gates on the tip (`check-generated.py --audits`) | **fails** at `check-operation-specs.py`: `Regenerate WorkspaceSeed/prompts/method-guides/catalog.json` (F1) |
| Unified gates on the tip plus the F1 port | every generator matches and every audit passes (see §5) |
| Byte identity of the port | after porting the branch's edits into `docs/techniques/operations/*.json`, `check-operation-specs.py --write` reproduces the branch's `workflows.json` and `catalog.json` with no diff |
| Request and response shapes the new Swift calls rely on | `/api/gemmascope/import-id` requires exactly `model, release, saeID, feature, label, residualNormArtifact`; `/api/neutral-pcs/build` reads `corpus` and the new `expectedModelID`; `/api/neutral/corpora` returns `{corpora, bases}`; `/api/jlens/lenses/import` reads an optional `tier` — all match `NeutralRemote.swift` and `JLensRemote.swift` |
| Vocabulary in the diff and all six commit messages; `git diff --check`; public scan | clean |
| The `serve --help` defect the handoff reports | reproduced: `steerlab-server serve --help` starts the server (auth banner, artifact-root warning, uvicorn) and mints a token file and a `.steerlab` bookkeeping directory in the working directory; a separate CLI defect, not this branch's |
| Full Python suite on the ported worktree (main venv, cwd = worktree/Server) | green; the count is in §5 |
| Full serial Xcode beta suite on the ported worktree (`TEST_RUNNER_STEERLAB_TEST_PYTHON` set, external derived data, coverage mapping off) | `TEST SUCCEEDED`: 290 SteeringKit + 4,611 ExperimentKit, matching the handoff's claim |

## 3. What the branch does

**Agent and interview contract (UX-02, UX-03).** The workspace agent guide
gains a "Collaborate at the researcher's level" section: the agent is the
experimental object, explanations follow the researcher's level, and choosing
a model or concept is not authorization to generate datasets, call paid
models or launch coworkers. The three shared study interviews open with the
same rule. A designated-reference authoring prompt ships as workspace data
with separate target and reference files and an explicit approval boundary.

**Vector creation in one place (UX-11, UX-12, UX-13, UX-14, UX-16).** The
Concept Vector Builder offers Train with OptVec, Import an SAE feature, a
J-lens library, and its own editable recipe selector. OptVec's panel opens
with train, evaluate and campaign actions that present the maintained
method interview through one item-based sheet presentation. The SAE sheet
resolves a Neuronpedia link against the installed SAELens directory through
a new read-only route and CLI verb, never fetching the URL, and requires a
measured calibration artifact before review. J-lens derivation and SAE
import refresh the originating vector catalog and say where the artifact
is, or say plainly that the current library was not refreshed. In the
Playground, a per-vector switch is the whole action: switching one vector
on under an old muted mix enables only that vector.

**Projection bases (UX-01, UX-15).** The local neutral-component build runs
its corpus load, PCA and save off the main actor with cancellation checks
between layers and a captured destination, so a workspace switch mid-build
cannot misfile the artifact. Server workspaces gain the engine's corpus and
basis catalog, a durable build job with an expected-model guard, and a basis
picker whose selection carries engine, model and revision so an incompatible
basis is refused at generation and at agent save rather than reused.

**Reading and explaining (UX-04 to UX-10, UX-17).** Cached repositories are
offered as generative models only when their configuration says so, on both
engines. Results and the reference-data section gain a bounded, UTF-8-safe
page reader. Templates says "template" throughout. Calibration and
projection move under an advanced disclosure with labeled fields, an
effective-prompt preview and explicit copy actions. Reading-position choices
and raw-versus-chat rendering carry explanations. Adapter training leads
with Training data and Validation data and folds the bookkeeping paths into
Storage details.

## 4. Findings

**F1 — generated science declarations were edited by hand.** The improved
interview help and JSONL examples, the OptVec titles and purposes, and the
Gemma Scope catalog row (its `mac` and `http` text and the new
`resolve-feature` route) were written directly into
`WorkspaceSeed/prompts/method-guides/{workflows,catalog}.json` and the
packaged copies, not into `docs/techniques/operations/*.json`, which is the
source `check-operation-specs.py` generates them from. On the tip the
repository's single gate entry point fails, and running the generator
reverts every one of those improvements. The handoff lists the individual
resource checks it ran; the one-entry-point gate that `ADDING-A-TECHNIQUE.md`
names was not among them, which is how this slipped. Fix, landed with this
review: port the same text into the seven affected specifications
(`optvec-train`, `optvec-eval`, `optvec-geometry`, `optvec-fracture`,
`optvec-interpret`, `optvec-gradient`, `gemmascope`) and regenerate;
the regenerated files are byte-identical to the branch's, so no seed, packaged
copy or compiled identity changes.

The same edits also trip `audit-operation-registration.py`, the audit that
proves the September registration migration preserved every declaration
against its base `d84968c`. That audit compared whole catalog and interview
rows, so it forbade any later change to an existing operation's help text or
catalog pointers, and it cannot be re-pinned to a later base because it
parses the pre-migration literal registry at the base. Landed with this
review: the audit now compares what has execution meaning (every catalog key
except the researcher-facing `mac` and `http` pointers, every original action
unchanged with new actions permitted, and each interview field's identity,
kind, requirement and default) and leaves titles, purposes, labels, help and
examples to the operation-specs gate. Two negative controls were run by hand:
changing a field's kind and changing an action's service role both still
fail it.

**F2 — no changelog entry.** A 72-file researcher-facing change lands with
a bullet under Unreleased; added at landing.

**N1 — "template" in the app, "design" everywhere else.** The app now says
template in every label, but the Python client's family is `design`, the
Mac CLI's family is `design`, and `STUDY-DESIGN-AUTHORING.md` and the shared
interviews say design. A researcher reading the app and their agent's
commands will see two words for one thing. Worth one decision and one sweep,
either way.

**N2 — the Oxford-comma rule sits first among the ground rules.** The root
`AGENTS.md` lists it above "help honest research" in the section that
"outranks speed." It is a style rule; a line under a style heading, or in
the contributing notes, would carry the same instruction without that
ranking.

**N3 — model inventory now reads every cached repository's configuration on
each enumeration.** `localModelIDs` opens `config.json` for each snapshot of
each cached repository; `isCached` keeps the marker-only enumeration, so the
two no longer share one answer, as the updated test comment says. The cost
is a directory walk per refresh; acceptable, and worth remembering if a
picker refresh ever feels slow on a large cache.

**N4 — a local build finished after a workspace switch leaves a stale
status.** The basis is saved into the captured destination, correctly, and
the catalog refresh is skipped, correctly, but `neutralPCStatus` keeps the
in-progress text. Cosmetic.

**N5 — the engine's expected-model guard fails the job rather than refusing
the request.** A mismatch between the loaded model and `expectedModelID`
raises inside the build, so the researcher sees a failed job carrying the
message. A synchronous 400 before the job is created would be the
instrument's usual shape.

**N6 — items the handoff itself leaves open, restated so they are not lost:**
the runner-managed reference-input and evidence round trip for UX-15 is
wired in the UI but not completed or qualified; general custom J-lens file
import is absent; the researcher's previously missing derived direction was
not located; and every UX item still has the live acceptance walk in
`RESEARCHER-EXPERIENCE-HANDOFF.md` §"Live acceptance still required" ahead
of it.

## 5. Landing shape as proposed at `9d0f4fe`

Fast-forward, then one landing commit carrying the F1 port of the seven
operation specifications, the audit scoping, the F2 changelog bullet, and
this review. Suite results on that ported worktree, before the branch's own
follow-up superseded it:

- Python: 6,204 passed, 9 skipped, 8 warnings (matching the handoff's claim).
- Swift: `TEST SUCCEEDED`, 290 SteeringKit and 4,611 ExperimentKit tests.

## 6. The follow-up, `a65d0d6`, and what actually landed

The refactoring agents carried the repairs on the branch before this
review's landing commit existed, so the port described under F1 was
discarded rather than applied twice. Read in full (30 files, +347/−74):

- **F1.** The seven specifications own the help, examples, titles and the
  Gemma Scope catalog route. The registration audit keeps its historical
  base and its owner-body, binding and input-role checks, and now compares
  declarations minus their presentation keys: every catalog key except `mac`,
  `http` and `actions`, every original action present unchanged with
  duplicate action ids refused, and every interview field minus `label`,
  `help` and `example`, in order. It carries its own negative controls,
  mutating field id, kind, requirement and default, action path and service
  role, and removing a field or an action, each of which must fail, plus a
  prose edit that must pass. Stricter than the scoping proposed above.
- **Additional finding, theirs.** The OptVec `neutralTexts` and
  `probePrompts` examples were choice rows that their text loader skips; the
  specifications now carry text-field examples and a test loads both through
  `load_neutral_texts`. A deliberate generated-resource change, so the seeds,
  packaged copies and compiled identity moved with it.
- **F2.** A changelog bullet under Unreleased.
- **N1.** Terminology mapped rather than migrated: the root contract, the
  workspace guide, the three interviews and `STUDY-DESIGN-AUTHORING.md` say
  that the app's "template" is what the `design` command family operates on.
  A command rename stays out of scope.
- **N2.** The Oxford-comma rule moved from the ground rules to a Writing
  style section, with the template wording beside it.
- **N4.** A local build finished after a workspace switch now reports success
  and the saved path in the original workspace.
- **N5.** A mismatched `expectedModelID` is refused with HTTP 400 before a
  job exists; the recheck under the model lock stays. A new test module
  covers the refusal, matching and legacy requests, and a model change after
  admission.
- **N3, N6.** Left as observations and open acceptance items, as they should
  be.

Verified on `a65d0d6` in a clean worktree: the unified gates and audits, the
public scan, whitespace, and vocabulary in the diff and the commit message;
the full Python suite and the full serial Xcode beta suite (counts below).
The branch changed shipped Python and the compiled identity twice, so the
app and its Python payload are rebuilt together before the app is used with
this main.

Suite results on `a65d0d6`:

- Python: 6,210 passed, 9 skipped, 8 warnings (matching the follow-up's claim).
- Swift: `TEST SUCCEEDED`, 290 SteeringKit and 4,611 ExperimentKit tests.
