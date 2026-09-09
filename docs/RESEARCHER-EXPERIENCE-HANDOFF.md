# Researcher experience implementation — review handoff

Date: 2026-09-09
Branch: `codex/researcher-experience`
Base: `e4a06b3` on main

This branch implements the researcher-facing changes in
[the findings and acceptance plan](RESEARCHER-EXPERIENCE-FIX-PLAN.md).
That plan remains the acceptance checklist; implementation and automated tests
are not substitutes for the live journeys below. Main and the installed app
have not been changed. Review and integration go through the maintainer's
coding/auditing agents.

## What changes for a researcher

| Findings | Result in this branch |
|---|---|
| UX-02, UX-03 | Workspace agent instructions and shared interviews explain the agent/model/intervention vocabulary and distinguish extraction data, validation, preservation checks and final testing. Agents must offer existing files, pasted data, a prompt, or explicitly approved generation/delegation. A research question does not authorize workers or dataset generation. Scope, model choice, worker roles and cost are discussed first. |
| UX-04 | The Mac inventory reads architecture metadata before offering cached repositories as generative models. Python inventory also excludes explicitly non-generative architectures. Base completion models remain eligible. Cache-presence and exact-revision loading checks retain their previous roles. |
| UX-05 | Results offers an in-app reader for the complete local file, including JSONL. It reads bounded pages, preserves UTF-8 boundaries and supports long or malformed records as text. Remote previews retain the verified evidence-import path before local reading. |
| UX-06 | Templates uses “template” in its buttons, help and editor return actions. Templates, Concepts & Vectors, Adapter Training and OptVec have section-specific explanatory viewers. Historical workspace activity remains available in a disclosure, and explicit viewer pinning is respected. |
| UX-07 | Designated-reference extraction has a Cowork authoring prompt with separate target/reference files and an explicit generation-approval boundary. It does not pretend to be a grand-mean corpus. |
| UX-08 | The Gemma Scope analysis pane states what comparison does, which vector to select, which fields are resolved metadata, and where to import a feature directly. Similarity is framed as a hypothesis, not validation. |
| UX-09, UX-10 | Calibration/projection moves to an advanced disclosure at the bottom of Concepts & Vectors. Counts, saved-data preview, destination selection, paste/import, prompt authoring and optional basis building have distinct explanations. Prompt fields are grouped with real labels and tooltip examples; an explicit preview shows effective defaults. Reading-position parameters have names and explanations; raw versus chat rendering explains what changes. |
| UX-11 | Concept Vector Builder offers direct SAE import through the Python workbench. A Neuronpedia link resolves against installed SAELens metadata, or identifiers can be entered directly. A review identifies the feature, model, calibration and possible weight download before submission. Existing import owners retain their scientific checks. |
| UX-12 | J-lens derivation refreshes the originating vector catalog after success and reports where to find the result. A failed refresh or workspace switch does not claim that the current library contains the artifact. SAE imports use the same refresh rule. |
| UX-13, UX-14 | Concept Vector Builder has its own editable recipe selector. Playground vector controls remain usable without finding a separate master switch. Turning on one vector from an old muted mix does not turn on all the other vectors. |
| UX-01, UX-15 | Local neutral-component estimation runs off the UI actor, with cancellation between numerical operations and a captured save destination. The app exposes the existing Python corpus/basis catalog, durable build job, cancellation request and basis picker. Selected bases enter the existing agent/generation composition path. Changing engine, model or a known model revision prompts correction instead of silently reusing an incompatible basis. |
| UX-16 | OptVec opens with Train a vector, Evaluate a trained vector and Plan a campaign. These open the maintained method interviews, not another training implementation. Help distinguishes optimization from a steering-strength sweep and from J-space analysis. |
| UX-17 | The main adapter file controls are Training data and Validation data, each with Choose, drop support and source browsing. Project, output and optional raw-source locations live in collapsed Storage details. Tooltips explain the two datasets, formats and separation; Analyze Training Plan remains the authority for parsed counts and recommendations. |

## Boundaries that matter in review

- No dataset generation, model training, model/SAE download, scientific run,
  release installation or merge was performed for this implementation.
- Agent instructions are guidance for a cooperating agent. They do not impose a
  new runtime approval mechanism or promise that any external agent obeys them.
- The direct SAE path uses the existing Python import convention and calibration
  requirement. It does not claim that an SAE feature from one checkpoint can be
  used on an arbitrary MLX conversion. Missing SAELens metadata yields explicit
  identifier guidance; URL lookup never fetches the supplied URL.
- The new read-only feature lookup is available to both service roles. Existing
  neutral and SAE workbench routes keep their existing roles. A runner-only
  deployment still cannot receive workbench authorship through these routes.
- Local reference-file selection does not upload data. The Python build form
  lists the engine's already-staged corpora. The fully managed runner input and
  evidence round trip in UX-15 is **not completed or qualified by this UI wiring**;
  preserve it as an open acceptance item, not a parity claim.
- The Python basis form explicitly uses all layers. The local default remains
  the middle third with an all-layers option. Artifact formats stay native to
  their owners; no conversion or identity rewriting is introduced.
- Local cancellation is cooperative: it can wait for a model operation or one
  layer's PCA. Once final publication has begun it may complete. Server jobs use
  their existing cancellation semantics. No UI promises immediate GPU release.
- No frozen manifests, saved runs or existing provenance stamps are rewritten.
  No numerical algorithm bodies were moved. PCA orchestration and cancellation
  changed intentionally; this is not presented as a mechanical AST-only refactor.

## Verification

- Full Python suite, worktree `Server/` with the existing main venv and
  `PYTHONPATH=.`: **6,203 passed, 9 skipped, 8 warnings**.
- Full serial Xcode beta suite: **290 SteeringKit tests and 4,611 ExperimentKit
  tests passed**. The real Python client is selected in the shell environment.
- Shared interview, complete workspace seed, science resource/substrate and
  compiled Python identity checks pass. Swift bridge release gate and task
  prompt parser audit pass. `git diff --check` passes.
- New checks cover muted vector activation, cancellation before PCA, bounded
  Unicode/JSONL reading, generative inventory, designated-reference prompts,
  exact/ambiguous/invalid feature links, CLI/HTTP lookup agreement and basis
  catalog identity/layer metadata. Existing projection and composition tests
  remain in the full suites.
- An early restricted Python run was interrupted while waiting in Hugging Face
  HTTP handling. The full successful run used the normal test permissions.
  Early Swift passes caught test-fixture assumptions and a view annotation;
  those were corrected. One intermediate run saw the expected compiled/source
  identity refusal while source edits were still underway; the successful run
  used matching generated identity and Python bytes.

Reproduce from the branch worktree, serially:

```sh
cd Server
PYTHONPATH=. /path/to/existing/venv/bin/python -m pytest -q
cd ..
DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer \
TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.920.1 \
TEST_RUNNER_STEERLAB_TEST_PYTHON=/path/to/existing/venv/bin/python \
xcodebuild test -skipMacroValidation -scheme SteerLab-Package \
  -destination 'platform=macOS' -parallel-testing-enabled NO \
  -derivedDataPath /private/tmp/researcher-experience-check \
  CLANG_COVERAGE_MAPPING=NO
```

Read the diff before landing. Both suites and relevant generated-resource gates
must pass on the reviewed tip. Any later change described as a mechanical move
also needs its corresponding AST audit. Fast-forward only after the maintainer's
reviewing agents approve; rebuild the app and its matching Python payload together.

## Isolated UI walkthrough

A separately identified, ad-hoc preview bundle was assembled from the tested
Debug binaries and matching payload. It used a new temporary workspace; the
installed app and its preferences were not replaced. This was a UI check, not
release-build or scientific qualification.

Verified interactively:

- Concept Vector Builder exposes an editable recipe selector, OptVec entry and
  SAE import entry. Calibration/projection starts collapsed at the bottom.
- The expanded projection group has one label per field, with examples in help.
  Entering two properties and requesting a preview placed those properties in
  the prompt; the stored corpus remained at 200 rows.
- View existing examples opened the JSONL reader. Next page reached byte
  115,331 of 115,331 and disabled Next; Previous remained available.
- Switching to Templates showed template terminology and its own explanation.
- Creating an empty adapter project in the disposable workspace exposed the two
  primary data rows with Browse and Choose. Storage details remained collapsed.
  No training or dataset generation was started.
- OptVec exposed train, evaluation and campaign actions. The initial walkthrough
  caught an empty first-click training sheet. The follow-up uses one item-based
  presentation value containing the workflow, workspace and client, instead of
  a Boolean plus separately updated optional state. SAE confirmation similarly
  passes its reviewed request through SwiftUI's presenting parameter.
- After rebuilding the preview from the tested follow-up, the first training
  click opened the populated form with research decisions, data selectors and
  review controls. Evaluation and campaign actions each opened their own
  populated form as well. All three were dismissed without publishing a request
  or submitting compute. This closes the observed empty-sheet regression; it
  does not replace the authoring-to-evidence acceptance journey below.

## Live acceptance still required

1. At ordinary and narrow window sizes, walk Templates, Adapter Training,
   Concept Vector Builder and the advanced reference section. Confirm values
   have one label, fields are usable, tooltips are readable, dataset browsing
   works, and the pinned viewer has the expected ownership.
2. In a disposable workspace with an approved cached model, build a local basis
   while navigating the UI. Cancel during capture and PCA. Inspect the saved
   basis, select/deselect removal, save an agent, reopen it and verify actual
   injection inputs. Repeat on an approved Python workbench with matched
   all-layer settings and record resource use.
3. Complete the runner-managed reference-input/evidence workflow and its queue,
   download and custody checks before marking all of UX-15 complete. Do not
   relax the service-role census to make a workbench form run on a runner.
4. With approved compatible artifacts, derive a J-lens direction and import an
   SAE feature; locate each exact artifact in Data and Playground, exercise a
   failed refresh and a workspace switch, and inspect scaling/provenance. The
   researcher's previously reported missing direction was not located or
   recovered by this implementation.
5. Run an OptVec authoring-to-evidence journey and an adapter train/validation
   inspection from a clean workspace. Use approved data; opening a form must not
   create examples, start a worker or submit compute.
6. Judge projection's scientific utility using baseline, original steering and
   projected steering on development and separate held-out data, with preserved
   capabilities and comparable strength conventions. UI access and unit tests
   do not establish behavioral benefit or broad backend qualification.


## Follow-up: researcher guidance in the authoring dialogs

The next review includes these responses to hands-on feedback:

- SAE import uses a padded, grouped form with two stages: identify the feature,
  then select measured steering calibration. Labels and help explain release,
  dictionary, feature number, and calibration; a Neuronpedia link can fill the
  source identifiers. A description is not represented as behavioral validation.
- Method authoring separates model, data, settings, notes, and review. Steps with
  no model or file inputs are omitted. The connected engine supplies a model
  picker; a separately prepared model can still be entered explicitly. Inspect
  selected model fills a returned exact revision without acquiring weights.
  Changing the model clears the prior revision. Local MLX no longer supplies a
  fallback loopback Python client to the OptVec form.
- Notes have examples and explain their actual use: stored rationale, not
  optimizer parameters. Existing nonempty-note admission remains unchanged.
  Required fields are named as required, rather than marked with unexplained
  asterisks. Defaults for settings are unchanged; data examples are syntax
  examples, never auto-selected study data.
- OptVec file roles and JSONL examples are maintained in the shared interview,
  so Python agents receive them too. The choice examples pass the training
  owner's row parser. Training increases the target/contrast log-probability
  margin; anchor training preserves baseline choice distributions, and
  capability training penalizes incorrect answers. Study outputs require
  explicit preparation as choice rows, not direct ingestion by this form.
- Review explains that it captures inputs and settings without executing. It
  shows effective settings, notes, and input paths, with fingerprints and exact
  engine JSON under disclosures. Saving chooses a unique requests subdirectory
  automatically; an override is available only under Storage details. Existing
  stale-input and create-only publication checks remain authoritative.
- A J-lens library button always appears in Concept Vector Builder. Its dialog
  reuses the existing catalog, acquisition, import, token lookup, and derivation
  owners. The model picker includes known and imported lenses. Another published
  model can be named explicitly, with the existing declared-tier parameter sent
  through to the import owner. This is not a qualification claim.
- **General custom J-lens file/repository import is still absent.** The existing
  engine importer is tied to the published repository's format and provenance.
  A follow-up must accept explicit source artifacts, retain their actual source
  and fitted-model metadata, verify dimensions and layer conventions, publish
  without replacing existing evidence, and expose the same operation to agents
  and the app. Merely relabeling files as a curated model is not an implementation.
- Repository instructions now explicitly require the Oxford comma in
  researcher-facing prose.


Follow-up verification: the full Python suite passed **6,204 tests**, with nine
skips, including the new shared-example parser check. The serial Swift suite
passed **290 SteeringKit and 4,611 ExperimentKit tests** on the final source,
including the disconnected-model hint and multiline note-example wrapping. Shared
science resources, compiled Python identity, bridge retirement, and the task
parser AST gate pass. In a separately identified preview, the model, data,
notes, and review pages were inspected visually, and the always-visible J-lens
button opened its library dialog. The Python-connected SAE layout and the new
uncurated import path still need interactive acceptance with the reviewer’s
engine; no feature import, model load, or lens download was executed.

An unrelated diagnostic was also observed: the engine's `serve --help` enters
server startup instead of printing help. The probe could not bind a socket and
exited; its newly created, ignored bookkeeping directory was moved out of the
checkout into a temporary directory. No study data was changed. Treat this as a
separate CLI-help defect, not a documented read-only discovery command.
