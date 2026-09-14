# UI/UX follow-up handoff — typed values, crowded rows, unclear buttons (2026-09-13)

Read-only review of `Sources/SteerLabApp` at branch
`claude/results-page-button-ux-94d33e` @ `0521032` (main `0208750` plus one
commit). It generalizes the three defects a researcher reported on the
Compute section's Server Jobs panel, which that commit fixed:

1. six full-width buttons packed into a 560 pt column truncated every label;
2. a columns-style `Form` mixing captions and button rows, with a minimum
   width larger than the window, clipped and misaligned;
3. forms asked for a file path, a model id, and a commit hash as free text
   where the workspace or the server already held the candidates.

Every line cited below was re-read in source on 2026-09-13. Line numbers
drift; search for the quoted label when a number is off. Nothing in this
document has been changed yet.

## Ground rules for the agents doing this work

- **Reuse what exists.** The app already has the right affordance for every
  class of problem here; the defect is non-uniform application. Copy the
  sibling, do not invent a third idiom:
  - workspace file chooser: `WorkspacePathChooseButton`
    (`Sources/SteerLabApp/WorkspaceFileChooser.swift:106`), used at five sites;
  - picker over workspace files plus a typed fallback: `InstrumentPathField`
    (`Sources/SteerLabApp/OptimizationComposerView.swift:1529`, currently
    `private`), fed by `OptimizationComposer.scanInstrumentFiles()`
    (`Sources/ExperimentKit/OptimizationComposer.swift:696`);
  - server-side revision resolution: the pattern in
    `ScientificExecutionSheet.resolveRevision`, `MethodAuthoringSheet.inspectModel`
    (`client.modelLoadPreflight`), and `JudgingSectionView.resolveJudgeRevision`
    (`client.cachedRevision(forModel:)`);
  - site GPU vocabulary picker: `ScientificGPUSelection`
    (`Sources/SteerLabApp/ScientificGPUSelection.swift:4`);
  - button rows that must not truncate: `ViewThatFits(in: .horizontal)` with
    an `HStack` first and a `VStack` fallback
    (`ModelVariantsPanelView.swift:377`, `HomeDashboardView.swift:158`,
    `ConceptsPanelView.swift:1163`), or a `Menu` for the rarely used verbs
    (`ServerJobsPanelView.actionsMenu`);
  - a disabled control's reason rendered as a visible caption, not only a
    tooltip (`ResultsExplorerPane.swift:203`,
    `ClusterProfileCoauthoringSheet.swift:132`);
  - destructive confirmation titles that name the object
    (`ModelVariantsPanelView.swift:153`).
- **Split-view minimum height.** Panels that sit directly in an `HSplitView`
  column (Compute, Studies, and others) must not add rows whose presence
  changes with async state; use constant-height slots or captions that go
  transparent. See the write-up on `ServerJobsPanelView.jobsRegion` and
  `docs/UI-UX-AUDIT-2026-09-06.md` headline 8. A moving minimum is a crash
  on macOS 27.
- **Copy conventions.** `.help` strings are lowercase fragments. Researcher
  prose uses the Oxford comma. Reusable study settings are "templates".
  Every new or renamed control keeps a `.help`; every icon-only control gets
  an `accessibilityLabel`.
- **No identifying words.** Site, host, and person names, and `/Users` paths,
  are forbidden in anything committed (the denylist in
  `Tests/ExperimentKitTests/AgentContractTests.swift`). Run
  `python3 scripts/ci/public_scan.py` before every commit.
- **Do not change behaviour.** These are entry-UI and layout changes. Manifest
  fields, engine validation, and gates stay where they are; a picker writes
  the same string a researcher would have typed.
- **Typed fallback stays.** Wherever a picker replaces free text, keep the
  typed path behind a `DisclosureGroup` (the pattern in
  `ScientificExecutionSheet.batteryRows`): a file staged only on the server,
  or a model not yet installed, must remain expressible.

## Work packages (disjoint file ownership)

WP-0 lands first and alone; the others can run in parallel afterwards. An
agent owns every file listed under its package and touches no other view
file. Anything that needs a change outside the owned set is reported, not
done (the 2026-09-06 lesson: cross-boundary defects survive
ownership-partitioned programs).

### WP-0 — shared components (land first)

Files: `Sources/SteerLabApp/WorkspaceFileChooser.swift`,
`Sources/SteerLabApp/OptimizationComposerView.swift` (the move only),
one new file `Sources/SteerLabApp/ModelRevisionResolveButton.swift`,
and `Sources/ExperimentKit/` for any helper worth a unit test.

- **0.1 Promote the instrument path field.** Move `InstrumentPathField` out of
  `OptimizationComposerView.swift:1529` into `WorkspaceFileChooser.swift` as
  an internal `WorkspacePathField(label:options:text:help:startingSubdirectory:)`
  that renders the picker, the typed fallback behind a disclosure, and a
  `WorkspacePathChooseButton`. Update the five call sites in the composer
  (`:449`, `:468`, `:482`, `:1155`, `:1167`). Options come from
  `OptimizationComposer.scanInstrumentFiles()` or from a directory listing the
  caller supplies (`ScientificDiagnosticInputs.batteryFiles(root:)` shows the
  shape for batteries).
- **0.2 One revision resolver.** A small view, `ModelRevisionResolveButton(
  client:modelID:revision:note:)`, that runs `client.modelLoadPreflight`, fills
  the binding only with a 40-hex commit, and writes an honest note (cached /
  resolved-but-not-cached / not resolvable / error), copied from
  `ScientificExecutionSheet.resolveRevision`. Local-workspace callers pass the
  Mac cache lookup that `ExperimentStore.judgePinPrefill` already uses.
- **0.3 Columns-style Forms (done for the scientific sheet).** A
  columns-style `Form` sizes its content column to the ideal width of its
  widest child, so one long single-line caption makes the whole sheet wider
  than its minimum and both edges clip. `ScientificExecutionSheet.swift` was
  the last `Form {` without a `.formStyle` and now uses `.grouped`. Any new
  sheet Form should too, and a caption in a columns form needs a width cap,
  not only `fixedSize(horizontal: false, vertical: true)`.

### WP-A — Optimization and agents

Files: `OptimizationRunsView.swift`, `OptimizationComposerView.swift`
(after WP-0 lands), `ModelVariantsPanelView.swift`, `OptVecPanelView.swift`.

- **A.1** `OptimizationRunsView.swift:1931`, `:1934`, `:2039`, `:2060` —
  `TextField("Dev prompts file")`, `TextField("Capability battery file")`,
  `TextField("Choice prompts — <concept>")`. Typed workspace-relative paths in
  the sweep-spec editor; the composer offers the identical fields as a
  picker. Fix: `WorkspacePathField` from WP-0. Check: a typo is impossible
  from the picker; the typed fallback still normalizes through
  `SweepSpecForm.workspaceRelativeNormalized`.
- **A.2** `ModelVariantsPanelView.swift:512`, `:526` — robustness
  `TextField("Capability battery")` and `TextField("Coherence prompts")` with
  an inspect row but no chooser. Fix: `WorkspacePathChooseButton` beside each
  (`startingSubdirectory: "prompts/batteries"` and `"prompts"`), preserving
  the preset-to-Custom switch in the binding's setter.
- **A.3** `ModelVariantsPanelView.swift:890` — `TextField("Base revision")`
  with help "optional pinned model revision, if known". Fix: WP-0 resolver
  beside it, bound to the base model chosen in the picker above.
- **A.4** `ModelVariantsPanelView.swift:2051` — the `ViewThatFits` fallback
  keeps all five agent buttons in one `HStack`, so when neither candidate
  fits the last one truncates. Fix: second candidate stacks the buttons
  vertically, as `actionButtons` at `:377` does.
- **A.5** `OptimizationComposerView.swift:1244` — the primary button title
  concatenates the substrate label ("Declare & Optimize on …") beside two
  more long buttons in a 560 pt column. Fix: `ViewThatFits` with a vertical
  fallback; `.lineLimit(1)` plus `.help(optimizeButtonTitle)` on the button.
- **A.6** `OptimizationComposerView.swift:1650`, `:1655` — OpenRouter judge
  slug and provider as bare text (default judge kind is `openrouter`, so this
  is the default path). Lower confidence: the app may hold no OpenRouter
  catalog. Fix, if a catalog call exists beside `OpenRouterCatalog.endpoints`:
  a searchable picker with free text as fallback; otherwise offer previously
  pinned slugs and providers from the workspace as a menu.
- **A.7** `OptVecPanelView.swift:828` — `TextField("Eval run (optional)")`
  under two proper pickers, while `panel.runs` lists every run with its kind.
  Fix: `Picker` over `panel.runs.filter { $0.kind == .eval }` with a
  "Choose…" empty tag; keep the artifact-provenance default.

### WP-B — Studies

Files: `StudyArtifactAttachmentSheet.swift`, `SAERosterSheet.swift`,
`StudyRunControlsView.swift`, `StudyRecentJobsView.swift`,
`StudyMeasurementsView.swift`, `StudyConceptsSection.swift`,
`StudyTaskPromptsEditor.swift`, `StudyLiveRunView.swift`,
`ValidateStudyButtonRow.swift`, `StudyPreparationControlsView.swift`,
`StudyManagementSection.swift`, `ModelRevisionRow.swift`,
`DataReadinessSection.swift`, `TemplatesPanelView.swift`,
`TemplateInstantiationSheet.swift`.

- **B.1** `StudyArtifactAttachmentSheet.swift:23` — `TextField("Vector path
  (without extension)")`; the help tells the researcher to strip the
  extension by hand and warns about `..`. Fix: `WorkspacePathChooseButton`
  (`startingSubdirectory: "vectors"`, or wherever the catalog files them)
  that drops the extension on selection. Lines `:27`, `:29`, `:31` — concept
  name, source concept, evaluation run: pickers over `manifest.concepts`
  (plus "new name…") and over the workspace's run directories.
- **B.2** `SAERosterSheet.swift:17`, `:19` — roster path and draft study name
  typed with the workspace root in hand. Fix: chooser for the path; picker of
  draft study names. Lines `:21`–`:26`: four text buttons in one `HStack`, and
  "Pin reviewed roster" is `.disabled(planHash == nil)` with no reason. Fix:
  a read row (Check, Show) and a review→pin row, plus a caption "review the
  pin first".
- **B.3** `StudyRunControlsView.swift:402` — `TextField("e.g. A100")` for
  the GPU type whose own help says the vocabulary is the site profile's GPU
  types. Fix: `ScientificGPUSelection` (its options come from
  `client.scientificGPUPlacement()`), with an "other…" typed fallback.
- **B.4** `StudyRunControlsView.swift:478` — `TextField("job id to
  reconnect")` while `StudyRecentJobsView.swift:16` renders
  `recentServerJobs` below it. Fix: picker over recent jobs (verb · study ·
  state) with the typed field kept for ids from a previous session.
- **B.5** `StudyRunControlsView.swift:576` — `remoteJobActionsRow`: three
  buttons in a disclosure inside the 560 pt column, two disabled with the
  reason only in `.help`; "Import Evidence" also appears at
  `StudyRecentJobsView.swift:48` acting on a different job. Fix: wrap or
  split; caption "no job in flight — submit or reconnect first"; qualify the
  label with the job id.
- **B.6** `StudyMeasurementsView.swift:63`, `:65` — conditions and seats as
  comma-separated text; a misspelled name silently records nothing. Fix:
  multi-select menu of toggles over `manifest.conditions` and the seat
  casting (pattern: `TemplateInstantiationSheet.swift:262`).
- **B.7** `StudyConceptsSection.swift:167` — `TextField("extra corpus members
  (comma-separated…)")` directly under a `Picker` over the same
  `sources.filter(\.hasStories)` at `:155`. Fix: multi-select menu over that set.
- **B.8** `StudyTaskPromptsEditor.swift:38` — five text buttons in one row
  (Load Prompts, Save & Pin Prompts, Import JSONL…, Import table…, Generate
  factorial design…). Fix: an "Import…" `Menu` holding the three import
  routes, or `ViewThatFits`.
- **B.9** Bare "Stop": `StudyLiveRunView.swift:64` (stops judging, under
  "Stop Run" at `:39`), `ValidateStudyButtonRow.swift:89`,
  `StudyPreparationControlsView.swift:123`, `StudyRunControlsView.swift:243`.
  Fix: "Stop Judging", "Stop Validation", "Stop Extraction", "Stop Run"; the
  confirmation title names the same object.
- **B.10** `StudyManagementSection.swift:202` and `ModelRevisionRow.swift:36`
  — commit hash typed with only "Set Revision". Fix: WP-0 resolver beside
  each, resolving from the active substrate's cache (server) or the Mac
  cache (local), leaving "empty = auto-pin at freeze" as the default.
- **B.11** `DataReadinessSection.swift:339`, `:417`, `:470` — three bare
  "Pin"/"Unpin" pairs in one section. Fix: "Pin battery", "Pin baseline",
  "Pin validation subset". Lines `:355`, `:432`, `:488`: `.disabled(!isDraft)`
  on whole groups with nothing saying the study is frozen. Fix: one caption
  above the group. Lines `:320`, `:395`, `:449`: long placeholders plus
  three controls in one caption-font row; fix with `ViewThatFits` and move
  the explanatory half of the placeholder into the existing caption line.
- **B.12** `TemplatesPanelView.swift:378` — Instantiate…, Edit template…,
  Rename…, Delete in one row inside a grouped form. Fix: keep the first two
  inline; Rename and Delete into a `Menu`.
- **B.13** `TemplateInstantiationSheet.swift:132` — the treated-agent picker's
  empty tag reads "baseline" and the button beside it is dead with no reason.
  Fix: empty tag "select an agent…"; `.help` and a caption on the button.

### WP-C — Cluster, Compute, Playground

Files: `ClusterSiteEditor.swift`, `ClusterSetupWizard.swift`,
`ClusterHealthCard.swift`, `ChatView.swift`.

- **C.1** `ClusterSiteEditor.swift:175` — `TextField("Default partition")`
  while `partitionsTable` at `:200` holds the names in the same form. Fix:
  `Picker` over `model.partitions.map(\.name)` plus an empty "site default"
  tag. Same treatment for `:225` allowed GPU types (multi-select over
  `gpuTable` rows), `:173` default gres, `:318` required headers (the legal
  vocabulary lives only in the tooltip; render it as toggles), `:526` torch
  variant, `:669` transfer method, `:683` auth mode, `:633` storage roles.
  The file already uses `Picker` for two vocabularies at `:366` and `:377`.
- **C.2** `ClusterSiteEditor.swift:205`, `:263` — hand-rolled table headers
  aligned by matching fixed widths, truncating in `.caption2` with no `.help`.
  Fix: `Grid` with `gridColumnAlignment`, or `.help` on each header.
- **C.3** `ClusterSetupWizard.swift:531` — `TextField("partition")` for the
  setup job; the selected site enumerates partitions. Fix: picker defaulting
  to the first non-GPU partition. `:475` remote bundle root and `:610`/`:613`
  env prefix and Python version duplicate site-profile fields with nothing
  saying which wins; show the site's values read-only with an "override for
  this run" disclosure.
- **C.4** `ClusterHealthCard.swift:30` — `Button("Refresh connection
  details")` runs a full `cluster.connect()` with no `.help`. Fix: "Reconnect"
  with a `.help` naming the consequence. `:493` — the rescan is disabled
  without a token and the reason lives only in `.help`; render it as a caption.
- **C.5** `ChatView.swift:1363` — Load, Install Model…, Cancel Load, Cancel
  Download in one row in the 340 pt Playground column; the two Cancel buttons
  appear exactly when they are needed. Fix: `ViewThatFits` with a vertical
  fallback.
- **C.6** `ChatView.swift:2031`, `:2135`, `:2316` — `TextField("", value:
  slot.alpha…)` with no `accessibilityLabel`. Fix: `.accessibilityLabel(
  alphaFieldLabel)` and "Strength λ". Same for
  `StudySamplingControls.swift:21` and `StudySetupSection.swift:168` (WP-B
  may take those two; coordinate).
- **C.7** `ChatView.swift:1044` — "Reset Chat" is disabled unless a model is
  loaded; clearing a transcript needs no model. Fix: drop the gate, or show
  `composerUnavailableReason` beside it.

### WP-D — Analysis, Probes, Judging, Data

Files: `JudgingSectionView.swift`, `JudgeModelPicker.swift`,
`JSpacePanelSection.swift`, `JLensSupportSection.swift`,
`JLensTraceViews.swift`, `InterventionPoliciesView.swift`,
`ProbesPanelView.swift`, `ExclusionRulesEditorView.swift`,
`FittingCorpusPreparationSheet.swift`, `ConceptsPanelView.swift`,
`FactorialDesignSheet.swift`, `DatasetInventoryView.swift`,
`ArtifactImportButton.swift`, `ServerNeutralBasisControls.swift`.

- **D.1** `JudgingSectionView.swift:206` — one `HStack` holding nine
  controls with fixed widths (name 140, kind 110, slug, provider 140,
  Discover, spinner, provider picker 260, key badge, Remove). The flexible
  slug field collapses first. Fix: a `VStack` of identity line / transport
  line / discovery line, or a `Grid`; the discovery note at `:640` gets its
  own row.
- **D.2** `JudgeModelPicker.swift:61`, `:65` — OpenRouter slug and provider
  as bare text with no Discover, while `JudgingSectionView.swift:622` has
  `discoveryControls` and its help says to use Discover rather than type.
  Fix: reuse the discovery controls here.
- **D.3** `JSpacePanelSection.swift:138` — Download published lens, Import
  downloaded lens, Refresh library in one row with no `.help`; the first two
  are steps 1 and 2, both dead with no model chosen and no reason shown.
  Fix: number them, add `.help`, render the "choose a model first" caption,
  stack with `ViewThatFits`. `:126` — the "Another published model" typed id
  could list the published repository's folders from the server instead.
- **D.4** `JSpacePanelSection.swift:141` "Refresh library" and
  `JLensSupportSection.swift:113` "Refresh Lenses" both call
  `client.jlensCatalog()`; `JLensTraceViews.swift:101` is a bare "Refresh".
  Fix: one label for the catalog refresh (share the function); "Refresh
  Runs" for the trace one.
- **D.5** `JLensTraceViews.swift:489` — six always-visible table headers
  without the `.help` their two logit-lens siblings have. Fix: `.help` on all.
- **D.6** `InterventionPoliciesView.swift:84` — token IDs as comma-separated
  integers against a tokenizer the app can query (`JSpacePanelSection.swift:274`
  "Show token options"). Fix: reuse the token lookup rows to build the list;
  keep the raw field as an advanced fallback. `:95` — "Review policy" and
  "Review advanced settings" differ by a hidden flag and the third button
  appears and disappears; `:124` — one Save whose label and target flip on
  hidden state; `:42` — a top-right "Done" on a sheet with a real Save. Fix:
  one Review whose scope follows the loaded settings, two distinct
  always-visible saves, "Close" instead of "Done", and inline disabled reasons.
- **D.7** `ProbesPanelView.swift:26` — bare "Refresh" with no `.help`, the
  only unhelped button in the file. Fix: "Refresh library" and a `.help`.
- **D.8** `ExclusionRulesEditorView.swift:141` — the endpoint a rule reads
  is typed, though the study declares its endpoints; a typo yields a rule
  that never fires. Fix: menu of declared endpoints, free text for others.
- **D.9** `FittingCorpusPreparationSheet.swift:113` — "New corpus folder"
  typed with no chooser; `:64`–`:66` — dataset id, dataset revision
  (defaulting to `main`), and file globs transcribed from a web page. Fix:
  chooser scoped to the workspace for the folder; resolve and pin the
  dataset commit after the id resolves, and list the repo's files where the
  Hub API allows. `:41` — "Done" on a sheet whose real action is "Save corpus
  and use for fitting"; rename "Close".
- **D.10** `ConceptsPanelView.swift:2667`, `:2681` — concept and topic drafts
  typed by hand while `existingConcepts` is a `Picker` three times in the same
  view; a misspelled topic silently creates an empty balance cell. Fix:
  editable combo (picker of known values plus "other…").
- **D.11** `FactorialDesignSheet.swift:323` — generated-item preview cut at
  90 characters and `lineLimit(1)` with no `.help` or selection; the
  substituted variable is usually past the cut. Fix: `.help(item.text)`,
  `.textSelection(.enabled)`, or wrap to three lines.
- **D.12** `DatasetInventoryView.swift:112` — a 110-character caption at
  `lineLimit(1)` with no `.help` (its twin in `ActivityFeedColumn.swift:47`
  has one). Fix: `.help` with the full text; same at `:177`.
- **D.13** `ArtifactImportButton.swift:91` — the primary import is dead
  until step 2 has run, with no reason and no `.help`; "Done" beside it reads
  as commit. Fix: adopt `importDisabledReason` from
  `ClusterProfileCoauthoringSheet.swift:132`; rename "Close".
- **D.14** `ServerNeutralBasisControls.swift:64` — "Build on this engine" is
  disabled by three invisible facts (busy, no model, fewer than four
  examples). Fix: caption "this reference set has N examples; the build needs
  at least 4".

### WP-E — Multi-agent, Results, Pipeline, Setup

Files: `MultiAgentPanelView.swift`, `RunResultsViews.swift`,
`SectionContainers.swift` (results detail headers only),
`PipelineComposerView.swift`, `ResearchSetupSheet.swift`.

- **E.1** `MultiAgentPanelView.swift:141` — New Scenario, New Contract
  Panel, Save Scenario, Delete… in one row, then two more long buttons at
  `:186`, in a form in the 560 pt column. Fix: `ViewThatFits` on both rows.
- **E.2** `MultiAgentPanelView.swift:322` — "Delete selected multi-agent
  scenario?" names no file and carries no consequence message; the agents
  panel fixed the same dialog at `ModelVariantsPanelView.swift:153`. Fix:
  title with the panel's name and a `message:` naming the studies that pin it.
- **E.3** `MultiAgentPanelView.swift:831`, `:842` — Run and Stop for a
  rehearsal never say "rehearsal". Fix: "Run Rehearsal" / "Stop Rehearsal".
  `:1508` — "Routed to" renders raw seat ids; map through `panel.agents`.
- **E.4** `RunResultsViews.swift:811` — per-condition run error at
  `lineLimit(1)` middle-truncated with no `.help`. Fix: `.help(error)`.
- **E.5** `SectionContainers.swift:1241`, `:2086` — results detail headers
  truncate the run id with no `.help`, while both list rows have one. Fix:
  `.help(item.name)` and `.help(run.id)`.
- **E.6** `PipelineComposerView.swift:301` — up to seven word-labelled
  checkboxes plus an info button on one row. Fix: `ViewThatFits` or a
  two-row `Grid`.
- **E.7** `ResearchSetupSheet.swift:87` — `.frame(width: 660, height: 700)`
  is a fixed size, not a minimum. Fix: `minWidth`/`idealWidth` and
  `minHeight`/`idealHeight` (the results explorer explains why a 780 floor
  is too tall for a 13-inch display).

## Already fine — do not re-litigate

Three pinned-file rows in `DataReadinessSection.swift` (choosers present);
`JLensTraceViews.swift:90` run picker before the typed id;
`FineTuningPanelView` directory rows (Choose… present);
`EvaluationClarityViews.swift:52` rubric path (chooser present);
`JudgingSectionView.swift:541` judge revision (Resolve present);
`InterventionPoliciesView.swift:80` vector path (Choose vector… present);
`DatasetCreationSheet.swift` (pickers and Choose… throughout);
`ClusterSetupWizard.swift:464` local payload (Choose… present);
`ClusterSiteEditor.swift` remote-host paths (venv, env-file, token-file,
remote repo, workspace, cache, archive, metadata roots) — these name paths on
the cluster, where a Finder chooser cannot apply.

## Verification every package must pass

```sh
export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
export TOOLCHAINS=$(xcodebuild -showComponent MetalToolchain 2>/dev/null | sed -n 's/^Toolchain Identifier: //p')
swift build --build-tests --scratch-path <non-iCloud scratch>
swift test --skip-build --scratch-path <non-iCloud scratch> --filter <suites you touched>
python3 scripts/ci/public_scan.py
```

- Build from a worktree needs the non-iCloud scratch path and both
  variables above; the Metal toolchain identifier comes from
  `xcodebuild -showComponent MetalToolchain`.
- Before landing, the full serial Swift suite:
  `xcodebuild test -skipMacroValidation -scheme SteerLab-Package -destination
  'platform=macOS' -parallel-testing-enabled NO`. `ClientSetupTests` needs
  `STEERLAB_TEST_PYTHON=<a Python with the Server extras>`.
- If any researcher-facing label that the workspace agent guide cites
  changes, edit `docs/AGENTS-WORKSPACE-DRAFT.md`, then run
  `python3 scripts/ci/check-workspace-bootstrap.py --write`, then
  `python3 scripts/ci/check-python-client-identity.py --write`, and rebuild.
  Both gates without `--write` must pass afterwards; the Python test
  `Server/tests/test_python_client_identity.py` checks the second.
- One `CHANGELOG.md` entry under Unreleased per package, in the voice of the
  existing entries.
- A live pass in the app on a scratch workspace (`steerlab-cli workspace init
  <scratch> --json`), never on a real study workspace: open every sheet and
  row you touched at the column's minimum width and confirm nothing truncates,
  every disabled control shows its reason, and the typed fallback still works.

## Reporting

Report per finding id: fixed / skipped with reason / needs a change outside
the owned files (name the file and what it needs). Do not silently widen
scope; a finding whose fix turns out to need an engine or manifest change is
reported, not made.
