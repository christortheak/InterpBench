# Audit 07 — Agents section: Library region, New Agent region (manual + optimize composer), sweep grid

Files read in full (every line): `ModelVariantsPanelView.swift` (1895), `OptimizationComposerView.swift` (1560), `SweepGridView.swift` (126), `SweepPanelSection.swift` (114), `AddConditionEditor.swift` (181), `SeedsListRow.swift` (83). Cross-checked against `InjectionModeControls.swift`, `AgentChips.swift`, `NoticesViews.swift`, `FileReferenceRow.swift`, `JudgeModelPicker.swift`, `ModelJobGPUWarning.swift`, `WorkbenchSection.swift`, `ChatView.swift`, and the ExperimentKit stores/panels they call (`ModelVariantStore`, `FineTuningPanel`, `ExperimentPanel`, `ExperimentStore.sweepGridProblem`, `OptimizationComposer.declare`, `StudyServerJobCoordinator`, `InjectionModeCopy`). Base dir: `<checkout>/Sources/SteerLabApp/`.

## Coverage table

| File | Controls found | With `.help` (own or group) | Missing help | Icon-only missing `accessibilityLabel` |
|---|---|---|---|---|
| ModelVariantsPanelView.swift | 55 (incl. 1 dialog button, 3 DisclosureGroups, `InjectionModeControls` counted once, `NoticesBellButton`/`JudgeModelPicker` counted once each) | 30 | 25 | 1 (`:826` trash) |
| OptimizationComposerView.swift | 42 (5 `InstrumentPathField` instances = 5 pickers + 5 text fields; `JudgeDraftRow` = 8; `ControlScopeControls` = 2) | 26 | 16 | 1 (`:1355` minus.circle — has `.help`, no a11y label) |
| SweepGridView.swift | 0 (read-only grid; every cell `Text` carries `.help`) | — | 0 | 0 |
| SweepPanelSection.swift | 1 | 0 | 1 | 0 |
| AddConditionEditor.swift | 10 (+2 shared `InfoButton`s, not counted) | 5 | 5 (all five have a visible caption instead) | 0 |
| SeedsListRow.swift | 2 | 1 | 1 | 0 |

## Findings

- **[BUG]** `OptimizationComposerView.swift:1075-1121` — `declare(andOptimize:)` / Declare buttons — the layer×alpha grid is never range-validated before the draft study is written. `declareDisabledReason` (`:197-247`) and `declare()` only check that the fields *parse* (`SweepSpecForm.parseNumberList`); the real audit (`ExperimentStore.sweepGridProblem`: fractions in [0,1], alphas > 0, both ascending, max tokens > 0) runs inside `panel.setSweepSpec` — which `OptimizationComposer.declare` (ExperimentKit `OptimizationComposer.swift:915-923`) calls only *after* `ExperimentStore.create` has already saved the manifest. A typed `1.2` fraction, `0` alpha, or a descending ladder passes the enabled Declare button and then fails with "…draft study 'X' was created with its pins but without the declared criterion; declare one on it in Agents → Optimizations, or delete the draft in Studies" — an orphan draft per attempt, and the retry with the same auto-suggested name (`:894-900`) then refuses with "experiment 'X' already exists". `SweepSpecForm.validate(spec)` is public (`ExperimentPanel.swift:3293`) and unused here. Should feed `declareDisabledReason` (visible caption + disabled button) or at minimum run before `OptimizationComposer.declare`.
- **[BUG]** `OptimizationComposerView.swift:1027-1033, 1114-1119` — "Declare & Optimize" — `ExperimentPanel.runSweep` (`ExperimentPanel.swift:1155`) silently `return`s when `localJobs.isSweeping / isRunning / isValidating`. The composer neither disables the button while a sweep is running nor surfaces the refusal: the study is declared, `onDeclared` switches the region to Optimizations with the new run selected, and nothing says the sweep did not start. Optimizations' own button is gated by `sweepDisabledReason` and shows a `ProgressView`; the composer should either share that gate ("a sweep is already running — …") or `runSweep` should `note` the refusal.
- **[BUG]** `ModelVariantsPanelView.swift:725` + `InjectionModeControls.swift:85-90` — "Alpha in residual-norm units" toggle vs per-row "Alpha" field — unit-blind alpha. The agent-level toggle flips the meaning of every injection row's "Alpha" field between norm units and raw activation units, but the row's label is bare "Alpha" and its group `.help` (`InjectionModeCopy.alphaHelp`) states unconditionally "α in units of the layer's residual-stream norm" — wrong whenever the toggle is off. The toggle itself has no `.help`/caption saying what it changes. The label should carry the unit ("Alpha (norm units)" / "Alpha (raw)") and the help should follow the toggle.
- **[DESIGN]** `ModelVariantsPanelView.swift:836-855` ("Add Vector" seeds `alpha: 1`) and `InjectionModeControls.swift:43-49` (Steer/Ablate switch resets α to `2`, comment "α is typically 1–3") vs `OptimizationComposerView.swift:1224-1247` (`AlphaMagnitudeWarning`: α ≥ 1 "injects a vector at least as large as the entire residual stream — almost certainly a typo (0.22, not 2.2)"; composer default ladder 0.05–0.13) — the manual editor and the optimize composer, two tabs of the same "New Agent" region, disagree by an order of magnitude on what a sane α is, and the manual editor's defaults are exactly what the composer flags as a typo (memory: "playground stops unit-blind α=2"). Manual defaults should be norm-unit-scale (e.g. 0.1) when `alphaInNormUnits` is on, or the row should show the same magnitude warning.
- **[DESIGN]** `ModelVariantsPanelView.swift:300-303, 1254-1275` — "Save New Agent" — no name-collision handling. `ModelVariantStore.save` (`ModelVariantStore.swift:662-666`) mints a unique run directory per save, so saving a second agent with an existing name silently creates a duplicate; the Library then shows two rows with identical names distinguishable only by `dateLabel`. No warning, no "rename?" prompt, no de-duplicating suffix. (There is no Duplicate action in this cluster; iterating on an agent means re-saving under the same name, which is exactly the collision path.)
- **[DESIGN]** `OptimizationComposerView.swift:1055-1063` — declare caption "…the dev-prompts and battery files are named by path only (see the note above)" contradicts the note it points at (`:997-1004` `sweepInputPinNote`: "every instrument file is packaged with the study and hash-pinned at freeze (dev prompts, battery, and choice prompts alike)"). The doc comment at `:977-989` records that the "not hash-pinned" claim has been false since 2026-07-20; the declare caption still carries it. Delete the clause.
- **[DESIGN]** `OptimizationComposerView.swift:1027-1041` — "Declare & Optimize" in a server workspace whose server is *unpaired* — the button calls `panel.runSweep` directly, which routes to `StudyServerJobCoordinator.run` and refuses ("…direct runs execute the server-resident copy only. Pair…") via a notice *after* the study has been declared and the region has switched. Optimizations knows this case and shows "Submit Bundle: sweep — in Studies…" instead (`OptimizationRunsView.swift:634-638, 700-711`). The composer should either offer the bundle route or say in the button label/help where the sweep will run (Optimizations labels its button "Optimize on <substrate>"; the composer's says only "start its sweep immediately").
- **[DESIGN]** `ModelVariantsPanelView.swift:498-511` — "Run Robustness Check" — disabled with no explanation when `!hasResolvableRobustnessTarget` (no agent picked, or the picked one no longer resolves). Only the judge precondition (`robustnessJudgeDisabledReason`) is rendered as a visible label; the button has no `.help` at all. Add a caption for the missing-target case (the Library's "Run robustness" button at `:1760-1772` does carry the judge reason in `.help`).
- **[DESIGN]** `ModelVariantsPanelView.swift:741-743` — Temperature `Slider` (0…1.5, step 0.1) inside `LabeledContent("Temperature")` — no numeric readout anywhere; the researcher cannot tell whether the saved agent is at 0.0 or 0.1 without reading the JSON. Add a value label (and `.help`).
- **[DESIGN]** `OptimizationComposerView.swift:1257-1269` — "Selection objective" `Picker` — option labels are ~110 characters ("judge score — paired judging vs baseline (outcome instrument; recommended when the claim is about a substantive outcome)") and two options share the identical parenthetical. In a pop-up picker at the 560pt controls minimum the selected label truncates to the first few words. Keep the labels short ("judge score", "logprob shift", "marker density (smoke test)") and put the recommendation in the existing `.help`/captions.
- **[DESIGN]** `OptimizationComposerView.swift:1337-1363` — `JudgeDraftRow` — one `HStack` holding name field + 110pt kind picker + (for openrouter) two more text fields + remove button, inside a Form whose column minimum is 560pt: the three text fields get ~120pt each, so the required placeholders "Model slug (required)" / "Provider (required)" are cut off. The "kind" picker is `labelsHidden` with no `.help`, so its meaning rests on the option text alone. Consider a two-line row (name + kind / model + provider).
- **[POLISH]** `ModelVariantsPanelView.swift:318-321` — `.help` on "Apply in this workspace" reads "seed the steering controls from this definition; sends compose an inline spec on <server>" — garbled ("sends compose an inline spec").
- **[POLISH]** Old vocabulary in user-visible copy (Agents / Playground are current): `ModelVariantsPanelView.swift:322` "Use in Steering", `:332` "Reset From Steering" (Steering = pre-rename Playground; the neighbouring `:329` help already says "steering controls" which is fine as a noun, but the button titles name the old section), `:342` "save, apply, or reset this agent (variant artifact)", `:761` "No vectors in this variant.", `:75`/`:1151` default agent name "variant-1", `:260` "New… starts from the current steering setup". Notices this view surfaces from ExperimentKit: "deleted model variant …", "select a model variant to delete", "could not delete model variant" (`FineTuningPanel.swift:1502-1510`), "variant '…' selected; load … to apply its adapter and vectors" (`ChatService.swift:1412-1414`). The Robustness section's "baseline vs variant" (`:410, :486, :538, :1837-1858`) uses "variant" as the treatment-arm name — defensible, but it is the same word. No "Screens"/"Concept Lab"/"Geometry" strings remain in these files.
- **[POLISH]** `ModelVariantsPanelView.swift:101-106` — Delete confirmation `confirmationDialog("Delete selected agent?")` — no message and does not name the agent being deleted (`panel.selectedVariant?.artifact.name` is available). The only visible feedback afterwards is the single-slot status line + bell.
- **[POLISH]** `OptimizationComposerView.swift:1027-1041` — Declare row mixes control sizes: two regular-size Declare buttons and a `.controlSize(.small)` "Open Optimizations" in one `HStack`.
- **[POLISH]** `OptimizationComposerView.swift:647` — `criterionJSONError = "\(error)"` renders a raw Swift `DecodingError` (e.g. `keyNotFound(CodingKeys(stringValue: …), Swift.DecodingError.Context(…))`) to the user. Same pattern, lower impact: `:1121` `formError = "\(error)"` is fine for `ExperimentError` (CustomStringConvertible) but raw for the `StimulusSet` IO errors `declare` can throw; `ModelVariantsPanelView.swift:1285` "could not save agent: \(error)"; `:1807` "Couldn't attach the agent: \(error)"; `SeedsListRow.swift:80` "Details: \(error)".
- **[POLISH]** `OptimizationComposerView.swift:579-583` — "Seed from form" — vague label with no object and no `.help` (it overwrites the JSON text with the form's current criterion); "Apply JSON" (`:577`) also has no `.help`, and applying overwrites every form field with no confirmation.
- **[POLISH]** `ModelVariantsPanelView.swift:1539-1568` / `:222-232` — region self-explanation (extra item a): the segmented "Library / New Agent / Optimizations" picker makes the current region obvious and the Library's empty state explains itself, but a populated Library section has no intro line, and "Edit" (`:1746-1750`) opens the definition under a region titled "New Agent" whose section header also says "New Agent" while the Save button reads "Save Changes" — editing an existing agent lives under a heading that says it is new.
- **[POLISH]** `ModelVariantsPanelView.swift:739-740` — "Qwen thinking mode" toggle is disabled for non-Qwen base models with no `.help`/caption; and `:724` "Layer band: N" `Stepper` (1…11, step 2) has no explanation of what a band is (the only mention is buried in `InjectionModeCopy.alphaHelp`: "The layer is widened by the variant's band width").
- **[POLISH]** `ModelVariantsPanelView.swift:613-614` — server agent `Text(variant.path)` `.lineLimit(1).truncationMode(.middle)` with no `.help` carrying the full path (selection is enabled, but a middle-truncated path cannot be read); `:1734` selected-agent name `.lineLimit(1)` likewise.
- **[POLISH]** Layout at the 560pt controls minimum (unverified — not run): `ModelVariantsPanelView.swift:299-341` actions row holds up to four regular-size buttons ("Save Changes", "Apply in this workspace", "Reset From Steering", "Delete…") with no wrapping; `:1728-1778` selected-agent row holds the name plus five small buttons ("Chat", "Edit", "Open optimization run", "Run robustness", "Add to study") where the only compressible element is the name; `:433-472` Prompts/Max-tokens row has three `.fixedSize()` labels/captions (`:438, :451, :455`) — a long coherence-file name in "of N in <file>" cannot compress.
- **[POLISH]** `ModelVariantsPanelView.swift:1582-1601` — agent roster is a `ScrollView`/`LazyVStack` capped at 360pt inside a `.formStyle(.grouped)` Form (itself a scroll view): nested scrolling; trackpad scroll over the roster is captured by the inner list. Deliberate per the comment (constant height, macOS 27 hazard), but the scroll-capture cost stands.
- **[POLISH]** `OptimizationComposerView.swift:575` — `TextEditor.frame(minHeight: 120)` inside a `DisclosureGroup` (appears/disappears) inside the Form in the controls split column. The Form scrolls, so the column's minimum height most likely does not vary (unverified), but this is the shape the beta-fatal class warns about; a `maxHeight`/idealHeight would be safer.
- **[POLISH]** Index/self ids: `OptimizationComposerView.swift:911` `ForEach(… .enumerated(), id: \.offset)` for drift badges; `ModelVariantsPanelView.swift:1877` `ForEach(report.warnings, id: \.self)`; `AddConditionEditor.swift:156` `ForEach(lastControlMatrixNotes, id: \.self)`; `SweepPanelSection.swift:67` `ForEach(resolved.files, id: \.label)`. Display-only lists, so only duplicate strings would misbehave.
- **[POLISH]** `AddConditionEditor.swift:113-123, 142-155` (extra item e) — the editor is inline, not a sheet: there is no Cancel/Clear and no primary styling on "Add Condition" (not `.borderedProminent`, no `.keyboardShortcut(.defaultAction)`); it sits alone in an `HStack` visually detached from "Add Baseline" / "Scaffold Control Matrix". After a successful add only `conditionName` is cleared (`ExperimentPanel.swift:2186`); layer/α persist (useful for ladders, but unstated). The inline refusal rendering (`:135-141`) and stale-refusal clearing are good.
- **[POLISH]** `SeedsListRow.swift:33-36, 50-51` — seeds `TextField` has no `.help` (only "Set" carries `seedsHelp`); the field is seeded `onAppear` and on `manifest.name` change only, so seeds changed by another path on the same draft leave it stale; success feedback is implicit (Set greys out) with no status line.
- **[POLISH]** `SweepPanelSection.swift:26-28` — "Edit sweep spec" (`.buttonStyle(.link)`) — no `.help`; it is the only action in the section.
- **[A11Y]** `ModelVariantsPanelView.swift:826-831` — injection-row trash `Button` (`Image(systemName: "trash")`, `.borderless`) — icon-only with neither `.help` nor `.accessibilityLabel`; removes the row instantly, no undo.
- **[A11Y]** `OptimizationComposerView.swift:1355-1361` — judge remove `Button` (`Image(systemName: "minus.circle")`, `.plain`) — has `.help("remove this judge")` but no `.accessibilityLabel`.
- **[A11Y]** `SweepGridView.swift:64-68` — the winning cell is marked only by a 1.5pt accent-coloured border (plus fill intensity); the `.help` tooltip names the state but there is no non-colour glyph. Struck-through ineligible cells are fine.

## Missing tooltips (exhaustive)

ModelVariantsPanelView.swift
- `ModelVariantsPanelView.swift:105` — "Delete" (confirmation-dialog button; dialog title is the only context)
- `ModelVariantsPanelView.swift:223` — Picker "Creation mode" (explainer text visible below; no `.help`)
- `ModelVariantsPanelView.swift:387` — DisclosureGroup "Files"
- `ModelVariantsPanelView.swift:498` — Button "Run Robustness Check" / "Checking…"
- `ModelVariantsPanelView.swift:666` — TextField "Name"
- `ModelVariantsPanelView.swift:717` — Picker "Adapter"
- `ModelVariantsPanelView.swift:724` — Stepper "Layer band: N"
- `ModelVariantsPanelView.swift:725` — Toggle "Alpha in residual-norm units"
- `ModelVariantsPanelView.swift:727` — Picker "Neutral basis"
- `ModelVariantsPanelView.swift:734` — Picker "Prompt mode"
- `ModelVariantsPanelView.swift:739` — Toggle "Qwen thinking mode"
- `ModelVariantsPanelView.swift:742` — Slider "Temperature"
- `ModelVariantsPanelView.swift:744` — TextField "System prompt"
- `ModelVariantsPanelView.swift:759` — DisclosureGroup "Injections"
- `ModelVariantsPanelView.swift:826` — icon-only trash button (remove injection) — also no accessibilityLabel
- `ModelVariantsPanelView.swift:836` — Button "Add Vector"
- `ModelVariantsPanelView.swift:1558` — Button "New Agent" (empty state)
- `ModelVariantsPanelView.swift:1559` — Button "Optimize a Vector" (empty state)
- `ModelVariantsPanelView.swift:1563` — Button "Train Adapter" (empty state)
- `ModelVariantsPanelView.swift:1588` — roster row Button (`.plain`, selects the agent)
- `ModelVariantsPanelView.swift:1625` — Picker "Base model" (library filter)
- `ModelVariantsPanelView.swift:1632` — Toggle "Sweep-promoted only"
- `ModelVariantsPanelView.swift:1633` — Toggle "Has adapter"
- `ModelVariantsPanelView.swift:1634` — Toggle "Runnable here" (non-obvious: depends on the active substrate)
- `ModelVariantsPanelView.swift:1864` — DisclosureGroup "Judge N: <winner>" (report rows)

OptimizationComposerView.swift
- `OptimizationComposerView.swift:440` — Button "Add judge"
- `OptimizationComposerView.swift:572` — DisclosureGroup "Edit criterion as JSON"
- `OptimizationComposerView.swift:573` — TextEditor (criterion JSON)
- `OptimizationComposerView.swift:577` — Button "Apply JSON"
- `OptimizationComposerView.swift:579` — Button "Seed from form"
- `OptimizationComposerView.swift:735` — Button "Open Data"
- `OptimizationComposerView.swift:941` — TextField "Max tokens per generation"
- `OptimizationComposerView.swift:1022` — TextField "Question or purpose (optional)"
- `OptimizationComposerView.swift:1305` — TextField "<label> — workspace-relative path" (×5 instances: choice prompts per concept `:333`, single `:355`, none `:366`, dev prompts `:945`, battery `:957`; the paired Picker has help, the free-text field does not)
- `OptimizationComposerView.swift:1338` — TextField "Judge name (required)"
- `OptimizationComposerView.swift:1341` — Picker "kind" (`labelsHidden`, no visible label either)
- `OptimizationComposerView.swift:1461` — ArtifactPickRow Toggle (the checkbox itself; only refused/drift captions carry help)

SweepPanelSection.swift
- `SweepPanelSection.swift:26` — Button "Edit sweep spec"

AddConditionEditor.swift (each has a visible caption row instead; listed for completeness)
- `AddConditionEditor.swift:18` — TextField "Condition name" (placeholder "optional"; no caption either)
- `AddConditionEditor.swift:27` — Picker "Concept"
- `AddConditionEditor.swift:61` — TextField "Layer"
- `AddConditionEditor.swift:72` — TextField "Strength (α)"
- `AddConditionEditor.swift:84` — Toggle "Relative strength" (InfoButton beside it)

SeedsListRow.swift
- `SeedsListRow.swift:33` — TextField "seeds (comma-separated, …)" (help is on the "Set" button only)

## Notes on strengths

- The optimize composer is unusually honest: every disabled state of Declare is written out as a visible caption (`declareDisabledReason`), instruments are explained in one breath (`instrumentRolesCaption`), the three constraint fields have genuinely explanatory help, and the alpha-magnitude warning is exactly the unit guard the checklist asks for — on this screen the "Alphas (norm units, …)" field is not unit-blind.
- The Robustness section renders its judge-precondition refusal inline instead of a mute grey button, states prefix truncation out loud, and its local "Stop" explains the no-partial-report rule.
- `SweepGridView` handles ineligible cells correctly (measured but struck through, never hidden) and states a control's absence as a fact rather than a blank; `SweepPanelSection` does the same for collapsed fractions.
- The injection picker keeps unresolvable refs rendered and non-pickable rather than silently blanking a binding, and the availability caption names the filter inputs when the list looks thin.
