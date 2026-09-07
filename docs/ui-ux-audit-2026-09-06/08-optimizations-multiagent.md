# 08 — Agents › Optimizations region + Multi-Agent section (scenario/protocol/pipeline builders)

Files read in full: `OptimizationRunsView.swift` (2214), `MultiAgentPanelView.swift` (1104), `PipelineComposerView.swift` (517), `StudyPipelinesView.swift` (171). Cross-checked against `SweepGridView.swift` (embedded "Measured grid"), `ExperimentKit/MultiAgent/MultiAgentPanel.swift`, `ExperimentKit/ExperimentPanel.swift`, `ExperimentKit/StudyServerJobCoordinator.swift`, `ExperimentKit/StudyPipelineController.swift`, `ExperimentKit/PipelineState.swift`, `InfoPopover.swift`, and the host call sites (`ModelVariantsPanelView.swift:177`, `ChatView.swift:126`, `ExperimentsPanelView.swift:122,313`). Read-only; no edits, no builds.

## Coverage table

| File | Controls found | With `.help` (own or group) | Missing `.help` | Icon-only missing `accessibilityLabel` |
|---|---|---|---|---|
| `OptimizationRunsView.swift` | 32 | 20 | 12 | 0 (no icon-only controls) |
| `MultiAgentPanelView.swift` | 53 | 16 | 37 | 0 (Move Up/Down, Add Seat/Turn, Run/Stop are `Label` with text) |
| `PipelineComposerView.swift` | 15 (+3 shared `InfoButton`) | 6 | 9 | 0 (`InfoButton` carries `accessibilityLabel("Explain")`, InfoPopover.swift:25, but no `.help`) |
| `StudyPipelinesView.swift` | 2 | 2 | 0 | 0 |

Counts exclude DisclosureGroup headers and the sub-views defined in other files (`SweepPanelSection`, `SweepDoseMonotonicityView`, `ControlScopeControls`, `AlphaMagnitudeWarning`, `FileReferenceRow`, `ExecutionPlanEchoRow`), which belong to other clusters. `SweepGridView` cells do carry `.help` (SweepGridView.swift:69).

## Findings

- **[BUG]** `MultiAgentPanelView.swift:47` — Scenario `Picker` — switching scenarios silently discards unsaved edits. The binding sets `panel.selectedScenarioID`, whose `didSet` (MultiAgentPanel.swift:29-35) calls `loadEditorFromSelection()` (:764-785), which overwrites name/description/materials/agents/turns with no `hasUnsavedChanges` check. Every other entry point guards and confirms ("New Scenario" :78-86, "New Contract Panel" :88-92, "Start from this protocol" :410-414, each with a confirmationDialog). Additionally the picker's "New…" entry (`tag(String?.none)`, :53) is a no-op: `loadEditorFromSelection` returns early on nil, so the picker reads "New…" while the editor still shows the previous scenario. Should: route the picker's setter through the same discard-confirm path (hold the pending id like `pendingProtocol`), and make "New…" call `newScenario(discardingChanges: false)` or drop it.
- **[BUG]** `PipelineComposerView.swift:191,380-387` — "Declare pipeline" `Toggle` — flipping it OFF calls `saveDeclaration(nil)` immediately, which writes the manifest on disk (`StudyPipelineController.saveDeclaration`, :63-72, notice "pipeline declaration removed"). No confirmation, no `role: .destructive`, and the stages/gates the researcher declared vanish from the draft on a single stray click. Should: confirmationDialog before removal, or let the toggle only collapse the editor and require "Save Pipeline" to persist the removal.
- **[BUG]** `OptimizationRunsView.swift:652-668` — "Optimize on \<substrate\>" — no re-entry guard on the server route. `sweepDisabledReason` (:715-738) checks only `localJobs.isSweeping/isRunning/isValidating`; `ExperimentPanel.runSweep` (ExperimentPanel.swift:1155) guards the same local flags; `StudyServerJobCoordinator.run` (:14-35) has no `activeSweepJob` guard. While a server sweep is in flight the button stays enabled and each click submits another durable sweep job; the UI simultaneously shows "Optimize on X" and "Cancel Optimization" (:674, which appears exactly because `activeSweepJob != nil`). The spinner (:669-672) is also local-only. Should: add `panel.remoteJobs.activeSweepJob != nil` to `sweepDisabledReason` ("a sweep job is running on \<substrate\> — follow it in the activity pane") and show the `ProgressView` for that state.
- **[BUG]** `PipelineComposerView.swift:156-177` — "Run Draft Pipeline (exploratory)" — runs the SAVED `manifest.pipeline` while the editor may hold unsaved stage/gate edits: there is no dirty tracking (`draft` is never compared to `PipelineDraft.parse(manifest.pipeline)`), and the button is gated on `savedViolations` only. This is the exact class the sweep editor closed ("Finding 5", OptimizationRunsView.swift:1590-1594 disables Optimize while `isDirty` and says why). Should: compute `isDirty`, disable Run with the "unsaved edits — Run executes the SAVED chain" caption.
- **[BUG]** `PipelineComposerView.swift:176` — Run button `.disabled(!savedViolations.isEmpty)` with no visible reason. The red list (:267-271) renders `draftViolations`, which can be empty after the researcher fixes the draft but before saving, leaving a grey Run with no explanation. Show the saved block's violations (or "save the pipeline first") next to the button.
- **[BUG]** `OptimizationRunsView.swift:2187` — `OverridePromotionSheet` body text — `"Cell L\(promotion.layer) α\(promotion.alpha)"` interpolates a raw `Double`, unlike every other alpha in the file (`format`, :1392, 0–3 fraction digits). Can render `α0.30000000000000004`-style values or disagree with the grid's `0.13`. Use the same formatter.
- **[DESIGN]** `OptimizationRunsView.swift:1339-1357` with `:1899` and `:168-175` — Create Agent outcome/refusal placement — `promoteFromDisplayedRun` refuses via `panel.refuse(.sweepSpec, …)` (ExperimentPanel.swift:154-160 → `draft.formErrors[.sweepSpec]` + notice). `formErrors[.sweepSpec]` is rendered only inside `SweepSpecEditorSection.saveControls` (:1899), which exists only for local DRAFT optimizations; promotion normally happens on frozen studies, where the message lands only in the bottom `panel.status` Section (many hundreds of points below the button) and the bell. Success ("promoted 'X' → agent 'Y'…", ExperimentPanel.swift:1024) is likewise a notice only — nothing beside the button confirms, and there is no "Open in Library" affordance. Server-route promotion (`Task { await promoteOnActiveServer }`, ExperimentPanel.swift:1008) has no busy indicator and no re-entry guard.
- **[DESIGN]** `OptimizationRunsView.swift:863` vs `:1234` — two grids of the same run's cells on one screen: "Optimization grid — \<run\>" (clickable; green=winner, red tint=fails, grey=pass; legend :1124-1130) and "Measured grid — \<run\>" (`SweepGridView`: accent-intensity heatmap, accent border=winner, strikethrough=fails; NO legend for shading, border or strikethrough — only the per-cell tooltip, SweepGridView.swift:80-99). Same numbers, two colour vocabularies. Consolidate into one grid (heatmap + selection affordance) and give it one legend.
- **[DESIGN]** `OptimizationRunsView.swift:994` and `SweepGridView.swift:36` — grid axis header `"L \ α"` — alpha units are stated in the spec fields ("Alphas (norm units, comma-separated)", :1603; "Alphas (norm units)", :583) but nowhere on either grid or its Section title. A reader of the grid alone cannot tell 0.13 is a residual-norm fraction. Add "(α in norm units)" to the Section title or the `objective:` caption (SweepGridView.swift:27).
- **[DESIGN]** `OptimizationRunsView.swift` (0 `InfoButton` uses), `MultiAgentPanelView.swift` (0 uses) vs `PipelineComposerView.swift:203,216,315` — the ⓘ pattern the composer originated is not used by its siblings. The optimizations panel carries equally dense concepts only as hover tooltips and orange captions (relative vs absolute coherence floor :1761-1772, matched-norm control :1773-1777, judge pins :1838-1857, "declared ahead" objectives :1695-1699, lifecycle chips :838-848 whose `.help` is the empty string for known states); the multi-agent editor inlines long paragraphs as `.caption` text (:141-146, :174-178, :636-648). Recommend `InfoButton` beside "Declared criterion", "Optimization (sweep) spec", "Lifecycle", "Rehearsal" and "Turn Script" headers.
- **[DESIGN]** `MultiAgentPanelView.swift:182-208` — Turn Script — no way to reorder turns. Seats have Move Up/Down (:515-526, `moveSeat`); `MultiAgentPanel` has no `moveTurn` (verified: only `agents.swapAt` at MultiAgentPanel.swift:463). Turn order is semantic — contract inputs must come from EARLIER turns and the editor itself warns "no earlier turn produces this; validate refuses it" (:960-967). The only fix today is Remove Turn + re-add. Add Move Up/Down for turns.
- **[DESIGN]** `MultiAgentPanelView.swift:551-560` — Rehearsal "Run" — disabled with no visible reason for three of its four gates (`agents.isEmpty`, `turns.isEmpty`, `rehearsalModelID.isEmpty`); only `runUnavailableReason` (server target) is shown (:565-569). On a fresh local workspace with no models the "Rehearsal model" picker (:585-598) renders with zero options and Run is grey under the tooltip "Play the scripted scenario locally." Provide one computed reason string like `sweepDisabledReason`.
- **[DESIGN]** `MultiAgentPanelView.swift:97-102` — "Save Scenario" — disabled with no visible reason when the name is blank (the legacy case has the banner). Add a caption or `.help`.
- **[DESIGN]** `PipelineComposerView.swift:66` — "Discard pipeline edits and reload" — discards unsaved edits with no confirmation, no `role`, and sits as the FIRST control in the section, above the editor, where the primary action is expected. Move below "Save Pipeline"; confirm when dirty (once dirty tracking exists).
- **[DESIGN]** `PipelineComposerView.swift:301-328` — promotion-rule fields bind to `panel.draft` (shared study draft) while pipeline fields bind to the local `draft`; two Save buttons in one section ("Save Pipeline" :272, "Save Promotion Rule" :323) and the Discard button resets only one of them. No caption says which fields each Save covers. Give the rule its own sub-header ("Promotion rule" + its `InfoButton`) or one Save.
- **[DESIGN]** `MultiAgentPanelView.swift:876-879` — free-template `TextEditor` — the variable vocabulary ("Available variables: {{scenario.materials}}, {{agent.name}}, …") exists ONLY as a hover tooltip on the editor; it is the one thing a template author must know. Make it a visible caption (the contract editor does this for its task field's substitutions in a tooltip too, :916-918).
- **[DESIGN]** `MultiAgentPanelView.swift:542` vs `:1050` — "Rehearsal" holds Run/Stop; outputs appear under "Live Run" (empty state "Run a scenario to watch turn outputs appear here."). Two names for one activity. Rename "Live Run" → "Rehearsal transcript" or merge.
- **[DESIGN]** `OptimizationRunsView.swift:2093-2103` — "Selection objective" `Picker` options are full sentences, two of them identical ("… — outcome instrument (recommended when the claim is about a substantive outcome)"), inside a 440pt sheet. Menu items will be very wide or truncated; `objectiveCaption` (:2110-2129) already exists for the recommendation text.
- **[DESIGN]** `OptimizationRunsView.swift:1901-1903` — Save refusal "The displayed sweep is stale or unavailable. Reopen the editor from the intended draft." names a repair the UI does not offer: there is no reload button and the editor is keyed by `.id(optimization.id)` (:570), so re-selecting the same run does not reseed. Add "Reload spec", mirroring the composer's :66.
- **[DESIGN]** `OptimizationRunsView.swift:410-437` — optimization row `.help` — every row shows the same concept paragraph ("experiments whose manifest declares a sweep spec … not a new object type") instead of the action ("select this optimization run"). Move the concept to the section caption/InfoButton.
- **[DESIGN]** `OptimizationRunsView.swift:1193` and `:1295` — "Create Agent" appears twice on one screen for the same concept (grid-selected winner and recommendation row), both calling `promoteFromDisplayedRun`. Consider keeping the grid button only for override cells.
- **[DESIGN]** `OptimizationRunsView.swift:2044`, `:2201` — sheet "Cancel" buttons lack `.keyboardShortcut(.cancelAction)` (Esc); MultiAgentPanelView's sheets do it (:429-430, :460-461).
- **[DESIGN]** `StudyPipelinesView.swift:32,40` — `.prefix(20)` / `.prefix(10)` silently truncate the pipeline lists; no "N more" indicator.
- **[DESIGN]** Label casing inconsistent within one file. `MultiAgentPanelView.swift`: "New Scenario"/"Save Scenario"/"Add Seat"/"Remove Turn" (Title Case) vs "New from protocol template…"/"Save as protocol template…"/"Save protocol"/"Migrate to an environment…"/"Start from this protocol"/"Use the run's budget" (sentence case). `PipelineComposerView.swift`: "Save Pipeline"/"Save Promotion Rule" vs "Discard pipeline edits and reload".
- **[A11Y]** `OptimizationRunsView.swift:1116-1121` — clickable grid — "fails constraint" is a red tint only (no glyph, no strikethrough; the tooltip carries it). `SweepGridView` uses strikethrough for the same state (SweepGridView.swift:59). Adopt strikethrough or a "✕" in the clickable grid so the state survives colour loss.
- **[A11Y]** `OptimizationRunsView.swift:1044` — "winner" badge at `.system(size: 8, weight: .bold)` — below legible size.
- **[POLISH]** `StudyPipelinesView.swift:76` — `"last ledger write: \(updated)"` — `updatedAt` is `String?` (PipelineState.swift:93); the raw server timestamp is shown with no date formatter or relative time.
- **[POLISH]** `StudyPipelinesView.swift:53-56` — run name `.lineLimit(1)` with no `.help` carrying the full name.
- **[POLISH]** `MultiAgentPanelView.swift` — `.help` strings are sentence case with trailing periods throughout (:93, :112, :122, :502, :510, :560, :758, :852, :879, :908, :916, :928, :932, :1003), unlike the lowercase-fragment convention followed by the other three files.
- **[POLISH]** `OptimizationRunsView.swift:847` — lifecycle chip `.help(state == nil ? "not derivable from the server API" : "")` — empty tooltip for the common case; known states ("Optimized", "Recommended") are never explained.
- **[POLISH]** `OptimizationRunsView.swift:1615` — "Max tokens per generation" is the only spec field without `.help`.
- **[POLISH]** `OptimizationRunsView.swift:674-677` — "Cancel Optimization" — after a local cancel request the button greys out with no "cancelling…" indicator (server route posts a notice; local only flips `sweepCancelRequested`).
- **[POLISH]** `OptimizationRunsView.swift:418` — row title `"\(name) · \(statusLabel)"` shows raw enum values (`draft`/`frozen`/`complete`) and the literal `?` for a server record with no status (:121). Say "status unknown".
- **[POLISH]** `MultiAgentPanelView.swift:800-811` — per-turn "Max tokens" field displays the rehearsal budget when the turn has none, so "no override" and "overridden to the same number" look identical; the only cue is the "Use the run's budget" button appearing. Use a placeholder ("run's budget: 512") instead of a value.
- **[POLISH]** `MultiAgentPanelView.swift:528`, `:820` — "Remove Seat"/"Remove Turn" — `role: .destructive` present, but no confirmation and no undo in a Form; a turn's whole contract or a seat's system prompt goes in one click. Consider confirmation when the item is non-empty.
- **[POLISH]** `MultiAgentPanelView.swift:228` (via `deleteSelectedScenario`, MultiAgentPanel.swift:425) and `:97` (via `saveScenario`, :412) — failures set `status = "\(error)"`, a raw error dump. `PipelineComposerView.swift:514` likewise: "Could not read the pipeline declaration: \(error)".
- **[POLISH]** `StudyPipelinesView.swift:17-19` — "Refresh Pipelines" — async with no busy indicator. (Safe: the controller uses a generation token, :39-40, so the last request wins.)
- **[POLISH]** `OptimizationRunsView.swift:1211`, `:2191` — "promotedBy: manualOverride" and "promotedBy: criterion" (:1219) expose the manifest key verbatim in user-facing copy; acceptable for the audience but worth a plain-words half ("recorded as a manual override").

Checked and NOT a finding: no old section names ("Screens", "Variants", "Concept Lab", "Geometry", "Steering") in any user-visible string across the four files; "screen" appears only as the funnel-stage term (screen→confirm) at :433, :802 and PipelineComposerView.swift:325. Every referenced UI location exists ("Remote options", "Study Focus", "Studies › Evaluation", "Install model… (Compute menu)", "activity pane", "Submit Bundle", "Pipelines"). No `.frame(minHeight:)` sits at split-column level: all fixed minimums (MultiAgentPanelView.swift:153, 420, 457, 878, 915, 925; prompt preview constant `height: 240` at :1014) are inside the scrolling grouped `Form` or inside sheets, so the column minimum does not vary with state. The "Create Agent" basis is explained (provenance line :1370-1387, control line :1389-1394, dose-monotonicity view :1284-1291, button help :1359-1367). Multi-agent buttons are all verb-labelled; seat→agent casting is deliberately absent and the section says so (:174-178). Bundle submission (`StudyBundleSubmissionController.submit`) goes through an `admission()` guard (:56, :66), so Run Pipeline double-submit appears covered (unverified beyond the guard's presence).

## Missing tooltips (exhaustive)

`OptimizationRunsView.swift:652` — "Optimize" / "Optimize on \<substrate\>" (visible caption below instead)
`OptimizationRunsView.swift:701` — "Submit Bundle: sweep — in Studies…" (visible caption below instead)
`OptimizationRunsView.swift:1615` — TextField "Max tokens per generation"
`OptimizationRunsView.swift:1891` — "Save Sweep Spec" / "Save Sweep Spec (unsaved changes)"
`OptimizationRunsView.swift:2044` — "Cancel" (Declare an Optimization sheet)
`OptimizationRunsView.swift:2060` — "Create a draft in Studies…"
`OptimizationRunsView.swift:2065` — Picker "Draft study"
`OptimizationRunsView.swift:2073` — "Declare Optimization"
`OptimizationRunsView.swift:2080` — "New draft in Studies…"
`OptimizationRunsView.swift:2196` — TextField "Reason (required — stamped into the birth certificate)"
`OptimizationRunsView.swift:2201` — "Cancel" (Manual override sheet)
`OptimizationRunsView.swift:2203` — "Create Agent with override"
`MultiAgentPanelView.swift:47` — Picker "Scenario"
`MultiAgentPanelView.swift:78` — "New Scenario"
`MultiAgentPanelView.swift:97` — "Save Scenario"
`MultiAgentPanelView.swift:103` — "Delete…"
`MultiAgentPanelView.swift:135` — TextField "Name"
`MultiAgentPanelView.swift:137` — TextField "Description"
`MultiAgentPanelView.swift:151` — TextEditor (Shared Materials)
`MultiAgentPanelView.swift:168` — "Add Seat"
`MultiAgentPanelView.swift:192` — "Add Turn"
`MultiAgentPanelView.swift:311` — Toggle per materials-checklist item (group caption below, no `.help`)
`MultiAgentPanelView.swift:410` — "Start from this protocol"
`MultiAgentPanelView.swift:429` — "Cancel" (protocol picker sheet)
`MultiAgentPanelView.swift:449` — TextField "Protocol name"
`MultiAgentPanelView.swift:455` — TextEditor (materials checklist draft)
`MultiAgentPanelView.swift:460` — "Cancel" (save protocol sheet)
`MultiAgentPanelView.swift:462` — "Save protocol"
`MultiAgentPanelView.swift:491` — TextField "Role name"
`MultiAgentPanelView.swift:515` — "Move Up"
`MultiAgentPanelView.swift:521` — "Move Down"
`MultiAgentPanelView.swift:528` — "Remove Seat"
`MultiAgentPanelView.swift:545` — "Stop"
`MultiAgentPanelView.swift:614` — TextField "Max tokens" (rehearsal)
`MultiAgentPanelView.swift:622` — Slider (rehearsal temperature)
`MultiAgentPanelView.swift:670` — "Migrate to an environment…" (banner text explains; no `.help`)
`MultiAgentPanelView.swift:731` — "Dismiss" (migration report)
`MultiAgentPanelView.swift:749` — TextField "Title" (turn)
`MultiAgentPanelView.swift:751` — Picker "Speaking seat"
`MultiAgentPanelView.swift:760` — Picker "Route output to"
`MultiAgentPanelView.swift:772` — Toggle per seat "Seats that see this output"
`MultiAgentPanelView.swift:789` — Toggle "Include shared materials"
`MultiAgentPanelView.swift:790` — Toggle "Include speaker context"
`MultiAgentPanelView.swift:800` — TextField "Max tokens" (per turn)
`MultiAgentPanelView.swift:808` — "Use the run's budget"
`MultiAgentPanelView.swift:820` — "Remove Turn"
`MultiAgentPanelView.swift:864` — "Dismiss" (turn notices)
`MultiAgentPanelView.swift:923` — TextEditor (contract output format; caption above, no `.help`)
`MultiAgentPanelView.swift:947` — Toggle per available input "Earlier outputs to show this speaker"
`PipelineComposerView.swift:66` — "Discard pipeline edits and reload"
`PipelineComposerView.swift:368` (via :429) — TextField "Validation floor" (caption below, no `.help`)
`PipelineComposerView.swift:368` (via :438) — TextField "Declared minimum" (caption below, no `.help`)
`PipelineComposerView.swift:368` (via :223) — TextField "Distinctness cap" (caption below, no `.help`)
`PipelineComposerView.swift:272` — "Save Pipeline"
`PipelineComposerView.swift:305` — TextField "promotion FDR threshold (e.g. 0.05)"
`PipelineComposerView.swift:309` — Toggle "dose-monotone"
`PipelineComposerView.swift:311` — Toggle "exceeds random floor"
`PipelineComposerView.swift:318` — TextField "capability gate (free text, …)"

## Notes on strengths

- Disabled-state honesty is a deliberate pattern here: `sweepDisabledReason` (OptimizationRunsView.swift:715-738) states why Optimize is grey in one sentence, and the sweep editor makes Optimize unreachable while edits are unsaved and says so (:1590-1594, :1905-1911).
- Discard/delete flows in MultiAgentPanelView all confirm with a message that names what is lost (:230-286), and the migration flow explains that the original file's bytes are untouched (:254-268).
- The recommendation row explains its basis fully (winner cell, criterion, metric, dev-split hash, control margin, dose monotonicity) before offering Create Agent; the override path demands a written reason and says what stamp results (:2181-2212).
- Pipeline gates are explained in plain words via InfoButton with a precise "no gates" message that distinguishes "none declared" from "none apply" (PipelineComposerView.swift:246-266).
