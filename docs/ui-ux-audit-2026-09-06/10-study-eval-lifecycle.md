# Audit 10 — Studies evaluation & judging, study lifecycle controls (validate → freeze → run → live → results)

Files read completely (4,041 lines): StudyEvaluationSection, JudgingSectionView, JudgeModelPicker, ExclusionRulesEditorView, ValidateStudyButtonRow, ValidationDepthView, EvaluationClarityViews, StudyDesignActionsView, StudyPreparationControlsView, StudyFreezeControlsView, StudyRunControlsView, StudyLiveRunView, StudyRecentJobsView, StudyServerRunsView, StudyResultsView, StudyResultReviewWindow, ExecutionPlanEchoRow, EvidenceCustodyView, NumericParserPickerView, OrdinalScaleInstrumentView. Cross-checked: StudyControlCopy.swift, InfoPopover.swift (StudyInfo), ExperimentsPanelView.swift (hosting: everything sits inside a scrolling `Form`, L49), ExperimentKit/SubstrateRouting.swift (SubstrateRouting + FreezeRouting), ExperimentKit/UnifiedStudyRunner.swift, ExperimentKit/ExperimentPanel.swift (validateStudy, cancelRemoteJob, saveProtocol, residencyCalloutMessage, pairedJudgeDisabledReason), ExperimentKit/StudyRemoteJobController.swift, ExperimentKit/StudyLocalJobController.swift, ExperimentKit/StudySubmissionOptions.swift, ExperimentKit/StudySubmissionPresentation.swift, ExperimentKit/ExecutionPlanEcho.swift, ModelJobGPUWarning.swift, WorkspaceFileChooser.swift, SectionContainers.swift (key rows).

## Coverage table

InfoButtons are counted as help, not as controls needing help. Confirmation-dialog buttons and Menu entries are excluded from control counts (noted separately). "Group" = a `.help` on an enclosing HStack/LabeledContent that covers the control.

| File | Controls | With .help (own/group) | Missing help | Icon-only w/o accessibilityLabel |
|---|---|---|---|---|
| StudyEvaluationSection.swift | 5 | 5 | 0 | 0 |
| JudgingSectionView.swift | 14 | 9 | 5 | 1 (trash, L224) |
| JudgeModelPicker.swift | 3 | 3 (2 group) | 0 | 0 |
| ExclusionRulesEditorView.swift | 7 (+3 menu entries) | 5 | 2 | 0 |
| ValidateStudyButtonRow.swift | 2 | 2 | 0 | 0 |
| ValidationDepthView.swift | 3 | 1 | 2 | 0 |
| EvaluationClarityViews.swift | 7 (+4 menu entries) | 7 (1 group) | 0 | 1 (chevron menu, L253; plus WorkspacePathChooseButton defined in WorkspaceFileChooser.swift:127) |
| StudyDesignActionsView.swift | 3 | 2 | 1 | 0 |
| StudyPreparationControlsView.swift | 3 | 3 | 0 | 0 |
| StudyFreezeControlsView.swift | 2 | 2 | 0 | 0 |
| StudyRunControlsView.swift | 18 | 12 | 6 (only the DisclosureGroup's generic tooltip covers them) | 0 |
| StudyLiveRunView.swift | 3 (+2 DisclosureGroups, both with help) | 3 | 0 | 0 |
| StudyRecentJobsView.swift | 3 | 3 | 0 | 0 |
| StudyServerRunsView.swift | 1 | 1 | 0 | 0 |
| StudyResultsView.swift | 9 | 0 | 9 | 0 |
| StudyResultReviewWindow.swift | 0 (1 DisclosureGroup) | — | — | 0 |
| ExecutionPlanEchoRow.swift | 1 | 0 | 1 | 0 |
| EvidenceCustodyView.swift | 2 | 0 | 2 | 0 |
| NumericParserPickerView.swift | 2 | 2 | 0 | 0 |
| OrdinalScaleInstrumentView.swift | 2 | 2 | 0 | 0 |
| **Total** | **90** | **62** | **28** | **2 (+1 cross-file)** |

## Extra-item verdicts for this cluster

- **(a) Freeze.** The button confirms (`confirmationDialog`, destructive "Freeze"/"Freeze on <site>", one-way stated in the title). There is **no `--force` option in the app** — `StudyControlCopy.remoteFreezeHelp` says so explicitly ("No force option in the app — forcing requires the CLI") and `freezeHelp` names "CLI: freeze --force skips validation gates"; grep confirms no force path in StudyFreezeControlsView or StudyFreezeController wiring. Gap: the confirmation does not restate the current unmet gates/advisories (finding 8/9).
- **(b) Validate.** Depth picker has a long `.help` and a "declared: …" status line (no ⓘ, unlike sibling declarations). Disabled state has no visible reason (finding 10). Server-routed validate has no busy state / re-entry guard (finding 4). Refusal display lives in StudyIssuesSection / panel notes (outside this cluster).
- **(c) Run/submit.** Local-vs-server routing is very clear (segmented "Run on", button label "Run Study" vs "Run on <site>: <verb>", ExecutionPlanEchoRow). Double-submit guard present (`runner.isSubmitting` in `busy`; `guard !isSubmitting` in UnifiedStudyRunner.submitBundle:63). No submit confirmation except when the GPU gate fires (accepted — the echo row is the pre-statement). **Walltime/gres fields have no labels or units (finding 6).** Post-submit: status line names the job id ("… job <id>, following in the activity pane") and `LabeledContent("Remote job")`, but there is no link to Compute (finding 13).
- **(d) Live run.** Progress (spinner + word count + streaming output), Stop with `role: .destructive` but no confirmation (cooperative stop, partial artifacts retained — acceptable, noted). No stale/disconnected state for the remote log (finding 21).
- **(e) Judging.** Judge model picker present (pinned rows + ad-hoc fallback), key state badges with correct "Compute section" pointers (verified: `ClaudeAPIKeyRow`/`ExternalJudgeKeyRow` at SectionContainers.swift:156-157). **Noncompliance policy is not surfaced anywhere in the app (finding 7).** Exclusion rules editor: add (menu, disables already-declared rules), remove, live validation reusing engine violations, attention-check preflight — good.
- **(f) Results review window.** Navigation: none (single scroll list, no prev/next/filter/search). Copy: `textSelection` on every text block, no copy-all. **Close: none (finding 1).**
- **(g) Old names.** "Submit Bundle" (retired button) in the residency callout; "variant" in robustness copy and freezeHelp; `claude` judge described as a current kind though no longer offered. Details in findings 11, 16, 18.

## Findings

- **[BUG]** `StudyResultReviewWindow.swift:24-61` — `ResultReviewWindow` sheet (presented by `StudyResultsView.swift:13` via `.sheet(item:)`) — has no Close/Done button and no `.keyboardShortcut(.cancelAction)`; the body is title + Divider + ScrollView only. On macOS a SwiftUI sheet is not reliably Escape-dismissible without a cancelAction button, so "Review Responses"/"Review Judge Responses" can trap the user in a modal (verify at runtime; if Escape happens to work it is still undiscoverable). — Add a "Close" button in the header HStack with `.keyboardShortcut(.cancelAction)`.
- **[BUG]** `StudyRunControlsView.swift:451` — Button "Cancel Job" — cancels the durable server job (`panel.cancelRemoteJob()` → `client.cancelJob`, StudyRemoteJobActions.swift:56-66) with no confirmation, no `role: .destructive`, no `.help`; a queued Slurm job irreversibly loses its queue slot. Feedback is a status line only. — Confirmation dialog naming the job id/verb/site ("Cancel job <id> (<verb>) on <site>?"), destructive role, help text.
- **[BUG]** `ValidationDepthView.swift:98-135` — Picker "Validation read depth" `.onChange(of: mode)` fires on the programmatic `syncFromManifest()` (L98/L99) as well as user picks. Switching from a study that declares a layer/fraction to one on the default rule sets `mode = .defaultRule`, which enters the `.defaultRule` branch (L121-127) and calls `ExperimentStore.setValidationReadDepth(experimentName:)` on the NEWLY selected study: a needless manifest rewrite for a draft, and for a frozen study a thrown draft-gate error (`ManifestMutationPolicy.admitDraftEdit`, ExperimentStore.swift:817) rendered as "Couldn't set the validation read depth — the study must still be a draft…" (L240-244) though the user only clicked a different study. The view is shown for frozen studies too (StudyPreparationControlsView.swift:40-42 has no status gate). Reasoned from code, not runtime-verified. — Write from the Picker's binding setter (user action) instead of `onChange`, or guard the handler with a `syncing` flag.
- **[BUG]** `ValidateStudyButtonRow.swift:39-63` — Button "Validate Study" — in a server workspace the busy state and re-entry guard are absent: `disabled` reads `panel.localJobs.isValidating`, which only the LOCAL controller sets (StudyLocalJobController.swift:226); `panel.validateStudy()` routes server workspaces to `submitSelectedStudyRemotely`/`runStudyOnActiveServer` (ExperimentPanel.swift:2539-2544), neither of which sets a flag this row reads, and the spinner/Stop row is explicitly hidden for server workspaces (L54). The label never becomes "Validating Study…" and a second click packages/uploads/submits a second bundle job (no `isSubmitting` exists in StudyBundleSubmissionController — grep). — Consult the bundle/server-execution in-flight state the way the Run button consults `runner.isSubmitting`.
- **[DESIGN]** `StudyRunControlsView.swift:388-391` — TextFields "GPU gres" and "walltime" — no label, no units, no format hint, no own `.help`. Because the defaults are non-empty ("A100", "04:00:00" — StudySubmissionOptions.swift:14,20), the placeholders never show: the user sees two bare boxes reading `A100` and `04:00:00` with nothing saying which is GPU type and which is walltime, nor whether walltime is HH:MM:SS or D-HH:MM:SS. Extra item (c) fails here. — `LabeledContent("GPU type (gres)")` / `LabeledContent("Walltime (HH:MM:SS)")` with help naming the Slurm format and where the site's GPU vocabulary comes from.
- **[DESIGN]** `StudyRunControlsView.swift:379-382` — Picker "Executor" ("local"/"slurm") — no `.help`; "local" here means "inside the server's controller process", two rows under a "Run on" picker whose other arm is "This Mac". Easy to misread as running locally. — Rename options ("controller (no scheduler)" / "Slurm batch job") or add help; the GPU gate's fix path already exists for the wrong choice, but the choice should be legible first.
- **[DESIGN]** `JudgingSectionView.swift:66-170` — judging section — the judge noncompliance policy (noncompliant judge answers become recorded rows, runs complete, capped) is stated nowhere in the app: no "noncompli" string exists in SteerLabApp (grep), only the record contract in ExperimentKit/StudyRecordContracts.swift:834-880. A researcher pinning external judges never learns what happens when a judge refuses/garbles. — One sentence under "Pinned judges — …" (L116-123) or in `StudyInfo.judges`.
- **[DESIGN]** `StudyFreezeControlsView.swift:42-53` — Freeze `confirmationDialog` — title-only (no `message`, no `titleVisibility: .visible`), and it does not restate the readiness state at the moment of the click: with `coordinator.freezeReadiness.ready == false` the button is still enabled (by design — the gate refuses), so the user confirms a one-way action and then reads a refusal. The forced-override dialog in StudyRunControlsView.swift:295-311 restates its failing checks; freeze should restate unmet gates + prominent advisories (`FreezeRouting.present`) in the message.
- **[DESIGN]** `StudyFreezeControlsView.swift:120-131` — readiness Label — `displayLine()` (ExperimentStore.swift:6027-6036) shows at most 3 unmet gates then "+N more"; the full list is only in the hover `.help`. A gate the user must satisfy before a one-way action must not live only in a tooltip. — Render `unmetGates` as rows like `freezeAdvisoryRows`.
- **[DESIGN]** `ValidateStudyButtonRow.swift:39-52` + `StudyPreparationControlsView.swift:45-72` — Button "Validate Study" — disabled when `!panel.violations.isEmpty`, `isRunning`, `isExtracting` or `missingOnServer`, but only the last has a visible explanation (the callout). The neighbouring Extract button renders `ExperimentPanel.extractDisabledReason` (L103-133). — Add a `validateDisabledReason` caption in the same idiom.
- **[DESIGN]** `StudyPreparationControlsView.swift:159-162` (text from `ExperimentPanel.residencyCalloutMessage`, ExperimentPanel.swift:393-397) — residency callout — says "Submit Bundle sends a portable copy", but no "Submit Bundle" button exists in Studies any more (StudyRunControlsView.swift:332-337: "No submit button lives here: the ONE Run control above is the submission path"). Old name in user-visible copy. — "Run on <site> (Remote options) submits a portable hash-pinned bundle".
- **[DESIGN]** `JudgingSectionView.swift:440-445` vs `456-460` — `judgeRevisionHelp` says "Filled from the ACTIVE compute substrate's model cache — Resolve re-reads it", but Resolve is `.disabled` whenever a revision is already present, so it can never re-read. Help contradicts behaviour. — Either keep Resolve enabled (overwrite with a note) or drop the "re-reads" claim.
- **[DESIGN]** `StudyRunControlsView.swift:478-481` + `StudyRunControlsView.swift:16-18` — post-submit "where did it go" — the status line and `LabeledContent("Remote job")` give the job id, and captions say "reconnect from Compute", but there is no control that takes the user to Compute or the job. — A "Show in Compute" link button next to the job id (the panel already exposes `openOptimizations`/`openTemplates`-style callbacks; add one for Compute).
- **[DESIGN]** `JudgingSectionView.swift:224-229` — trash Button (remove judge) — deletes a pinned judge (name, model, provider, revision, dtype) instantly with no confirmation and no undo, while every other removal on this page is a text "Remove" button (StudyEvaluationSection.swift:110, ExclusionRulesEditorView.swift:92); no `.help`. — Use "Remove" text (or keep the icon with help + accessibilityLabel) and confirm when the row carries a resolved revision/provider.
- **[DESIGN]** Same action, different labels — "Import evidence" (`StudyRecentJobsView.swift:45`) vs "Import Evidence" (`StudyRunControlsView.swift:453`); "Show remote options" bordered small (`StudyPreparationControlsView.swift:168`) vs "Show Remote options" link-style caption (`ExecutionPlanEchoRow.swift:34`); "Add Judge" (`JudgingSectionView.swift:130`) vs "Add rule" (`ExclusionRulesEditorView.swift:105`) vs "Add reader instrument (repeReaderScore)" (`StudyEvaluationSection.swift:137`); refresh family "Refresh Results" / "Refresh Job States" / "Refresh Server Runs" / "Refresh receipts" (`StudyResultsView.swift:35`, `StudyRecentJobsView.swift:65`, `StudyServerRunsView.swift:24`, `EvidenceCustodyView.swift:48`). — Pick one casing rule (Title Case for buttons is the majority here).
- **[DESIGN]** `StudyResultsView.swift:24-150` — none of the nine controls in this file has a tooltip; the five `Link`s (L136-150) open `file://` URLs in whatever app owns .jsonl/.json with no hint that they leave the app, and there is no "Reveal in Finder". — Help on each; consider a reveal affordance.
- **[POLISH]** Old name "variant" in user-visible copy — `StudyResultsView.swift:192-200` ("% variant · % baseline", "Distinct-2 … variant"), `StudyResultsView.swift:207` ("Judge: baseline N · variant N · ties N"), `StudyControlCopy.swift:31` freezeHelp "(variant artifact)". Agents replaced Variants. — "agent".
- **[POLISH]** Stale `claude` judge copy — `JudgingSectionView.swift:266-273` localJudgeHelp "distinct from the Claude judge (API)"; `InfoPopover.swift` `StudyInfo.judges` ("Three judge kinds. 'claude' and 'openrouter' are API calls…") and `StudyInfo.judgeKindsAndKeys` lead with 'claude', but the kind picker no longer offers it (L196-203, rendered only as "claude (legacy)" for existing rows). — Describe openrouter/local as the offered kinds and claude as legacy.
- **[POLISH]** `ExclusionRulesEditorView.swift:157` — Button "Apply" — vague object; and `Cancel` (L161) only appears while adding a new range, so editing an existing range has no revert. — "Apply range"; show Cancel/Revert whenever fields differ from the stored rule.
- **[POLISH]** `StudyLiveRunView.swift:39-45, 64-70` — "Stop Run"/"Stop" — destructive role but no confirmation (extra item (d)); acceptable because the stop is cooperative and partial artifacts are kept, and every Stop on the page behaves the same. Noting for completeness, not asking for a dialog.
- **[POLISH]** `StudyRunControlsView.swift:497-511` — remote job log — after "Stop Log" or a stream failure (StudyRemoteJobController.swift:203-205 sets only `remoteStatus`) the log block stays on screen unchanged, so a stopped/failed stream looks live; the only tell is the small status text at L491. — A "stream stopped/failed" badge on the "job log — <id>" header.
- **[POLISH]** `EvidenceCustodyView.swift:31, 59, 73` — `Text(entry.createdAt)` shows the stored raw string (HousekeepingDates.format) with no date formatter; failure messages interpolate raw `\(error)`. Also `Verify retained evidence`/`Refresh receipts` have no help (see list).
- **[POLISH]** `StudyServerRunsView.swift:47` — `Text(task).lineLimit(1)` — truncates the run's task with no `.help` carrying the full text.
- **[POLISH]** `StudyDesignActionsView.swift:40` — "Minted with N sibling study(s)" — awkward pluralization; `:95` help concatenates a lowercase fragment with a sentence-case, period-terminated sentence; `:97` "Open Templates" has no help.
- **[POLISH]** `StudyFreezeControlsView.swift:185-195` — dialog title is a two-sentence paragraph ("Freeze 'x'? This is one-way — afterwards…"); on macOS this renders as the bold headline. — Keep "Freeze 'x'?" as title, move the rest to `message:` (pairs with finding 8).
- **[POLISH]** `StudyRunControlsView.swift:120-141` — segmented "Run on" — the greyed server arm is rendered as "<site> — <serverHint>", where serverHint is "connect first — pick <site> in the toolbar connection dot" (SubstrateRouting.swift:135-138): a ~60-char string inside one segment will truncate at the 560pt panel minimum. — Shorten to "<site> (connect first)" and keep the long hint in the row below (it is already rendered at L153-157 only when selection == .server).
- **[POLISH]** `EvaluationClarityViews.swift:45-51` — rubric Menu label — `.lineLimit(1)` combined with `.fixedSize()` means a long workspace-relative path cannot truncate and will push the row past the column width (unverified at runtime).
- **[POLISH]** `StudyRecentJobsView.swift:21` — jobs discovered by refresh get `study: "—"` (StudyRemoteJobController.swift:111), so rows read "run · — · running". — Show the study name from the job record or omit the segment.
- **[POLISH]** `ValidationDepthView.swift:146-150` — value TextField + "Set" — no tooltips; the Set-required flow is explained only by the status line ("not yet declared — enter a value and press Set"). Sibling declarations in Evaluation each carry an ⓘ; this one has none.
- **[A11Y]** `JudgingSectionView.swift:224-229` — trash Button — icon-only, no `accessibilityLabel`, no `.help` (VoiceOver reads "button").
- **[A11Y]** `EvaluationClarityViews.swift:243-257` — case-family suggestions Menu — label is `Image(systemName: "chevron.down.circle")` with `.help` but no `accessibilityLabel`. Same pattern in `WorkspacePathChooseButton` (WorkspaceFileChooser.swift:127-133, used at EvaluationClarityViews.swift:57) — icon-only with help, no accessibilityLabel (cross-file).
- **[A11Y]** `JudgingSectionView.swift:183, 471, 555` — Pickers with `""` titles (kind, dtype, provider) — no accessible name; add `.accessibilityLabel("judge kind" / "judge dtype" / "serving provider")`.
- **[A11Y]** `StudyLiveRunView.swift:156-161, 182-188` — judgment result capsule — meaning carried by colour (blue = condition, orange = baseline) plus the word itself; acceptable since the text is present, noting only that `.secondary` for ties is indistinguishable from disabled.

Checked and OK (not findings): `StudyRunControlsView.swift:510` `.frame(minHeight: 120, maxHeight: 260)` is conditional, but it sits inside the scrolling `Form` (ExperimentsPanelView.swift:49), not directly in a split column, so it is not the macOS-27 fatal class. `StudyLiveRunView`'s conditional Section carries no fixed heights. `ForEach(panel.draft.judges.indices, id: \.self)` is index-keyed but every read/write is stale-index-guarded (L34-63). `ExclusionRulesEditor` `ForEach(rules, id: \.rule)` — rule ids are unique by engine rule. All `write {}` helpers surface errors as rendered text, never swallowed.

## Missing tooltips (exhaustive)

Controls (28):
- `JudgingSectionView.swift:178` — TextField "name" (judge name)
- `JudgingSectionView.swift:183` — Picker "" (judge kind: openrouter / local / claude (legacy))
- `JudgingSectionView.swift:211` — TextField "model (blank = default)" (legacy claude judge)
- `JudgingSectionView.swift:224` — trash Button (remove judge) — icon-only
- `JudgingSectionView.swift:497` — TextField "model slug (required)" (OpenRouter judge)
- `ExclusionRulesEditorView.swift:157` — Button "Apply" (range rule)
- `ExclusionRulesEditorView.swift:161` — Button "Cancel" (range editor)
- `ValidationDepthView.swift:146` — TextField (layer index(es) / depth fraction(s))
- `ValidationDepthView.swift:149` — Button "Set"
- `StudyDesignActionsView.swift:97` — Button "Open Templates" (link style)
- `StudyRunControlsView.swift:379` — Picker "Executor" (only the Remote-options DisclosureGroup tooltip, L513, covers it)
- `StudyRunControlsView.swift:388` — TextField "GPU gres" (disclosure-level tooltip only)
- `StudyRunControlsView.swift:390` — TextField "walltime" (disclosure-level tooltip only)
- `StudyRunControlsView.swift:450` — Button "Test Connection" (disclosure-level tooltip only)
- `StudyRunControlsView.swift:451` — Button "Cancel Job" (disclosure-level tooltip only)
- `StudyRunControlsView.swift:453` — Button "Import Evidence" (disclosure-level tooltip only)
- `StudyResultsView.swift:24` — Picker "Run" (result run selector)
- `StudyResultsView.swift:35` — Button "Refresh Results"
- `StudyResultsView.swift:90` — Button "Review Judge Responses"
- `StudyResultsView.swift:117` — Button "Review Responses (N)"
- `StudyResultsView.swift:136` — Link "generations.jsonl"
- `StudyResultsView.swift:139` — Link "judgments.jsonl"
- `StudyResultsView.swift:142` — Link "report.json"
- `StudyResultsView.swift:145` — Link "judge-report.json"
- `StudyResultsView.swift:148` — Link "robustness-report.json"
- `ExecutionPlanEchoRow.swift:34` — Button "Show Remote options" (link style)
- `EvidenceCustodyView.swift:35` — Button "Verify retained evidence"
- `EvidenceCustodyView.swift:48` — Button "Refresh receipts"

Menu entries without help (menu items rarely show tooltips; listed for completeness): `ExclusionRulesEditorView.swift:106, 110, 114`; `EvaluationClarityViews.swift:40, 43, 245, 251`.

DisclosureGroups without help (StudyLiveRunView gives its two disclosures help; these do not): `StudyResultsView.swift:67, 98, 108, 213`; `EvidenceCustodyView.swift:21`; `StudyDesignActionsView.swift:40`; `StudyResultReviewWindow.swift:128`.

## Notes on strengths

- The lifecycle gates are honest and well-explained: `ExecutionPlanEchoRow` states what Run will submit before the click, the forced-override dialog restates the failing preflight checks, `FreezeRouting`/`SubstrateRouting` labels name the executing substrate ("Freeze (on <site>)…", "Run on <site>: run"), and refusals from stores are rendered as plain-language text with the raw detail appended, never swallowed.
- Judging is unusually careful: an explicit "none-state" paragraph, "Judging incomplete" vs "Paired judging declared" summaries, key-state badges that use the same presence checks as the run-time gate, provider Discovery to avoid slug mismatches, and stale-index-safe judge bindings.
- Local Stop buttons everywhere share one idiom (destructive role, disabled after the request, help text stating what survives).
- Help strings are dense but generally plain-worded, and every `.help` I checked against its action closure matches behaviour except the Resolve "re-reads" claim above.
