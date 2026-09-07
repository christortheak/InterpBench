# Audit 09 — Templates section and Studies authoring

Base: `<checkout>/Sources/SteerLabApp/`. All 19 assigned files read in full (4,568 lines). Claims about behaviour outside the cluster (management verbs, `setStudyType`, `seatCastingRefusal`, `note()` routing, rename/collision paths, `hiddenContentNote`, `StudyInfo` keys) were verified by reading the ExperimentKit sources named inline.

## Coverage table

Counts are of controls defined in the file itself (sub-views owned by other files — `ModelRevisionControls`, `TemperatureRow`, `SeedsListControls`, `AddConditionEditor`, `WorkspacePathChooseButton`, `TabularImportButton`, `NoticesBellButton`, `ReadingPositionField`, `ExtractionRenderingField` — are counted only where noted). "With help" includes group-level `.help` on an enclosing container.

| File | Controls | With .help (own/group) | Missing help | Icon-only w/o accessibilityLabel |
|---|---|---|---|---|
| TemplatesPanelView.swift | 14 | 6 | 8 (4 of them alert/dialog buttons) | 0 |
| TemplateInstantiationSheet.swift | 16 | 8 | 8 | 1 (`minus.circle`, L229) |
| FactorialDesignSheet.swift | 20 | 9 | 11 | 3 (`trash` L409, `minus.circle` L444, `trash` L463 — all have .help, none has accessibilityLabel) |
| ExperimentsPanelView.swift | 6 | 5 | 1 | 0 |
| StudySetupSection.swift | 16 | 15 | 1 | 0 |
| StudyManagementSection.swift | 12 | 9 | 3 | 0 |
| StudyArmsSection.swift | 10 | 6 | 4 | 1 (`trash`, L290) |
| StudyTypeSection.swift | 9 | 4 | 5 | 0 |
| StudyTypeOverviewColumn.swift | 2 | 2 | 0 | 1 (`doc.on.doc`, L100) |
| StudyConceptsSection.swift | 8 | 8 | 0 | 1 (`trash`, L44 — has .help) |
| StudyInjectionConditionsSection.swift | 3 | 3 | 0 | 1 (`trash`, L111 — has .help) |
| StudySeatsSection.swift | 3 | 3 | 0 | 0 |
| StudyTaskPromptsEditor.swift | 7 (+2 sub-views with own help) | 6 | 1 | 0 (sub-view `WorkspacePathChooseButton` is icon-only with .help, no a11y label — other file) |
| StudySamplingControls.swift | 2 | 2 | 0 | 0 |
| StudyIssuesSection.swift | 0 | — | — | 0 |
| StudyRenameWindow.swift | 4 | 0 | 4 | 0 |
| StudyPromptImportSheet.swift | 4 | 1 | 3 | 0 |
| StudyArtifactAttachmentSheet.swift | 7 | 0 | 7 | 0 |
| StudyControlCopy.swift | 0 (copy only) | — | — | — |
| **Total** | **143** | **87** | **56** | **8** |

## Findings

### Bugs

- **[BUG]** `StudyRenameWindow.swift:76-83` — "Rename" button — on failure (name collision `experiment 'x' already exists`, stale manifest, workspace change) `management.rename(...)` returns `false` and routes the refusal to `draft.formErrors[.rename]` (rendered in `StudyManagementSection.swift:151`, BEHIND the modal sheet) and `panel.note` (the status Section at the bottom of the form, also behind the sheet). The sheet stays open with no visible message, so a collision looks like a dead button. — Render `panel.draft.formErrors[.rename]` inside the sheet under the Name field, or close the sheet before refusing.
- **[BUG]** `StudyTypeSection.swift:170-176` — "Import as Draft" (Paste Study JSON sheet) — `panel.importStudyJSON` failure (`ExperimentPanel.swift:2998-3003`) goes only to `panel.note(...)`; the modal sheet stays open and shows nothing (`packPreviewError` is only set by Preview). Same shape as the rename bug. — Show the import refusal inline in the sheet (reuse `packPreviewError`).
- **[BUG]** `StudyManagementSection.swift:78` — `Picker("Draft", …)` — the label says "Draft" but the picker enumerates `panel.management.experiments` at every status; frozen and complete studies appear under a control titled "Draft" (their status is only visible inside the row text `[frozen]`). — Label it "Study".
- **[BUG]** `StudySetupSection.swift:51-177` — baseline-model picker (and revision, precision, prompt mode, system prompt, reasoning effort) — the whole block is inside `if panel.draft.studyKind == .modelOutput`, so the `.multiAgent` branches of its label (L60-61 "Default model for seats") and help (L85-89 "used only by panel seats that name no base model…") are unreachable dead code. Consequence for the user: a multi-agent study has NO model control at all in the editor — its `modelID` is fixed at creation — while `StudySeatsSection.swift:59-61` tells them "no saved agents use this study's base model (X)… build one in Agents" and `seatCastingRefusal` says "pick this study's base model first". There is no control that can pick it. — Either render the model picker for both kinds (the dead copy suggests that was the intent) or explain in Seats that the model is set at creation and Duplicate/Advanced-create is the way to change it.
- **[BUG]** `StudyManagementSection.swift:127-132` — "Delete…" — `deleteReview = try? management.reviewStudy(named:)`; when review throws, `confirmDeleteDraft` stays `false` and the click does nothing, silently (no note, no inline message). — Surface the review failure (`panel.note`) like `openRename` does at L340-342.
- **[BUG]** `TemplateInstantiationSheet.swift:412, 434` — "Load Only" / "Load and Submit" — `panel.note(model.lastSummary ?? "", …)` posts an EMPTY notice into the persistent notices feed when `lastSummary` is nil; additionally a clean Load Only is stamped `.success` while a clean Load and Submit is stamped `.info` — inconsistent severity for the same outcome. — Guard on a non-nil summary; use one severity.
- **[BUG]** `ExperimentsPanelView.swift:332-339` — "Cancel Server Job" (`role: .destructive`) — cancels a durable server job on a single click with no confirmation dialog; every other destructive action in this cluster (Delete study, Delete design) confirms. — Add a `confirmationDialog` naming the job id and study.
- **[BUG]** `StudyTypeSection.swift:161-169` — "Preview" — enabled with an empty editor; `StudyPackAuthoring.preview` on empty bytes yields a raw decoder error via `String(describing: error)`. — Disable Preview when the trimmed text is empty (the Import button already checks this) and show a plain message.

### Draft vs frozen (extra item a)

Verified per control. Correctly disabled/hidden on frozen: every Study Setup field (`.disabled(manifest.status != .draft)` L26-270), Save/Discard buttons (L284-294, hidden), `ModelRevisionControls` (hidden when not draft), `SeedsListControls` (`.disabled(!isDraft)`), sampling fields, Scenario picker + baseline toggle, Arms add/remove (inside `isDraft`), Concepts attach/detach (inside `status == .draft`), Injection Conditions add/remove/controls, Seats pickers (replaced by `LabeledContent`), Save Casting (disabled with the refusal text rendered inline at L93-97 — the model for how to do it), task-prompt path/Save & Pin/Import/Table/Factorial (all `.disabled`), Attach vector artifact (hidden). Remaining problems:

- **[DESIGN]** `StudySetupSection.swift:13` / `StudyTypeSection.swift:25` — Study Type and Study Setup sections — on a frozen study every field is greyed with NO visible reason anywhere in these sections; the only cue is the provenance section header `"<name> — frozen"` far below (`ExperimentsPanelView.swift:156`). Disabled-with-no-reason is exactly what the checklist asks for. — Add one caption row at the top of Study Setup on non-draft studies ("frozen — settings are part of the record; Duplicate as Draft to change them"), mirroring `seatCastingRefusal`.
- **[DESIGN]** `StudyTypeSection.swift:27-37` — "Study type" picker — stays ENABLED on frozen studies; `ExperimentPanel.setStudyType` (L41-43) then only sets a view override and returns. The hover help says "on drafts, is saved into the study" and the ⓘ says "on frozen studies it only changes what the page displays", but nothing inline tells the user the change they just made is not persisted. — Show a caption on non-drafts ("view only — the frozen study's declared type is X").
- **[POLISH]** `StudyTaskPromptsEditor.swift:39-40` — "Load Prompts" — the only always-enabled button in a row of disabled ones on a frozen study; legitimate (read-only) but its help ("read the JSONL file into the editor below") does not say it is safe on frozen studies. Minor.

### Create / duplicate / delete / rename (extra item b)

- Delete study: confirmation dialog with the name, run-count warning, trash semantics — good. Delete design: confirmation with name — good. Duplicate: auto-suffixed `-2`, `-3` (`StudyManagementController.swift:522-530`), no confirmation needed — good. New Study: placeholder `new-study-<date>` with uniqueness — good.
- **[DESIGN]** `StudyManagementSection.swift:198-199` — "Create Draft" (Advanced) — a name collision surfaces only through `panel.note` ("Couldn't create the draft — check the name isn't already in use…", `StudyManagementController.swift:185-189`), i.e. the status line at the bottom of the form and the bell — not inline at the control like `formErrors[.rename]`/`[.template]` are. No help on the button. — Route through `refuse(.create…)` and render inline.
- **[POLISH]** `TemplatesPanelView.swift:97-104` — Rename design alert — "Rename" is enabled with an empty field; a collision or empty name is refused via `formErrors[.template]`, which renders in the "New Design" section (L164) two sections above the actions, after the alert has closed. Acceptable but easy to miss. No help on "Rename…" (L326) or "Delete" (L330).
- **[POLISH]** `StudyRenameWindow.swift:82` — "Rename" disabled while `!hasChange`, with no help saying why; and on frozen studies the button still says "Rename" though the only effect is setting a display label (the caption explains, the button does not). — "Set Label" when `!sheet.isDraft`.

### Study-type picker / hidden data (extra item c)

- Present and correct: `StudyTypeSection.swift:49-53` renders `panel.studyFocus.hiddenContentNote(for: manifest)` with `eye.slash` in orange directly under the picker, and `StudyTypeOverviewColumn.swift:233-238` repeats it in the Guide. `hiddenContentNote` (`StudyAuthoring.swift:378-403`) covers concepts, injection conditions and pinned scenario per type. This matches the `StudyInfo.studyType` promise ("called out right under the picker").
- **[POLISH]** `StudyTypeSection.swift:50-52` — hidden-content Label lacks `.fixedSize(horizontal: false, vertical: true)` (the Guide copy at L237 has it); a long note ("3 attached concept(s), 4 injection condition(s), a pinned multi-agent scenario") may truncate in the 560pt column.

### Templates: instantiate / cast flow (extra item d)

- **[DESIGN]** Four verbs for one flow: Templates says "Instantiate…" (`TemplatesPanelView.swift:306`), the sheet is titled "New studies from '…'" and its section "Studies to mint" (`TemplateInstantiationSheet.swift:103, 201`), row state says "minted" (L304), the buttons say "Load Only" / "Load and Submit" (L400, L424), and the help text says "mints" / "cast". "Load" is a misnomer for "create study directories". — Pick one: "Create Studies" / "Create and Submit".
- **[DESIGN]** `TemplateInstantiationSheet.swift:396-441` — footer hierarchy — "Load and Submit" carries `.keyboardShortcut(.defaultAction)` but no `.borderedProminent`; "Close" has `role: .cancel` but no `.keyboardShortcut(.cancelAction)` (unverified whether role alone binds Escape inside a `.sheet` on this OS). The primary of the two is arguably "Load Only" (the safe one) — making Submit the Return-key default risks queueing a batch with Return.
- **[DESIGN]** `TemplateInstantiationSheet.swift:84-85` — "Discard casting edits and reload design" — sits between the form and the totals line (not in the footer), is always enabled even with no edits, and has no help. — Move to the footer, disable when nothing changed.
- **[A11Y]** `TemplateInstantiationSheet.swift:226-232` — per-row remove button (`minus.circle`, borderless, `role: .destructive`) — no `.help`, no `.accessibilityLabel`; also removes a row with no confirmation (fine — undoable by Add Study).
- **[POLISH]** `TemplateInstantiationSheet.swift:150-163` — permutation `Menu` and "Add all permutations" — no help (the sibling "Add composition sweep" row has group help); the refusal text below is the only explanation.
- **[POLISH]** `TemplateInstantiationSheet.swift:237-242` — per-seat pickers in each row — no help; `StudyControlCopy.seatPickerHelp` exists and is used by the Seats section — reuse it.
- Factorial sheet validation and cell count: the count line (`FactorialDesign.cellCountLine`, "2 anchor × 1 template = 2 items") and 3-item preview update live (L269-291); parse errors, `validate()` failures and generate refusals all render in red with selectable text; "Generate & Pin" is gated on `generatable`; the overwrite toggle guards the destination file. Good.
- **[DESIGN]** `FactorialDesignSheet.swift:342-345` — "Cancel" / "Generate & Pin" — the prominent button has no `.defaultAction` and Cancel no `.cancelAction`; the other sheets in this cluster each pick a different subset (see consistency below).
- **[POLISH]** `FactorialDesignSheet.swift:181-184` — Toggle label "Counterbalance option order (each cell emitted twice, options reversed, orderFlipped recorded)" — leaks the internal field name `orderFlipped` into a control label. — "…and the flipped order is recorded per item".
- **[POLISH]** `FactorialDesignSheet.swift:285, 387` — `Label("\(error)")` / `loadProblem = "\(error)"` — raw Swift error descriptions shown to the user (the panel elsewhere uses `(error as? ExperimentError)?.reason ?? …`). `loadProblem` is also reused for Save-design failures (L201-203), and `problem` (L321) is never cleared when the design is edited after a refusal.
- **[POLISH]** `FactorialDesignSheet.swift:404, 433, 460, 468, 482` — factor name, level name, template id, template text editor, options field — no help; the placeholders carry the meaning, but the level-name and factor-name fields have hard-coded widths (`220`, `140`) that clip long names.

### Copy JSON / Copy LLM Prompt / Paste-as-new-draft (extra item e)

- **[DESIGN]** `StudyTypeSection.swift:66-75, 94-106` — "Copy Study JSON" / "Copy LLM Prompt" — the only feedback is `panel.note(...)`, which renders as the status Section at the BOTTOM of a very long form (`ExperimentsPanelView.swift:321-323`) plus the bell. At the top of the page where the buttons live nothing changes on click. — Transient inline "Copied" label next to the buttons.
- **[A11Y]** `StudyTypeOverviewColumn.swift:97-103` — copy icon (`doc.on.doc`) in the JSON viewer header — no feedback of any kind (not even a note), no `.accessibilityLabel`.
- **[DESIGN]** `StudyTypeSection.swift:170-176` — "Import as Draft" stays disabled until "Preview" succeeds, and nothing in the sheet says so; the researcher pastes, presses the default action (Return → Import) and nothing happens. — Caption "Preview first — import is enabled once the pack parses", or run the preview automatically on paste.
- **[POLISH]** `StudyTypeSection.swift:167` — `packPreviewError = String(describing: error)` — raw error dump in the paste sheet.
- **[POLISH]** `StudyTypeSection.swift:138-141` — paste `TextEditor` has `.frame(minWidth: 520, minHeight: 320)` and the sheet has no outer frame; the optional preview block (L142-156, up to 120pt) changes the sheet's content height after it is up. Inside a `.sheet` this is not the split-column fatal class, but it does resize the sheet under the user. Minor.

### Seats vs arms vocabulary (extra item f)

- **[DESIGN]** Two sections titled "Conditions" on a concept study: `StudyArmsSection.swift:141-142` header "Conditions" (holds AGENTS — "Add agent", `variantConditions`) and `StudyInjectionConditionsSection.swift:51-52` "Injection Conditions & Controls" (holds CONDITIONS). The ⓘ (`StudyInfo.conditionsArms`) explains arm→condition, but the section title should say what its rows are. — "Arms (agents)" or "Comparison arms" for the first.
- **[DESIGN]** Vocabulary map as rendered: Arms section says "Arms:" only for multi-agent (L45), "agent" for comparison rows; Seats section says seat / casting / occupant / "cast" / "compiled" / "bound (legacy)"; the instantiation sheet says arm ("baseline arm only", "each chosen agent becomes one arm") and casting; `StudyControlCopy.freezeHelp` says "agent (variant artifact)"; `StudyInfo.conditionsArms` says "steered or fine-tuned variants". "Variant" (the old Agents name) still appears in three user-visible strings: `StudyControlCopy.swift:31`, `InfoPopover.swift:148` (conditionsArms), and the Optimizations caption is fine. "Occupant" (`TemplateInstantiationSheet.swift:171`, `StudyControlCopy.swift:125`) is never defined for the user. — One glossary line in the Seats section ("seat = a role in the scenario; casting = which agent, or the baseline model, fills each seat; the cast panel is one arm").
- **[POLISH]** `StudySeatsSection.swift:26` — `Section("Seats")` is the only authoring section in the page without an `InfoSectionHeader`/ⓘ, though it introduces the most new vocabulary.
- **[POLISH]** `StudyArmsSection.swift:67-72` — "Add" (no object) and "Reload agent selection" — the group help (L74) only describes Add; "Reload agent selection" is the review-snapshot refresh and nothing explains when you would press it.

### Old names / dangling references (extra item g)

- **[DESIGN]** `StudySeatsSection.swift:29-30` — "add roles to it in the Panels editor first" and `SeatCasting.swift:334-337` (`legacyAdvisory`, rendered at `StudySeatsSection.swift:67-73`) "Migrate it in the Panels editor" — there is no "Panels" editor; the section is "Multi-Agent" (`WorkbenchSection.swift:19`) and its editor sections are "Scenario"/"Seats"/"Turn Script". — "in Multi-Agent".
- **[POLISH]** `StudyControlCopy.swift:31` — "each agent (variant artifact)" and `InfoPopover.swift:148` "steered or fine-tuned variants" — "variant" is the pre-rename term for agents; acceptable as the artifact's technical name but the parenthetical reads as an alias for a section that no longer exists.
- No occurrences of "Screens", "Concept Lab", "Geometry", or "Steering" (as a section name) in user-visible strings across the 19 files (grep-verified). Cross-references to "Agents → Optimizations", "Agents → New Agent", "Compute", "Templates", "Data & Prompts", "Install model… (Compute menu)" all resolve to existing surfaces (verified).
- **[POLISH]** `TemplatesPanelView.swift` — template/design used interchangeably in one view: section "New Design" contains a button "New Template" (L129/135); "Delete design" dialog says "Removes templates/<name>/" (L340); management notices say "renamed template" / "deleted template" (`StudyManagementController.swift:448, 465`); sidebar says "Templates"; everything else says "design". — Pick "design" in copy and keep "Templates" only as the section name, or vice versa.

### Other findings

- **[DESIGN]** Sheet action hierarchy is different in every sheet of this cluster: `TemplateInstantiationSheet` (defaultAction, no prominence, Close role cancel), `FactorialDesignSheet` (prominence, no defaultAction, plain Cancel), `StudyTypeSection` paste sheet (defaultAction, no prominence, plain Cancel), `RenameStudyWindow` (defaultAction, no prominence, Cancel role cancel), `ImportJSONLSheet` (prominence, no defaultAction, plain Cancel), `StudyArtifactAttachmentSheet` (neither, plain Cancel). None sets `.keyboardShortcut(.cancelAction)`. — One convention: primary = `.borderedProminent` + `.defaultAction`, Cancel = `role: .cancel` + `.cancelAction`.
- **[DESIGN]** "review"/"reviewed" — internal transaction vocabulary leaks into user copy: `ExperimentsPanelView.swift:34` "Select and review a draft study before importing prompts.", `StudyManagementSection.swift:341` "Reload the study before opening Rename.", `TemplatesPanelView.swift:236` "Reload and review this design before saving its description.", L243 "…review the notice and reload the saved design before retrying.", `StudyArtifactAttachmentSheet.swift:45` button "Attach reviewed vector", `StudyArmsSection.swift:71` "Reload agent selection". The user never performed anything called "review". — "Reload the study and try again" / "Attach vector".
- **[DESIGN]** `StudyArtifactAttachmentSheet.swift` (whole file) — zero `.help` on 7 controls; four bare TextFields whose placeholders are the only documentation ("Vector path (without extension)" — relative to what? "Source concept (only if the artifact requires it)" — which artifacts?); "Inspect vector" must be pressed before "Attach reviewed vector" enables, unexplained; `problem = "\(error)"` raw errors (L31, L51); fixed `.frame(width: 640)` with no height; no keyboard shortcuts. The sheet title says "vector", the opening button says "Attach vector artifact…" (`StudyTypeSection.swift:55`), and the button lives in the Study Type section rather than the Concepts section where attachments are listed.
- **[DESIGN]** `StudyConceptsSection.swift:57-59` — header "Build & Validate Concept Vectors" — the section attaches/detaches concept RECIPES; the file's own doc comment says "Extraction is not launched here", and building/validating live in the Preparation controls. — "Concepts (pinned recipes)".
- **[A11Y]** `StudyConceptsSection.swift:100-113` — method `Picker("")` — empty label; VoiceOver has nothing to read; the visible "Attach Concept…" label above belongs to a different picker.
- **[A11Y]** Icon-only buttons with `.help` but no `.accessibilityLabel`: `StudyArmsSection.swift:287-293` (trash, and this one has no `.help` either), `StudyConceptsSection.swift:41-46`, `StudyInjectionConditionsSection.swift:108-114`, `FactorialDesignSheet.swift:408, 443, 462`, `TemplateInstantiationSheet.swift:226`, `StudyTypeOverviewColumn.swift:97`.
- **[DESIGN]** `StudyArmsSection.swift:286-293` — trash on an attached agent row — removes the arm immediately (writes the manifest) with no confirmation and no help; the Injection Conditions trash at least has help ("remove this condition from the draft"). Same action, different affordance ("Delete…" with dialog for studies, silent trash for arms/conditions/concepts).
- **[DESIGN]** `StudyInjectionConditionsSection.swift:94, 97` — "+ sign control" / "+ random control" — the only buttons in the cluster using a "+ " prefix instead of "Add …"; the badge "direction control" (L80-85) has no help while its sibling "random-direction control" (L74-78) does.
- **[DESIGN]** `StudySetupSection.swift:148-158` — "Reasoning effort" — disabled when `PromptRendering.hasThinkingMode` is false, with no visible reason; the help text describes what the field does but never says "this model's chat template has no thinking mode". — Caption when disabled.
- **[POLISH]** `StudySetupSection.swift:14-19` vs `284-288` — two "Discard edits and reload" buttons with the same label at different positions depending on `selectedDraftNeedsReload`; the top one has no help.
- **[POLISH]** `StudySetupSection.swift:59-63` — picker title "Baseline model" — for a Compare-agents study the ⓘ/help call it "the required base for added agents"; for a concept study it is simply "the model". Fine, but see the multi-agent BUG above.
- **[POLISH]** `TemplatesPanelView.swift:266-269` — `shortDate` slices the ISO string by hand (`prefix(19)`, `T`→space) — unlocalized, timezone-less date shown as "Created". — `Date.FormatStyle`.
- **[POLISH]** `TemplatesPanelView.swift:228` — "Discard description edits and reload" sits ABOVE the field it discards and is always enabled; the description saves only on Return ("press return to save" caption, L251) with no Save button — a vertical-axis `TextField` where Return submits is easy to mistake for a newline field.
- **[POLISH]** `TemplatesPanelView.swift:144, 204` — "Study" and "Design" pickers — no help; the "Design" picker's row label `"name · intent · modelID"` (L215-218) is a long unbroken string in a 560pt column (`Picker` rows truncate with no `.help` carrying the full text).
- **[POLISH]** `ExperimentsPanelView.swift:51` — "Research methods and guides" — a bare `Button` as the first row of a grouped `Form`, above any `Section` (renders as an orphan cell); no help.
- **[POLISH]** `StudyIssuesSection.swift:100-101` — uses `VectorCatalog.projectRoot` while every sibling uses `ExperimentStore.workspaceRoot` (= `rootOverride ?? VectorCatalog.projectRoot`). `rootOverride` is only set in tests today, so not user-visible, but it is the one place in the page that would scan the wrong root if that ever changes.
- **[POLISH]** `StudyTaskPromptsEditor.swift:78-80` — `TextEditor.frame(minHeight: 220)` on content that appears/disappears (expanded in place by Data & Prompts' Edit button) inside the controls column. It sits inside the scrolling grouped `Form`, so the column's own minimum should not change (unverified on this beta) — noting it because it matches the fatal-class pattern textually.
- **[POLISH]** `StudyTaskPromptsEditor.swift:101-105` — "Import as JSONL…" (the paste-guard button) — no help; and it is a second entry point to the same sheet as "Import JSONL…" three rows above (deliberate, but the two labels differ by one word).
- **[POLISH]** `StudyPromptImportSheet.swift:67-74` — the "select a draft study first" branch is dead: `ExperimentsPanelView.swift:395` always passes `destination: "prompts/tasks/versions/"` and the study guard lives in `importJSONLPresented` (L29-41).
- **[POLISH]** `StudyManagementSection.swift:176-177` — help on the "new study name" TextField ("creates a draft pinned to the currently selected model") describes the Create Draft button, not the field.
- **[POLISH]** `StudyManagementSection.swift:264-265` — "Open Templates" `.link` button — no help; `StudyDesignActionsView` has the same button (verified) — consistent label at least.
- **[POLISH]** `StudyArmsSection.swift:21` — `agentReviewMessage = String(describing: error)` — raw error shown in orange.
- **[POLISH]** `.help` strings that start with a capital or end with a period, against the lowercase-fragment convention: `StudySetupSection.swift:160-165` ("The reasoning effort…accepts it."), L174-175 ("The reasoning block's…Required."), L206-210 (ends "checks."), L226-230 (ends "subtracts."), `StudyManagementSection.swift:257-260` ("From scratch opens… casting."), `StudyControlCopy.swift` most entries end with a period (`saveBackHelp`, `saveAsNewDesignHelp`, `duplicateHelp`, `deleteStudyHelp`, `unifiedRemoteRunHelp`, `studyDtypeHelp`, `seatPickerHelp`, `saveCastingHelp`, `permutedSiblingsHelp`, `canonicalNameHelp`). Not defects per the checklist; listed for consistency.
- **[POLISH]** `StudyControlCopy.swift:34` — `freezeHelp` ends "CLI: freeze --force skips validation gates" — a CLI flag in an app tooltip, with no product name (`steerlab-cli`); `remoteFreezeHelp` says "forcing requires the CLI". Fine but terse.

## Missing tooltips (exhaustive)

TemplatesPanelView.swift:97 — TextField "design name" (rename alert)
TemplatesPanelView.swift:98 — Button "Cancel" (rename alert)
TemplatesPanelView.swift:99 — Button "Rename" (rename alert)
TemplatesPanelView.swift:144 — Picker "Study" (source study for New from Study)
TemplatesPanelView.swift:204 — Picker "Design" (library selection)
TemplatesPanelView.swift:228 — Button "Discard description edits and reload"
TemplatesPanelView.swift:326 — Button "Rename…"
TemplatesPanelView.swift:330 — Button "Delete" (design)
TemplateInstantiationSheet.swift:84 — Button "Discard casting edits and reload design"
TemplateInstantiationSheet.swift:150 — Menu "Agents to permute…" / "N agent(s) chosen"
TemplateInstantiationSheet.swift:152 — Toggle per agent (permutation menu items)
TemplateInstantiationSheet.swift:161 — Button "Add all permutations"
TemplateInstantiationSheet.swift:226 — Button minus.circle (remove row; icon-only, no accessibilityLabel)
TemplateInstantiationSheet.swift:237 — Picker per seat (row casting)
TemplateInstantiationSheet.swift:251 — Toggle per agent ("Add agent" menu items)
TemplateInstantiationSheet.swift:399 — Button "Close"
FactorialDesignSheet.swift:181 — Toggle "Counterbalance option order…"
FactorialDesignSheet.swift:237 — Button "Add factor"
FactorialDesignSheet.swift:256 — Button "Add template"
FactorialDesignSheet.swift:342 — Button "Cancel"
FactorialDesignSheet.swift:343 — Button "Generate & Pin"
FactorialDesignSheet.swift:404 — TextField "factor name (e.g. anchor)"
FactorialDesignSheet.swift:406 — Button "Add level"
FactorialDesignSheet.swift:433 — TextField "level (e.g. high)"
FactorialDesignSheet.swift:460 — TextField "template id (e.g. t1)"
FactorialDesignSheet.swift:468 — TextEditor template text
FactorialDesignSheet.swift:482 — TextField "options, one per line…"
ExperimentsPanelView.swift:51 — Button "Research methods and guides"
StudySetupSection.swift:18 — Button "Discard edits and reload" (needs-reload variant)
StudyManagementSection.swift:160 — DisclosureGroup "Advanced"
StudyManagementSection.swift:198 — Button "Create Draft"
StudyManagementSection.swift:264 — Button "Open Templates" (link style)
StudyArmsSection.swift:81 — Button per concept ("Add sweep-created agent…" menu items)
StudyArmsSection.swift:154 — Picker "Confirm agent"
StudyArmsSection.swift:174 — Button "Attach Policy"
StudyArmsSection.swift:287 — Button trash (remove attached agent; icon-only, no accessibilityLabel)
StudyTypeSection.swift:55 — Button "Attach vector artifact…"
StudyTypeSection.swift:138 — TextEditor (paste JSON)
StudyTypeSection.swift:160 — Button "Cancel" (paste sheet)
StudyTypeSection.swift:161 — Button "Preview"
StudyTypeSection.swift:170 — Button "Import as Draft"
StudyTaskPromptsEditor.swift:101 — Button "Import as JSONL…" (paste-guard)
StudyRenameWindow.swift:48 — TextField "study name" (caption below, no .help)
StudyRenameWindow.swift:64 — TextField "display label (optional)" (caption below, no .help)
StudyRenameWindow.swift:75 — Button "Cancel"
StudyRenameWindow.swift:76 — Button "Rename"
StudyPromptImportSheet.swift:41 — TextEditor (JSONL text)
StudyPromptImportSheet.swift:84 — Button "Cancel"
StudyPromptImportSheet.swift:85 — Button "Import & Pin"
StudyArtifactAttachmentSheet.swift:22 — TextField "Vector path (without extension)"
StudyArtifactAttachmentSheet.swift:24 — TextField "Concept name in this study"
StudyArtifactAttachmentSheet.swift:25 — TextField "Source concept (only if the artifact requires it)"
StudyArtifactAttachmentSheet.swift:26 — TextField "Evaluation run (optional)"
StudyArtifactAttachmentSheet.swift:27 — Button "Inspect vector"
StudyArtifactAttachmentSheet.swift:44 — Button "Cancel"
StudyArtifactAttachmentSheet.swift:45 — Button "Attach reviewed vector"

## Notes on strengths

- Refusals are surfaced AT the control in most of this cluster: `deleteSelectedStudyRefusal` becomes the Delete button's help and gate, `seatCastingRefusal` is the Save Casting help AND an inline caption, `submitHelp` in the instantiation sheet names the specific blocker (no server vs. no casting).
- The instantiation sheet's totals line, per-row refusal/advisory/ready states, and constant frame (documented against the split-view fatal class) are exactly the right shape for a batch-size decision.
- Study Setup, Sampling, and Concepts have near-complete help coverage with plain-words-first copy, visible captions for the non-obvious consequences (no baseline arm → no effect sizes; stochastic design → derived seeds; local MLX stays greedy), and ⓘ popovers where hover text would be too much.
- The "hidden data" callout under the study-type picker is implemented as the ⓘ promises, and delete/rename confirmations state what happens to runs.
