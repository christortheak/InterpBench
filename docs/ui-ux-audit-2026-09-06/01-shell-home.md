# Audit 01 — App shell, Home dashboard, workspace controls, shared primitives

Reviewer scope: `SteerLabApp.swift`, `ChatView.swift` 1–330, `WorkbenchSection.swift`, `HomeDashboardView.swift`, `WorkspaceControls.swift`, `WorkspaceMismatchBanner.swift`, `UpdateSignpost.swift`, `NoticesViews.swift`, `InfoPopover.swift`, `AgentChips.swift`, `FileReferenceRow.swift`, `WorkspaceFileChooser.swift`, `InlineFileEditorSheet.swift`, `MarkdownMessageText.swift`, `ScienceGuidesView.swift`. Every file read in full (ChatView to line 330). All paths below are under `<checkout>/Sources/SteerLabApp/` unless stated.

Counting convention: a "control" is a Button/Toggle/Picker/Menu/TextField/SecureField/TextEditor/List-selection/acting row. Menu items are counted because this codebase itself puts `.help` on menu items (`SteerLabApp.swift:246`), so the convention is available. Alert buttons and app-menu commands are counted but flagged "(n/a)" — macOS cannot show a tooltip there.

## Coverage table

| File | Controls | With `.help` (own or group) | Missing `.help` | Icon-only missing `accessibilityLabel` |
|---|---|---|---|---|
| SteerLabApp.swift | 17 | 8 | 9 (2 are alert buttons, n/a) | 0 |
| ChatView.swift (1–330) | 2 (sidebar rows as one group; Pin button) | 2 | 0 | 0 (Pin uses `Label(...).labelStyle(.iconOnly)`, title survives for VoiceOver) |
| WorkbenchSection.swift | 0 (strings only) | – | – | – |
| HomeDashboardView.swift | 15 | 4 | 11 | 0 |
| WorkspaceControls.swift | 15 | 4 | 11 (1 alert button, n/a) | 0 |
| WorkspaceMismatchBanner.swift | 3 | 3 (root items covered by the Menu's group help) | 0 | 0 |
| UpdateSignpost.swift | 5 | 3 | 2 (both app-menu commands, n/a) | 1 |
| NoticesViews.swift | 2 | 2 | 0 | 1 |
| InfoPopover.swift | 1 (InfoButton) | 0 | 1 | 0 (has `accessibilityLabel("Explain")`) |
| AgentChips.swift | 0 (two badges; one has help, one does not) | – | – | – |
| FileReferenceRow.swift | 4 | 3 | 1 | 3 |
| WorkspaceFileChooser.swift | 1 | 1 | 0 | 1 |
| InlineFileEditorSheet.swift | 5 | 3 | 2 | 0 |
| MarkdownMessageText.swift | 0 | – | – | – |
| ScienceGuidesView.swift | 4 | 0 | 4 | 0 |
| **Total** | **74** | **33** | **41** | **6** |

Fatal split-column class (varying `minHeight` inside a split column): **none found** in this cluster. The only `minHeight`s are in sheets (`FileReferenceRow.swift:144`, `InlineFileEditorSheet.swift:41,61`, `ScienceGuidesView.swift:59`). `ChatView.swift:96` varies a min **width** per section (340/420/560) on user selection, not async state. `HomeDashboardView` is a plain grouped `Form` with no fixed heights; `ClusterHealthCard.swift` declares no `minHeight` (grep). `UpdateBanner` sits above the `NavigationSplitView`, not inside a column.

## Findings

- **[BUG]** `HomeDashboardView.swift:143` — Button "Train Adapter" — action is `navigate(.data)`, and `navigate` (`ChatView.swift:136-138`) only sets `section`; `dataTool` is never touched, so the button lands on Data › whichever tool was last shown (Inventory by default), not on "Adapter Training" (`SectionContainers.swift:35`). — Route through a tool-setting closure the way `openOptimizations` does for Agents › Optimizations (`ChatView.swift:142-145`), or rename the button "Open Data".
- **[BUG]** `InfoPopover.swift:282` — `StudyInfo.judges` copy — says "a frozen judged study needs at least 2 [judges] so the report can carry agreement statistics; one judge's quirks are not evidence". This contradicts `InfoPopover.swift:548` (`dataCategory(.judging)`: "at least one judge — a single-coder design is legal and freezes with an advisory") and the engine: freeze emits `ExperimentStore.singleJudgePanelAdvisory` (an advisory, `ExperimentKit/ExperimentStore.swift:7973`), and `AgentContract.swift:662` states "including exactly one: a single-coder design is a legal methodology". Two ⓘ popovers on the same study page tell the researcher opposite rules. — Rewrite line 282 to the single-coder-with-advisory rule.
- **[BUG]** `SteerLabApp.swift:401` and `WorkspaceControls.swift:237` — Button "Plan" — enabled with an empty model id; only the sibling "Install" is gated (`:407`, `:243`). `previewModelPreparation` (`ExperimentKit/ClusterConnectionStore.swift:1647`) has no empty guard and sends `""` to the server, so the click yields a server-side error instead of nothing. — Apply the same `.disabled(trimmed.isEmpty)` gate as Install.
- **[BUG]** `SteerLabApp.swift:385-390` — Button "Remove" (server editor) — forgets the server **and deletes its Keychain token** (`cluster.removeServer`) then dismisses, on a single click with no confirmation; `role: .destructive` alone does not prompt. — Add a `confirmationDialog` naming the server.
- **[BUG]** `NoticesViews.swift:47-50` — Button "Clear" — empties the notices ring *and its persisted file* (per its own help) with no confirmation and no `role: .destructive`. — Confirm, or at least mark destructive.
- **[DESIGN]** `SteerLabApp.swift:200` + `:271-277` — Picker titled "Workspace" inside the "Compute: …" toolbar menu, with help "which workspace the app is scoped to…" — but the toolbar's *other* menu (`WorkspaceSelector`, folder icon) is the actual data Workspace, and its help says "Compute picks the engine; Workspace picks the data" (`WorkspaceControls.swift:125-126`). The internal type name `ClusterConnectionStore.Workspace` leaks into user copy and the two toolbar menus contradict each other on what "workspace" means. — Title the picker "Compute target" (or hide the label) and open the help with "which engine the app computes on".
- **[DESIGN]** `WorkbenchSection.swift:50` and `:56` — sidebar help strings "(formerly Steering)" and "(formerly Geometry)" — stale "formerly" references to the retired section names (the checklist calls these out explicitly); they are the first tooltips a new user sees. — Drop the parentheticals.
- **[DESIGN]** `HomeDashboardView.swift` — duplicated controls on one screen: "Open Compute" ×3 (`:84`, `:237`, `:330`), "Open Playground" ×2 (`:115`, `:326`), "Optimize" ×2 (`:142`, `:327`), "Create study" ×2 (`:254`, `:329`). The "Next actions" section (`:323-334`) restates what every section above already offers, and only one twin of each pair carries help (`:117`, `:328`). — Keep one row of next actions or drop the per-section buttons; give whichever survives the help.
- **[DESIGN]** `HomeDashboardView.swift:254`, `:329` — Button "Create study" — only navigates (`navigate(.studies)`); nothing is created. Its non-empty twin (`:260`) is honestly "Open Studies". — Either land on the New Study flow or call it "Open Studies".
- **[DESIGN]** `HomeDashboardView.swift:156-171` vs `:268-277` — agent rows are inert `HStack`s while study rows are `Button`s that open the study with it selected (with help). Same list shape, different affordance; there is no way to open a listed agent from Home. — Make agent rows link to Agents › Library with the agent selected (`openAgentsLibrary` exists in `ChatView.swift:151`).
- **[DESIGN]** `ScienceGuidesView.swift:51` — Button "Open study designs" — the section is named "Templates" (`WorkbenchSection.swift:23`, help "the design library"). — "Open Templates".
- **[DESIGN]** `ScienceGuidesView.swift:53` — caption "Agents: science guide \(id) --json returns this same text. Engine-only operations require the listed engine; discovery does not execute or qualify a study." — "Agents:" here means coding agents/CLI, which collides with the Agents section; it dumps bare CLI syntax without the product name (`steerlab-cli science guide …`) into UI copy. — "Command line: `steerlab-cli science guide <id> --json` returns this same text…".
- **[DESIGN]** `ScienceGuidesView.swift:20` ("Done" → `.cancelAction`) vs `FileReferenceRow.swift:124` ("Done" → `.defaultAction`) vs `InlineFileEditorSheet.swift:112` ("Cancel" → no shortcut) — three sheets in this cluster bind the same dismiss verb three different ways; the editor sheet cannot be closed with Escape. — Bind Cancel/Done to `.cancelAction` consistently; keep `.defaultAction` for Save.
- **[DESIGN]** `WorkspaceControls.swift:240` ("Install") vs `:300` ("Download") — the server and local twin popovers use different verbs for the same fetch, and only the local one has `.keyboardShortcut(.defaultAction)` (`:303`) and a progress bar (`:314`); the server popover shows nothing after "Install" beyond the shared single-slot status line. — Align the verb, add the default shortcut, and show the queued-job state.
- **[DESIGN]** `WorkspaceControls.swift:47-50` — error alert titled "Workspace" with "OK" — title is a noun, not a failure ("Workspace" / "The file couldn't be…"). — "Could not open workspace".
- **[DESIGN]** `WorkspaceMismatchBanner.swift:56` — Button "Point server at this workspace" — the same action from the toolbar (`SteerLabApp.swift:241-243`) awaits `service.connectCluster()` after a successful switch; the banner path only gets the store's internal `connect()` (`ClusterConnectionStore.swift:1336`). Whether service-level panels refresh without the second call is (unverified). No busy indicator or re-entry guard on either path.
- **[DESIGN]** `InfoPopover.swift:242` — `StudyInfo.dataPrompts` "…the row's buttons view it, reveal it in Finder, or open it in your editor" — the pencil opens the in-app editor sheet (`FileReferenceRow.swift:75-84`); "your editor" is the retired system-editor route. — "…or edit it in the app".
- **[DESIGN]** `WorkbenchSection.swift:87` — `AgentsRegion.optimizations.help` "declared layer×alpha optimization runs: grids, recommendations, Create Agent" — jargon with no plain-words half (the corpus convention is "plain words first"). — "runs that search layers and strengths for the best steering point: their grids, the recommended point, and Create Agent".
- **[DESIGN]** Substrate naming drift in user-visible strings: "MLX (local)" (`ChatView.swift:262`), "Local (MLX)" (`SteerLabApp.swift:201`, `ExperimentKit/WorkspaceCompute.swift:52`), bare "MLX" (`WorkspaceControls.swift:118`), "in-process MLX" (`HomeDashboardView.swift:82`). — Pick one spelling.
- **[POLISH]** `HomeDashboardView.swift:51-56` — workspace path `Text` with `.lineLimit(1).truncationMode(.middle)` and no `.help` carrying the full path; a long path is unreadable even though selection is enabled. — Add `.help(workspace.rootURL.path)`.
- **[POLISH]** `HomeDashboardView.swift:336-340` — `shortDate` slices the ISO string by hand (`replacingOccurrences` + `prefix(16)`); not localized, no formatter, shows UTC without saying so.
- **[POLISH]** `HomeDashboardView.swift:325-331` — four `.small` buttons in one HStack inside a grouped Form at Home's 420pt minimum (`WorkbenchSection.swift:65`) — likely to clip at the floor (unverified at runtime). — Use a wrapping layout or two rows.
- **[POLISH]** `HomeDashboardView.swift:33-35` — `.onAppear { service.experiments.refresh() }` runs a synchronous scan on every appearance, while the agent-library scan right below correctly uses `.task` with the off-main-actor rationale.
- **[POLISH]** `SteerLabApp.swift:131` / `:171` vs `ChatView.swift:79,96,98` — window floor 1200 = sidebar min 160 + controls 560 + viewer 420 = 1140 (fits). With the sidebar dragged to its 240 max, 240 + 980 = 1220 > 1200, so at the window floor the `HSplitView` cannot honor both column minimums (unverified which column gives). Default 1440×900 is fine.
- **[POLISH]** `SteerLabApp.swift:317-323` — active server row gets a `circle.fill` icon *and* the inline Picker's own checkmark — double selection indicator (unverified rendering).
- **[POLISH]** `SteerLabApp.swift:411-414` — the model-preparation message block is mis-indented and sits **outside** `if isActiveServer` (the `}` at `:410` closes it), so it is evaluated for non-active servers too; guarded only by endpoint/model-id equality. Cosmetic today, but the indentation hides the scoping.
- **[POLISH]** `WorkspaceControls.swift:18` — `Section(workspace.rootURL.path)` uses the full absolute path as a menu section header; long paths widen the whole menu (unverified).
- **[POLISH]** `WorkspaceControls.swift:92`, `:159`, `:175` — `errorMessage = "\(error)"` — fine for `ExperimentError` (CustomStringConvertible), but any other thrown error (e.g. a `CocoaError` from `WorkspaceCompute.declare`'s file write) renders its raw description in the alert. — Use `localizedDescription`.
- **[POLISH]** `WorkspaceControls.swift:379-384` (SteerLabApp) — "Connect" is enabled with an empty URL; `addServer` falls back to `hostLabel(forURLString:)` of an empty string and `installModel` later reports "invalid server URL" — the failure surfaces only in the status line. — Gate Connect on a parseable URL.
- **[POLISH]** `InfoPopover.swift:19-33` — `InfoButton`: has `accessibilityLabel("Explain")` but no `.help` tooltip (the hover layer is documented as the secondary layer, yet the help button itself has none); the popover has no `ScrollView`, and the longest corpus entries (`controls`, `judges`, `caseFamily`, `judgeKindsAndKeys`) are 4–5 paragraphs at 380pt caption text — tall popovers on small displays.
- **[POLISH]** `InlineFileEditorSheet.swift:103-107` — "Open in Default App" ignores `NSWorkspace.shared.open(url)`'s Bool; this sheet exists precisely because researchers may have no handler for `.jsonl`/`.md`, and that failure is silent. — Show a status line on `false`.
- **[POLISH]** `InlineFileEditorSheet.swift:113-116` — "Save" enabled when text is unchanged; Cancel discards edits with no unsaved-changes prompt.
- **[POLISH]** `FileReferenceRow.swift:131,136` with `:164-171` — `contents` is a computed property that reads the file from disk; body reads it twice per render (`contents.text`, `contents.truncated`). — Cache in `@State` on appear.
- **[POLISH]** `AgentChips.swift:30-39` — `AgentKindBadge` shows a bare kind word ("sweep-promoted", "override", "baseline") with no `.help`, while `AgentChipView` (`:16`) has one. — Add a help per kind.
- **[POLISH]** `MarkdownMessageText.swift:21` — `ForEach(... id: \.offset)` index identity for streamed blocks; `:44-45` code-block background `.black.opacity(0.12)` is near-invisible on a dark bubble (unverified).
- **[POLISH]** `ScienceGuidesView.swift:25-30` — no empty state when the catalog loads zero methods without throwing (blank list, blank detail); `:74` re-parses `ScienceCatalog.catalog()` on every selection change; `:47,50` hard-code method ids (`"optimization"`, `"jspace"`, `"multi-agent"` — all present in `WorkspaceSeed/prompts/method-guides/catalog.json`, so correct today).
- **[POLISH]** `WorkbenchSection.swift:48` — Home help "workspace dashboard: where am I, what exists, what next" reads as a design-note fragment; `:54` "scenario and protocol builder" is thin. — Sentence-shaped fragments.
- **[POLISH]** `WorkbenchSection.swift:166` — viewer title "Vector Geometry" keeps the retired section word "Geometry" in a visible title (descriptive rather than a section name, so low).
- **[A11Y]** `UpdateSignpost.swift:119-125` — dismiss button whose whole label is `Image(systemName: "xmark")` — has `.help` but no `.accessibilityLabel`; VoiceOver reads "xmark". — `.accessibilityLabel("Dismiss update notice")`.
- **[A11Y]** `NoticesViews.swift:11-24` — bell button, icon-only, no `.accessibilityLabel`; unseen-error state is conveyed by glyph + color (not color-only, fine) but the label never says "N unseen errors".
- **[A11Y]** `FileReferenceRow.swift:55-61`, `:63-73`, `:75-84` — eye / folder / pencil buttons, icon-only, `.help` present, no `.accessibilityLabel`.
- **[A11Y]** `WorkspaceFileChooser.swift:114-133` — `folder.badge.plus` button, icon-only, `.help` present, no `.accessibilityLabel`.
- **[A11Y]** `FileReferenceRow.swift:55-84`, `InfoPopover.swift:19-24`, `NoticesViews.swift:11-25` — `.buttonStyle(.plain)` glyph buttons at caption/small image scale; hit targets are the glyph bounds (~14–16pt), under the ~20pt guideline (unverified exact size). — Pad with `.contentShape`/`.frame(minWidth: 20, minHeight: 20)`.

### Extra items for this cluster

- **(a) Toolbar.** `WorkspaceSelector` (`WorkspaceControls.swift:10-208`) and `SubstrateSelector` (`SteerLabApp.swift:187-339`) audited above; the "Workspace"-label collision between them is the main finding. `GPUSessionToolbarControl` / `ClusterConnectionDot` belong to other reviewers.
- **(b) App menu / `UpdateCommands`** (`UpdateSignpost.swift:137-147`, placed after `.appInfo` at `SteerLabApp.swift:176`): "Check for Updates…" + "Check for Updates Automatically" toggle. Correct placement and wording; the manual report alert (`SteerLabApp.swift:135-150`) offers "OK" + "Open Releases…" only when a page exists, with honest sentences from `UpdateCheckPolicy.manualReport` (`ExperimentKit/UpdateCheck.swift:571-583`). "Check for Updates…" is disabled while a check runs with no spinner — acceptable in a menu. No defect.
- **(c) Window vs section minimums.** `.frame(minWidth: 1200, minHeight: 700)` / `.defaultSize(1440×900)` (`SteerLabApp.swift:131,171`); sidebar 160/185/240 (`ChatView.swift:79`); controls min 340/420/560 (`WorkbenchSection.swift:62-68`); viewer min 420 (`ChatView.swift:98`). Fits at the sidebar's min/ideal; over-committed by 20pt at the sidebar's max — see the POLISH item above. The 1140-vs-1200 headroom at ideal sidebar width is only 35pt, so any future viewer/controls minimum bump breaks the floor.
- **(d) Sidebar help strings** (`WorkbenchSection.swift:46-59`): two stale "(formerly …)" strings; Home reads as a note fragment; Data is 30+ words (long but accurate — "New Dataset" exists, 5 files); Templates/Studies/Results/Compute are good. `AgentsRegion.help` (`:83-89`) — Optimizations line is jargon-only.
- **(e) Dangling doc paths.** All four files are absent from `docs/` (verified with `ls`). In this cluster they appear only in comments: `WorkbenchSection.swift:5` (`docs/UI_REDESIGN_AGENT_WORKBENCH_EXPERIENCE.md`) and `:72` (`docs/AGENT_CREATION_SWEEP_UI_RECOMMENDATION.md`). Elsewhere (other clusters, comments only): `OptimizationRunsView.swift:10, :1640` (`docs/STATUS.md`), `ModelVariantsPanelView.swift:6`, `FineTuningPanelView.swift:297` (`docs/CLUSTER-LORA-READINESS.md`). **None appears in a user-visible string anywhere under `Sources/SteerLabApp`** (grep of every `.swift`). The `StudyInfo` corpus references no doc paths at all; the only path-like strings are `prompts/parsers/parser-registry.json`, `prompts/rubrics/`, `~/.steerlab/judge-key`, all live.

### Corpus verification (InfoPopover `StudyInfo`)

Every button label the corpus names was confirmed to exist in the app (`Copy LLM Prompt`, `Scaffold Control Matrix`, `Add Baseline`, `+ sign control`, `+ random control`, `Create from template`, `Save Evaluation Settings`, `Run Paired Judge`, `Save & Pin`, `Import & Pin`, `Save scratchpad as rubric file`, `Install model…`). "Compute section" as the home of the Anthropic / external judge keys (`:437`, `:443`) is correct (`SectionContainers.swift:383`). No old section names (Steering / Concept Lab / Variants / Geometry / Screens) appear in the corpus. Beyond the two items above (judge count at `:282`, "your editor" at `:242`) no typos or unbalanced quotes were found; the only oddity is the mid-sentence hard wrap at `:588` (source formatting, invisible to users).

## Missing tooltips (exhaustive)

- `SteerLabApp.swift:141` — alert Button "OK" (n/a — alert)
- `SteerLabApp.swift:143` — alert Button "Open Releases…" (n/a — alert)
- `SteerLabApp.swift:200` — Picker "Workspace" (Compute menu, inline)
- `SteerLabApp.swift:209` — menu Button "Add Server…"
- `SteerLabApp.swift:214` — menu Button "Edit “<name>”…"
- `SteerLabApp.swift:253` — submenu "Recent Server Workspaces"
- `SteerLabApp.swift:255` — submenu Button `<root>` (one per recent root)
- `SteerLabApp.swift:401` — Button "Plan" (server editor)
- `SteerLabApp.swift:404` — Button "Install" (server editor)
- `HomeDashboardView.swift:84` — Button "Open Compute" (Compute section)
- `HomeDashboardView.swift:141` — Button "Open Agents"
- `HomeDashboardView.swift:142` — Button "Optimize" (Recent agents empty state)
- `HomeDashboardView.swift:143` — Button "Train Adapter"
- `HomeDashboardView.swift:150` — Button "Open Agent Library"
- `HomeDashboardView.swift:237` — Button "Open Compute" (Running now)
- `HomeDashboardView.swift:254` — Button "Create study" (Recent studies empty state)
- `HomeDashboardView.swift:260` — Button "Open Studies"
- `HomeDashboardView.swift:326` — Button "Open Playground" (Next actions)
- `HomeDashboardView.swift:329` — Button "Create study" (Next actions)
- `HomeDashboardView.swift:330` — Button "Open Compute" (Next actions)
- `WorkspaceControls.swift:19` — menu Button "New Workspace…"
- `WorkspaceControls.swift:21` — menu Button "Open Workspace…"
- `WorkspaceControls.swift:27` — menu Button "Reveal in Finder"
- `WorkspaceControls.swift:49` — alert Button "OK" (n/a — alert)
- `WorkspaceControls.swift:70` — Picker "Computes on" (labels hidden; no help on the picker or its rows)
- `WorkspaceControls.swift:233` — TextField "HF repo id (e.g. Qwen/Qwen3-4B)" (Install model popover)
- `WorkspaceControls.swift:237` — Button "Plan" (Install model popover)
- `WorkspaceControls.swift:240` — Button "Install" (Install model popover)
- `WorkspaceControls.swift:288` — TextField "HF repo id (e.g. mlx-community/…)" (Add Model popover)
- `WorkspaceControls.swift:300` — Button "Download"
- `WorkspaceControls.swift:309` — Button "Cancel Download"
- `UpdateSignpost.swift:141` — app-menu Button "Check for Updates…" (n/a — menu command)
- `UpdateSignpost.swift:145` — app-menu Toggle "Check for Updates Automatically" (n/a — menu command)
- `InfoPopover.swift:19` — InfoButton (questionmark.circle; has accessibilityLabel, no `.help`)
- `FileReferenceRow.swift:124` — Button "Done" (viewer sheet)
- `InlineFileEditorSheet.swift:39` — TextEditor (file body)
- `InlineFileEditorSheet.swift:112` — Button "Cancel"
- `ScienceGuidesView.swift:20` — Button "Done"
- `ScienceGuidesView.swift:25` — List (method selection rows)
- `ScienceGuidesView.swift:48` — Button "Open Optimizations"
- `ScienceGuidesView.swift:51` — Button "Open study designs"

## Notes on strengths

- Help text in this cluster is unusually substantive: `SubstrateSelector`, `WorkspaceSelector`, `WorkspaceMismatchBanner`, and the model pickers each explain scope, consequence, and the gate that refuses ("refused while server jobs are running") rather than restating the label.
- Destructive-adjacent actions name their cost up front: "Download ~X GB and load" (`ChatView.swift:296-304`), the 3–35 GB note and resumable-cancel in `AddLocalModelButton`, and the drift consequence rendered visibly (not hover-only) in `InlineFileEditorSheet`.
- The viewer pin (`ChatView.swift:219-254`) carries origin-aware help and a badge that says whose data a pinned pane shows; the environment-pinned workspace explains its disabled items inline in the menu.
- Every truncated file name/hash in `FileReferenceRow` carries its full value in `.help`; the editor refuses lossy loads (non-UTF-8, >2 MB) with a stated reason instead of corrupting on save.
