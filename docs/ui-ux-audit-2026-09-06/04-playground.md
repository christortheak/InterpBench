# 04 — Playground: chat transcript, steering controls, composer, activity feed

Scope read in full: `ChatView.swift` 260–2932 (plus 1–259 skimmed for state context only), `ActivityFeedColumn.swift`, `TemperatureRow.swift`, `ReaderPinView.swift`, `InjectionModeControls.swift`. Cross-checked against `ExperimentKit/ChatService.swift`, `ExperimentKit/InjectionModeCopy.swift`, `ExperimentKit/FineTuningPanel.swift`, `ModelVariantsPanelView.swift` (the only `InjectionModeControls` caller), `WorkbenchSection.swift`, `docs/CLI-REFERENCE.md`. No files edited, no builds run.

Views instantiated inside the steering panel but defined elsewhere (`WorkspaceModelPicker`, `InstallModelButton`, `AddLocalModelButton` in WorkspaceControls.swift; `GPUSessionSection` in GPUSessionControls.swift) are excluded from the counts below.

## Coverage table

| File | Controls found | With `.help` (own or group) | Missing `.help` | Icon-only / label-less missing `accessibilityLabel` |
|---|---|---|---|---|
| ChatView.swift (260–2932) | 64 interactive controls + 3 context-menu items + 2 confirmation-dialog buttons | 49 | 15 (context-menu and dialog buttons not counted — conventional) | 8 (pencil 2321, copy 2328, trash 1821, trash 1986, `Toggle("")` 1791/1959, `Picker("")` 1794/1986→1962) |
| ActivityFeedColumn.swift | 1 (folder button 265; `MessageBubble` reused, counted in ChatView) | 1 | 0 | 1 (folder 268) |
| TemperatureRow.swift | 2 (Slider, TextField) | 2 (group-level `.help` on the `LabeledContent`, 26) | 0 | 0 |
| ReaderPinView.swift | 2 (Remove 58, Menu 83) + N menu-item buttons | 2 | 0 | 0 |
| InjectionModeControls.swift | 5 (Picker 37, Stepper 73, TextField Layer 79, TextField Alpha 86, TextField λ 97) | 5 (Picker own; others via group `.help` on the HStacks, 90/106) | 0 | 0 |

Counts come from reading every control site; the enumeration is in "Missing tooltips" below.

## Findings

- **[BUG]** `ChatView.swift:925-937` + `ChatView.swift:2858-2870` — composer Return key / `sendDraft()` — `sendDraft` clears `draft = ""` and `stagedLongPrompt = nil` *before* calling `service.send(text)` / `service.seedAssistantTurn(text)`, and `PromptInputTextView.keyDown` fires `onSubmit` with no check of the Send/Seed button's disabled predicate. `ChatService.send` (ChatService.swift:2361-2380) returns early with only `errorMessage` set when `turnConstraintReason(for: .user)` is non-nil (e.g. consecutive user turns on a gemma-3 template), and `seedAssistantTurn` (ChatService.swift:2524-2531) returns early when `seedUnavailableReason` is non-nil (e.g. a server without `chat.seededTurns`). In both cases the box is still editable (`canSendPrompt` is true), the button is greyed, but pressing Return destroys the typed text and leaves only the red banner. — Guard `sendDraft` on the same predicates the buttons use (`turnConstraintReason`, `seedUnavailableReason`, `activeDraftText.isEmpty`) or clear the draft only after the service accepts.
- **[BUG]** `InjectionModeControls.swift:43-49` — Mode picker `onChange` — switching Ablate → Steer resets `strength` to the literal `2` ("α is typically 1–3" is raw-unit reasoning). The only caller (ModelVariantsPanelView.swift:819) binds this to a draft whose default is `alphaInNormUnits = true` (ModelVariantsPanelView.swift:86); α = 2 in norm units is past the "collapse above ~1" range stated in the Playground's own help (ChatView.swift:1428-1433). Playground's equivalent (`ChatView.swift:1725-1737`) instead re-asks the artifact via `applyDefaultsForSelectedVector`. — Reset to the artifact's default (or a norm-aware default), not a literal 2.
- **[BUG]** `InjectionModeControls.swift:85-90` — "Alpha" TextField — unit-blind: the label carries no norm/raw indication, and the group `.help` (`InjectionModeCopy.alphaHelp`, "α in units of the layer's residual-stream norm…") asserts norm units unconditionally while the calling panel has an "Alpha in residual-norm units" toggle (ModelVariantsPanelView.swift:725) that can be OFF — the tooltip is then wrong. Playground labels the same field "α (norm units)" / "α (raw)" (ChatView.swift:2159-2169). — Take a units flag and label like Playground; make the help conditional.
- **[BUG]** `ChatView.swift:1653-1656` — Playground "Temperature" — a bare `Slider` with no numeric readout. `TemperatureRow.swift:5-9` documents this exact defect ("finding 7a … '0' and '0.1' … looked identical") and fixes it for Studies, but Playground kept the bare slider. — Reuse `TemperatureRow` (or add the TextField readout).
- **[BUG]** `ChatView.swift:878-881` — "Reset Chat" — irreversibly empties the transcript (`ChatService.resetChat`, 2353-2359) with no confirmation and no `role: .destructive`; it also does not clear `errorMessage`, so a stale red banner survives the reset. — Confirm when `conversationTranscript` is non-empty; clear the banner on reset.
- **[BUG]** `ChatView.swift:387-393` — error banner — `Text(error)` is the only surface for Send/Seed/edit/download errors, yet it is not selectable (`.textSelection` missing) and has no dismiss control; it persists until the next successful action. — Add `.textSelection(.enabled)` and a dismiss button.
- **[BUG]** (unverified — pattern match on the fatal class) split-column minimum height varies with async state, outside any compressible container: `ChatView.swift:699-720` (multi-agent `liveRunFailure` text and the unbounded `liveRunWarnings` VStack — one row per warning — sit *above* the ScrollView), `ChatView.swift:803-822` (robustness failure + judge status rows), `ChatView.swift:387-393` (error banner), `ChatView.swift:463-467` (`StagedLongPromptCard`, up to ~6 caption lines), `ChatView.swift:557-594` (four conditional composer caption rows), `ActivityFeedColumn.swift:25-27` (`runningBar` appears only while something runs; a horizontal ScrollView still has an intrinsic vertical minimum — the same file's line 45 comment says minimums must not vary). None uses `.frame(minHeight:)`, so the risk is below the known-fatal case, but the rule is "keep column minimums constant". — Move failure/warning stacks inside the ScrollView; put composer captions in a fixed-height or compressible container.
- **[DESIGN]** `ChatView.swift:954-957`, `ActivityFeedColumn.swift:179-182` — `copyToClipboard` (Copy Transcript 883, Copy Run 675, Copy 778, "Copy this turn" 2328, context "Copy turn" 2360) — no feedback of any kind after the click. — Swap the label to "Copied" / checkmark for ~1.5 s.
- **[DESIGN]** old section names / retired terms in user-visible copy:
  - `ChatView.swift:1262`, `:1266` — Save Agent help: "save the current **Steering tab** configuration…" → Playground.
  - `ChatView.swift:1467` — Use adapter help: "adapter artifacts are registered in **Fine-Tuning**"; `:1493` — "Train one in **Fine-Tuning** (server job)". No section is called Fine-Tuning (sections: Home, Playground, Data, Agents, Multi-Agent, Templates, Studies, Results, Analysis, Compute; adapter training lives under Data per WorkbenchSection.swift:51).
  - `ChatView.swift:1839`, `:2011` — Layer slider help: "see the norm-by-layer curve in the **Concepts panel**" → Data (or Analysis).
  - `ReaderPinView.swift:76` — empty hint: "fit one from the **Concept Lab**" → Data.
  - `ChatView.swift:1266` "(**variant** artifact)", `:1725`→`:1793` "include this vector in the composed **variant** spec", `:1888` "through the composed **variant** spec", `:1679` "**model variants** created by the app, **web app**, or CLI" — "variant" is the retired user-facing name (Agents is current); "web app" (unverified) — no web app exists in this app target.
  - `ChatView.swift:891`, `:898` — "Copy/Save the full **steering transcript**…", and the exported markdown header "# SteerLab Steering Transcript" (ChatService.swift:1299, produced by these buttons) → "Playground transcript".
  - `ChatView.swift:1217` — `Section("Steering")` — the Playground Form group is titled with the retired section name, directly under a sidebar entry whose help says "(formerly Steering)". Acceptable as a concept word; consider "Steering controls".
- **[DESIGN]** `ChatView.swift:2471-2497` — TranscriptTurnEditor action buttons — "Save as is", "Return control to user", "Continue generation from here", "Restart conversation from here" have no `.help`; "Save as is" is ambiguous (it means: save the edit, leave the branch intact), and "Return control to user" silently truncates every later turn and flips the composer to User (ChatService.swift:2602-2616) — that consequence is stated nowhere in the sheet; the orange `branchUnavailableReason` label (2465) explains only why Continue/Restart is disabled. The `.defaultAction` (⌘↩) is on the most consequential branching action. — Per-button help; a one-line consequence note ("Later turns are removed from context; the original text stays available under the turn"); consider making "Save as is" the default.
- **[DESIGN]** `ChatView.swift:2376` — "Original turn (not in context)" disclosure — "context" is model-context jargon and reads as if the turn is missing; no `.help`. — "Before edit (the model no longer sees this)" plus help.
- **[DESIGN]** `ChatView.swift:1246-1270` — Save Agent section — the "agent name" TextField has no help; the button is disabled with no visible reason when `!workspaceHasRunnableModel`; the outcome of `captureVariant` (FineTuningPanel.swift:1400-1445) goes to `fineTuning.status` and the workspace notices with source "Agents" (`note`, FineTuningPanel.swift:135-138) — nothing under the button in Playground shows success or refusal. — Render `fineTuning.status` inline under the button (or "Saved — open in Agents").
- **[DESIGN]** same action, different labels: "Copy Transcript" (883) / "Copy Run" (678) / "Copy" (781) / "Copy this turn" (2328) / "Copy turn" (2360); "Refresh artifacts" (1664) vs slot "Refresh" (1865) vs help text "then Refresh artifacts" (1275); "Add vector" (1237, lowercase) vs New Agent's "Add Vector" (ModelVariantsPanelView.swift:834); "Remove turn" / trash icon / "Remove" (ReaderPinView.swift:58). — One verb and casing per action.
- **[DESIGN]** `ChatView.swift:1763` vs `InjectionModeControls.swift:109` — ablation scope caption — "Every layer, every token — including the whole prompt" vs "All layers — ablation is not aimed at one layer". `InjectionModeControls` exists (per its doc comment) so Playground/New Agent/Studies "cannot drift", yet Playground reimplements the picker inline (`slotModePicker`/`slotAblationControls`, 1717-1766) and `isCompact` ("the Playground's slot list") has no caller anywhere. — Adopt the shared view in Playground or align the strings.
- **[DESIGN]** `ChatView.swift:1401-1404` — Stepper "Layer band: N" — bounds 1…11 stepping by 2 (odd only); the help ("inject across N consecutive layers…") never states the range or the odd-only rule. — Add "(1–11, odd)".
- **[DESIGN]** `ChatView.swift:485-497` — composer disabled states — while a long prompt is staged the box is non-editable but still paints the ordinary placeholder ("message (server: …)"); while generating the whole box is disabled (`!canSendPrompt`), so the next message cannot be pre-typed. — Placeholder "long prompt staged — Clear to type"; consider gating only Send during generation. Also note `canSendPrompt` (ChatView.swift:366-375) is stricter than `ChatService.send`'s own `canSend` (ChatService.swift:2363-2372 also allows a selected-but-unloaded server model with an inline spec) — the composer greys out a send the service would accept (unverified whether intentional).
- **[DESIGN]** `ChatView.swift:501` — composer "Stop" — no `.help`; its ⌘. shortcut is not surfaced; the help should say the partial text stays in the transcript (ChatService.swift:2400-2403).
- **[DESIGN]** `ChatView.swift:2760` — StagedLongPromptCard "Clear" — no `.help`; "Clear" collides with "clear the conversation" wording in Reset Chat's help. — "Discard staged prompt".
- **[DESIGN]** `.help` convention mixed within one file: sentence case + trailing period at `ChatView.swift:674` ("Stop the active multi-agent run after the current generation observes cancellation."), `:687`, `:790`, `:891`, `:898`, `:2326` ("Edit this turn"), `:2332` ("Copy this turn") vs lowercase fragments everywhere else. `:674` "observes cancellation" is engineer jargon → "stops after the current generation finishes".
- **[DESIGN]** `ChatView.swift:1172-1174` — compute status line — `.lineLimit(1).truncationMode(.middle)` on "Compute: Local (MLX) — <model id> · change in toolbar"; the `.help` does not carry the full text (it says "the workspace is global; switch it with the Compute selector…"), so a truncated model id has no full-text fallback.
- **[POLISH]** `ChatView.swift:659-662`, `:768-771` — `Text(panel.name)` / `robustnessTargetName` — `.lineLimit(1)` with no `.help` carrying the full name.
- **[POLISH]** `ChatView.swift:712` `ForEach(… enumerated(), id: \.offset)`, `:2653` `ForEach(report.warnings, id: \.self)` — index/self ids on the warning lists (duplicate warning strings collapse in the second). Display-only.
- **[POLISH]** `ChatView.swift:2653-2657` — robustness summary warnings (item f) — `Label`s are not `.textSelection(.enabled)` while the multi-agent warnings (713-716) are; no help/expansion for long warning text.
- **[POLISH]** `ChatView.swift:1664-1672` — "Refresh artifacts" is `.controlSize(.small)` while every sibling control in the Form is default size.
- **[POLISH]** `ChatView.swift:2067-2070` — "Measure norms" disabled while `concepts.isWorking` with no visible reason.
- **[POLISH]** `ChatView.swift:591-593` — draft-size caption "· long prompt uses chunked study path" — internal jargon ("study path").
- **[POLISH]** `ChatView.swift:1275-1279` — empty-vectors caption: the server branch ends "then Refresh artifacts"; the local branch gives the CLI hint (`steerlab-cli --config prompts/configs/toy-french.json` — verified valid, CLI-REFERENCE.md:2805, file exists) but does not say to refresh afterwards.
- **[POLISH]** `ActivityFeedColumn.swift:147-149` — live-log entries reuse `MessageBubble`, so a log line's copy button reads "Copy this turn" / "Copy turn".
- **[POLISH]** `ChatView.swift:730-735`, `:827-832` — `ContentUnavailableView` with `minHeight: 360` inside the LazyVStack; harmless because it sits inside a ScrollView, noted only because it is a conditional fixed minimum.
- **[POLISH]** `ChatView.swift:1658-1662` — "Qwen thinking mode" toggle — enabled by a model-id substring; help covers the gate. (unverified) whether this still reflects the engine's current reasoning-effort control.
- **[A11Y]** icon-only buttons with `.help` but no `accessibilityLabel`: `ChatView.swift:2321-2326` (pencil "Edit"), `:2328-2332` (copy), `:1819-1826` and `:1984-1991` (trash "remove box"), `ActivityFeedColumn.swift:265-271` (folder "reveal in Finder"). VoiceOver reads the symbol name.
- **[A11Y]** `ChatView.swift:1791`, `:1959` `Toggle("")` and `:1794`, `:1962` `Picker("")` with `labelsHidden()` — empty labels give VoiceOver nothing; use real labels ("Enabled", "Vector") and hide them.
- **[A11Y]** color-only meaning: `ChatView.swift:2905-2912` probe highlight blue (aligned) vs red (opposite) — the underline is applied to both signs, so sign is color-only; `:2595-2597` variant vs baseline output background blue vs gray (the title text does name the side — partial); `:2576-2577` green `checkmark.circle.fill` completion with no label.

## Missing tooltips (exhaustive)

ChatView.swift:485 — `PromptInputBox` composer text view (custom NSTextView; no `.help`, no accessibility label)
ChatView.swift:501 — Button "Stop" (composer, ⌘.)
ChatView.swift:1247 — TextField "agent name" (Save Agent)
ChatView.swift:1794 — Picker "" vector selector (server slot; `labelsHidden`)
ChatView.swift:1962 — Picker "" vector selector (local slot; `labelsHidden`)
ChatView.swift:2369 — DisclosureGroup "Original turn (not in context)"
ChatView.swift:2405 — DisclosureGroup "Long prompt" (user long-prompt bubble)
ChatView.swift:2457 — TextEditor (turn editor)
ChatView.swift:2471 — Button "Cancel" (turn editor — standard, low priority)
ChatView.swift:2474 — Button "Save as is"
ChatView.swift:2481 — Button "Return control to user"
ChatView.swift:2486 — Button "Continue generation from here"
ChatView.swift:2493 — Button "Restart conversation from here"
ChatView.swift:2629 — DisclosureGroup "Judge N: …" (robustness summary)
ChatView.swift:2760 — Button "Clear" (StagedLongPromptCard)

Conventionally exempt, listed for completeness: context-menu items ChatView.swift:2360 "Copy turn", :2362 "Edit turn…", :2365 "Remove turn"; confirmation-dialog buttons :1195 (download-and-load), :1198 "Cancel"; ReaderPinView.swift:85 menu-item buttons.

ActivityFeedColumn.swift — none missing.
TemperatureRow.swift — none missing.
ReaderPinView.swift — none missing.
InjectionModeControls.swift — none missing (all covered by group-level `.help`; quality issues above).

## Notes on strengths

- Steering-box captions are unusually honest: the α label names its units and, in raw mode, turns orange with a "(raw)" suffix (2159-2169); the injection preview line says "send refuses" rather than implying a fallback (2211-2214); `slotNotAppliedCaption` shares one rule with the send-time refusal so caption and behaviour cannot drift.
- Disabled buttons in the composer carry the *reason* as their tooltip (Send/Seed/Continue at 504-556), and the same reason is echoed as a visible caption — a good pattern the turn editor should copy.
- Provenance badges (seeded / edited / continued) with dashed borders and per-state help text (2262-2300) make researcher-authored turns visibly distinct everywhere they render.
- Norm-units toggle deliberately stays turnable-OFF when unavailable (1411-1413), with the field-report rationale in the comment.
