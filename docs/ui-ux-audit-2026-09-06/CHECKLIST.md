# SteerLab SwiftUI app — UI/UX audit checklist (READ-ONLY review; make NO code edits)

## Context
- Repo: ~/SteerLab/InterpBench. App sources: Sources/SteerLabApp/*.swift (macOS 26.4+, SwiftUI, 42K lines).
- Shell: `ChatView` = NavigationSplitView with a sidebar of 10 sections (Home, Playground, Data, Agents, Multi-Agent,
  Templates, Studies, Results, Analysis, Compute), a CONTROLS pane (left, per-section) and a VIEWER pane (right,
  follows the section unless pinned). `WorkbenchSection.swift` defines sections; `SectionContainers.swift` hosts Data/Compute/Results.
- Current user-facing names: Playground (was Steering), Data (was Concept Lab), Agents (was Variants), Analysis (was Geometry).
  "Screens" is no longer a top-level section (it became Agents › New Agent / Optimizations). Any user-visible copy using the OLD names is a finding.
- Shared help primitives: `InfoPopover.swift` has `InfoButton` (visible ⓘ popover) and `InfoSectionHeader`. Hover `.help(...)` tooltips are
  the secondary layer. The project convention is that `.help` strings are lowercase fragments; note inconsistencies (sentence case, trailing periods) but do not treat lowercase as a defect.
- Known fatal class on this macOS beta: a split-view column whose SwiftUI minimum HEIGHT changes mid-display-cycle (or exceeds the window) crashes the app.
  Flag any view inside a split column whose min height varies with async state (conditional rows with fixed minHeight, `.frame(minHeight:)` on content that appears/disappears).
- Read the ENTIRE file(s) assigned to you, line by line. Do not sample. Use `sed -n` / `cat` via Bash to read. Use grep to cross-check.

## What to audit (be exhaustive — the person explicitly wants EVERY control checked)

### A. Tooltip / help coverage (list EVERY control lacking one, with file:line and its label)
Controls = Button, Toggle, Picker, Menu, TextField, SecureField, TextEditor, Slider, Stepper, DatePicker, Link, tappable Labels/Images with onTapGesture, context-menu items, toolbar items, segmented pickers, sidebar/list rows that act.
- Missing `.help(...)` entirely (check the control's own modifier chain AND an enclosing container that applies .help to the group — a group-level .help counts, note it).
- Icon-only controls (`labelStyle(.iconOnly)`, `Image(systemName:)` as the whole label, `Label(...).labelStyle(.iconOnly)`) missing `.accessibilityLabel` AND/OR `.help`.
- Help text quality: repeats the label verbatim, is vague ("do it"), uses jargon with no plain-words half, references removed features/old section names, references non-existent docs paths, is stale (dates, "formerly"), or contradicts the button's actual behavior in code.
- A `.help` that is the ONLY explanation for a non-obvious gate/refusal (should probably be visible text or an InfoButton).

### B. Button correctness (read each action closure)
- No-op or placeholder actions; actions whose only effect is a print.
- Errors swallowed: `try?` where failure should surface; empty `catch {}`; errors set to a state var that is never rendered.
- Enabled when it cannot succeed (no `.disabled` gate while the action early-returns on a nil/empty precondition) — the user sees a click do nothing.
- Disabled with no visible/ hover explanation of why.
- Destructive/irreversible actions (delete, clear, cancel job, overwrite, reset, force-*) with no confirmation dialog or role: .destructive.
- Async work with no busy indicator and no re-entry guard (double-click runs twice).
- Actions that mutate state but never persist, or persist but never refresh dependent views.
- Copy-to-clipboard with no feedback.
- Buttons in sheets: missing Cancel, missing `.keyboardShortcut(.cancelAction)` / `.defaultAction`, sheet not dismissible, sheet with no fixed/ minimum size so it collapses.

### C. View/state bugs
- `@State` initialised from an init parameter (stale after parent changes); bindings into computed properties; `ForEach` with non-unique or index ids (reorder/delete bugs); `.id()` misuse; `.onAppear` loads with no cancellation vs `.task`; `Task {}` capturing stale values; alerts with empty message; force unwraps / `!` on optional UI state; `lineLimit(1)` truncation of important dynamic text with no `.help` carrying the full text; hard-coded widths that clip localized/long text; `.frame(minHeight:)` in split-view columns (see fatal class); `Text` interpolating a value that can be nil ("Optional(...)" leak); date/number formatting without a formatter; color-only status.
- Layout: controls that will overflow at the section's minimum width (560pt for dense panels; Playground 340; Home 420); HStacks with many buttons and no wrapping; missing ScrollView on tall forms; nested ScrollViews; `Form` mixed with custom layouts.

### D. Design / UX consistency
- Same action, different labels across screens (e.g., "Delete" vs "Remove" vs trash icon; "Run" vs "Start" vs "Submit").
- Vague button titles with no object ("OK", "Go", "Apply", "Do it", "Save" where it's ambiguous what is saved).
- Missing empty states, missing loading states, raw error/JSON dumped to the user, no feedback after success.
- Primary/secondary action hierarchy (`.buttonStyle(.borderedProminent)` on the primary; `.defaultAction`); action placement inconsistent with the rest of the app.
- Text with typos, doubled words, unbalanced quotes/parentheses, "TODO"/dev notes visible to the user, internal codenames (W7, WP2, F4, WS3, chip, phase 4, "live-testing finding") leaking into user-facing strings (comments are fine; user-visible strings are not).
- Old section names / removed features in user-visible copy. Dangling doc references in user-visible copy.
- Controls that exist twice for the same thing on one screen; controls with no visible label.
- `.controlSize` inconsistency within a row; mixed `.font(.caption)` sizes for equivalent captions.

### E. Accessibility / keyboard
- Icon-only without accessibilityLabel; color-only meaning; hit targets under ~20pt; no keyboard shortcut on the primary action of a sheet; `.focusable` issues; text not selectable where users need to copy (paths, hashes, errors).

## Output format
Write a Markdown report to the OUTPUT path you were given, then return the SAME content as your final message. Structure:

1. `## Coverage table` — per file: total controls found, controls with .help (own or group-level), controls missing help, icon-only controls missing accessibilityLabel. Counts must come from reading, not guessing.
2. `## Findings` — one bullet per finding, most severe first, format:
   `- **[BUG|DESIGN|POLISH|A11Y]** \`File.swift:LINE\` — <control/element> — <what is wrong> — <what it should be>`
   Be specific and cite exact lines. Quote the user-visible string when relevant. Do not pad; do not include speculative findings you could not confirm in the code — if unsure, say "(unverified)".
3. `## Missing tooltips (exhaustive)` — every control missing .help, one line each: `File.swift:LINE — <label or description>`.
4. `## Notes on strengths` — 3-5 lines max, only if genuinely notable (helps calibrate).

Do NOT edit any file in the repository. Do NOT run xcodebuild. Do not spawn subagents.
