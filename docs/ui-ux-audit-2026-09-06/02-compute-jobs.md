# 02 — Compute section: keys rows, model capabilities, server jobs, GPU session, local engine setup

Files read in full (every line): `SectionContainers.swift` 150–600, `ServerJobsPanelView.swift` (938), `GPUSessionControls.swift` (469), `ComputeChoiceAccessory.swift` (93), `ModelJobGPUWarning.swift` (156), `LocalEngineSetupSheet.swift` (449). Cross-checked in ExperimentKit: `AnthropicKeyStore`, `JudgeKeyStore`, `HuggingFaceTokenStore`, `ChatService.syncJudgeKeyNow`, `GPUSession.swift` (display labels, sizing, preflight copy), `ClusterClient.ClientError`, `RemoteJobStatusClass`, `WorkspaceRunImport.SetupError`, `EvidenceImportOrigin.changedRepair`; in the app: `ChatView` (column hosting), `SteerLabApp` (toolbar), `WorkspaceControls` (`InstallModelButton`, compute-binding menu), `ClusterConnectionDot` (sheet presentation); in the engine: `api/gpu_session.py` (gres fallback).

## Coverage table

| File | Controls found | With `.help` (own or group) | Missing `.help` | Icon-only missing `accessibilityLabel` |
|---|---|---|---|---|
| SectionContainers.swift (150–600) | 12 (InstallModelButton, Refresh, 3 SecureField, 6 Save/Clear, 1 Picker) | 7 | 5 | 0 (KeyStoredBadge is an indicator, not a control; it has both help and a11y label) |
| ServerJobsPanelView.swift | 15 (Refresh, Import runs, Import evidence ×2, Retrieve partial data, Resume, Retry missing judgments, shard chip, Stop, Stream, Cancel, 3 context-menu items) | 10 | 5 | 0 |
| GPUSessionControls.swift | 12 (+4 confirmation-dialog buttons where `.help` is not applicable) | 6 | 6 | 0 |
| ComputeChoiceAccessory.swift (AppKit) | 1 (NSSegmentedControl) | 0 | 1 | 0 |
| ModelJobGPUWarning.swift | 0 standing controls (6 confirmation-dialog buttons; `.help` not applicable) | – | – | 0 |
| LocalEngineSetupSheet.swift | 7 (+2 dialog buttons) | 4 | 3 | 0 |

## Findings

### Split-view minimum-height hazard (extra item c) — the Compute column does NOT keep a constant minimum

`ComputeSectionView` is placed directly in the `HSplitView` (`ChatView.swift:88-91`), not inside a ScrollView, so every row above the `Divider` at `SectionContainers.swift:161` adds to the column's reported minimum. `ServerJobsPanelView` was carefully made constant (`jobsRegion` 280 floor, always-present status slot, import detail as a tooltip rather than a row, pipelines rows inside the List), but the rows stacked above it were not:

- **[BUG]** `SectionContainers.swift:277-286` — `ModelCapabilitiesRow` record list — renders one `Text(...).lineLimit(2)` per probed model via `ForEach`, unbounded, with no ScrollView and no fixed frame, and re-reads on `.onChange(of: service.workspaceModelOptions.count)` (line 292) — i.e. it grows by up to 2 lines × N models at the moment the server's model list lands asynchronously after connect. This is exactly the class documented at `ServerJobsPanelView.swift:95-107` and `319-329` ("ANY ~40pt of extra fixed height reproduced it"). (Hazard identified from code; not reproduced.) — Should be a constant-height slot: e.g. a single summary line ("3 probed models") with the per-model lines in a popover/`.help`, or a `ScrollView` with a fixed `.frame(height:)` as `LocalEngineSetupSheet` does.
- **[BUG]** `SectionContainers.swift:195-231` — `pairingWarningRow` — a conditional row that appears/disappears with async cluster state (`activeServerPairingWarning` / `activeServerPairingDescription` come from `/api/info` after connect). The comment says "Bounded height" but bounded is not constant: the warning branch (`.callout`, 2 lines, vertical padding 6) and the description branch (`.caption`, 2 lines, padding 4) have different heights, and both differ from "nothing". — Make it an always-present slot with a fixed line count (the pattern `ServerJobsPanelView.swift:60-69` already uses for `status`).
- **[BUG]** `SectionContainers.swift:433-438` — `ExternalJudgeKeyRow` sync-result `Label` — conditional, no `lineLimit`, set asynchronously by `syncJudgeKeyNow()` and also during `connectCluster()` (`ChatService.swift:3017`), so it can appear mid-connect. — Constant slot with `lineLimit`, full text in `.help`.
- **[POLISH]** `SectionContainers.swift:522-527` — `HuggingFaceTokenRow` `fileError` `Label` — conditional, no `lineLimit`; set synchronously on Save/Clear (lower risk than the three above) but still moves the column minimum by a multi-line row. — Same constant-slot treatment.

### Job cancel / GPU stop (extra item b)

- **[BUG]** `ServerJobsPanelView.swift:311-315` (context menu "Cancel Job") and `:424-430` (log-header "Cancel", `stop.fill`) — both call `cancel(_:)` → `client.cancelJob` immediately with no confirmation; the only gate is `finishedAt == nil`. Cancelling a Slurm job kills a possibly hours-long run; the same app confirms GPU "Release" (`GPUSessionControls.swift:119-135`), toolbar GPU "Stop" when jobs are running (`:293-304`), and engine restart (`LocalEngineSetupSheet.swift:71-86`). — Add a `confirmationDialog` naming the job kind and id (and the shard, when a chip is selected).
- **[BUG]** `GPUSessionControls.swift:137-141` — Playground `Button("Stop")` — calls `controller.stop()` directly, while the toolbar `Stop` for the SAME session (`:343-362`) first counts unfinished server jobs via `GPUSessionStopCheck` and confirms. Two surfaces, one session, different safety. — Route the Playground button through the same `requestStop()` logic.
- **[DESIGN]** `SectionContainers.swift:372, 465, 547` — three `Clear` buttons — delete a stored credential (the judge one also removes it from the cluster at next sync, per its own help) with no confirmation. Recoverable by re-entry, so low severity, but "Clear" gives no pause before a Keychain delete that propagates. — Consider a confirmation on the judge-key Clear at minimum.

### Button correctness

- **[BUG]** `GPUSessionControls.swift:422, 437-439, 459-468` — start sheet `Walltime (HH:MM:SS)` / `Start` — the only validation is non-empty; a malformed walltime is sent, the sheet has already dismissed (`dismiss()` before the `Task`), and the failure lands only in `controller.lastActionError`, which is rendered in exactly one place: the Playground's GPU Session section (`:177-182`, confirmed by grep — no other reference in the app). When the sheet is opened from the toolbar control (`SteerLabApp.swift:161`) while another section is showing, the error is invisible: the toolbar just returns to "GPU Off". — Validate the `HH:MM:SS` shape in the sheet (disable Start with a reason), and surface `lastActionError` in the toolbar control (e.g. red bolt + `.help`, or a transient popover).
- **[BUG]** `GPUSessionControls.swift:459-464` — `start()` sends `gres: gpuType.map { "gpu:\($0):1" }` — when the suggestion has `gpuType == nil` (site without a VRAM table; `GPUSession.swift:497-500` then returns `gres: slurm?.defaultGres` and the rationale "site declares no GPU VRAM data — using the site default", which the sheet displays at line 410), the request drops that default and sends `gres: nil`. The other start path, `startWithDefaults` (`:47-51`), sends `suggestion.gres`. The engine backstops nil with `STEERLAB_SESSION_GRES`/its own default (`api/gpu_session.py:787`), so the session still starts, but the sheet's caption promises the site default while the request omits it (unverified whether the two defaults always agree). Also with `gpuTypes` declared but `gpuVRAMGB` empty, the `Picker` (`:416`) shows with a nil selection and no highlighted row. — Fall back to `model.suggestion.gres` when `gpuType` is nil; hide or pre-select the Picker when no type could be suggested.
- **[BUG]** `SectionContainers.swift:397-401, 574-578` — `Save` (Claude key, HF token) — Keychain write failures are swallowed: `AnthropicKeyStore.save` and `HuggingFaceTokenStore.save` both discard `writeKeychain`'s result (`_ = storage.writeKeychain(trimmed)`, `AnthropicKeyStore.swift:113`, `HuggingFaceTokenStore.swift:123`). The row then clears the draft and `refresh()` shows "no key stored" with no message, indistinguishable from never having typed. `JudgeKeyStore.save` returns a `Bool` (`JudgeKeyStore.swift:62-66`) that `ExternalJudgeKeyRow.save()` (`:485-490`) ignores. — Surface a "could not write to the Keychain" line in the row's caption slot when the write fails.
- **[BUG]** `GPUSessionControls.swift:436` and `LocalEngineSetupSheet.swift:432` — sheet `Cancel` / `Close` — no `.keyboardShortcut(.cancelAction)`, so Esc does not dismiss either sheet (both have `.defaultAction` on the primary). In the engine sheet the button labelled "Cancel" (`:427`) aborts provisioning rather than dismissing, so `.cancelAction` must go on `Close`, not on that Cancel. — Add `.keyboardShortcut(.cancelAction)` to the dismissing button in each sheet.
- **[DESIGN]** `ServerJobsPanelView.swift:301, 934-937` — context menu "Copy Job ID" — writes the pasteboard with no feedback; the always-present status slot is right there and unused. — `status = "copied job id \(id)"` (or a transient checkmark).
- **[DESIGN]** `SectionContainers.swift:485-496` — judge-key Save/Clear — kick off `service.syncJudgeKeyNow()` with no busy state; a successful push sets `judgeKeySyncResult = ""` (`ChatService.swift:2987-2988`) which renders nothing, so there is never a positive "pushed to <site>" confirmation, only silence vs. an orange warning. A stale warning from an earlier connect also persists (service-level string) until the next sync. — Show "syncing…" then "pushed to <site> at <time>" in the caption slot.
- **[DESIGN]** `ServerJobsPanelView.swift:57` — "Import runs" `.help(importDetail ?? Self.importHelp)` — after the first import the tooltip is permanently replaced by the last report; the description of what the button does is gone until relaunch. — Keep the description and append the last report under it, or move the report into a popover.

### Log viewer (extra item d)

- **[DESIGN]** `ServerJobsPanelView.swift:434-441` — log `ScrollView` — text is selectable (good) but there is no auto-scroll: streamed lines append (`:586-591`) and the view never scrolls to the bottom (no `ScrollViewReader`/`scrollTo`, no `.defaultScrollAnchor(.bottom)`), so a live stream fills below the fold. No "Copy log" action either (only manual selection of a 2,000-line `Text`). The 2,000-line cap (`:589-591`) silently drops the head with no "(earlier lines dropped)" marker. — Add bottom-anchored auto-scroll while streaming, a Copy-log button with status feedback, and a truncation marker line.
- **[DESIGN]** `ServerJobsPanelView.swift:410-414` — `Button("Stop")` in the log header — sits beside `Cancel` (which cancels the JOB); "Stop" stops only the log stream and has no `.help` and no object in its label. — "Stop stream" (or an icon with help), and keep it visually distinct from the destructive Cancel.

### Polling / refresh / staleness (extra item e)

- **[DESIGN]** `ServerJobsPanelView.swift:38-44, 465-501` — job list — Refresh is visible and shows busy state (spinner + disabled), but the list only refreshes on manual Refresh, target change, after an action, or when a stream ends; there is no auto-poll and no "as of <time>" stamp, and the status slot only says "N jobs". A queued job can sit showing "queued" indefinitely with nothing telling the researcher the row is minutes old. — Add a last-refreshed timestamp to the status line (and/or a modest auto-refresh while any job is in flight).
- **[POLISH]** `ServerJobsPanelView.swift:116-125` — `jobsRegion` branches — during the very first refresh (`isRefreshing == true`, `jobs` empty) the `else` branch renders an empty `List` plus a log box reading "No log output yet.", then swaps to the "No Jobs" empty state when the fetch lands. Layout-safe (constant 280 frame) but a visible flash. — Show a loading placeholder inside the same frame while `isRefreshing && jobs.isEmpty`.

### Secret-entry fields (extra item a) — mostly sound, with the a11y gaps below

Verified good: all three fields are `SecureField`s; drafts are cleared after Save/Clear; the stored secret is never read back into the field, a label, or a tooltip; captions state only presence/kind; all three stores trim whitespace/newlines so a pasted key with a trailing newline is safe, and the Save-enabled gate trims the same way; the HF file-error text carries the path and `localizedDescription`, never the token.

- **[A11Y]** `SectionContainers.swift:365, 456, 539` — three `SecureField`s — the visible label ("Claude API key", "External judge key", "Hugging Face token") is a sibling `Text`, not the field's label, so VoiceOver announces the placeholder ("sk-ant-…", "hf_…") and nothing else; no `.help` on the field either. — `.accessibilityLabel("Claude API key")` etc., plus a `.help` naming the token type and where to get it.
- **[A11Y]** `SectionContainers.swift:450-455` — `Picker("", selection: $kind).labelsHidden()` — no accessible name and no `.help`; a screen reader hears "OpenRouter, pop-up button" with no idea what the choice governs. — `.accessibilityLabel("Judge key provider")` and `.help("which service issued the key — decides how the cluster's inline judge authenticates")`.

### Copy, layout, and consistency

- **[DESIGN]** `SectionContainers.swift:380-395, 474-483, 556-572` — the three status captions — each is a 40–70-word paragraph in `.caption2` mixing state, policy, instructions, and ALL-CAPS emphasis ("THIS Mac", "SPEND-CAPPED", "READ token"), e.g. the Claude caption runs "a key is stored — stored in the macOS Keychain and never sent to a server: Claude judging, stimulus generation, and sweep credential checks all run on THIS Mac (cluster generations are judged here after download; pin a local judge for cluster-side judging)". — One short status line ("a key is stored · Keychain only, never sent to a server") plus an `InfoButton` carrying the policy paragraph.
- **[DESIGN]** `SectionContainers.swift:266` — `Button("Refresh")` (Model capabilities) — no `.help`, and a second "Refresh" (with icon) lives 200pt lower in the same column refreshing the job list (`ServerJobsPanelView.swift:41`). Vague verb, no object. — "Re-read records" or keep "Refresh" with `.help("re-read prompts/models/ in this workspace")`.
- **[DESIGN]** `SectionContainers.swift:171` vs `ServerJobsPanelView.swift:29` — `statusLine` and `connectionSummary` — for a server target both render `service.cluster.status` ("Compute: <server> / <status> · N models available" then "Server Jobs / <status>"), so the same connection status appears twice within one column. — Drop one (the panel sub-caption is the natural candidate).
- **[DESIGN]** `SectionContainers.swift:185-187`, `ServerJobsPanelView.swift:111-115`, `SteerLabApp.swift:195-200` — vocabulary — the header help says "switch the compute target with the Compute selector in the window toolbar", the empty state says "Choose a server workspace in the toolbar, then connect", the toolbar menu reads "Compute: …" with an inline picker titled "Workspace". Three names for one control. The header help also says "this section shows the active target's jobs and logs", which is untrue for the Local target (the panel then shows "No Active Server"). — Pick one term (the toolbar's "Compute") and use it in both strings; qualify the header help for Local.
- **[POLISH]** `ServerJobsPanelView.swift:27-31, 111-120` — Local target — the column stacks "Server Jobs" / "Local workspace selected" / "No Active Server" / "Choose a server workspace…": three framings of one fact, and a headline promising server jobs on a local target. — For `.local`, a single empty state ("Jobs appear here when a server compute target is active") without the "Server Jobs" headline.
- **[DESIGN]** `LocalEngineSetupSheet.swift:423, 427, 432-444` — `Re-check` / `Cancel` / `Close` / `Set Up` (→ `Re-verify`) — four buttons, two near-synonyms: "Re-check" re-plans without acting, "Re-verify" runs the whole flow; "Cancel" aborts work while "Close" dismisses. Only Cancel has `.help`. — "Re-check" → `.help("re-read what is installed without changing anything")`; rename "Cancel" to "Stop Setup"; add help to Close and Set Up/Re-verify.
- **[DESIGN]** `LocalEngineSetupSheet.swift:246, 260` — `"unload failed: \(error)"`, `"cancel failed: \(error)"` — `\(error)` on `ClusterClient.ClientError.badResponse` prints `server returned 502: <raw body>` (`ClusterClient.swift:2195`), i.e. the JSON body, and on any other error the Swift debug form rather than `localizedDescription`. `ServerJobsPanelView` already does this right with `ClusterClient.unwrappingDetail` (`:856, 873`). — Use `unwrappingDetail(...).description` for `ClientError` and `localizedDescription` otherwise.
- **[POLISH]** `ServerJobsPanelView.swift:566` — status `"evidenceContextChanged: Reconnect to the originating server…"` — an internal error code as the visible prefix. — Drop the code (or put it in the `.help`).
- **[POLISH]** `LocalEngineSetupSheet.swift:369, 393` — `"site qualify — <by> on <platform>"`, `"engineVersion (from /api/info): …"` — CLI verb and JSON field names in user copy. — "Acceptance checks — …" and "Engine version: …".
- **[POLISH]** `GPUSessionControls.swift:86, 90` — `Section("GPU Session")` whose first row reads "GPU Session: Ready" — the words appear twice. — Row text "Ready · 1h 42m" is enough under that header.
- **[POLISH]** `GPUSessionControls.swift:113, 137, 276` — ellipsis convention — "Start…" and "Restart Engine…" use the trailing ellipsis for a further-dialog action, but "Release (verified gone)" always confirms and the toolbar "Stop" conditionally confirms, neither with an ellipsis; "(verified gone)" in a button title reads oddly next to the confirmation that already says so. — "Release…" with the verification in the dialog (already there).
- **[POLISH]** `GPUSessionControls.swift:318-331` — toolbar `Stop` — disabled during `.ending`, but its help still says "end the GPU session (asks first…)". — "already ending" while disabled for that reason.
- **[POLISH]** `ServerJobsPanelView.swift:44, 423` — `.help("Refresh the durable job list from the active server.")`, `.help("Stream live log output for the selected job.")` — sentence case with trailing periods; every other help string in this cluster is a lowercase fragment. — Match the convention.
- **[POLISH]** `SectionContainers.swift:270-273` — empty-state text "…until then declarations are gated on the model id and say so" — jargon with no plain-words half. — "…until then, the app decides what a model supports from its name alone and says so in each declaration".

### View/state details

- **[BUG]** `SectionContainers.swift:171-175` — header `statusLine` — `.lineLimit(1).truncationMode(.middle)` on dynamic text that includes `service.cluster.status` (e.g. "connection failed: SSH tunnel manager is unavailable · 0 models available"); the enclosing `.help` is the static "switch the compute target…" explanation, so the truncated text has no full-text carrier. — `.help(statusLine)` on the Text (it wins over the group help on hover).
- **[BUG]** `ServerJobsPanelView.swift:291-296` — job `error` `Text` — `.lineLimit(2)`, no `.help`, and inside a `List` row so it cannot be selected/copied; server failure reasons are routinely longer than two caption lines. — `.help(error)` and a "Copy error" context-menu item.
- **[POLISH]** `ServerJobsPanelView.swift:275-284, 401-404` — `Text(job.id)`, `Text("scheduler \(executorJobID)")`, `Text(selectedJobID ?? …)` — all `.lineLimit(1)` with no `.help`; long ids truncate (the context menu's Copy Job ID mitigates the first). — `.help` with the full id on each.
- **[A11Y]** `ServerJobsPanelView.swift:65-69` — the status slot — the ONLY place refusals and errors render ("cancel failed: …", "import refused: …", "resume failed: …", the repair sentence) and it is `.lineLimit(1)` without `.textSelection(.enabled)`; the researcher cannot copy a refusal or repair action. — Enable selection (the constant height is unaffected).
- **[A11Y]** `GPUSessionControls.swift:163-166` — `Button("Dismiss")` — `.buttonStyle(.plain)` at `.caption` size: hit target roughly 12×40pt; no `.help`. — `.controlSize(.small)` bordered, or a proper close affordance.
- **[A11Y]** `ServerJobsPanelView.swift:375-389` — shard chips `.controlSize(.mini)` — ~16pt tall targets. Acceptable for chips but below the ~20pt guideline; keep the surrounding padding generous.
- **[A11Y]** `ComputeChoiceAccessory.swift:30-35` — `NSSegmentedControl` "Computes on:" — no `toolTip`, no `setAccessibilityLabel`, and the "Computes on:" `NSTextField` is not linked as its title element, so the control is announced as an unnamed segmented control. — `control.setAccessibilityLabel("Computes on")` (or `setAccessibilityTitleUIElement(label)`) and `control.toolTip = …`.
- **[POLISH]** `LocalEngineSetupSheet.swift:298` — `ForEach(engine.downloadPreamble, id: \.self)` — string identity; two identical preamble lines would collide (unverified whether the provisioner can emit duplicates).
- **[POLISH]** `ServerJobsPanelView.swift:910-915` — `formatTimestamp` — allocates a `DateFormatter` per call, several per row per render. Not a UX defect; a cached formatter is the usual fix.

No old section names, no internal codenames (WP3/WS2/WS3/§ refs are comment-only), and no dangling doc paths were found in the user-visible strings of these six files.

## Missing tooltips (exhaustive)

Standing controls with no `.help` on the control or an enclosing group:

- `SectionContainers.swift:266` — `Button("Refresh")` (Model capabilities)
- `SectionContainers.swift:365` — `SecureField("sk-ant-…")` (Claude API key)
- `SectionContainers.swift:450` — `Picker("", selection: $kind)` OpenRouter/Anthropic (also no accessibility label)
- `SectionContainers.swift:456` — `SecureField("sk-or-…"/"sk-ant-…")` (External judge key)
- `SectionContainers.swift:539` — `SecureField("hf_…")` (Hugging Face token)
- `ServerJobsPanelView.swift:301` — context-menu `Button("Copy Job ID")`
- `ServerJobsPanelView.swift:305` — context-menu `Button("Resume from Checkpoint")`
- `ServerJobsPanelView.swift:311` — context-menu `Button("Cancel Job", role: .destructive)`
- `ServerJobsPanelView.swift:410` — `Button("Stop")` (stop log stream)
- `ServerJobsPanelView.swift:424` — `Button(role: .destructive)` `Label("Cancel", systemImage: "stop.fill")` (cancel job)
- `GPUSessionControls.swift:163` — `Button("Dismiss")` (session notice)
- `GPUSessionControls.swift:416` — `Picker("GPU type")` (start sheet)
- `GPUSessionControls.swift:422` — `TextField("Walltime (HH:MM:SS)")` (start sheet)
- `GPUSessionControls.swift:423` — `Stepper("Idle timeout: N min")` (start sheet)
- `GPUSessionControls.swift:436` — `Button("Cancel")` (start sheet)
- `GPUSessionControls.swift:437` — `Button("Start")` (start sheet; disabled with no reason when walltime is blank)
- `ComputeChoiceAccessory.swift:31` — `NSSegmentedControl` "Computes on:" (no `toolTip`)
- `LocalEngineSetupSheet.swift:423` — `Button("Re-check")` (disabled while running, no reason)
- `LocalEngineSetupSheet.swift:432` — `Button("Close")`
- `LocalEngineSetupSheet.swift:433` — `Button(startTitle)` "Set Up" / "Re-verify"

Confirmation-dialog buttons (help not renderable; listed for completeness): `GPUSessionControls.swift:126, 129, 298, 301`; `ModelJobGPUWarning.swift:119, 123, 126, 128, 138, 141`; `LocalEngineSetupSheet.swift:76, 79`.

## Notes on strengths

- `ServerJobsPanelView.jobsRegion` / `jobList` / `logViewer` / the status slot are a textbook treatment of the split-view minimum-height class, with the incident reasoning written where the next editor will read it; the pipelines-awaiting rows living inside the List is the right call.
- Credential handling is careful end to end: SecureFields, trimmed stores, empty-means-delete, no echo, the `KeyStoredBadge` with both `.help` and `.accessibilityLabel`, and the HF hub-file coexistence logic ("only if it holds this same token").
- Refusals and server detail are surfaced verbatim via `unwrappingDetail` rather than as JSON in the jobs panel, and every async row action captures `jobsOrigin` and re-checks it before mutating state.
- The GPU "Release (verified gone)" path and "Restart Engine…" both confirm with a message that names the exact manual check, and `ModelJobGPUWarning` composes two preflight rules into one dialog with the more specific concern winning.
