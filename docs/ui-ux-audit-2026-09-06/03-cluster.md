# 03 — Cluster connection: toolbar dot, health card, setup wizard, site editor, profile coauthoring, maintenance windows

Files read in full (3,098 lines): `ClusterConnectionDot.swift` (658), `ClusterHealthCard.swift` (532), `ClusterSetupWizard.swift` (862), `ClusterSiteEditor.swift` (652), `ClusterSitePreviewView.swift` (136), `ClusterProfileCoauthoringSheet.swift` (102), `MaintenanceWindowsEditor.swift` (156). Base: `<checkout>/Sources/SteerLabApp/`. Cross-checked against `ExperimentKit/ClusterConnectionStore.swift`, `ClusterTunnel.swift`, `ClusterProvisioner.swift`, `ClusterSiteProfile.swift`, `ClusterEnvironmentRenderer.swift`, `ClusterClient.swift`, `ShardedJobs.swift`, `SteerLabApp.swift` (toolbar), `HomeDashboardView.swift`, `GPUSessionControls.swift`, and `Server/scripts/bootstrap.sh`.

## Coverage table

Counts are control *instances* from reading (alert buttons excluded; table row templates counted once per table; the three job-class blocks in the site editor counted separately because each renders its own fields).

| File | Controls | With `.help` (own or group) | Missing `.help` | Icon-only missing `accessibilityLabel` |
|---|---|---|---|---|
| ClusterConnectionDot.swift | 23 (17 menu items + Site picker + menu label + 3 in HF token sheet + 1 disabled placeholder pair) | 3 (menu label L175, Set Up Local Engine L98, Start Local Python Server L114) | 20 | 0 |
| ClusterHealthCard.swift | 4 | 4 | 0 | 0 |
| ClusterSetupWizard.swift | 32 (rail step buttons counted as 1) | 8 | 24 | 0 |
| ClusterSiteEditor.swift | ~132 | ~78 (72 `.help` call sites; `linesEditor`, Port, Idle-minutes sites are reused) | ~54 | 2 (`minus.circle` row-remove templates L213, L255) |
| ClusterSitePreviewView.swift | 5 (DisclosureGroups) | 0 | 5 | 0 |
| ClusterProfileCoauthoringSheet.swift | 5 | 0 | 5 | 0 |
| MaintenanceWindowsEditor.swift | 7 (3 + 4 per-row template) | 1 (trash L112) | 6 | 1 (trash L109) |

Extra-item verdicts for this cluster:

- **(a) Connection dot states** — every state has a distinct WORD next to the glyph (`connectionTitle` L300-311): Local / Not Connected / Authenticate / Connecting / Connected / Degraded, with glyphs `network.slash` / `network.slash` / `lock` / `network` / `network` / `exclamationmark.triangle.fill`. Not color-only. Two gaps: `Connecting` and `Connected` share the `network` glyph (the word carries the difference, acceptable); and **the local Python server's starting/running-but-unconnected phases have no dot state at all** — the dot keeps reading "Local" (grey) until auto-connect flips the active workspace; the phase is only visible by opening the menu (L82). See DESIGN finding below. The bigger problem is that for direct-transport sites the state is *derived by substring-matching a shared status string* (first BUG below).
- **(b) Wizard navigation** — Cancel (`.cancelAction`), Back, Continue and (steps 2-6) Skip Step are in a footer rendered on every step (L776-794). Failed steps are retryable: every Run button is disabled only while `.running` (L423, 552-563, 662, 702, 737). But Continue is gated only on step 1 (L796-800) — see DESIGN finding: a pending step can be passed silently.
- **(c) Secrets** — no field writes a secret into a profile. `Token-file path` (ClusterSiteEditor L443) is a path indirection (help says so). Bearer token → Keychain (`setStoredToken`, ClusterConnectionStore L1023); HF token → Keychain (`setStoredHFToken`, L1012) and materialised remotely over stdin (`ClusterTunnel.installHFToken` L531). `~/.steerlab-token` is read over the tunnel straight into Keychain (wizard L830-835), never into the transcript. One deviation noted below (HF sheet pre-fills the SecureField with the stored token).
- **(d) Site editor help strings** — all 72 call sites read against their fields. Env-var names cited (`STEERLAB_SLURM_GRES/PARTITION/ACCOUNT`, `STEERLAB_ROOT`, `HF_HOME`) are all emitted by `ClusterEnvironmentRenderer.swift` (L374, 439, 576, 581, 608). "default 8080" (L127) matches `SiteEditorModel` L404. The default-partition fallback (L163-165) matches `resolvedDefaultPartition` (ClusterSiteProfile L319-325). `bootstrap --repo` (wizard L416) exists (`bootstrap.sh:54`). No help references a removed flag. Three mismatches found: `Add partition`/`Add GPU type` help describes the *table*, not the button; "WS5" codename at L564; "fetched by the app, never by a job" at L539 has no app-side fetch behind it. Details below.
- **(e) `.frame(minHeight:)`** — none inside the toolbar `Menu` and none in a split column. All four occurrences are inside fixed-minimum sheets: wizard transcript L266 (`minHeight: 80`, conditional on a non-empty transcript, sheet `minHeight: 620` at L39); MaintenanceWindowsEditor L49; site editor Notes L88 and `linesEditor` L608 (inside collapsible DisclosureGroups; sheet `minHeight: 620` at L73). `ClusterHealthCard` lives in `HomeDashboardView`'s `Form` (a split column) and its rows appear/disappear with async state, but none carries a fixed height — clear of the fatal class.

## Findings

- **[BUG]** `ClusterConnectionDot.swift:342-347` and `ClusterHealthCard.swift:87-90` — connection state for direct-transport sites — the dot's label/colour/glyph and the health card's "Connection" row are derived by substring-sniffing `cluster.status` (`contains("failed"|"invalid"|"rejected")` → Degraded; `hasSuffix("...")` → Connecting; else Connected). That string is overwritten by unrelated store operations: `"requesting install of X..."` (ClusterConnectionStore L1675) turns the primary status affordance amber "Connecting"; `"install failed: …"`/`"install rejected (…)"` (L1684, 1687) and `"agent sync: N recipes failed to import"` (L1583) turn it red "Degraded" with the warning triangle while the connection is fine; `"switching server workspace to …..."` (L1333) reads as "Connecting"; conversely `"workspace switch refused (…)"` (L1340) and `"could not save the site registry"` (L1788) leave it green "Connected". Should derive from `cluster.capabilities != nil` plus an explicit connection-phase value (the SSH path already has `TunnelState`), never from a shared free-text line.
- **[BUG]** `ClusterConnectionDot.swift:71` — `Button("Stop GPU Session")` — cancels the worker job with no confirmation, whereas the adjacent toolbar `GPUSessionToolbarControl` routes its "Stop" through `requestStop()`/`GPUSessionStopCheck` with a `confirmationDialog` and a `role: .destructive` button (`GPUSessionControls.swift:119-126, 276`). Same destructive action, two safeguards. Should reuse the guarded path or at least `role: .destructive` + confirmation.
- **[BUG]** `ClusterConnectionDot.swift:51` — `Button("Authenticate…") { tunnel.openAuthTerminal() }` — the `Bool` result is discarded (`@discardableResult`, ClusterTunnel L331); when Terminal cannot be opened the click does nothing visible. The wizard handles the same call with a message and shows the copyable ssh command (L364-375: "could not open Terminal — run the command above yourself"); the menu path has neither the message nor any way to see the command. Should surface the failure and expose the command.
- **[BUG]** `ClusterSetupWizard.swift:350-353, 387-390` — Authenticate step on a direct-transport site — the step says "Direct transport — nothing to authenticate. Continue." but never stamps its record, so the end-of-wizard Summary prints "· Authenticate — pending" (ClusterProvisioner L2096). The Controller-job step handles the same situation correctly by stamping itself skipped (L717-721). Should stamp `.skipped("direct transport")` the same way.
- **[BUG]** `ClusterConnectionDot.swift:204-207, 491-499` — export failure is routed through `importError`, so a failed export appears under an alert titled **"Site Import"** with body "could not export site: …". Also `fileExporter`'s completion (L201-203) ignores its `Result`, so a write failure at save time is swallowed. Should have its own title (or a neutral "Site Profile") and surface the exporter result.
- **[BUG]** `MaintenanceWindowsEditor.swift:153` — `errorLine = "\(error)"` — for `ClientError` this happens to hit `description`, but for any other thrown error (URLError, DecodingError, CancellationError) the user sees the raw Swift dump (`Error Domain=NSURLErrorDomain Code=-1001 …` / `keyNotFound(CodingKeys(…))`). The comment says "Server validation messages, verbatim" — should be `error.localizedDescription`, which `ClientError: LocalizedError` also satisfies.
- **[DESIGN]** `ClusterSetupWizard.swift:796-800` — "Continue" — `canAdvance` only requires a site on step 1; on Authenticate/Push/Bootstrap/Validate/Controller a `.pending` step can be passed with Continue (or by clicking any rail step, L150-151) with no validation message. The wizard's own rule for skipping is "loud, never silent" (Skip Step help L786), yet Continue-on-pending is exactly a silent skip; the Summary shows "— pending" rather than SKIPPED. Should gate Continue on a terminal status, or route Continue-on-pending through `provisioner.skip` with a visible note.
- **[DESIGN]** `ClusterSiteEditor.swift:225-227` — "Add partition" `.help` — text reads "allowed GPU types: comma-separated subset of the inventory below; empty means the whole site vocabulary. qos overrides the site-wide QOS." — that describes the two *columns* (L206, L208), which have no help themselves; the add button gets a tooltip that has nothing to do with adding. Same shape at `:267-270` ("Add GPU type" help describes the inventory's purpose and VRAM/compute-capability semantics). Move the column semantics onto the row fields (or the header row) and give the add buttons a one-line add help.
- **[DESIGN]** `ClusterSiteEditor.swift:564` — "bootstrap.sh path (optional)" `.help` — "site-local path of the **WS5** bootstrap script once provisioned" — internal work-stream codename in user-visible copy. Should read "the bootstrap script".
- **[DESIGN]** `ClusterSiteEditor.swift:481-482` vs `:536-540` — three maintenance-source fields across two sections: Constraints › "Maintenance source" (`constraints.maintenanceSource`, help "URL or free text") and Policy › "Maintenance source URL" + "Maintenance note" (`policy.maintenance.sourceURL/sourceNote`). Overlapping meaning, no help on "Maintenance note", and L539's help "fetched by the app, never by a job" has no consumer in the app (grep finds `maintenanceSourceURL`/`sourceURL` only in the editor and model; unverified whether the engine reads it). Either consolidate, or make the help say what actually happens.
- **[DESIGN]** `ClusterSetupWizard.swift:332-333, 712-713` — topology text — renders `site.topology.rawValue` (`daemonInJob`, `loginDaemon`, `externalServer`) to the user, while the site editor's picker uses "Daemon in a job" / "Login-node daemon" / "External server" (ClusterSiteEditor L137-139). Should use the same human labels.
- **[DESIGN]** `ClusterConnectionDot.swift:131` "Site" picker vs `SteerLabApp.swift:200` `SubstrateSelector` "Workspace" picker (collapsed label "Compute: …") vs `ClusterHealthCard.swift:20` "Cluster health — \(substrateLabel)" vs `MaintenanceWindowsEditor.swift:29` — the same local/servers selection sits in two adjacent toolbar menus under two names, and the concept is called Site, Workspace, Compute, and substrate across this cluster. Likewise "Edit Site…" (dot L148, opens `ClusterSiteEditor`) and "Edit “name”…" (SubstrateSelector L214, opens a different server editor) edit the same entry through two editors. Pick one noun and one editor.
- **[DESIGN]** `ClusterConnectionDot.swift:300-311` + `:82` — local Python server phases — while `localServer.phase` is `.starting` (venv creation "many minutes", per the Start help) or `.running` but not yet auto-connected, the dot reads "Local"/grey; the only signal is `localServer.statusLine` inside the menu. The primary status affordance should carry a "Starting local server" state (item (a)).
- **[DESIGN]** `ClusterSetupWizard.swift:555-569` — real bootstrap button — locked until a dry-run completes, but before the first dry-run the only explanation is the `.help` tooltip ("locked until a dry-run with the current settings completes"); the visible status line only explains it after a dry-run (`awaitingConfirmation`, ClusterProvisioner L1906-1909). Gate explained only by tooltip — a caption under the buttons would do.
- **[DESIGN]** `ClusterProfileCoauthoringSheet.swift:75-84` — "Close" has no `.keyboardShortcut(.cancelAction)`, "Import reviewed profile" no `.defaultAction`; the disabled Import has no visible reason before a draft is loaded (the "Questions remain" headline only exists after a review). The site editor (L626-629) and HF sheet (L629) set both shortcuts — inconsistent.
- **[DESIGN]** `MaintenanceWindowsEditor.swift:77-83` — "Cancel"/"Save" lack `.cancelAction`/`.defaultAction` (same inconsistency as above).
- **[DESIGN]** `ClusterSiteEditor.swift:620-626` — "Cancel" with `model.isDirty` — the footer shows "unsaved changes" but Cancel/Esc discards them with no confirmation; the whole profile form (100+ fields) can be lost by one keypress.
- **[A11Y]** `ClusterSiteEditor.swift:210-214, 252-256` and `MaintenanceWindowsEditor.swift:106-112` — icon-only remove buttons (`minus.circle`, `trash`) have `.help` but no `.accessibilityLabel`; VoiceOver gets the symbol name. Add `.accessibilityLabel("Remove partition")` etc.
- **[A11Y]** `MaintenanceWindowsEditor.swift:94-102` — two `DatePicker`s with `.labelsHidden()` and no `.help`; the sighted cue for which is start and which is end is only the "→" glyph between them.
- **[POLISH]** `ClusterConnectionDot.swift:93-95, 126-127` — `Button("Setting Up Local Engine…") {}.disabled(true)` and `Button("Stop Local Python Server") {}.disabled(true)` — disabled placeholder buttons whose ellipsis titles promise an action; plus L93+L95 are two items for one in-flight state. A `Text` status row is the honest control.
- **[POLISH]** `ClusterConnectionDot.swift:637-641, 648-651` (HFTokenInstallSheet) — `onAppear` pre-fills the `SecureField` with the Keychain token, so "Install on Cluster" is live immediately and a click re-saves and re-pushes the old secret. The type's own doc (L572-573) says "the sheet reports presence, not contents". Not a JSON leak (Keychain confirmed), but it contradicts the stated design; "a token is stored — Reinstall / Replace" is clearer. Also `setStoredHFToken` uses `try? ClusterTokenStore.save` (store L1020), so a Keychain failure is silent while the sheet reports success.
- **[POLISH]** `ClusterSetupWizard.swift:654-656`, `ClusterSiteEditor.swift:535`, `ClusterSiteEditor.swift:340` (via `ShardedJobs.swift:132`) — literal backticks in user-visible text: the wizard line is a `+`-concatenated `String` (no Markdown parsing) so "runs `steerlab-server profile validate`" renders with backticks; `.help` tooltips never render Markdown ("matched against `hostname`", "`sacctmgr show qos …`").
- **[POLISH]** `ClusterSiteEditor.swift:340` — "Max parallel GPU jobs" help (from `ShardedSubmission.siteFieldHelp`) ends "(empty = uncapped stepper, max N)" — "stepper" is a control on the study-submit screen, not in this editor; jargon with no plain-words half here.
- **[POLISH]** `ClusterSiteEditor.swift:206, 208` — partition-row placeholders "all" and "site" read like literal values to type (they mean "empty = whole vocabulary" / "empty = site-wide QOS"); the meaning is only in the misplaced Add-button help above.
- **[POLISH]** `ClusterSetupWizard.swift:754, 778` — after Connect succeeds both "Close" (`.defaultAction`) and the footer's "Cancel" (`.cancelAction`) are shown; "Cancel" on a finished wizard is misleading.
- **[POLISH]** `ClusterSetupWizard.swift:408` — "Local server payload" is a filesystem path typed by hand with no Browse/choose-folder button.
- **[POLISH]** `ClusterHealthCard.swift:173` — storage rows use the raw role key as the row label ("hfCache", "metadata", "workspace").
- **[POLISH]** `ClusterHealthCard.swift:329` — "Edit…" is a vague title with no object; "Edit Windows…" would stand alone in a VoiceOver rotor.
- **[POLISH]** `ClusterHealthCard.swift:385-387` — "Import now" is also disabled when `origin != cluster.evidenceImportOrigin` (workspace/site changed under the listing) but the static help doesn't explain that case.
- **[POLISH]** `MaintenanceWindowsEditor.swift:37` "No windows declared." (sentence case, period) vs `ClusterHealthCard.swift:346` "no windows declared" (lowercase fragment) — same phrase, two styles.
- **[POLISH]** `ClusterSitePreviewView.swift:23, 71-81` — `@State environmentExpanded` seeded from the `expandsEnvironment` init parameter; the stored property is dead after init. Harmless today (no parent changes it), but the pattern the checklist flags.
- **[POLISH]** Nested scroll views (deliberate per the file's layout note, listed for completeness): `ClusterSetupWizard.swift:200` detail `ScrollView` wraps the transcript `ScrollView` (L259) and the preview panes' `ScrollView`s (via L336); `ClusterSiteEditor.swift:53` grouped `Form` wraps the preview panes (L592) and three `TextEditor`s; `ClusterProfileCoauthoringSheet.swift:36` wraps the panes (L68). Inner panes capture the wheel once the cursor is over them.
- **[POLISH]** `ClusterProfileCoauthoringSheet.swift:63-64` — `ForEach(review.advisories, id: \.self)` / `blockers` — duplicate strings would collide as ids (unverified whether the reviewer can emit duplicates).

## Missing tooltips (exhaustive)

ClusterConnectionDot.swift
- `ClusterConnectionDot.swift:51` — Button "Authenticate…"
- `ClusterConnectionDot.swift:55` — Button "Disconnect"
- `ClusterConnectionDot.swift:57` — Button "Connect"
- `ClusterConnectionDot.swift:71` — Button "Stop GPU Session"
- `ClusterConnectionDot.swift:93` — Button "Setting Up Local Engine…" (disabled placeholder)
- `ClusterConnectionDot.swift:95` — Button "Show Setup Progress…"
- `ClusterConnectionDot.swift:124` — Button "Stop Local Python Server"
- `ClusterConnectionDot.swift:126` — Button "Stop Local Python Server" (disabled placeholder)
- `ClusterConnectionDot.swift:131` — Picker "Site"
- `ClusterConnectionDot.swift:140` — Button "Add \(preset.name) preset…"
- `ClusterConnectionDot.swift:142` — Button "Import Site JSON…" (the wizard's twin at L312 has one)
- `ClusterConnectionDot.swift:144` — Button "Export “\(name)”…"
- `ClusterConnectionDot.swift:148` — Button "Edit Site…"
- `ClusterConnectionDot.swift:152` — Button "Install HF Token…"
- `ClusterConnectionDot.swift:157` — Button "Set Up Cluster…"
- `ClusterConnectionDot.swift:616` — SecureField "hf_…" (HF token)
- `ClusterConnectionDot.swift:627` — Button "Cancel" (HF sheet)
- `ClusterConnectionDot.swift:628` — Button "Install on Cluster"

ClusterSetupWizard.swift
- `ClusterSetupWizard.swift:150` — rail step buttons (×7, "Site" … "Connect + register")
- `ClusterSetupWizard.swift:280` — Picker "Site"
- `ClusterSetupWizard.swift:294` — Button "Add \(preset.name)"
- `ClusterSetupWizard.swift:303` — Button "New Site…"
- `ClusterSetupWizard.swift:305` — Button "From documentation…"
- `ClusterSetupWizard.swift:318` — Button "Edit Selected…"
- `ClusterSetupWizard.swift:420` — Button "Run Push"
- `ClusterSetupWizard.swift:450` — TextField "partition" (CPU partition)
- `ClusterSetupWizard.swift:454` — TextField "CPUs"
- `ClusterSetupWizard.swift:460` — TextField "memory"
- `ClusterSetupWizard.swift:464` — TextField "HH:MM:SS" (Setup walltime)
- `ClusterSetupWizard.swift:519` — TextField "Environment prefix (optional)"
- `ClusterSetupWizard.swift:520` — TextField "Python version"
- `ClusterSetupWizard.swift:522` — Button "Edit Site Settings…"
- `ClusterSetupWizard.swift:541` — Toggle "Rewrite existing environment file"
- `ClusterSetupWizard.swift:542` — Toggle "Submit GPU hello job afterward"
- `ClusterSetupWizard.swift:549` — Button "Run Dry-Run"
- `ClusterSetupWizard.swift:659` — Button "Run Validate"
- `ClusterSetupWizard.swift:699` — Button "Submit Controller Job"
- `ClusterSetupWizard.swift:734` — Button "Connect"
- `ClusterSetupWizard.swift:754` — Button "Close"
- `ClusterSetupWizard.swift:778` — Button "Cancel"
- `ClusterSetupWizard.swift:788` — Button "Back"
- `ClusterSetupWizard.swift:790` — Button "Continue"

ClusterSiteEditor.swift
- `ClusterSiteEditor.swift:106` — Picker "Kind" (transport)
- `ClusterSiteEditor.swift:136` — Picker "Daemon runs as" (visible explanation text below, no tooltip)
- `ClusterSiteEditor.swift:152` — Picker "Scheduler"
- `ClusterSiteEditor.swift:203` — TextField "name" (partition row)
- `ClusterSiteEditor.swift:204` — TextField "hours" (partition max walltime)
- `ClusterSiteEditor.swift:206` — TextField "all" (allowed GPU types)
- `ClusterSiteEditor.swift:208` — TextField "site" (partition qos)
- `ClusterSiteEditor.swift:247` — TextField "A100" (GPU type)
- `ClusterSiteEditor.swift:248` — TextField "80" (VRAM GB)
- `ClusterSiteEditor.swift:250` — TextField "sm_80" (compute capability)
- `ClusterSiteEditor.swift:276` — DisclosureGroup "Directives & required headers"
- `ClusterSiteEditor.swift:295` — DisclosureGroup "Scheduler commands"
- `ClusterSiteEditor.swift:309` — DisclosureGroup "Job defaults & interruption"
- `ClusterSiteEditor.swift:314` — TextField "Default CPUs per task"
- `ClusterSiteEditor.swift:319` — TextField "Auto-resubmit limit"
- `ClusterSiteEditor.swift:322` — Picker "Signal target" (values `step`/`batch-forward`/`batch-direct` shown raw, no explanation anywhere)
- `ClusterSiteEditor.swift:338` — DisclosureGroup "Limits & submission"
- `ClusterSiteEditor.swift:343` — TextField "Max running jobs"
- `ClusterSiteEditor.swift:348` — Toggle "Submit from the bundle directory"
- `ClusterSiteEditor.swift:349` — TextField "Job-name prefix"
- `ClusterSiteEditor.swift:355` — DisclosureGroup "Job classes"
- `ClusterSiteEditor.swift:379` — TextField "Partition" (×3 job classes)
- `ClusterSiteEditor.swift:380` — TextField "CPUs per task" (×3)
- `ClusterSiteEditor.swift:381` — TextField "Memory" (×3)
- `ClusterSiteEditor.swift:382` — TextField "Walltime" (×3)
- `ClusterSiteEditor.swift:383` — TextField "gres" (×3)
- `ClusterSiteEditor.swift:404` — Picker "Module system"
- `ClusterSiteEditor.swift:420` — DisclosureGroup "Python details"
- `ClusterSiteEditor.swift:423` — TextField "Python version"
- `ClusterSiteEditor.swift:426` — TextField "Conda profile script"
- `ClusterSiteEditor.swift:427` — TextField "Conda env name"
- `ClusterSiteEditor.swift:428` — TextField "Venv path"
- `ClusterSiteEditor.swift:432` — DisclosureGroup "Packages"
- `ClusterSiteEditor.swift:440` — DisclosureGroup "Paths & hosts"
- `ClusterSiteEditor.swift:445` — TextField "Remote repo path"
- `ClusterSiteEditor.swift:489` — DisclosureGroup "Storage details"
- `ClusterSiteEditor.swift:500` — Picker "Model-hub offline mode"
- `ClusterSiteEditor.swift:511` — TextField "Housekeeping scan file cap"
- `ClusterSiteEditor.swift:512` — TextField "Free-space warn (GB)"
- `ClusterSiteEditor.swift:513` — TextField "Free-space fail (GB)"
- `ClusterSiteEditor.swift:514` — TextField "Pre-stage minimum free (GB)"
- `ClusterSiteEditor.swift:515` — TextField "Maintenance calendar stale days"
- `ClusterSiteEditor.swift:540` — TextField "Maintenance note"
- `ClusterSiteEditor.swift:549` — DisclosureGroup "Server posture overrides"
- `ClusterSiteEditor.swift:626` — Button "Cancel"

ClusterSitePreviewView.swift
- `ClusterSitePreviewView.swift:31` — DisclosureGroup "Environment file"
- `ClusterSitePreviewView.swift:38` — DisclosureGroup "Scheduler headers"
- `ClusterSitePreviewView.swift:45` — DisclosureGroup "Scheduler commands"
- `ClusterSitePreviewView.swift:50` — DisclosureGroup "GPU vocabulary"
- `ClusterSitePreviewView.swift:59` — DisclosureGroup "Unresolved facts"

ClusterProfileCoauthoringSheet.swift
- `ClusterProfileCoauthoringSheet.swift:20` — Button "Copy agent and reviewer prompts"
- `ClusterProfileCoauthoringSheet.swift:33` — Button "Review agent draft…"
- `ClusterProfileCoauthoringSheet.swift:51` — DisclosureGroup "Sources and declarations"
- `ClusterProfileCoauthoringSheet.swift:75` — Button "Close"
- `ClusterProfileCoauthoringSheet.swift:76` — Button "Import reviewed profile"

MaintenanceWindowsEditor.swift
- `MaintenanceWindowsEditor.swift:52` — Button "Add window"
- `MaintenanceWindowsEditor.swift:77` — Button "Cancel"
- `MaintenanceWindowsEditor.swift:79` — Button "Save"
- `MaintenanceWindowsEditor.swift:94` — DatePicker "start" (labels hidden)
- `MaintenanceWindowsEditor.swift:99` — DatePicker "end" (labels hidden)
- `MaintenanceWindowsEditor.swift:103` — TextField "label (optional)"

## Notes on strengths

- The dot's state is normalised into one enum (`ConnectionState`, L325-359) so word, glyph and tint cannot drift; the label is text-bearing rather than a bare circle — the right fix for the colour-blind reader, and the tooltip carries the full sentence.
- Secrets discipline is real: both tokens go to Keychain, the HF token travels on stdin, `~/.steerlab-token` is imported over the tunnel and never logged, and the editor's only token field is a path.
- The health card is honest about its numbers: the "whole-filesystem vs quota" caption (L138-146), the "site policy undeclared" purge wording (L244-246), the stale-calendar and stale-scan warnings, and per-bundle import failures with attempt counts — all rendered, none swallowed.
- Every async action in this cluster has a busy title and a re-entry guard (Import now, Refresh, Save, Install on Cluster, every wizard Run button), and every wizard step is retryable after failure.
