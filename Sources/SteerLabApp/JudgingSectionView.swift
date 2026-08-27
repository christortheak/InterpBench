import ExperimentKit
import SteeringKit
import SwiftUI

// The unified judging section (2026-07-21 researcher walkthrough, issue 4):
// one coherent block that explains the judging model, shows an explicit
// calm none-state, renders the pinned-judge panel with honest "needs API
// key" states, and admits the legacy single-judge picker only in the one
// case the engine actually reads it (no pinned judges — `resolvedJudges`
// synthesizes a single unpinned judge from it, otherwise ignores it).
// Extracted from `ExperimentsPanelView.judgingControls`, which sits at the
// type-checker's limits. Copy and structure only — the judge JSON contract
// and engine resolution are untouched.
struct JudgingSectionControls: View {
    @Bindable var service: ChatService
    let manifest: ExperimentManifest
    @Bindable var panel: ExperimentPanel

    /// Discovered serving endpoints per judge row (2026-07-24). View state
    /// only — nothing here is pinned until the researcher picks, and the
    /// pin itself still lives in the manifest.
    @State private var discovered: [Int: [OpenRouterCatalog.Endpoint]] = [:]
    @State private var discoveryNote: [Int: String] = [:]
    @State private var discovering: Int?
    /// Revision-resolution state per judge row (2026-07-24). View state
    /// only — the pin itself lives in the manifest.
    @State private var resolving: Int?
    @State private var resolveNote: [Int: String] = [:]

    private var isDraft: Bool { manifest.status == .draft }

    // MARK: Stale-index safety

    /// Deleting a judge shrinks `panel.judges` one graph update before the
    /// index-keyed ForEach re-diffs, and SwiftUI re-evaluates the REMOVED
    /// row's bindings once in between — a raw `panel.judges[index]` in a
    /// binding getter then traps (field crash 2026-08-05: the trash-can
    /// button, `openRouterJudgeFields`). Every row read and write goes
    /// through these three: a stale read renders a blank placeholder for at
    /// most one frame, a stale write is dropped.
    private func judge(_ index: Int) -> ExperimentManifest.JudgeRef {
        panel.judges.indices.contains(index)
            ? panel.judges[index]
            : .init(name: "", kind: "openrouter")
    }

    private func withJudge(
        _ index: Int, _ mutate: (inout ExperimentManifest.JudgeRef) -> Void
    ) {
        guard panel.judges.indices.contains(index) else { return }
        mutate(&panel.judges[index])
    }

    private func judgeBinding<Value>(
        _ index: Int,
        get: @escaping (ExperimentManifest.JudgeRef) -> Value,
        set: @escaping (inout ExperimentManifest.JudgeRef, Value) -> Void
    ) -> Binding<Value> {
        Binding(
            get: { get(judge(index)) },
            set: { value in withJudge(index) { set(&$0, value) } })
    }

    var body: some View {
        HStack(spacing: 6) {
            Text("Judging (optional)")
                .font(.caption.bold())
            InfoButton(text: StudyInfo.judges)
        }

        if panel.judges.isEmpty {
            // The explicit none-state: a valid design, not an empty list.
            Text(
                manifest.studyKind == .multiAgent
                    ? "No judging declared — the scenario transcripts are "
                        + "recorded without judge scoring. That is a valid "
                        + "state; add judges only if you want blinded "
                        + "transcript scoring against a rubric."
                    : "No judging declared — the study's declared endpoints "
                        + "(parsed answers, answer-token probabilities) are "
                        + "measured on their own. That is a complete design; "
                        + "add judges only if you want blinded "
                        + "condition-vs-baseline scoring against a rubric.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        } else {
            // The effective declaration, stated plainly (2026-07-22
            // incident): pinned judges + a chosen rubric file ARE paired
            // judging — Save Evaluation Settings writes the explicit
            // evaluation block; judges without a rubric file declare
            // nothing the evaluate stage could run with.
            let rubricFile = panel.judgeRubricFile
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if rubricFile.isEmpty {
                Text(
                    "Judging incomplete — \(panel.judges.count) "
                        + "judge\(panel.judges.count == 1 ? "" : "s") pinned "
                        + "but no rubric file chosen; the declaration needs "
                        + "both. Choose a rubric file below, or remove the "
                        + "judges.")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                Text(
                    "Paired judging declared — \(panel.judges.count) "
                        + "judge\(panel.judges.count == 1 ? "" : "s"), rubric "
                        + URL(filePath: rubricFile).lastPathComponent
                        + ". Save Evaluation Settings writes this "
                        + "declaration into the study.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack(spacing: 6) {
                Text("Pinned judges — durable identities, hashed into the "
                    + "study (frozen judged studies need at least 2)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                InfoButton(text: StudyInfo.judgeKindsAndKeys)
            }
        }

        ForEach(panel.judges.indices, id: \.self) { index in
            judgeRow(index: index)
        }
        if isDraft {
            Button("Add Judge") { panel.addJudge() }
                .help(
                    "add a pinned judge to the panel; each judge runs a "
                        + "full pass and the report carries pairwise "
                        + "agreement statistics")
        }

        // The legacy single-judge selector exists ONLY as the fallback the
        // engine uses when no judges are pinned (it synthesizes one
        // unpinned judge from it for Results' Run Paired Judge). With a
        // pinned panel it is ignored by resolution, so it is hidden — the
        // panel state keeps its value either way.
        if panel.judges.isEmpty {
            JudgeModelPicker(
                title: "Ad-hoc judge model",
                help:
                    "used only by Run Paired Judge in Results while no judges "
                    + "are pinned above — one unpinned judge for "
                    + "exploratory judging, never freeze-grade evidence",
                offers: panel.judgeModelOffers,
                selection: $panel.judgeModel,
                openRouterModel: $panel.adHocOpenRouterModel,
                openRouterProvider: $panel.adHocOpenRouterProvider)
        }

        RubricFileControls(manifest: manifest, panel: panel)

        HStack(alignment: .top, spacing: 6) {
            TextField(
                "structured fields for judge JSON",
                text: $panel.evaluationStructuredPrompt,
                axis: .vertical
            )
            .lineLimit(3 ... 8)
            .help(
                "optional: fields you name here (e.g. holding_changed "
                    + "boolean) are ALSO returned machine-readably by each "
                    + "judge and summarized across judgments — the app "
                    + "builds the JSON-output instruction itself")
            InfoButton(text: StudyInfo.structuredJudgeFields)
        }
    }

    // MARK: Judge rows

    @ViewBuilder
    private func judgeRow(index: Int) -> some View {
        HStack {
            TextField(
                "name",
                text: judgeBinding(index, get: { $0.name }, set: { $0.name = $1 })
            )
            .frame(maxWidth: 140)
            Picker(
                "",
                selection: Binding(
                    get: { judge(index).kind },
                    // Kind switches route through the panel's per-kind
                    // stash (field bug 2026-08-07): the outgoing kind's
                    // fields are kept for the session and the incoming
                    // kind's restored — never carried live into a kind
                    // that does not own them. Stale-index safe: the panel
                    // guards the row's existence.
                    set: { panel.setJudgeKind(at: index, to: $0) })
            ) {
                // `claude` is no longer OFFERED (2026-07-24): external
                // judging standardises on OpenRouter, which reaches
                // Anthropic models via provider `anthropic` and — unlike
                // the direct path, which has no provider dimension at all
                // — reports and verifies which backend served the verdict.
                // An EXISTING claude judge still lists its own kind, or the
                // picker would silently rewrite a pinned judge on first
                // render. The kind still works; it is just not offered.
                if judge(index).kind == "claude" {
                    Text("claude (legacy)").tag("claude")
                }
                Text("openrouter").tag("openrouter")
                Text("local").tag("local")
            }
            .frame(maxWidth: 110)
            if judge(index).kind == "claude" {
                TextField(
                    "model (blank = default)",
                    text: judgeBinding(
                        index,
                        get: { $0.model ?? "" },
                        set: { $0.model = $1 }))
            } else if judge(index).kind == "openrouter" {
                openRouterJudgeFields(index: index)
            } else {
                localJudgeModelPicker(index: index)
            }
            keyStateBadge(kind: judge(index).kind)
            if isDraft {
                Button {
                    panel.removeJudge(at: index)
                } label: {
                    Image(systemName: "trash")
                }
                .buttonStyle(.plain)
            }
        }
        .disabled(!isDraft)
    }

    /// "Needs API key" state for API-backed judge kinds, using the SAME
    /// presence checks as `pairedJudgeDisabledReason` so the badge and the
    /// run-time refusal can never disagree. Purely informative — declaring
    /// the judge stays legal; the key is needed when judging runs.
    @ViewBuilder
    private func keyStateBadge(kind: String) -> some View {
        switch kind {
        case "claude" where ClaudeStimulusGenerator.apiKey == nil:
            Label("needs API key", systemImage: "key.slash")
                .font(.caption2)
                .foregroundStyle(.orange)
                .help(
                    "no Anthropic key found — judging will refuse at run "
                        + "time until one is set (Compute section, stored "
                        + "in the macOS Keychain, or ANTHROPIC_API_KEY). "
                        + "Declaring the judge now is fine")
        case "openrouter" where JudgeKeyStore.resolveKey(kind: "openrouter") == nil:
            Label("needs API key", systemImage: "key.slash")
                .font(.caption2)
                .foregroundStyle(.orange)
                .help(
                    "no OpenRouter key found — judging will refuse at run "
                        + "time until the external judge key is set "
                        + "(Compute section) or OPENROUTER_API_KEY is in "
                        + "the environment. Declaring the judge now is fine")
        default:
            EmptyView()
        }
    }

    // MARK: Kind-specific fields (moved verbatim from ExperimentsPanelView)

    private static let localJudgeHelp =
        "the model that acts as the LOCAL judge, scoring baseline-vs-condition "
        + "output pairs against the rubric — distinct from the Claude judge "
        + "(API). Blank = the study model: it judges the steered output with "
        + "the same model that generated it, through the already-loaded "
        + "weights. A DIFFERENT local model needs an engine holding two "
        + "models (server with STEERLAB_MAX_LOADED_MODELS ≥ 2) and refuses "
        + "at start otherwise"

    /// Strict workspace-scoped installed-models picker for a LOCAL judge row
    /// (same rule as WorkspaceModelPicker: a value outside the inventory is
    /// rendered "(not installed)" but can never be picked anew). The saved
    /// manifest value stays a plain model-id string.
    @ViewBuilder
    private func localJudgeModelPicker(index: Int) -> some View {
        let installed = service.workspaceModelOptions
        let current = judge(index).model ?? ""
        Picker(
            "",
            selection: judgeBinding(
                index,
                get: { $0.model ?? "" },
                set: { $0.model = $1.isEmpty ? nil : $1 })
        ) {
            Text("study model (default)").tag("")
            ForEach(installed, id: \.self) { model in
                Text(model).tag(model)
            }
            if WorkspaceScoping.selectionOutsideInventory(current, inventory: installed) {
                Text(
                    service.cluster.computeTarget == .server
                        ? "\(current) (not installed)" : current
                )
                .tag(current)
                .selectionDisabled()
            }
        }
        .help(Self.localJudgeHelp)
        .onChange(of: judge(index).model ?? "") { previous, picked in
            autofillJudgePins(
                index: index, model: picked, previousModel: previous)
        }
        foreignLocalJudgePins(index: index)
    }

    /// Fill a foreign local judge's pins from what the host already knows,
    /// rather than asking the researcher to type a 40-character commit hash
    /// (external review round 4, finding 5). The revision comes from the
    /// local model cache's `refs/main` — the same source freeze uses to
    /// auto-pin the study model.
    ///
    /// The RULE lives in `ExperimentStore.judgePinPrefill` (tested there);
    /// this only supplies the cache lookup and writes the result back.
    /// Resolve this judge's revision from the ACTIVE substrate's model cache
    /// (external review round 5, finding 5).
    ///
    /// The judge picker lists SERVER-installed models when a server
    /// workspace is active, so resolving only against the Mac's cache left
    /// exactly the cluster-only judge — the common case — needing a
    /// hand-copied hash. The server already reports each cached repo's
    /// `refs/main` in its housekeeping scan; this asks it.
    ///
    /// Failure is reported, never papered over: a model neither substrate
    /// holds keeps a nil revision and says so, because a fabricated commit
    /// would freeze cleanly and die after a queue wait.
    private func resolveJudgeRevision(index: Int, model: String) async {
        resolving = index
        resolveNote[index] = nil
        defer { resolving = nil }

        if case .server = service.cluster.activeWorkspace,
            let client = service.cluster.client
        {
            do {
                if let commit = try await client.cachedRevision(forModel: model) {
                    withJudge(index) { $0.revision = commit }
                    fillJudgeDtypeIfBlank(index: index)
                    resolveNote[index] =
                        "from \(service.cluster.substrateLabel)'s model cache"
                } else {
                    resolveNote[index] =
                        "\(service.cluster.substrateLabel) does not hold "
                        + "\(model) — install it there, or paste the commit"
                }
            } catch {
                resolveNote[index] =
                    "could not read \(service.cluster.substrateLabel)'s model "
                    + "cache: \(error.localizedDescription)"
            }
            return
        }
        autofillJudgePins(index: index, model: model)
        resolveNote[index] = (judge(index).revision ?? "").isEmpty
            ? "not in this Mac's model cache — paste the commit"
            : "from this Mac's model cache"
    }

    /// The dtype half of the prefill, for the server path where the revision
    /// came from elsewhere.
    private func fillJudgeDtypeIfBlank(index: Int) {
        if (judge(index).dtype ?? "").isEmpty {
            let filled = ExperimentStore.judgePinPrefill(
                model: judge(index).model,
                previousModel: nil,
                studyModel: manifest.modelID,
                revision: judge(index).revision,
                dtype: nil,
                resolveRevision: { _ in nil })?.dtype
            withJudge(index) { $0.dtype = filled }
        }
    }

    private func autofillJudgePins(
        index: Int, model: String, previousModel: String? = nil
    ) {
        guard let filled = ExperimentStore.judgePinPrefill(
            model: model,
            previousModel: previousModel,
            studyModel: manifest.modelID,
            revision: judge(index).revision,
            dtype: judge(index).dtype,
            resolveRevision: { SteeredContainerLoader.cachedRevision(for: $0) })
        else { return }
        withJudge(index) {
            $0.revision = filled.revision
            $0.dtype = filled.dtype
        }
    }

    /// Revision + dtype pins for a local judge naming a model OTHER than
    /// the study model (2026-07-24). A study-model judge inherits the
    /// study's pin, so it needs neither; a foreign one has nothing to
    /// inherit, and freeze now refuses it unpinned on both engines —
    /// without a revision, two judging sessions can load different
    /// defaults while both records say "none", so a resumed evaluation
    /// cannot prove its reused verdicts came from the same judge.
    ///
    /// Shown only when they are actually required, so the common case (the
    /// study model as judge) stays a single picker.
    @ViewBuilder
    private func foreignLocalJudgePins(index: Int) -> some View {
        let declared = (judge(index).model ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if !declared.isEmpty, declared != manifest.modelID {
            judgeRevisionField(index: index, model: declared)
            judgeDtypePicker(index: index)
            if let note = resolveNote[index] {
                Text(note)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            } else if (judge(index).revision ?? "").isEmpty {
                Text("no commit pinned — Resolve, or paste one")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private static let judgeRevisionHelp =
        "the model commit this judge loads. Required for a judge naming a "
        + "model other than the study model: there is no study pin to "
        + "inherit, and an unpinned judge cannot be shown to be the same "
        + "judge across two sessions. Filled from the ACTIVE compute "
        + "substrate's model cache — Resolve re-reads it."

    private static let judgeResolveHelp =
        "read this model's commit (refs/main) from the cache of whichever "
        + "substrate is active — the server's when a server workspace is "
        + "selected, this Mac's otherwise. A model neither holds has no "
        + "revision to read: install it there, or copy the commit from the "
        + "model's page."

    private static let judgeDtypeHelp =
        "the loader dtype this judge is loaded at. bf16 and fp16 are "
        + "different judges, so this is a pin, not a hint — the server "
        + "honours it rather than its own default, and refuses to load if "
        + "it cannot."

    @ViewBuilder
    private func judgeRevisionField(index: Int, model: String) -> some View {
        TextField(
            "revision (required)",
            text: judgeBinding(
                index,
                get: { $0.revision ?? "" },
                set: { $0.revision = $1.isEmpty ? nil : $1 }))
            .frame(maxWidth: 120)
            .help(Self.judgeRevisionHelp)
        Button(resolving == index ? "Resolving…" : "Resolve") {
            Task { await resolveJudgeRevision(index: index, model: model) }
        }
        .disabled(!(judge(index).revision ?? "").isEmpty
            || resolving == index)
        .help(Self.judgeResolveHelp)
    }

    @ViewBuilder
    private func judgeDtypePicker(index: Int) -> some View {
        // The engines' closed vocabulary, never a second hand-kept list
        // (review round 4, finding 2): an unknown dtype now refuses at
        // freeze AND at load, so a picker that could offer one would be
        // offering a guaranteed failure.
        let vocabulary = ExperimentStore.judgeDtypeVocabulary
        Picker("", selection: judgeBinding(
            index,
            get: { $0.dtype ?? "" },
            set: { $0.dtype = $1.isEmpty ? nil : $1 }))
        {
            Text("dtype…").tag("")
            ForEach(vocabulary, id: \.self) { name in
                Text(name).tag(name)
            }
        }
        .labelsHidden()
        .frame(maxWidth: 110)
        .help(Self.judgeDtypeHelp)
    }

    /// Model-slug + provider fields for an OPENROUTER judge row. No
    /// defaults (server rule, 2026-07-19): both are pins — verify() flags a
    /// blank, and the engines refuse at sweep/evaluate start.
    ///
    /// The provider is DISCOVERED, not typed (2026-07-24). It has to be
    /// OpenRouter's routing slug, which appears on no Hugging Face page and
    /// cannot be guessed from a display name (`Google` routes as
    /// `google-vertex`); typing it was the single likeliest way to lose a
    /// judged run to a string mismatch, hours in.
    @ViewBuilder
    private func openRouterJudgeFields(index: Int) -> some View {
        TextField(
            "model slug (required)",
            text: judgeBinding(
                index, get: { $0.model ?? "" }, set: { $0.model = $1 }))
        TextField(
            "provider (required)",
            text: judgeBinding(
                index, get: { $0.provider ?? "" }, set: { $0.provider = $1 }))
            .frame(maxWidth: 140)
            .help(
                "the OpenRouter serving provider pinned for this judge "
                    + "(allow_fallbacks is off): the same model slug served "
                    + "by another backend — or at another quantization — is "
                    + "a different judge. Use Discover rather than typing "
                    + "it: this is OpenRouter's routing SLUG, which is not "
                    + "always its display name.")
        discoveryControls(index: index)
    }

    /// Discover button, picker, and result note. Split out of
    /// `openRouterJudgeFields` to keep both bodies inside the type-checker's
    /// budget — this file's controls already sit at that limit.
    @ViewBuilder
    private func discoveryControls(index: Int) -> some View {
        let modelIsBlank = (judge(index).model ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        Button("Discover") {
            Task { await discoverProviders(index: index) }
        }
        .controlSize(.small)
        .disabled(modelIsBlank || discovering == index)
        .help(
            "ask OpenRouter which providers serve this model slug, and at "
                + "what quantization. Needs no API key. Picking from the "
                + "list guarantees the pin is spelled the way the router "
                + "expects.")
        if discovering == index {
            ProgressView().controlSize(.small)
        }
        if let options: [OpenRouterCatalog.Endpoint] = discovered[index],
            !options.isEmpty
        {
            providerPicker(index: index, options: options)
        }
        if let note: String = discoveryNote[index] {
            Text(note)
                .font(.caption2)
                .foregroundStyle(note.hasPrefix("✓") ? Color.green : Color.orange)
                .textSelection(.enabled)
        }
    }

    @ViewBuilder
    private func providerPicker(
        index: Int, options: [OpenRouterCatalog.Endpoint]
    ) -> some View {
        let binding = judgeBinding(
            index, get: { $0.provider ?? "" }, set: { $0.provider = $1 })
        Picker("", selection: binding) {
            Text("choose a provider").tag("")
            ForEach(options) { endpoint in
                Text(endpoint.summary).tag(endpoint.provider)
            }
        }
        .labelsHidden()
        .frame(maxWidth: 260)
        .help(
            "each row is one serving endpoint: routing slug, the "
                + "quantization it runs, and its context length. "
                + "Quantization is why this pin exists — the same model at "
                + "fp8 and bf16 can reach different verdicts.")
    }

    /// Fetch the endpoints serving this judge's model slug. Failure is
    /// reported in the row, never thrown away: "we could not check" and
    /// "the pin is wrong" are different facts and the researcher needs to
    /// tell them apart.
    private func discoverProviders(index: Int) async {
        let model = (judge(index).model ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !model.isEmpty else { return }
        discovering = index
        discoveryNote[index] = nil
        defer { discovering = nil }
        do {
            let endpoints = try await OpenRouterCatalog.endpoints(forModel: model)
            discovered[index] = endpoints
            if endpoints.isEmpty {
                discoveryNote[index] =
                    "OpenRouter lists no providers serving '\(model)' right now"
                return
            }
            let slugs = Set(endpoints.map(\.provider))
            let current = OpenRouterProviderIdentity.canonical(
                judge(index).provider ?? "")
            if !current.isEmpty, !slugs.contains(current) {
                discoveryNote[index] =
                    "the pinned provider '\(current)' does not serve this "
                    + "model — pick one of the \(slugs.count) below"
            } else if endpoints.filter({ $0.provider == current }).count > 1 {
                discoveryNote[index] =
                    "✓ '\(current)' serves this model, but at more than one "
                    + "quantization — the provider pin alone does not fix "
                    + "which one judges"
            } else if !current.isEmpty {
                discoveryNote[index] = "✓ '\(current)' serves this model"
            } else {
                discoveryNote[index] =
                    "\(endpoints.count) provider(s) serve this model"
            }
        } catch {
            discoveryNote[index] =
                "could not reach OpenRouter's catalogue (\(error)) — the pin "
                + "is unchecked, not wrong"
        }
    }
}
