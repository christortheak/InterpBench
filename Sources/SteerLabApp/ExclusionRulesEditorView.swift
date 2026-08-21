import ExperimentKit
import SwiftUI

/// Evaluation-section list editor for `manifest.exclusionRules` — which
/// answers the analysis may set aside, declared up front (the closed
/// vocabulary: failedAttentionCheck / unparseableEndpoint / outOfRange)
/// instead of hand-editing manifest JSON.
///
/// Pattern per `OrdinalScaleInstrumentControls`: reads the manifest, writes
/// through the store's draft-edit gate (`ExperimentStore.setExclusionRules`
/// — the ordinary manifest-editing pathway), plain-language rows, engine
/// violations reused verbatim for validation feedback, read-only once
/// frozen (rules are part of the frozen manifest).
///
/// Wiring: one line in `ExperimentsPanelView`'s `analysisSettings`:
///
///     ExclusionRulesEditor(manifest: manifest, panel: panel)
struct ExclusionRulesEditor: View {
    let manifest: ExperimentManifest
    let panel: ExperimentPanel

    @State private var errorText: String?
    /// outOfRange bounds editor state (committed via Apply, never
    /// per-keystroke — a manifest save per keystroke would be noise).
    @State private var endpointField = ""
    @State private var minField = ""
    @State private var maxField = ""
    @State private var editingRange = false
    /// The number of task-prompt items declaring an attentionCheck (nil =
    /// file unreadable; other readiness rows own that finding). Loaded off
    /// the body in `.task` — the prompts file can be large.
    @State private var checkedItemCount: Int?

    private var isDraft: Bool { manifest.status == .draft }
    private var rules: [ExclusionRule] { manifest.exclusionRules ?? [] }
    private func rule(_ id: String) -> ExclusionRule? {
        rules.first { $0.rule == id }
    }

    var body: some View {
        // Exclusions join per-record model outputs at analysis; multi-agent
        // studies do not apply them yet (documented residual).
        if manifest.studyKind == .modelOutput {
            content
                .task(id: taskPromptsProbeKey) { await probeAttentionChecks() }
                .onChange(of: manifest.name) { resetRangeEditor() }
        }
    }

    @ViewBuilder private var content: some View {
        HStack(spacing: 6) {
            Text(rules.isEmpty ? "Excluded answers: none declared" : "Excluded answers")
                .font(.caption)
                .foregroundStyle(.secondary)
            InfoButton(text: StudyInfo.exclusionRules)
            Spacer()
            if isDraft {
                addMenu
            }
        }
        ForEach(rules, id: \.rule) { rule in
            ruleRow(rule)
        }
        if editingRange || rule(ExclusionEngine.ruleOutOfRange) != nil {
            if isDraft {
                rangeEditor
            }
        }
        attentionCheckWarning
        if let errorText {
            Label(errorText, systemImage: "exclamationmark.triangle")
                .font(.caption)
                .foregroundStyle(.orange)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    // MARK: - Declared rule rows

    private func ruleRow(_ rule: ExclusionRule) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Label(
                ExclusionRulesUI.editorDescription(of: rule),
                systemImage: "line.3.horizontal.decrease.circle")
                .font(.caption)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
            Spacer()
            if isDraft {
                Button("Remove") { remove(rule.rule) }
                    .font(.caption)
                    .help(
                        "removes '\(rule.rule)' from the declared exclusion "
                            + "rules (draft-only — frozen studies keep their "
                            + "declared rules)")
            }
        }
    }

    /// Add menu over the closed vocabulary; rules already declared are
    /// disabled (each rule at most once — the engine's own refusal).
    private var addMenu: some View {
        Menu("Add rule") {
            Button("Drop answers that fail their attention check") {
                append(ExclusionRule(rule: ExclusionEngine.ruleFailedAttentionCheck))
            }
            .disabled(rule(ExclusionEngine.ruleFailedAttentionCheck) != nil)
            Button("Drop answers that cannot be parsed") {
                append(ExclusionRule(rule: ExclusionEngine.ruleUnparseableEndpoint))
            }
            .disabled(rule(ExclusionEngine.ruleUnparseableEndpoint) != nil)
            Button("Drop answers outside a declared range…") {
                editingRange = true
                loadRangeFields(from: nil)
            }
            .disabled(rule(ExclusionEngine.ruleOutOfRange) != nil)
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .font(.caption)
        .help(
            "declares an exclusion rule (manifest `exclusionRules`) — "
                + "applied at analysis time only; raw responses are never "
                + "touched, and the report states what each rule dropped")
    }

    // MARK: - outOfRange bounds editor

    /// The declared endpoint + min/max fields, committed via Apply with the
    /// engine's plain-language violations as live feedback — never a
    /// duplicate of the validation logic.
    @ViewBuilder private var rangeEditor: some View {
        HStack(spacing: 6) {
            Text("Kept range:")
                .font(.caption)
                .foregroundStyle(.secondary)
            TextField(ExclusionRulesUI.defaultEndpoint, text: $endpointField)
                .textFieldStyle(.roundedBorder)
                .frame(width: 130)
                .help(
                    "which parsed value the rule reads — blank means the "
                        + "default (\(ExclusionRulesUI.defaultEndpoint), the "
                        + "months-of-sentence endpoint)")
            TextField("min", text: $minField)
                .textFieldStyle(.roundedBorder)
                .frame(width: 64)
                .help("smallest kept value — blank for no lower bound")
            Text("to")
                .font(.caption)
                .foregroundStyle(.secondary)
            TextField("max", text: $maxField)
                .textFieldStyle(.roundedBorder)
                .frame(width: 64)
                .help("largest kept value — blank for no upper bound")
            Button("Apply") { applyRange() }
                .font(.caption)
                .disabled(!rangeProblems.isEmpty)
            if editingRange, rule(ExclusionEngine.ruleOutOfRange) == nil {
                Button("Cancel") {
                    editingRange = false
                    resetRangeEditor()
                }
                .font(.caption)
            }
        }
        .onAppear { loadRangeFields(from: rule(ExclusionEngine.ruleOutOfRange)) }
        ForEach(rangeProblems, id: \.self) { problem in
            Label(problem, systemImage: "exclamationmark.triangle")
                .font(.caption)
                .foregroundStyle(.orange)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// Live validation of the candidate bounds: local number-parse feedback
    /// first, then the engine's own violations for the candidate rule set.
    private var rangeProblems: [String] {
        var problems: [String] = []
        if !minField.trimmingCharacters(in: .whitespaces).isEmpty,
            parsedBound(minField) == nil
        {
            problems.append("min must be a number (got '\(minField)')")
        }
        if !maxField.trimmingCharacters(in: .whitespaces).isEmpty,
            parsedBound(maxField) == nil
        {
            problems.append("max must be a number (got '\(maxField)')")
        }
        guard problems.isEmpty else { return problems }
        return ExclusionEngine.violations(candidateRules())
            .filter { $0.contains("outOfRange") }
    }

    private func parsedBound(_ text: String) -> Double? {
        Double(text.trimmingCharacters(in: .whitespaces))
    }

    private func candidateRange() -> ExclusionRule {
        let endpoint = endpointField.trimmingCharacters(in: .whitespaces)
        return ExclusionRule(
            rule: ExclusionEngine.ruleOutOfRange,
            endpoint: endpoint.isEmpty ? nil : endpoint,
            min: parsedBound(minField),
            max: parsedBound(maxField))
    }

    /// The declared rules with outOfRange replaced (or appended) by the
    /// editor's candidate.
    private func candidateRules() -> [ExclusionRule] {
        var updated = rules.filter { $0.rule != ExclusionEngine.ruleOutOfRange }
        updated.append(candidateRange())
        return updated
    }

    private func loadRangeFields(from rule: ExclusionRule?) {
        endpointField = rule?.endpoint ?? ""
        minField = rule?.min.map(ExclusionRulesUI.formatBound) ?? ""
        maxField = rule?.max.map(ExclusionRulesUI.formatBound) ?? ""
    }

    private func resetRangeEditor() {
        editingRange = false
        loadRangeFields(from: rule(ExclusionEngine.ruleOutOfRange))
        errorText = nil
    }

    // MARK: - Attention-check preflight warning

    /// The gentle inline version of the run-start gate: a declared
    /// failedAttentionCheck rule with no checked items will REFUSE at run
    /// start — say so before the researcher runs, not at the failing gate.
    /// Non-blocking here (the file may still be edited); nil count (file
    /// unreadable) stays quiet — other readiness rows own that finding.
    @ViewBuilder private var attentionCheckWarning: some View {
        if rule(ExclusionEngine.ruleFailedAttentionCheck) != nil,
            checkedItemCount == 0
        {
            Label(
                "the attention-check rule is declared, but no item in the "
                    + "current task-prompts file declares an attentionCheck — "
                    + "the run will refuse at start. Add checks to items "
                    + "(\"attentionCheck\": {\"expected\": \"…\"}) or remove "
                    + "the rule.",
                systemImage: "exclamationmark.triangle")
                .font(.caption)
                .foregroundStyle(.orange)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        } else if rule(ExclusionEngine.ruleFailedAttentionCheck) != nil,
            let count = checkedItemCount
        {
            Text("\(count) task-prompt item\(count == 1 ? " declares" : "s declare") an attention check")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    /// Probe key: re-count when the referenced file or its pinned bytes
    /// change, or when the rule set starts/stops needing checks.
    private var taskPromptsProbeKey: String {
        [
            manifest.name,
            manifest.taskPromptsFile ?? "",
            manifest.taskPromptsHash ?? "",
            rule(ExclusionEngine.ruleFailedAttentionCheck) != nil ? "y" : "n",
        ].joined(separator: "|")
    }

    private func probeAttentionChecks() async {
        guard rule(ExclusionEngine.ruleFailedAttentionCheck) != nil else {
            checkedItemCount = nil
            return
        }
        let file = manifest.taskPromptsFile
        checkedItemCount = await Task.detached(priority: .utility) {
            ExclusionRulesUI.attentionCheckItemCount(taskPromptsFile: file)
        }.value
    }

    // MARK: - Writes (the one draft-edit pathway)

    private func append(_ rule: ExclusionRule) {
        write { try ExperimentStore.setExclusionRules(rules + [rule], experimentName: manifest.name) }
    }

    private func remove(_ id: String) {
        if id == ExclusionEngine.ruleOutOfRange { editingRange = false }
        write {
            try ExperimentStore.setExclusionRules(
                rules.filter { $0.rule != id }, experimentName: manifest.name)
        }
    }

    private func applyRange() {
        write {
            try ExperimentStore.setExclusionRules(
                candidateRules(), experimentName: manifest.name)
        }
        if errorText == nil { editingRange = false }
    }

    /// One draft-edit write path: store write, panel refresh, plain-language
    /// failure line with the raw detail underneath.
    private func write(_ edit: () throws -> Void) {
        do {
            try edit()
            errorText = nil
            panel.refresh()
        } catch {
            errorText =
                "Couldn't update the exclusion rules — the study must still "
                + "be a draft (frozen studies keep their declared rules). "
                + "Details: \(error)"
        }
    }
}
