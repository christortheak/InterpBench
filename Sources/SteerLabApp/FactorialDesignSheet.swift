import ExperimentKit
import SwiftUI
import UniformTypeIdentifiers

/// "Generate factorial design…" (Usability Plan Phase 4, items 20–21): a
/// sheet to author a factorial/counterbalancing design — factors with
/// levels, per-level `{{VAR}}` substitutions, prompt templates, an
/// option-order counterbalance toggle — with a live cell-count line and a
/// 3-item preview, then Generate & Pin. All crossing, substitution,
/// validation, write, and pin rules live in `FactorialDesign` /
/// `FactorialImport` (ExperimentKit, unit-tested); these views render them.
///
/// Own file on purpose: `ExperimentsPanelView` is at the type-checker's
/// limits — the panel view only drops a `FactorialDesignButton` next to the
/// existing import affordances.

/// The affordance the panel view wires: one button that owns the sheet.
struct FactorialDesignButton: View {
    let panel: ExperimentPanel
    var disabled: Bool = false

    @State private var showSheet = false

    var body: some View {
        Button("Generate factorial design…") { showSheet = true }
            .disabled(disabled)
            .help(
                "cross factors (e.g. anchor low/high × frame gain/loss) with "
                    + "prompt templates containing {{VAR}} placeholders — the "
                    + "generator emits ordinary literal prompt items with "
                    + "factor metadata, lands them at the study's task-prompts "
                    + "destination, and pins the file like any hand-authored "
                    + "prompts")
            .sheet(isPresented: $showSheet) {
                FactorialDesignSheet(panel: panel)
            }
    }
}

// MARK: - Editing drafts (UI state; converted to the spec on demand)

private struct LevelDraft: Identifiable {
    let id = UUID()
    var name = ""
    /// One `VAR = value` per line.
    var substitutionsText = ""
}

private struct FactorDraft: Identifiable {
    let id = UUID()
    var name = ""
    var levels: [LevelDraft] = [LevelDraft(), LevelDraft()]
}

private struct TemplateDraft: Identifiable {
    let id = UUID()
    var templateID = "t1"
    var text = ""
    /// One option per line (empty = no options).
    var optionsText = ""
    var target = ""
}

/// Draft → spec. Parsing problems (bad substitution lines) surface as a
/// plain string the preview area shows; everything else is validated by
/// `FactorialDesign.validate()` inside `generate()`.
private func buildDesign(
    factors: [FactorDraft], templates: [TemplateDraft], counterbalance: Bool
) -> Result<FactorialDesign, FactorialDesign.Problem> {
    var specFactors: [FactorialDesign.Factor] = []
    for factor in factors {
        var levels: [FactorialDesign.Level] = []
        for level in factor.levels {
            var substitutions: [String: String] = [:]
            for raw in level.substitutionsText.components(separatedBy: "\n") {
                let line = raw.trimmingCharacters(in: .whitespaces)
                if line.isEmpty { continue }
                guard let equals = line.firstIndex(of: "=") else {
                    return .failure(
                        FactorialDesign.Problem(
                            "level '\(level.name)' of factor '\(factor.name)': "
                                + "'\(line)' is not a substitution — write one "
                                + "VARIABLE = replacement text per line"))
                }
                let key = String(line[..<equals]).trimmingCharacters(in: .whitespaces)
                let value = String(line[line.index(after: equals)...])
                    .trimmingCharacters(in: .whitespaces)
                substitutions[key] = value
            }
            levels.append(
                FactorialDesign.Level(name: level.name, substitutions: substitutions))
        }
        specFactors.append(FactorialDesign.Factor(name: factor.name, levels: levels))
    }
    let specTemplates = templates.map { draft in
        let options = draft.optionsText.components(separatedBy: "\n")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        let target = draft.target.trimmingCharacters(in: .whitespaces)
        return FactorialDesign.Template(
            id: draft.templateID, text: draft.text,
            options: options.isEmpty ? nil : options,
            target: target.isEmpty ? nil : target)
    }
    return .success(
        FactorialDesign(
            factors: specFactors, templates: specTemplates,
            counterbalanceOptionOrder: counterbalance))
}

/// Spec → draft (Load design…).
private func drafts(
    from design: FactorialDesign
) -> (factors: [FactorDraft], templates: [TemplateDraft], counterbalance: Bool) {
    let factors = design.factors.map { factor in
        var draft = FactorDraft()
        draft.name = factor.name
        draft.levels = factor.levels.map { level in
            var levelDraft = LevelDraft()
            levelDraft.name = level.name
            levelDraft.substitutionsText = level.substitutions
                .sorted { $0.key < $1.key }
                .map { "\($0.key) = \($0.value)" }
                .joined(separator: "\n")
            return levelDraft
        }
        return draft
    }
    let templates = design.templates.map { template in
        var draft = TemplateDraft()
        draft.templateID = template.id
        draft.text = template.text
        draft.optionsText = (template.options ?? []).joined(separator: "\n")
        draft.target = template.target ?? ""
        return draft
    }
    return (factors, templates, design.counterbalanceOptionOrder)
}

// MARK: - The sheet

private struct FactorialDesignSheet: View {
    let panel: ExperimentPanel

    @Environment(\.dismiss) private var dismiss
    @State private var factors: [FactorDraft] = [
        {
            var factor = FactorDraft()
            factor.name = "anchor"
            factor.levels[0].name = "low"
            factor.levels[1].name = "high"
            return factor
        }()
    ]
    @State private var templates: [TemplateDraft] = [TemplateDraft()]
    @State private var counterbalance = false
    @State private var replaceExisting = false
    @State private var problem: String?
    @State private var showLoadPicker = false
    @State private var showSavePicker = false
    @State private var loadProblem: String?

    private var built: Result<FactorialDesign, FactorialDesign.Problem> {
        buildDesign(
            factors: factors, templates: templates, counterbalance: counterbalance)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Generate a factorial design")
                .font(.headline)
            explainer
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    factorsSection
                    templatesSection
                }
                .padding(.vertical, 2)
            }
            .frame(minWidth: 640, minHeight: 320)
            Toggle(
                "Counterbalance option order (each cell emitted twice, "
                    + "options reversed, orderFlipped recorded)",
                isOn: $counterbalance)
                .font(.caption)
            countAndPreview
            footer
        }
        .padding(16)
        .fileImporter(
            isPresented: $showLoadPicker, allowedContentTypes: [.json]
        ) { result in
            loadDesign(result)
        }
        .fileExporter(
            isPresented: $showSavePicker,
            document: designDocument,
            contentType: .json,
            defaultFilename: "factorial-design"
        ) { result in
            if case .failure(let error) = result {
                loadProblem = error.localizedDescription
            }
        }
    }

    /// What a factorial design IS, in plain words, plus what the output is
    /// — visible, not hover-only (the ⓘ rule).
    private var explainer: some View {
        Text(
            "A factorial design varies several things at once — each "
                + "factor (e.g. anchor) has levels (low, high) that fill "
                + "{{PLACEHOLDER}} variables in your prompt templates. Every "
                + "combination of levels is generated for every template, so "
                + "effects of each factor can be measured on the same items. "
                + "The OUTPUT is ordinary pinned prompt data: literal, "
                + "fully-substituted items with a factors label per item — "
                + "nothing changes at run time. The design spec itself is "
                + "also saved next to the output as …-design.json for "
                + "provenance (a record of how the file was made — the pin "
                + "is on the generated prompts file, not the spec).")
            .font(.caption)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    // MARK: Factors

    private var factorsSection: some View {
        GroupBox("Factors") {
            VStack(alignment: .leading, spacing: 10) {
                ForEach($factors) { $factor in
                    FactorEditor(
                        factor: $factor,
                        onRemove: { factors.removeAll { $0.id == factor.id } })
                }
                Button("Add factor") { factors.append(FactorDraft()) }
                    .controlSize(.small)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    // MARK: Templates

    private var templatesSection: some View {
        GroupBox("Prompt templates") {
            VStack(alignment: .leading, spacing: 10) {
                ForEach($templates) { $template in
                    TemplateEditor(
                        template: $template,
                        onRemove: {
                            templates.removeAll { $0.id == template.id }
                        })
                }
                Button("Add template") {
                    var draft = TemplateDraft()
                    draft.templateID = "t\(templates.count + 1)"
                    templates.append(draft)
                }
                .controlSize(.small)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    // MARK: Count line + preview

    @ViewBuilder
    private var countAndPreview: some View {
        switch built {
        case .failure(let refused):
            Label(refused.message, systemImage: "xmark.octagon")
                .font(.caption)
                .foregroundStyle(.red)
                .textSelection(.enabled)
        case .success(let design):
            Text(design.cellCountLine)
                .font(.caption)
                .foregroundStyle(.secondary)
            switch Result(catching: { try design.generate() }) {
            case .success(let items):
                previewList(items)
            case .failure(let error):
                Label("\(error)", systemImage: "xmark.octagon")
                    .font(.caption)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }
        }
    }

    private func previewList(_ items: [FactorialDesign.GeneratedItem]) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            ForEach(items.prefix(3), id: \.id) { item in
                Text("\(item.id): \(String(item.text.prefix(90)))")
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            if items.count > 3 {
                Text("… and \(items.count - 3) more")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    // MARK: Footer (load/save, replace, generate)

    private var footer: some View {
        VStack(alignment: .leading, spacing: 8) {
            Toggle(
                "Replace the existing file if its contents differ",
                isOn: $replaceExisting)
                .font(.caption)
                .help(
                    "without this, generating refuses when the study's "
                        + "task-prompts file already exists with different "
                        + "contents — nothing is overwritten silently")
            if let problem {
                Label(problem, systemImage: "xmark.octagon")
                    .font(.caption)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }
            if let loadProblem {
                Text(loadProblem)
                    .font(.caption2)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }
            HStack {
                Button("Load design…") { showLoadPicker = true }
                    .help(
                        "read a saved design JSON — a spec an LLM authored "
                            + "works too (a study pack's files map can carry it)")
                Button("Save design…") { showSavePicker = true }
                    .disabled(designDocument == nil)
                    .help("save the design spec as JSON to reload or share")
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Generate & Pin") { generate() }
                    .buttonStyle(.borderedProminent)
                    .disabled(!generatable)
            }
        }
    }

    private var generatable: Bool {
        guard case .success(let design) = built else { return false }
        return (try? design.generate()) != nil
    }

    private var designDocument: DesignJSONDocument? {
        guard case .success(let design) = built,
            let data = try? design.encoded()
        else { return nil }
        return DesignJSONDocument(data: data)
    }

    private func generate() {
        problem = nil
        guard case .success(let design) = built else { return }
        if let refused = panel.generateFactorialTaskPrompts(
            design: design, replacingExisting: replaceExisting)
        {
            problem = refused
        } else {
            dismiss()
        }
    }

    private func loadDesign(_ result: Result<URL, any Error>) {
        loadProblem = nil
        switch result {
        case .success(let url):
            let accessing = url.startAccessingSecurityScopedResource()
            defer { if accessing { url.stopAccessingSecurityScopedResource() } }
            do {
                let design = try FactorialDesign.decode(try Data(contentsOf: url))
                let loaded = drafts(from: design)
                factors = loaded.factors
                templates = loaded.templates
                counterbalance = loaded.counterbalance
            } catch {
                loadProblem = "\(error)"
            }
        case .failure(let error):
            loadProblem = error.localizedDescription
        }
    }
}

// MARK: - Row editors

private struct FactorEditor: View {
    @Binding var factor: FactorDraft
    let onRemove: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                TextField("factor name (e.g. anchor)", text: $factor.name)
                    .frame(maxWidth: 220)
                Button("Add level") { factor.levels.append(LevelDraft()) }
                    .controlSize(.small)
                Button(role: .destructive, action: onRemove) {
                    Image(systemName: "trash")
                }
                .controlSize(.small)
                .help("remove this factor")
            }
            ForEach($factor.levels) { $level in
                LevelEditor(
                    level: $level,
                    onRemove: {
                        factor.levels.removeAll { $0.id == level.id }
                    })
            }
        }
        .padding(6)
        .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 6))
    }
}

private struct LevelEditor: View {
    @Binding var level: LevelDraft
    let onRemove: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 6) {
            TextField("level (e.g. high)", text: $level.name)
                .frame(width: 140)
            TextField(
                "VAR = replacement text (one per line)",
                text: $level.substitutionsText, axis: .vertical)
                .font(.system(.caption, design: .monospaced))
                .lineLimit(1...4)
                .help(
                    "each line fills one {{VAR}} placeholder for this level, "
                        + "e.g. DEMAND = 9 years")
            Button(role: .destructive, action: onRemove) {
                Image(systemName: "minus.circle")
            }
            .controlSize(.small)
            .help("remove this level")
        }
        .padding(.leading, 12)
    }
}

private struct TemplateEditor: View {
    @Binding var template: TemplateDraft
    let onRemove: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                TextField("template id (e.g. t1)", text: $template.templateID)
                    .frame(maxWidth: 160)
                Button(role: .destructive, action: onRemove) {
                    Image(systemName: "trash")
                }
                .controlSize(.small)
                .help("remove this template")
            }
            TextEditor(text: $template.text)
                .font(.system(.caption, design: .monospaced))
                .frame(minHeight: 70)
                .overlay(alignment: .topLeading) {
                    if template.text.isEmpty {
                        Text("Prompt text with {{VAR}} placeholders…")
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.tertiary)
                            .padding(.top, 8)
                            .padding(.leading, 5)
                            .allowsHitTesting(false)
                    }
                }
            HStack(alignment: .top, spacing: 8) {
                TextField(
                    "options, one per line (optional; may use {{VAR}})",
                    text: $template.optionsText, axis: .vertical)
                    .font(.system(.caption, design: .monospaced))
                    .lineLimit(1...4)
                TextField("target (optional)", text: $template.target)
                    .font(.system(.caption, design: .monospaced))
                    .frame(maxWidth: 200)
                    .help(
                        "the correct/expected option, by value — follows its "
                            + "option when counterbalancing reverses the order")
            }
        }
        .padding(6)
        .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 6))
    }
}

/// Save design… payload (canonical JSON bytes from `FactorialDesign.encoded`).
private struct DesignJSONDocument: FileDocument {
    static let readableContentTypes: [UTType] = [.json]

    var data: Data

    init(data: Data) {
        self.data = data
    }

    init(configuration: ReadConfiguration) throws {
        data = configuration.file.regularFileContents ?? Data()
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: data)
    }
}
