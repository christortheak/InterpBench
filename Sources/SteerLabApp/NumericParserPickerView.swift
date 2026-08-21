import ExperimentKit
import SwiftUI

/// Evaluation-section controls for the declared numeric-answer parser
/// (manifest `numericParser` + pinned `parserRegistryHash`): how a number
/// is read out of each response, chosen from the workspace's parser
/// registry instead of hand-editing manifest JSON.
///
/// The pattern is `OrdinalScaleInstrumentControls`: reads the manifest,
/// writes through the store's draft-edit gate
/// (`ExperimentStore.setNumericParser` — the ordinary manifest-editing
/// pathway, never a parallel save path), plain-language failure line with
/// the raw detail underneath, read-only once frozen.
///
/// Wiring: one line in `ExperimentsPanelView`'s `analysisSettings` (the
/// case-family field's home — the two declarations both select how the
/// analysis reads what came back):
///
///     NumericParserControls(manifest: manifest, panel: panel)
struct NumericParserControls: View {
    let manifest: ExperimentManifest
    let panel: ExperimentPanel
    @State private var errorText: String?

    private var isDraft: Bool { manifest.status == .draft }
    private var declared: String { manifest.numericParser ?? "" }
    private var entries: [ParserRegistryUI.Entry] { ParserRegistryUI.entries() }

    var body: some View {
        // Numeric parsing reads per-record model outputs — a model-output
        // concern (the multi-agent panel endpoint is a documented residual).
        if manifest.studyKind == .modelOutput {
            picker
            selectionCaption
            registryNotes
            if !declared.isEmpty || manifest.parserRegistryHash != nil {
                FileReferenceRow(
                    label: "parser registry",
                    path: ParserRegistry.registryFile,
                    pinnedHash: manifest.parserRegistryHash,
                    allowsOpenInEditor: true)
                driftAffordance
            }
            if let errorText {
                Label(errorText, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var picker: some View {
        HStack(spacing: 6) {
            Picker(
                "Numeric answer parser",
                selection: Binding(
                    get: { declared },
                    set: { newValue in
                        write {
                            try ExperimentStore.setNumericParser(
                                newValue.isEmpty ? nil : newValue,
                                experimentName: manifest.name)
                        }
                    })
            ) {
                Text("None (built-in)").tag("")
                ForEach(entries) { entry in
                    Text("\(entry.name) — \(entry.plainKind)").tag(entry.name)
                }
                // A declared parser the registry no longer defines still
                // renders as the selection (with the problem stated below)
                // instead of silently snapping to another entry.
                if !declared.isEmpty, !entries.contains(where: { $0.id == declared }) {
                    Text("\(declared) (not in the registry)").tag(declared)
                }
            }
            .disabled(!isDraft)
            .help(
                "how a numeric answer is read out of each response — a named "
                    + "entry from prompts/parsers/parser-registry.json, written "
                    + "into the manifest as `numericParser` and pinned by hash. "
                    + "'None' keeps the built-in behavior (case family "
                    + "'sentencing' → parsedMonths)")
            InfoButton(text: StudyInfo.numericParser)
        }
    }

    @ViewBuilder private var selectionCaption: some View {
        if declared.isEmpty {
            Text(
                "none — built-in behavior: only case family 'sentencing' "
                    + "parses a numeric endpoint (parsedMonths)")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        } else if let entry = entries.first(where: { $0.id == declared }),
            !entry.description.isEmpty
        {
            Text(entry.description)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// Honest empty state + the declared parser's problems, in plain words
    /// (the engine's own verify-surface checks — never re-implemented here).
    @ViewBuilder private var registryNotes: some View {
        if entries.isEmpty, let problem = ParserRegistryUI.loadProblem(),
            declared.isEmpty
        {
            Label(problem, systemImage: "info.circle")
                .font(.caption2)
                .foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
        }
        ForEach(ParserRegistryUI.problems(for: manifest), id: \.self) { problem in
            Label(problem, systemImage: "exclamationmark.triangle")
                .font(.caption)
                .foregroundStyle(.orange)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// The standard drift affordance: the pinned registry no longer matches
    /// the file. The problem line above says so (engine wording); a draft
    /// additionally gets the one-click deliberate repair.
    @ViewBuilder private var driftAffordance: some View {
        if isDraft, !declared.isEmpty, ParserRegistryUI.registryDrifted(manifest) {
            Button("Re-pin registry (accept the current file)") {
                write {
                    try ExperimentStore.setNumericParser(
                        declared, experimentName: manifest.name)
                }
            }
            .font(.caption)
            .help(
                "re-pins parserRegistryHash to the registry file's current "
                    + "bytes — a deliberate acceptance of the edit; runs refuse "
                    + "while the pin and the file disagree")
        }
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
                "Couldn't update the numeric parser — the study must still "
                + "be a draft, and the parser must be defined in the "
                + "registry. Details: \(error)"
        }
    }
}
