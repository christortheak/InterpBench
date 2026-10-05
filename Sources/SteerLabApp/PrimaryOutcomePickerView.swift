import ExperimentKit
import SwiftUI

/// Evaluation-section control for the study's declared primary outcome
/// (manifest `primaryOutcome`): the outcome the study is about, which every
/// results summary then leads with.
///
/// The pattern is `NumericParserControls`: reads the manifest, writes through
/// the store's draft-edit gate (`ExperimentStore.setPrimaryOutcome` — the
/// same writer the `experiment set-primary-outcome` verb uses, never a
/// parallel save path), plain-language failure line with the raw detail
/// underneath, read-only once frozen.
///
/// The menu lists what the study's settings can produce
/// (`HeadlineOutcome.producible`), so a choice here cannot be refused; a
/// declared outcome the settings no longer produce stays visible, with the
/// problem said in words, instead of silently snapping to another entry.
///
/// Wiring: one line in `StudyEvaluationSection`'s `analysisSettings`.
struct PrimaryOutcomeControls: View {
    let manifest: ExperimentManifest
    let panel: ExperimentPanel
    @State private var errorText: String?

    private var isDraft: Bool { manifest.status == .draft }
    private var declared: String { manifest.primaryOutcome ?? "" }
    private var producible: HeadlineOutcome.Producible {
        HeadlineOutcome.producible(manifest)
    }
    private var declaredIsProducible: Bool {
        declared.isEmpty
            || HeadlineOutcome.canProduce(manifest, outcome: declared)
    }

    var body: some View {
        picker
        caption
        if let errorText {
            Label(errorText, systemImage: "exclamationmark.triangle")
                .font(.caption)
                .foregroundStyle(.orange)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// One menu row: the stored outcome name and the words shown for it.
    private struct Choice: Identifiable {
        let id: String
        let label: String
    }

    /// The menu's rows: what the study's settings can produce, plus a
    /// declared outcome the menu would not otherwise show — a
    /// reasoning-style feature (those are named in the taxonomy file), or
    /// one the settings no longer produce — so it still renders as the
    /// selection.
    private var choices: [Choice] {
        var rows = producible.names.map {
            Choice(id: $0, label: EffectNarrative.metricPhrase($0))
        }
        if !declared.isEmpty, !producible.names.contains(declared) {
            let suffix =
                declaredIsProducible ? "" : " (not produced by these settings)"
            rows.append(
                Choice(
                    id: declared,
                    label: EffectNarrative.metricPhrase(declared) + suffix))
        }
        return rows
    }

    private var selection: Binding<String> {
        Binding(
            get: { declared },
            set: { newValue in
                write {
                    try ExperimentStore.setPrimaryOutcome(
                        newValue.isEmpty ? nil : newValue,
                        experimentName: manifest.name)
                }
            })
    }

    private static let helpText =
        "the outcome this study is about. Every results summary leads "
        + "with it and says it was declared by the researcher. "
        + "Written into the study as `primaryOutcome` and frozen "
        + "with it. Leave it undeclared and summaries lead by the "
        + "default order: a judged outcome, then a choice or numeric "
        + "outcome, then a reader or probe score, then reasoning "
        + "style, then marker density, then surface measures such "
        + "as word count"

    private var picker: some View {
        Picker("Primary outcome", selection: selection) {
            Text("Not declared (default order)").tag("")
            ForEach(choices) { choice in
                Text(choice.label).tag(choice.id)
            }
        }
        .disabled(!isDraft)
        .help(Self.helpText)
    }

    private var undeclaredText: String {
        let base =
            "not declared. Results summaries will lead by the default "
            + "order and say so."
        return base + commandLineHint
    }

    private var notProducibleText: String {
        "These settings cannot produce '\(declared)'. Summaries will fall "
            + "back to the default order and say so. Choose an outcome from "
            + "the menu, or restore the setting that produces it."
    }

    private var declaredText: String {
        let base =
            "Results summaries will lead with this outcome and say it was "
            + "declared by the researcher."
        return base + commandLineHint
    }

    /// Reasoning-style features are named in the pinned taxonomy file, so
    /// the menu cannot list them; the command line accepts any of them.
    private var commandLineHint: String {
        guard !producible.patterns.isEmpty, isDraft else { return "" }
        return " A reasoning-style feature can be declared from the command "
            + "line: steerlab-cli experiment set-primary-outcome "
            + "\(manifest.name) rs_<feature>."
    }

    @ViewBuilder private var caption: some View {
        if declared.isEmpty {
            Text(undeclaredText)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        } else if !declaredIsProducible {
            Label(notProducibleText, systemImage: "exclamationmark.triangle")
                .font(.caption)
                .foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
        } else {
            Text(declaredText)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
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
                "Couldn't update the primary outcome. The study must still "
                + "be a draft, and its settings must be able to produce the "
                + "outcome. Details: \(Self.detail(error))"
        }
    }

    /// `ExperimentError` is CustomStringConvertible, not LocalizedError, so
    /// its `reason` is the readable half; anything else gets its localized
    /// description rather than a Swift dump.
    private static func detail(_ error: some Error) -> String {
        (error as? ExperimentError)?.reason ?? error.localizedDescription
    }
}
