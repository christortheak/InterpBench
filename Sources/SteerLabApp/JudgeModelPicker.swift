import ExperimentKit
import SwiftUI

/// The single-string judge picker, shared by the two panes that have one
/// (the Robustness Check's judge, and the study pane's ad-hoc judge).
///
/// Thin by rule: every string it renders is composed on the panel type
/// (`JudgeModelOffers`), so what a researcher reads is asserted in tests
/// rather than inferred from a screenshot. This view only lays it out.
///
/// Layout notes for macOS 27: labels carry `fixedSize` so they never wrap,
/// captions are vertically compressible, and nothing here sets a frame
/// minimum.
struct JudgeModelPicker: View {

    let title: String
    let help: String
    let offers: JudgeModelOffers.Offers
    @Binding var selection: String
    /// The OpenRouter model + provider fields, revealed only when an
    /// OpenRouter judge is selected. The pane owns the bindings because it
    /// owns the panel property they write through.
    @Binding var openRouterModel: String
    @Binding var openRouterProvider: String

    private var isOpenRouterSelection: Bool {
        switch JudgeModelSpelling.parse(selection) {
        case .openRouter, .openRouterUnpinned: true
        default: false
        }
    }

    var body: some View {
        Picker(title, selection: $selection) {
            Text("select…").tag("")
            ForEach(offers.models) { option in
                Text(option.label).tag(option.id)
            }
            if !offers.openRouter.isEmpty {
                Section("OpenRouter") {
                    ForEach(offers.openRouter) { option in
                        Text(option.label).tag(option.id)
                    }
                }
            }
        }
        .help(help)

        if let caption = offers.selectionCaption {
            Label(caption, systemImage: "exclamationmark.triangle.fill")
                .font(.caption)
                .foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
        }

        if isOpenRouterSelection {
            HStack(spacing: 8) {
                Text("Model")
                    .font(.caption)
                    .fixedSize()
                TextField("author/slug", text: $openRouterModel)
                Text("Provider")
                    .font(.caption)
                    .fixedSize()
                TextField("serving provider", text: $openRouterProvider)
            }
            .help(
                "an OpenRouter judge pins BOTH: the model slug (OpenRouter's "
                    + "id, not always the Hugging Face repo id) and the "
                    + "serving provider it must be answered by")
            if openRouterProvider.trimmingCharacters(in: .whitespaces).isEmpty {
                Text(
                    JudgeModelSpelling.unpinnedProviderRefusal(
                        model: openRouterModel))
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }

        if let hint = offers.openRouterHint {
            Text(hint)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
