import SwiftUI

/// The study-wide temperature control.
///
/// Was a bare `Slider` with no readout (finding 7a): "0" and "0.1" are one
/// step apart and looked identical, yet the difference decides whether the
/// study samples at all. Now the number is visible and directly typeable.
/// (A nonzero value used to make a study server-only; local runs now sample
/// with a seeded stream per record, so the help no longer says so.)
///
/// Lives in its own file because `ExperimentsPanelView` sits at the Swift
/// type-checker's limits; an inline `LabeledContent { HStack { … } }` there
/// tips it over ("unable to type-check this expression in reasonable time").
struct TemperatureRow: View {
    @Binding var value: Double
    /// The row's tooltip. Defaults to the study wording; the Playground
    /// passes its own, because a Playground chat's temperature has none of
    /// the study consequences (audit 2026-09-06, headline 11).
    var help: String =
        "study-wide generation temperature. At 0 the model always picks its "
        + "most likely next word, so a repeat gives the same text. Above 0 "
        + "it samples, with a separate seeded stream for each record; how "
        + "closely a repeat matches depends on the backend and the model "
        + "configuration"

    var body: some View {
        LabeledContent("Temperature") {
            HStack(spacing: 8) {
                Slider(value: $value, in: 0 ... 1.5, step: 0.1)
                    .accessibilityLabel("Temperature")
                TextField("Temperature", value: $value, format: Self.format)
                    .labelsHidden()
                    .frame(width: 56)
                    .multilineTextAlignment(.trailing)
            }
        }
        .help(help)
    }

    private static let format = FloatingPointFormatStyle<Double>()
        .precision(.fractionLength(0 ... 2))
}
