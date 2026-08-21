import SwiftUI

/// The study-wide temperature control.
///
/// Was a bare `Slider` with no readout (finding 7a): "0" and "0.1" are one
/// step apart and looked identical, yet the difference decides where the
/// study can run at all — local measured runs require exactly 0, so any
/// nonzero value silently makes the study server-only. Now the number is
/// visible and directly typeable.
///
/// Lives in its own file because `ExperimentsPanelView` sits at the Swift
/// type-checker's limits; an inline `LabeledContent { HStack { … } }` there
/// tips it over ("unable to type-check this expression in reasonable time").
struct TemperatureRow: View {
    @Binding var value: Double

    var body: some View {
        LabeledContent("Temperature") {
            HStack(spacing: 8) {
                Slider(value: $value, in: 0 ... 1.5, step: 0.1)
                TextField("", value: $value, format: Self.format)
                    .frame(width: 56)
                    .multilineTextAlignment(.trailing)
            }
        }
        .help(
            "study-wide generation temperature. Measured agent-comparison "
                + "runs currently require 0 for reproducibility, so a nonzero "
                + "value routes the study to the Python server")
    }

    private static let format = FloatingPointFormatStyle<Double>()
        .precision(.fractionLength(0 ... 2))
}
