import ExperimentKit
import SwiftUI

/// Renders `ExecutionPlanEcho.Plan` immediately above a submit control.
///
/// A mismatch is deliberately NOT a refusal: declaring a chain and then
/// submitting a single verb is legal and sometimes exactly what is wanted
/// (re-running one stage). It just must not be silent — the whole failure
/// mode being fixed is a submission that quietly did something other than
/// what the visible declarations implied.
///
/// Own file: `ExperimentsPanelView` is at the type-checker's limit.
struct ExecutionPlanEchoRow: View {
    let plan: ExecutionPlanEcho.Plan
    /// Opens the Remote options disclosure — where the verb that caused a
    /// mismatch actually lives. Naming the fix is not enough when the
    /// control is collapsed out of sight.
    var revealOptions: (() -> Void)?

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Label(plan.summary, systemImage: "arrow.right.circle")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if let mismatch = plan.mismatch {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Label(mismatch, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                    if let revealOptions {
                        Button("Show Remote options", action: revealOptions)
                            .buttonStyle(.link)
                            .font(.caption)
                    }
                }
            }
        }
    }
}
