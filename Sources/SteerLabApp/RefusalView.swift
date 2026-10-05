import ExperimentKit
import SwiftUI

/// The one way the app shows a refusal: the plain reason, then what to do —
/// as a button when the host view can do it — and the command-line repair
/// behind a disclosure for people who use one.
///
/// The host passes `perform` only for the actions it really offers; a
/// refusal whose action the host cannot take shows the words alone.
struct RefusalView: View {
    let refusal: RefusalPresentation
    var perform: ((RefusalPresentation.AppAction) -> Void)?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Label {
                Text(refusal.reason)
                    .fixedSize(horizontal: false, vertical: true)
            } icon: {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
            }
            .font(.caption)
            Text(refusal.whatToDo)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if let action = refusal.appAction, let perform {
                Button(action.title) { perform(action) }
                    .controlSize(.small)
            }
            if let command = refusal.commandLine {
                DisclosureGroup("For the command line") {
                    Text(command)
                        .font(.caption.monospaced())
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .font(.caption)
                .help("the same repair, written for steerlab-cli or steerlab")
            }
        }
        .textSelection(.enabled)
    }
}
