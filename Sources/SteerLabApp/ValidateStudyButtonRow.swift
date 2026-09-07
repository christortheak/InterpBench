import ExperimentKit
import SwiftUI

/// The Validate Study button row, extracted from
/// `ExperimentsPanelView.validationControls` (that function sat at the
/// SwiftUI type-checker's budget; the Mac-authority gate wiring pushed it
/// over — house rule: extract subviews rather than fight the checker).
///
/// Behavior, not just layout, is routing-sensitive (2026-07-21
/// Mac-authority round):
///
/// - Local workspace: in-process validation (cancellable between units).
/// - Paired server workspace: the direct server-resident `validate` verb
///   (GPU-session delegation included) — session-only GPU warning.
/// - KNOWN-unpaired server Compute: `panel.validateStudy()` submits
///   validate as a hash-pinned BUNDLE job, so the gate additionally checks
///   the submission's OWN resource request (executor + gres — the
///   no-GPU-ALLOCATION dialog with its Fix-options affordance).
struct ValidateStudyButtonRow: View {
    let service: ChatService
    let panel: ExperimentPanel
    /// Mac-authority mode: validate travels as a bundle job.
    let bundleValidate: Bool
    /// Residency gate (paired/direct path only — bundles need no residency).
    let missingOnServer: Bool
    let help: String
    @Binding var pendingModelJob: PendingModelJob?
    @Binding var runOnServerExpanded: Bool

    private var bundleOptions: ModelJobSubmissionPreflight.BundleOptions? {
        guard bundleValidate else { return nil }
        return ModelJobSubmissionPreflight.BundleOptions(
            executor: panel.submission.remoteExecutor,
            gres: panel.submission.remoteGres,
            verb: "validate",
            dryRun: false)
    }

    /// In flight on EITHER substrate (audit headline 5): the local
    /// controller's flag never covered the server routes, so the label
    /// stayed "Validate Study" and a second click submitted a second job.
    private var isValidating: Bool {
        panel.localJobs.isValidating || panel.isValidatingOnServer
    }

    private var disabled: Bool {
        isValidating || panel.localJobs.isRunning || panel.localJobs.isExtracting
            || !panel.violations.isEmpty
            || missingOnServer
    }

    /// Why the button is unavailable, in one line beside it — the idiom
    /// `ExperimentPanel.extractDisabledReason` already uses for the
    /// neighbouring Extract button. Busy states say so in the label, and the
    /// residency case has its own callout below, so both stay nil here.
    private var disabledReason: String? {
        guard !isValidating, !missingOnServer else { return nil }
        if panel.localJobs.isRunning {
            return "a study run is in progress — validation waits for it to finish"
        }
        if panel.localJobs.isExtracting {
            return "vector extraction is in progress — validation waits for it to finish"
        }
        if !panel.violations.isEmpty {
            let count = panel.violations.count
            return "\(count) verification problem\(count == 1 ? "" : "s") listed above — "
                + "validation runs only on a study that verifies"
        }
        return nil
    }

    @ViewBuilder
    var body: some View {
        HStack(spacing: 8) {
            Button(isValidating ? "Validating Study…" : "Validate Study") {
                // Re-entry guard on the same predicate the button reads: a
                // fast second click must not package and submit twice.
                guard !disabled else { return }
                submit()
            }
            .buttonStyle(.bordered)
            .disabled(disabled)
            .help(help)
            if isValidating {
                ProgressView().controlSize(.small)
            }
            // A1: local validation is cancellable between units of work.
            if panel.localJobs.isValidating, !panel.isServerWorkspace {
                Button("Stop", role: .destructive) { panel.cancelValidation() }
                    .controlSize(.small)
                    .disabled(panel.localJobs.validationCancelRequested)
                    .help(
                        "stops after the current unit of work; partial artifacts "
                            + "stay marked cancelled and NO validation evidence is "
                            + "written — reported as cancelled, never as an error")
            }
        }
        if let disabledReason {
            Text(disabledReason)
                .font(.caption2)
                .foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func submit() {
        let panel = panel
        var fixOptions: (@MainActor () -> Void)?
        if bundleValidate {
            fixOptions = {
                panel.applyGPUAllocationFix()
                runOnServerExpanded = true
            }
        }
        ModelJobGPUGate.submit(
            "study validation", service: service,
            pending: $pendingModelJob,
            bundleOptions: bundleOptions,
            fixOptions: fixOptions
        ) { await panel.validateStudy() }
    }
}
