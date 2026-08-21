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
            executor: panel.remoteExecutor,
            gres: panel.remoteGres,
            verb: "validate",
            dryRun: false)
    }

    private var disabled: Bool {
        panel.isValidating || panel.isRunning || panel.isExtracting
            || !panel.violations.isEmpty
            || missingOnServer
    }

    var body: some View {
        HStack(spacing: 8) {
            Button(panel.isValidating ? "Validating Study…" : "Validate Study") {
                submit()
            }
            .buttonStyle(.bordered)
            .disabled(disabled)
            .help(help)
            // A1: local validation is cancellable between units of work.
            if panel.isValidating, !panel.isServerWorkspace {
                ProgressView().controlSize(.small)
                Button("Stop", role: .destructive) { panel.cancelValidation() }
                    .controlSize(.small)
                    .disabled(panel.validationCancelRequested)
                    .help(
                        "stops after the current unit of work; partial artifacts "
                            + "stay marked cancelled and NO validation evidence is "
                            + "written — reported as cancelled, never as an error")
            }
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
