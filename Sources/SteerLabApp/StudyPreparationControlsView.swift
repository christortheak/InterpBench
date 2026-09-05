import ExperimentKit
import SteeringKit
import SwiftUI
import UniformTypeIdentifiers

struct StudyPreparationControlsView: View {
    @Bindable var service: ChatService
    let manifest: ExperimentManifest
    @Binding var pendingModelJob: PendingModelJob?
    @Binding var runOnServerExpanded: Bool
    private var panel: ExperimentPanel { service.experiments }
    var body: some View {
        validationControls(manifest: manifest, panel: panel)
        if !manifest.concepts.isEmpty { extractControls(manifest: manifest, panel: panel) }
    }
    private func serverRunCaption(_ verb: String) -> String {
        "\(verb) runs on \(service.cluster.substrateLabel) as a durable job — reconnect from Compute"
    }
    @ViewBuilder
    private func validationControls(manifest: ExperimentManifest, panel: ExperimentPanel)
        -> some View
    {
        // Validate routes through the same server-resident path as Run in a
        // paired server workspace, so it shares the residency gate (the
        // callout in the run controls below explains the disabled state).
        // KNOWN-unpaired servers (Mac-authority mode, 2026-07-21) submit
        // validate as a hash-pinned BUNDLE job instead — no residency
        // needed, and the evidence bundle imports back into this workspace.
        let bundleValidate = panel.isKnownUnpairedServerWorkspace
        let missingOnServer =
            panel.isServerWorkspace && !bundleValidate
            && panel.serverHasSelectedStudy == false
        // The validation group: the declared read depth sits WITH the
        // button that launches validation (2026-08-01 review feedback —
        // it was orphaned in Evaluation below the save button, where its
        // connection to the validate verb was invisible).
        Text("Validation")
            .font(.caption.bold())
            .padding(.top, 4)
        if !manifest.concepts.isEmpty {
            ValidationDepthControls(manifest: manifest, panel: panel)
        }
        // Button row extracted (ValidateStudyButtonRow.swift): the gate
        // wiring pushed this function past the type-checker budget.
        ValidateStudyButtonRow(
            service: service,
            panel: panel,
            bundleValidate: bundleValidate,
            missingOnServer: missingOnServer,
            help: StudyControlCopy.validateHelp,
            pendingModelJob: $pendingModelJob,
            runOnServerExpanded: $runOnServerExpanded)

        Text(StudyControlCopy.validateCaption(variantsPresent: !manifest.variantConditions.isEmpty))
            .font(.caption2)
            .foregroundStyle(.secondary)
        if bundleValidate {
            Text(
                "unpaired server Compute — Validate submits the study as a "
                    + "hash-pinned bundle job on "
                    + "\(service.cluster.substrateLabel) (Remote options set "
                    + "executor/GPU/walltime); its evidence bundle imports "
                    + "back into this workspace and satisfies the local "
                    + "freeze gate for server runs"
            )
            .font(.caption2)
            .foregroundStyle(.secondary)
        } else if panel.isServerWorkspace {
            Text(serverRunCaption("Validate Study"))
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        // Direct validation executes the SERVER-RESIDENT copy only — the
        // callout explains the disabled state and points at the unified
        // Run's portable bundle path (which needs no residency).
        if missingOnServer {
            studyNotOnServerCallout(panel: panel)
        }

        if let validationDirectory = panel.lastValidationDirectory {
            Text(validationDirectory)
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .help("latest completed validation run directory")
        }
    }

    /// A11: the explicit Extract action — the CLI `experiment extract` verb
    /// in-panel. Deliberately offered on drafts AND frozen studies:
    /// extraction is deterministic re-derivation from the pinned recipe (the
    /// CLI verb gates only on verify()). Server workspaces submit the
    /// server's extract verb as a durable job.
    @ViewBuilder
    private func extractControls(manifest: ExperimentManifest, panel: ExperimentPanel) -> some View
    {
        let missingOnServer =
            panel.isServerWorkspace
            && panel.serverHasSelectedStudy == false
        let busy =
            panel.isExtracting || panel.isRunning || panel.isValidating
            || panel.isSweeping
        let reason = ExperimentPanel.extractDisabledReason(
            busy: busy,
            hasViolations: !panel.violations.isEmpty,
            missingOnServer: missingOnServer)
        HStack(spacing: 8) {
            Button(panel.isExtracting ? "Extracting…" : "Extract Vectors") {
                // Item 2: same no-GPU-session gate as Run/Validate —
                // extraction loads the pinned model on the server.
                let panel = panel
                ModelJobGPUGate.submit(
                    "vector extraction", service: service,
                    pending: $pendingModelJob
                ) { await panel.extractStudy() }
            }
            .buttonStyle(.bordered)
            .disabled(reason != nil)
            .help(StudyControlCopy.extractHelp)
            if panel.isExtracting, !panel.isServerWorkspace {
                ProgressView().controlSize(.small)
                Button("Stop", role: .destructive) { panel.cancelExtract() }
                    .controlSize(.small)
                    .disabled(panel.extractCancelRequested)
                    .help(
                        "stops after the current concept; completed vectors "
                            + "stay in the run directory, marked cancelled")
            }
        }
        if let reason {
            Text(reason)
                .font(.caption2)
                .foregroundStyle(.secondary)
        } else {
            Text(
                "re-derives every pinned concept's vectors into a new "
                    + "immutable runs/ directory — allowed on drafts and "
                    + "frozen studies alike (deterministic from the pins)"
            )
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
        if panel.isServerWorkspace {
            Text(serverRunCaption("Extract"))
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        if let extractDirectory = panel.lastExtractDirectory {
            Text(extractDirectory)
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .help("latest completed extract run directory")
        }
    }
    @ViewBuilder
    private func studyNotOnServerCallout(panel: ExperimentPanel) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Label(
                ExperimentPanel.residencyCalloutMessage(
                    study: panel.selectedName ?? "study",
                    substrate: service.cluster.substrateLabel),
                systemImage: "exclamationmark.triangle"
            )
            .font(.caption)
            .foregroundStyle(.orange)
            .textSelection(.enabled)
            Button("Show remote options") { runOnServerExpanded = true }
                .controlSize(.small)
                .help(
                    "expands the Remote options — the unified Run button "
                        + "submits a portable hash-pinned bundle, which works "
                        + "without server residency")
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
    }
}
