import ExperimentKit
import SwiftUI

/// Freeze readiness and explicit confirmation; policy decisions remain in FreezeRouting.
struct StudyFreezeControlsView: View {
    let manifest: ExperimentManifest
    let coordinator: StudyFreezeController
    let inputs: FreezeRouting.Inputs
    let violations: [String]
    let freezeLocally: () -> Void
    let freezeOnServer: () async -> Void
    let syncDraft: () async -> Void
    @State private var confirmFreeze = false

    private func freezeRoutingDecision() -> FreezeRouting.Decision { FreezeRouting.decide(inputs) }

    /// Freeze button + readiness for a draft: the button routes to the
    /// substrate the workspace scopes to ("Freeze (on <server>)…" in a
    /// server workspace — gates evaluated there against SERVER-substrate
    /// evidence), local readiness renders as before, cross-substrate
    /// evidence advisories are promoted to warnings at this decision point,
    /// and the server's own gate refusal / advisories from the last remote
    /// attempt render in the same idiom as the local readiness items.
    @ViewBuilder
    var body: some View {
        let decision = freezeRoutingDecision()
        Button(decision.buttonLabel) { confirmFreeze = true }
            .buttonStyle(.borderedProminent)
            // Server-routed: the SERVER's gates decide remote readiness —
            // local verification failures render as context below, never as
            // a disabled button (rule unit-tested in FreezeRouting).
            .disabled(
                FreezeRouting.freezeButtonDisabled(
                    decision: decision,
                    hasLocalViolations: !violations.isEmpty)
                    || coordinator.isFreezingOnServer || coordinator.isSyncingServerDraft
            )
            .help(
                decision.target == .server
                    ? StudyControlCopy.remoteFreezeHelp : StudyControlCopy.freezeHelp
            )
            .confirmationDialog(
                freezeDialogTitle(manifest.name, decision: decision),
                isPresented: $confirmFreeze,
                titleVisibility: .visible
            ) {
                Button(decision.confirmLabel, role: .destructive) {
                    if decision.target == .server {
                        Task { await freezeOnServer() }
                    } else {
                        freezeLocally()
                    }
                }
            } message: {
                // The dialog used to be title-only, so a one-way click could
                // be confirmed without the unmet gates or the prominent
                // advisories in front of the researcher (audit 10).
                Text(freezeDialogMessage(decision: decision))
            }
        if let note = decision.executorNote {
            Text(note)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        if let blocked = decision.blockedReason {
            Label(blocked, systemImage: "exclamationmark.triangle")
                .font(.caption)
                .foregroundStyle(.orange)
                .textSelection(.enabled)
        }
        // Server-routed freeze with local verification failures: the local
        // readiness is CONTEXT here (the server verifies ITS copy's pins at
        // the click) — informational, never a disabled button.
        if decision.target == .server,
            let note = FreezeRouting.localViolationsContextNote(
                count: violations.count,
                serverLabel: inputs.serverLabel)
        {
            Label(note, systemImage: "info.circle")
                .font(.caption)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        }
        // The manifest-identity guard's answers from the last remote-freeze
        // attempt: a block (the server's same-named copy is not the document
        // on screen — field-level summary + remedy) renders prominently; a
        // proceeded-with note (server-only copy / paired-unverifiable)
        // renders as info.
        if let identityWarning = coordinator.remoteFreezeIdentityWarning {
            Label(identityWarning, systemImage: "exclamationmark.triangle")
                .font(.caption)
                .foregroundStyle(.orange)
                .textSelection(.enabled)
            // The one-click remedy (2026-07-21 incident, part 3): the
            // mismatch block used to name a remedy the app didn't offer.
            // Push the manifest ON SCREEN as the server's draft copy, then
            // re-verify — draft manifests only, frozen copies refuse
            // server-side (freeze firewall).
            if coordinator.remoteFreezeCanSyncDraft {
                Button(
                    coordinator.isSyncingServerDraft
                        ? "Updating the server's copy…"
                        : "Update the server's copy"
                ) {
                    Task { await syncDraft() }
                }
                .controlSize(.small)
                .disabled(coordinator.isSyncingServerDraft)
                .help(
                    "push the manifest you are looking at to "
                        + "\(inputs.serverLabel) as its DRAFT copy "
                        + "and re-run the identity check — the freeze itself "
                        + "stays a separate, deliberate click")
            }
        }
        if let identityNote = coordinator.remoteFreezeIdentityNote {
            Label(identityNote, systemImage: "info.circle")
                .font(.caption)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        }
        // Read-only freeze readiness: the same gates freeze enforces,
        // reported before the one-way click. For a server-routed freeze the
        // gates are re-evaluated SERVER-side at the click; on a paired
        // workspace this local view reads the same shared tree.
        if let readiness = coordinator.freezeReadiness {
            Label(
                readiness.ready
                    ? "ready to freeze"
                    : "not ready to freeze — \(readiness.unmetGates.count) unmet "
                        + "gate\(readiness.unmetGates.count == 1 ? "" : "s")",
                systemImage: readiness.ready
                    ? "checkmark.seal" : "hourglass"
            )
            .font(.caption)
            .foregroundStyle(readiness.ready ? Color.green : Color.secondary)
            .help(
                readiness.ready
                    ? "every freeze gate is currently satisfied"
                    : "each unmet gate is listed below — freeze refuses until "
                        + "they are satisfied")
            // A gate standing between the researcher and a one-way action
            // must be readable without hovering (audit 10): every unmet gate
            // gets its own row, not the first three in a tooltip.
            ForEach(readiness.unmetGates, id: \.self) { gate in
                Label(gate, systemImage: "hourglass")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            }
            // Non-blocking advisories (e.g. hand-created variants without
            // sweep-selection provenance): visible next to the gates, never
            // a refusal. Cross-substrate validate-evidence advisories are
            // PROMINENT here — this is the freeze decision they exist for.
            freezeAdvisoryRows(readiness.advisories)
        }
        // Decision-time coherence check for a server-routed freeze on a
        // paired workspace: the same evidence, seen from the server's
        // perspective (validate-locally-then-freeze-on-server warns BEFORE
        // the click).
        if decision.target == .server,
            let advisory = coordinator.serverFreezeCrossSubstrateAdvisory
        {
            freezeAdvisoryRows([advisory])
        }
        // The server's own answer to the last remote-freeze attempt,
        // rendered exactly like the local readiness items: a gate refusal
        // reads as an unmet gate (verbatim server wording), advisories as
        // advisory rows.
        if let failure = coordinator.remoteFreezeGateFailure {
            Label(
                ExperimentStore.FreezeReadiness(unmetGates: [failure]).displayLine(),
                systemImage: "hourglass"
            )
            .font(.caption)
            .foregroundStyle(Color.secondary)
            .help(failure)
            .textSelection(.enabled)
        }
        freezeAdvisoryRows(coordinator.remoteFreezeAdvisories)
    }

    /// One rendering rule for freeze advisories (local readiness, the
    /// paired-workspace server-perspective check, and the server's response
    /// advisories): cross-substrate evidence advisories render as warnings
    /// with the one-line rule appended; the rest stay info rows. The split
    /// itself is unit-tested in `FreezeRouting.present`.
    @ViewBuilder
    private func freezeAdvisoryRows(_ advisories: [String]) -> some View {
        let presentation = FreezeRouting.present(advisories: advisories)
        ForEach(presentation.prominent, id: \.self) { advisory in
            Label(advisory, systemImage: "exclamationmark.triangle")
                .font(.caption)
                .foregroundStyle(.orange)
                .textSelection(.enabled)
        }
        ForEach(presentation.regular, id: \.self) { advisory in
            Label(advisory, systemImage: "info.circle")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    /// Title only names the object; everything else moved into `message`,
    /// which macOS renders as body text rather than a bold paragraph.
    private func freezeDialogTitle(_ name: String, decision: FreezeRouting.Decision) -> String {
        switch decision.target {
        case .thisMac: "Freeze '\(name)'?"
        case .server: "Freeze '\(name)' on \(inputs.serverLabel)?"
        }
    }

    /// What the researcher must have read before a one-way click: the
    /// consequence, where the gates are evaluated, and the state of those
    /// gates AT THE MOMENT of the click (the button stays enabled when they
    /// are unmet — the gate itself refuses).
    private func freezeDialogMessage(decision: FreezeRouting.Decision) -> String {
        var parts: [String] = []
        switch decision.target {
        case .thisMac:
            parts.append(
                "One-way: afterwards the study can only be duplicated, never "
                    + "edited.")
        case .server:
            parts.append(
                "One-way: afterwards the study can only be duplicated, never "
                    + "edited. \(inputs.serverLabel) evaluates the gates "
                    + "against ITS OWN substrate's validation evidence and "
                    + "stamps frozenBy: \"server\".")
        }
        if let readiness = coordinator.freezeReadiness, !readiness.ready {
            parts.append(
                "Unmet gate\(readiness.unmetGates.count == 1 ? "" : "s") right "
                    + "now — freeze will refuse until these are satisfied:\n"
                    + readiness.unmetGates.map { "• \($0)" }.joined(separator: "\n"))
        }
        var advisories = coordinator.freezeReadiness?.advisories ?? []
        advisories += coordinator.remoteFreezeAdvisories
        if decision.target == .server,
            let crossSubstrate = coordinator.serverFreezeCrossSubstrateAdvisory
        {
            advisories.append(crossSubstrate)
        }
        let prominent = FreezeRouting.present(advisories: advisories).prominent
        if !prominent.isEmpty {
            parts.append(
                "Worth reading first:\n"
                    + prominent.map { "• \($0)" }.joined(separator: "\n"))
        }
        return parts.joined(separator: "\n\n")
    }
}
