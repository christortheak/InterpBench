import ExperimentKit
import SwiftUI

/// Reads the workspace captured in the displayed run path, independently of
/// compute selection. The parent keys this view by that path.
struct EvidenceCustodyView: View {
    let runDirectory: URL
    @State private var inventory: EvidenceCustodyInventory?
    @State private var message: String?
    /// Whether `message` reports a FAILURE — an error must not render in the
    /// same calm grey as a success (audit convention).
    @State private var verificationFailed = false
    @State private var refreshID = UUID()
    @State private var verification: Verification?

    private struct Verification: Equatable {
        let digest: String
        let requestID = UUID()
    }

    private var root: URL { runDirectory.deletingLastPathComponent().deletingLastPathComponent() }

    var body: some View {
        DisclosureGroup("Evidence retained locally") {
            Text("Import receipts record what arrived. Verify to check that the archive and imported files are still here.")
                .font(.caption).foregroundStyle(.secondary)
            if let inventory {
                if inventory.entries.isEmpty {
                    Text("No import receipts for this run. Local runs and older imports may have none.")
                        .font(.caption)
                }
                ForEach(inventory.entries) { entry in
                    VStack(alignment: .leading, spacing: 4) {
                        Text(Self.receiptDate(entry.createdAt))
                            .font(.caption)
                            .help("when this evidence was imported — recorded as \(entry.createdAt)")
                        if entry.failureRecorded { Text("Failure evidence").font(.caption) }
                        else if entry.evidenceComplete == false { Text("Partial evidence").font(.caption) }
                        Text(entry.receiptSHA256).font(.caption2.monospaced()).textSelection(.enabled)
                            .help("the receipt's own SHA-256 — the identity Verify checks against")
                        Button("Verify Retained Evidence") {
                            message = nil
                            verification = .init(digest: entry.receiptSHA256)
                        }
                        .disabled(verification != nil)
                        .help(
                            "re-reads the archive and imported files this "
                                + "receipt names and re-checks their hashes — "
                                + "local bytes only, nothing is fetched or "
                                + "deleted")
                    }
                }
                ForEach(inventory.issues, id: \.self) { issue in
                    Text(issue).font(.caption).foregroundStyle(.orange)
                }
            }
            if verification != nil { ProgressView("Checking local bytes…") }
            if let message {
                Label(message, systemImage: verificationFailed ? "exclamationmark.triangle" : "checkmark.seal")
                    .font(.caption)
                    .foregroundStyle(verificationFailed ? Color.orange : Color.secondary)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Button("Refresh Receipts") { message = nil; refreshID = UUID() }
                .disabled(verification != nil)
                .help("re-reads this run's import receipts from the workspace")
        }
        .help(
            "what came home from a remote run and whether those bytes are "
                + "still on this Mac — custody, not scientific validity")
        .task(id: refreshID) {
            let root = root
            let runID = runDirectory.lastPathComponent
            let result = await Task.detached { () -> Result<EvidenceCustodyInventory, Error> in
                Result { try EvidenceCustodyStore.inventory(runID: runID, workspaceRoot: root) }
            }.value
            guard !Task.isCancelled else { return }
            switch result {
            case .success(let value):
                inventory = value
                verificationFailed = false
            case .failure(let error):
                inventory = nil
                verificationFailed = true
                message = "Could not inspect receipts: \(Self.detail(error))"
            }
        }
        .task(id: verification) {
            guard let request = verification else { return }
            let root = root
            let result = await Task.detached { () -> Result<EvidenceCustodyReceipt, Error> in
                Result { try EvidenceCustodyStore.loadVerified(receiptSHA256: request.digest, workspaceRoot: root) }
            }.value
            guard !Task.isCancelled, verification == request else { return }
            switch result {
            case .success:
                verificationFailed = false
                message = "Archive and imported files verified now. Scientific validity and remote cleanup permission are separate."
            case .failure(let error):
                verificationFailed = true
                message = "Local custody could not be verified: \(Self.detail(error)). Recover missing evidence from its source; do not edit this run or delete remote evidence."
            }
            verification = nil
        }
    }

    /// The receipt's stored ISO-8601 string, read as a date — the raw string
    /// stays reachable in the row's tooltip (audit 10).
    private static func receiptDate(_ raw: String) -> String {
        guard let date = HousekeepingDates.parse(raw) else { return raw }
        return date.formatted(date: .abbreviated, time: .shortened)
    }

    /// `ExperimentError` is CustomStringConvertible, not LocalizedError, so
    /// its `reason` is the readable half; anything else gets its localized
    /// description rather than a Swift dump (audit headline 17).
    private static func detail(_ error: some Error) -> String {
        (error as? ExperimentError)?.reason ?? error.localizedDescription
    }
}
