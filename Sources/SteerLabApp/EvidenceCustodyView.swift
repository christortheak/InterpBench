import ExperimentKit
import SwiftUI

/// Reads the workspace captured in the displayed run path, independently of
/// compute selection. The parent keys this view by that path.
struct EvidenceCustodyView: View {
    let runDirectory: URL
    @State private var inventory: EvidenceCustodyInventory?
    @State private var message: String?
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
                        Text(entry.createdAt).font(.caption)
                        if entry.failureRecorded { Text("Failure evidence").font(.caption) }
                        else if entry.evidenceComplete == false { Text("Partial evidence").font(.caption) }
                        Text(entry.receiptSHA256).font(.caption2.monospaced()).textSelection(.enabled)
                        Button("Verify retained evidence") {
                            message = nil
                            verification = .init(digest: entry.receiptSHA256)
                        }
                        .disabled(verification != nil)
                    }
                }
                ForEach(inventory.issues, id: \.self) { issue in
                    Text(issue).font(.caption).foregroundStyle(.orange)
                }
            }
            if verification != nil { ProgressView("Checking local bytes…") }
            if let message { Text(message).font(.caption).textSelection(.enabled) }
            Button("Refresh receipts") { message = nil; refreshID = UUID() }.disabled(verification != nil)
        }
        .task(id: refreshID) {
            let root = root
            let runID = runDirectory.lastPathComponent
            let result = await Task.detached { () -> Result<EvidenceCustodyInventory, Error> in
                Result { try EvidenceCustodyStore.inventory(runID: runID, workspaceRoot: root) }
            }.value
            guard !Task.isCancelled else { return }
            switch result {
            case .success(let value): inventory = value
            case .failure(let error): inventory = nil; message = "Could not inspect receipts: \(error)"
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
                message = "Archive and imported files verified now. Scientific validity and remote cleanup permission are separate."
            case .failure(let error):
                message = "Local custody could not be verified: \(error). Recover missing evidence from its source; do not edit this run or delete remote evidence."
            }
            verification = nil
        }
    }
}
