import Foundation

/// Read-only workbench adapters. The request names its workspace so a caller
/// cannot silently verify against another workspace after selection changes.
enum EvidenceCustodyHTTP {
    struct Response: Sendable {
        let status: String
        let body: Data

        static func json(_ value: some Encodable, status: String = "200 OK") throws -> Self {
            .init(status: status, body: try JSONEncoder().encode(value))
        }

        static func failure(_ code: String, reason: String, repair: String, status: String) -> Self {
            struct Failure: Encodable {
                let ok = false
                let code: String
                let error: String
                let repairAction: String
            }
            // These scalar fields are always JSON-encodable.
            return .init(status: status, body: (try? JSONEncoder().encode(
                Failure(code: code, error: reason, repairAction: repair))) ?? Data())
        }
    }

    private struct Request: Decodable {
        let workspaceRoot: String
        let runID: String?
        let receiptSHA256: String?
    }

    static func perform(body: Data, verifying: Bool, workspaceRoot: URL) -> Response {
        let keys: Set<String> = verifying ? ["workspaceRoot", "receiptSHA256"] : ["workspaceRoot", "runID"]
        guard let fields = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any],
            Set(fields.keys) == keys,
            let request = try? JSONDecoder().decode(Request.self, from: body),
            request.workspaceRoot.hasPrefix("/"),
            verifying ? request.receiptSHA256 != nil : request.runID != nil else {
            return .failure("invalidCustodyRequest", reason: "Name the workspace and custody target explicitly.",
                repair: "Supply exactly " + keys.sorted().joined(separator: ", ") + " in the JSON body.", status: "400 Bad Request")
        }
        do {
            guard try ManifestFileTransaction.canonicalPath(URL(fileURLWithPath: request.workspaceRoot))
                == ManifestFileTransaction.canonicalPath(workspaceRoot) else {
                return .failure("custodyWorkspaceChanged", reason: "The workbench is serving another workspace.",
                    repair: "Reconnect to the originating workspace and inspect its receipts again.", status: "409 Conflict")
            }
            if verifying {
                struct Verified: Encodable {
                    let ok = true
                    let verified = true
                    let receiptSHA256: String
                    let receipt: EvidenceCustodyReceipt
                }
                let digest = request.receiptSHA256!
                return try .json(Verified(receiptSHA256: digest,
                    receipt: EvidenceCustodyStore.loadVerified(receiptSHA256: digest, workspaceRoot: workspaceRoot)))
            }
            struct Inventory: Encodable {
                let ok = true
                let inventory: EvidenceCustodyInventory
            }
            return try .json(Inventory(inventory: EvidenceCustodyStore.inventory(runID: request.runID!, workspaceRoot: workspaceRoot)))
        } catch {
            return .failure("custodyUnverified", reason: String(describing: error),
                repair: "Check the originating workspace and receipt digest; recover missing evidence from its source without editing runs or deleting remote evidence.",
                status: "409 Conflict")
        }
    }
}
