import Foundation

/// Publication of a reviewed configuration, independent of connection or deployment.
public enum ClusterProfileAcceptance {
    public struct Accepted: Sendable {
        public let site: ClusterSiteRecord
        public let evidencePath: String
        public let advisories: [String]
    }

    public struct Refusal: Error, LocalizedError, Sendable {
        public let code: String
        public let reason: String
        public var repairAction: String {
            "Run cluster sites review again, resolve its questions, and accept the exact reviewed draft SHA-256."
        }
        public var errorDescription: String? { reason + " — " + repairAction }
    }

    @MainActor public static func accept(
        data: Data, expectedSHA256: String, repository: ClusterSiteRepository,
        now: Date = Date()
    ) throws -> Accepted {
        guard ClusterSupportPaths.sha256Hex(data) == expectedSHA256 else {
            throw Refusal(code: "profileDraftChanged", reason: "The companion bytes changed after review; nothing was imported.")
        }
        let review = try ClusterProfileCoauthoring.review(data: data)
        guard review.readyForImport else {
            throw Refusal(code: "profileQuestions", reason: "The companion still has unresolved questions or inconsistent declarations.")
        }
        return try ManifestFileTransaction.withLock(manifestURL: repository.directoryURL,
            workspaceRoot: repository.directoryURL.deletingLastPathComponent()) {
            let identity = ClusterConnectionStore.canonicalKey(ClusterConnectionStore.registryKey(forProfile: review.profile))
            if let existing = try repository.sites().first(where: { $0.canonicalIdentity == identity }) {
                throw ClusterLifecycleError.siteFileExists(siteID: existing.id, path: repository.fileURL(forSite: existing.id).path)
            }
            // Retain exact reviewed bytes BEFORE profile publication. A failed import
            // may leave review evidence, never a false assertion of acceptance. The
            // archive is not a credential store and is outside every study/run hash.
            let directory = repository.directoryURL.appending(component: ".authoring")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true,
                                                   attributes: [.posixPermissions: 0o700])
            let evidence = directory.appending(component: expectedSHA256 + ".json")
            if FileManager.default.fileExists(atPath: evidence.path) {
                guard try Data(contentsOf: evidence) == data else {
                    throw Refusal(code: "profileEvidenceChanged", reason: "The saved review evidence is inconsistent; inspect the private configuration archive.")
                }
            } else {
                let staging = directory.appending(component: UUID().uuidString + ".tmp")
                defer { try? FileManager.default.removeItem(at: staging) }
                try data.write(to: staging, options: .withoutOverwriting)
                try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: staging.path)
                try FileManager.default.moveItem(at: staging, to: evidence)
            }
            var warnings = review.advisories
            let site = try repository.importProfile(review.profile, now: now, warn: { warnings.append($0) })
            return Accepted(site: site, evidencePath: evidence.path, advisories: warnings)
        }
    }
}
