import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit

/// Open-issues §3 — `remote import-chain` must be atomic on transport failure.
///
/// The named defect was "0-file run directories in the local workspace after a
/// failed import": empty shells that pass every exists-check and then block a
/// naive re-import (RUNS-GUIDE trap class 4). The mechanism found in the code
/// is a staging/destination VOLUME mismatch. `EvidenceBundleImporter` extracted
/// into `VectorCatalog.projectRoot/.steerlab/imports/<uuid>` and published into
/// `ExperimentStore.runsDirectory` — and under the workspace rule those are
/// different trees (`ExperimentStore.workspaceRoot` honours the workspace
/// override; `projectRoot` does not). Foundation's `moveItem` across volumes is
/// not a rename: it copies and then deletes, so an interruption part-way
/// through leaves a partial — possibly empty — destination directory.
///
/// The fix mirrors `WorkspaceStore.create` (staging + one `moveItem`, landed
/// 2026-08-18): stage in a hidden SIBLING of the destination, verify, publish
/// with a single same-volume rename, and on any failure remove staging and
/// leave the destination ABSENT.
struct EvidenceImportAtomicityTests {

    private func sha(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Build a real evidence bundle for `runID` with the given files.
    /// `corruptEntryHashes` makes the bundle fail verification AFTER extraction
    /// — the closest stand-in for "the transfer did not deliver what the server
    /// stamped", which is what a dead tunnel produces.
    private func bundle(
        runID: String, files: [String: String], corruptEntryHashes: Bool = false
    ) throws -> URL {
        let fm = FileManager.default
        let staging = fm.temporaryDirectory.appending(
            component: "atomicity-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: staging) }
        var entries: [[String: Any]] = []
        let dir = staging.appending(components: "runs", runID)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        for (name, contents) in files.sorted(by: { $0.key < $1.key }) {
            let data = Data(contents.utf8)
            try data.write(to: dir.appending(component: name))
            entries.append([
                "path": "runs/\(runID)/\(name)",
                "sha256": corruptEntryHashes ? String(repeating: "0", count: 64)
                    : sha(data),
            ])
        }
        try JSONSerialization.data(withJSONObject: [
            "runID": runID, "entries": entries,
        ]).write(to: staging.appending(component: "steerlab-evidence.json"))
        let archive = fm.temporaryDirectory.appending(
            component: "evidence-\(UUID().uuidString).tar.gz")
        let tar = Process()
        tar.executableURL = URL(filePath: "/usr/bin/tar")
        tar.currentDirectoryURL = staging
        tar.arguments = ["-czf", archive.path, "."]
        try tar.run()
        tar.waitUntilExit()
        #expect(tar.terminationStatus == 0)
        return archive
    }

    private func leftoverStaging() -> [String] {
        let root = EvidenceBundleImporter.stagingRoot()
        let contents = (try? FileManager.default.contentsOfDirectory(
            atPath: root.path)) ?? []
        return contents.filter {
            $0.hasPrefix(EvidenceBundleImporter.stagingPrefix)
        }
    }

    /// The property the whole fix rests on: staging and destination are on the
    /// same volume, so the publish is a rename and not a copy. Asserted against
    /// the filesystem's own volume identifier rather than by reading paths.
    @Test func stagingIsOnTheSameVolumeAsTheDestination() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "evatomic") { _ in
            let fm = FileManager.default
            let runs = ExperimentStore.runsDirectory
            try fm.createDirectory(at: runs, withIntermediateDirectories: true)
            let staging = EvidenceBundleImporter.stagingRoot()
            func volume(_ url: URL) throws -> String {
                let identifier = try url.resourceValues(
                    forKeys: [.volumeIdentifierKey]).volumeIdentifier
                return String(describing: identifier)
            }
            #expect(try volume(staging) == volume(runs))
            // …and it is the workspace that CONTAINS runs/, not the checkout.
            #expect(runs.deletingLastPathComponent() == staging)
        }
    }

    @Test func aSuccessfulImportPublishesEveryByteAndLeavesNoStaging() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "evatomic") { _ in
            let fm = FileManager.default
            let runID = "20260818T000001000-exp-atomic-run"
            let archive = try bundle(
                runID: runID,
                files: ["generations.jsonl": "{\"a\":1}\n{\"a\":2}\n",
                        "config.json": "{\"schema\":4}"])
            defer { try? fm.removeItem(at: archive) }

            let imported = try EvidenceBundleImporter.importEvidenceBundle(archive)
            #expect(imported == ExperimentStore.runsDirectory
                .appending(component: runID))
            #expect(try String(
                contentsOf: imported.appending(component: "generations.jsonl"),
                encoding: .utf8) == "{\"a\":1}\n{\"a\":2}\n")
            #expect(try String(
                contentsOf: imported.appending(component: "config.json"),
                encoding: .utf8) == "{\"schema\":4}")
            #expect(leftoverStaging().isEmpty)
        }
    }

    /// The §3 assertion in its plainest form: a delivery that does not verify
    /// leaves NO destination directory — not an empty one, not a partial one.
    @Test func aFailedImportLeavesTheDestinationAbsent() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "evatomic") { _ in
            let fm = FileManager.default
            let runID = "20260818T000002000-exp-atomic-run"
            let archive = try bundle(
                runID: runID, files: ["generations.jsonl": "{\"a\":1}\n"],
                corruptEntryHashes: true)
            defer { try? fm.removeItem(at: archive) }

            #expect(throws: (any Error).self) {
                _ = try EvidenceBundleImporter.importEvidenceBundle(archive)
            }
            let destination = ExperimentStore.runsDirectory
                .appending(component: runID)
            // Absent, not empty: an empty directory is the trap.
            #expect(!fm.fileExists(atPath: destination.path))
            #expect(leftoverStaging().isEmpty)
        }
    }

    /// A truncated/substituted download is rejected on the whole-bundle hash
    /// pin, before extraction — and again, nothing lands.
    @Test func aBundleThatFailsItsHashPinLeavesNothingBehind() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "evatomic") { _ in
            let fm = FileManager.default
            let runID = "20260818T000003000-exp-atomic-run"
            let archive = try bundle(
                runID: runID, files: ["generations.jsonl": "{\"a\":1}\n"])
            defer { try? fm.removeItem(at: archive) }

            #expect(throws: (any Error).self) {
                _ = try EvidenceBundleImporter.importEvidenceBundle(
                    archive, expectedSHA256: String(repeating: "b", count: 64))
            }
            #expect(!fm.fileExists(atPath: ExperimentStore.runsDirectory
                .appending(component: runID).path))
            #expect(leftoverStaging().isEmpty)
        }
    }

    /// Re-importing after a failure must work — which is only true because the
    /// failure left the destination absent rather than an empty shell that the
    /// collision refusal (and the chain's skip-if-present rule) would honour.
    @Test func aRetryAfterAFailureSucceeds() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "evatomic") { _ in
            let fm = FileManager.default
            let runID = "20260818T000004000-exp-atomic-run"
            let broken = try bundle(
                runID: runID, files: ["generations.jsonl": "{\"a\":1}\n"],
                corruptEntryHashes: true)
            let good = try bundle(
                runID: runID, files: ["generations.jsonl": "{\"a\":1}\n"])
            defer {
                try? fm.removeItem(at: broken)
                try? fm.removeItem(at: good)
            }
            #expect(throws: (any Error).self) {
                _ = try EvidenceBundleImporter.importEvidenceBundle(broken)
            }
            let imported = try EvidenceBundleImporter.importEvidenceBundle(good)
            #expect(fm.fileExists(
                atPath: imported.appending(component: "generations.jsonl").path))
        }
    }
}

/// The download half of §3: bytes stage beside the destination, are checked
/// for having actually landed, and only then take the artifact's name. A dead
/// tunnel must leave no download directory at all.
struct ArtifactDownloadAtomicityTests {

    private final class StubProtocol: URLProtocol, @unchecked Sendable {
        nonisolated(unsafe) static var handler:
            (@Sendable (URLRequest) throws -> (Data, Int))?

        override class func canInit(with request: URLRequest) -> Bool { true }
        override class func canonicalRequest(for r: URLRequest) -> URLRequest { r }
        override func startLoading() {
            guard let handler = Self.handler else {
                client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
                return
            }
            do {
                let (data, status) = try handler(request)
                let response = HTTPURLResponse(
                    url: request.url!, statusCode: status, httpVersion: "HTTP/1.1",
                    headerFields: ["Content-Type": "application/gzip"])!
                client?.urlProtocol(self, didReceive: response,
                                    cacheStoragePolicy: .notAllowed)
                client?.urlProtocol(self, didLoad: data)
                client?.urlProtocolDidFinishLoading(self)
            } catch {
                client?.urlProtocol(self, didFailWithError: error)
            }
        }
        override func stopLoading() {}
    }

    private func client(
        handler: @escaping @Sendable (URLRequest) throws -> (Data, Int)
    ) -> ClusterClient {
        StubProtocol.handler = handler
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubProtocol.self]
        return ClusterClient(
            profile: ClusterConnectionProfile(
                baseURL: URL(string: "http://127.0.0.1:8080")!),
            session: URLSession(configuration: configuration))
    }

    @Test func aSuccessfulDownloadPublishesTheBytesAndLeavesNoStaging() async throws {
        let fm = FileManager.default
        let out = fm.temporaryDirectory.appending(
            component: "dl-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: out) }
        let payload = Data("evidence-bytes".utf8)
        let subject = client { _ in (payload, 200) }

        let landed = try await subject.downloadArtifact(
            path: "/srv/runs/r1/r1.evidence-bundle.tar.gz", to: out)

        #expect(try Data(contentsOf: landed) == payload)
        let contents = try fm.contentsOfDirectory(atPath: out.path)
        #expect(!contents.contains { $0.hasPrefix(".steerlab-download-staging-") })
        #expect(contents.count == 1)
    }

    /// The stale-tunnel case: nothing listening behind the forward. The verb
    /// must not leave a download directory behind for a later exists-check to
    /// mistake for a completed fetch.
    @Test func anUnreachableEndpointCreatesNoDownloadDirectory() async throws {
        let fm = FileManager.default
        let out = fm.temporaryDirectory.appending(
            component: "dl-\(UUID().uuidString)")
        let subject = client { _ in throw URLError(.cannotConnectToHost) }

        await #expect(throws: (any Error).self) {
            _ = try await subject.downloadArtifact(path: "/srv/a.tar.gz", to: out)
        }
        #expect(!fm.fileExists(atPath: out.path))
    }

    /// A 5xx behind the tunnel is the other stale-forward shape (a proxy
    /// answering for a dead server). Same rule: nothing created.
    @Test func aServerErrorCreatesNoDownloadDirectory() async throws {
        let fm = FileManager.default
        let out = fm.temporaryDirectory.appending(
            component: "dl-\(UUID().uuidString)")
        let subject = client { _ in (Data("bad gateway".utf8), 502) }

        await #expect(throws: (any Error).self) {
            _ = try await subject.downloadArtifact(path: "/srv/a.tar.gz", to: out)
        }
        #expect(!fm.fileExists(atPath: out.path))
    }

    /// An existing download directory (the usual case — `.steerlab/downloads`
    /// is reused) survives a failure; only what this call created is removed.
    @Test func anExistingDownloadDirectoryIsNotRemovedOnFailure() async throws {
        let fm = FileManager.default
        let out = fm.temporaryDirectory.appending(
            component: "dl-\(UUID().uuidString)")
        try fm.createDirectory(at: out, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: out) }
        try Data("keep me".utf8).write(to: out.appending(component: "older.tar.gz"))
        let subject = client { _ in throw URLError(.networkConnectionLost) }

        await #expect(throws: (any Error).self) {
            _ = try await subject.downloadArtifact(path: "/srv/a.tar.gz", to: out)
        }
        #expect(fm.fileExists(atPath: out.appending(component: "older.tar.gz").path))
        let contents = try fm.contentsOfDirectory(atPath: out.path)
        #expect(!contents.contains { $0.hasPrefix(".steerlab-download-staging-") })
    }
}
