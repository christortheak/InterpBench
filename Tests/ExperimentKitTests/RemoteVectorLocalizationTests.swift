import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit

/// Localizing a server-catalog vector into the local workspace: the saved
/// reference must be workspace-relative and backed by local bytes the
/// workspace can VOUCH for (the workspace is the source of truth; the server
/// only caches).
struct RemoteVectorLocalizationTests {

    private func temporaryWorkspace() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "rvl-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        return root
    }

    private func sha256(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8))
            .map { String(format: "%02x", $0) }.joined()
    }

    /// Records requested server paths across the @Sendable download closure.
    private final class PathRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private var paths: [String] = []
        func record(_ path: String) {
            lock.lock()
            paths.append(path)
            lock.unlock()
        }
        var recorded: [String] {
            lock.lock()
            defer { lock.unlock() }
            return paths
        }
    }

    /// A download that writes `body(serverPath)` under the server path's own
    /// basename, recording what was asked for.
    private func downloader(
        _ recorder: PathRecorder,
        body: @escaping @Sendable (String) -> String = { _ in "x" },
        name: (@Sendable (String) -> String)? = nil
    ) -> @Sendable (String, URL) async throws -> URL {
        return { path, directory in
            recorder.record(path)
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            let component = name?(path) ?? URL(filePath: path).lastPathComponent
            let destination = directory.appending(component: component)
            try Data(body(path).utf8).write(to: destination)
            return destination
        }
    }

    @Test func downloadsThePairAndReturnsTheRelativeReference() async throws {
        let root = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let fetched = PathRecorder()
        let reference = try await RemoteVectorLocalization.localize(
            serverID: "/scratch/user/ws/runs/2026-run/fear",
            workspaceRelativeID: "runs/2026-run/fear",
            workspaceRoot: root,
            download: downloader(fetched))
        #expect(reference.relativeID == "runs/2026-run/fear")
        // Tensor first, sidecar last — the sidecar is the discovery key on
        // both engines, so it must never appear without its tensor.
        #expect(fetched.recorded == [
            "/scratch/user/ws/runs/2026-run/fear.safetensors",
            "/scratch/user/ws/runs/2026-run/fear.json",
        ])
        #expect(RemoteVectorLocalization.isLocal(
            relativeID: "runs/2026-run/fear", workspaceRoot: root))
    }

    @Test func skipsTheDownloadWhenThePairIsAlreadyLocal() async throws {
        let root = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let runDir = root.appending(components: "runs", "2026-run")
        try FileManager.default.createDirectory(
            at: runDir, withIntermediateDirectories: true)
        try Data("s".utf8).write(to: runDir.appending(component: "fear.safetensors"))
        try Data("j".utf8).write(to: runDir.appending(component: "fear.json"))
        // The closure throws, so a successful return PROVES no download ran.
        let reference = try await RemoteVectorLocalization.localize(
            serverID: "/scratch/user/ws/runs/2026-run/fear",
            workspaceRelativeID: "runs/2026-run/fear",
            workspaceRoot: root
        ) { _, _ in throw CocoaError(.fileNoSuchFile) }
        #expect(reference.relativeID == "runs/2026-run/fear")
    }

    @Test func refusesUnsafeOrMissingRelativeIDs() async throws {
        let root = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        // Server responses are data: the relative id names where WE write.
        for bad in [
            "/abs/runs/x/fear", "runs/../secrets/fear", "prompts/x/fear", "runs",
        ] {
            await #expect(throws: RemoteVectorLocalization.LocalizationError.self) {
                _ = try await RemoteVectorLocalization.localize(
                    serverID: "/srv/runs/x/fear",
                    workspaceRelativeID: bad,
                    workspaceRoot: root
                ) { _, _ in throw CocoaError(.fileNoSuchFile) }
            }
        }
        await #expect(throws: RemoteVectorLocalization.LocalizationError.self) {
            _ = try await RemoteVectorLocalization.localize(
                serverID: "/srv/runs/x/fear",
                workspaceRelativeID: nil,
                workspaceRoot: root
            ) { _, _ in throw CocoaError(.fileNoSuchFile) }
        }
    }

    @Test func surfacesAPairThatLandedUnderTheWrongName() async throws {
        // Nothing advertised to verify the CONTENT by, so the only evidence
        // the right artifact arrived is the name the transfer chose: a
        // mismatch fails the save rather than dangling at packaging.
        let root = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let fetched = PathRecorder()
        await #expect(throws: RemoteVectorLocalization.LocalizationError.self) {
            _ = try await RemoteVectorLocalization.localize(
                serverID: "/srv/runs/2026-run/fear",
                workspaceRelativeID: "runs/2026-run/fear",
                workspaceRoot: root,
                download: downloader(fetched, name: { _ in "renamed-by-proxy" }))
        }
        #expect(!RemoteVectorLocalization.isLocal(
            relativeID: "runs/2026-run/fear", workspaceRoot: root))
    }

    // MARK: - Hash binding (2026-08-06 review, P2)

    @Test func verifiesDownloadedBytesAgainstTheAdvertisedHashes() async throws {
        let root = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let fetched = PathRecorder()
        let reference = try await RemoteVectorLocalization.localize(
            serverID: "/srv/runs/2026-run/fear",
            workspaceRelativeID: "runs/2026-run/fear",
            sidecarSha256: sha256("sidecar-bytes"),
            tensorSha256: sha256("tensor-bytes"),
            workspaceRoot: root,
            download: downloader(fetched, body: { path in
                path.hasSuffix(".json") ? "sidecar-bytes" : "tensor-bytes"
            }))
        #expect(reference.relativeID == "runs/2026-run/fear")
        let stem = root.appending(components: "runs", "2026-run", "fear")
        #expect(try String(contentsOf: URL(filePath: stem.path + ".json"),
                           encoding: .utf8) == "sidecar-bytes")
    }

    @Test func refusesAndPublishesNothingWhenADownloadHashesWrong() async throws {
        // Transactional: the tensor verifies, the sidecar does not — and the
        // workspace is left exactly as it was, not holding a stray half-pair.
        let root = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let fetched = PathRecorder()
        await #expect(throws: RemoteVectorLocalization.LocalizationError
            .downloadedBytesDiffer(
                path: "fear.json",
                advertised: sha256("what-the-catalog-says"),
                downloaded: sha256("truncated"))) {
            _ = try await RemoteVectorLocalization.localize(
                serverID: "/srv/runs/2026-run/fear",
                workspaceRelativeID: "runs/2026-run/fear",
                sidecarSha256: sha256("what-the-catalog-says"),
                tensorSha256: sha256("tensor-bytes"),
                workspaceRoot: root,
                download: downloader(fetched, body: { path in
                    path.hasSuffix(".json") ? "truncated" : "tensor-bytes"
                }))
        }
        #expect(!RemoteVectorLocalization.isLocal(
            relativeID: "runs/2026-run/fear", workspaceRoot: root))
        // Not even the verified member was published, and no staging debris
        // survives the failure.
        let runDir = root.appending(components: "runs", "2026-run")
        let left = (try? FileManager.default.contentsOfDirectory(
            atPath: runDir.path)) ?? []
        #expect(left.isEmpty)
    }

    @Test func refusesToOverwriteLocalBytesThatDisagree() async throws {
        // The workspace holds bytes the server does not. That is a CONFLICT
        // only the researcher can settle — never a stale cache to refresh.
        let root = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let runDir = root.appending(components: "runs", "2026-run")
        try FileManager.default.createDirectory(
            at: runDir, withIntermediateDirectories: true)
        try Data("mine".utf8).write(
            to: runDir.appending(component: "fear.safetensors"))
        try Data("mine".utf8).write(
            to: runDir.appending(component: "fear.json"))

        var thrown: RemoteVectorLocalization.LocalizationError?
        do {
            _ = try await RemoteVectorLocalization.localize(
                serverID: "/srv/runs/2026-run/fear",
                workspaceRelativeID: "runs/2026-run/fear",
                sidecarSha256: sha256("theirs"),
                tensorSha256: sha256("theirs"),
                workspaceRoot: root
            ) { _, _ in throw CocoaError(.fileNoSuchFile) }
        } catch let error as RemoteVectorLocalization.LocalizationError {
            thrown = error
        }
        #expect(thrown == .localBytesDiffer(
            path: runDir.appending(component: "fear.safetensors")
                .resolvingSymlinksInPath().path,
            advertised: sha256("theirs"), local: sha256("mine")))
        // Actionable: the message names BOTH hashes and the file.
        let message = thrown?.errorDescription ?? ""
        #expect(message.contains(sha256("theirs")))
        #expect(message.contains(sha256("mine")))
        // Local truth survives untouched.
        #expect(try String(contentsOf: runDir.appending(component: "fear.json"),
                           encoding: .utf8) == "mine")
    }

    @Test func refusesWhenOnlyHalfThePairIsPresentAndDisagrees() async throws {
        // The half-present case must verify BEFORE the fetch, or the download
        // would quietly replace the member that is already there.
        let root = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let runDir = root.appending(components: "runs", "2026-run")
        try FileManager.default.createDirectory(
            at: runDir, withIntermediateDirectories: true)
        try Data("mine".utf8).write(
            to: runDir.appending(component: "fear.json"))
        await #expect(throws: RemoteVectorLocalization.LocalizationError.self) {
            _ = try await RemoteVectorLocalization.localize(
                serverID: "/srv/runs/2026-run/fear",
                workspaceRelativeID: "runs/2026-run/fear",
                sidecarSha256: sha256("theirs"),
                tensorSha256: sha256("tensor-bytes"),
                workspaceRoot: root
            ) { _, _ in throw CocoaError(.fileNoSuchFile) }
        }
        #expect(try String(contentsOf: runDir.appending(component: "fear.json"),
                           encoding: .utf8) == "mine")
    }

    @Test func completesAHalfPresentPairThatVerifies() async throws {
        let root = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let runDir = root.appending(components: "runs", "2026-run")
        try FileManager.default.createDirectory(
            at: runDir, withIntermediateDirectories: true)
        try Data("sidecar-bytes".utf8).write(
            to: runDir.appending(component: "fear.json"))
        let fetched = PathRecorder()
        let reference = try await RemoteVectorLocalization.localize(
            serverID: "/srv/runs/2026-run/fear",
            workspaceRelativeID: "runs/2026-run/fear",
            sidecarSha256: sha256("sidecar-bytes"),
            tensorSha256: sha256("tensor-bytes"),
            workspaceRoot: root,
            download: downloader(fetched, body: { _ in "tensor-bytes" }))
        #expect(reference.relativeID == "runs/2026-run/fear")
        // Only the missing member was fetched.
        #expect(fetched.recorded == ["/srv/runs/2026-run/fear.safetensors"])
    }

    @Test func hashedDownloadsPublishUnderTheAdvertisedStem() async throws {
        // With the content verified, the name the transfer chose carries no
        // information — the workspace names the file, as it should.
        let root = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let fetched = PathRecorder()
        let reference = try await RemoteVectorLocalization.localize(
            serverID: "/srv/runs/2026-run/fear",
            workspaceRelativeID: "runs/2026-run/fear",
            sidecarSha256: sha256("sidecar-bytes"),
            tensorSha256: sha256("tensor-bytes"),
            workspaceRoot: root,
            download: downloader(
                fetched,
                body: { path in
                    path.hasSuffix(".json") ? "sidecar-bytes" : "tensor-bytes"
                },
                name: { _ in "content-disposition-nonsense" }))
        #expect(reference.relativeID == "runs/2026-run/fear")
        #expect(RemoteVectorLocalization.isLocal(
            relativeID: "runs/2026-run/fear", workspaceRoot: root))
        // …and no staging directory or stray name is left behind.
        let left = try FileManager.default.contentsOfDirectory(
            atPath: root.appending(components: "runs", "2026-run").path)
        #expect(Set(left) == ["fear.json", "fear.safetensors"])
    }

    @Test func announcesAnUnverifiedLocalizationOnPreFieldServers() async throws {
        let root = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let warnings = PathRecorder()
        let fetched = PathRecorder()
        let first = try await RemoteVectorLocalization.localize(
            serverID: "/srv/runs/2026-run/fear",
            workspaceRelativeID: "runs/2026-run/fear",
            workspaceRoot: root,
            warn: { warnings.record($0) },
            download: downloader(fetched))
        #expect(warnings.recorded.count == 1)
        #expect(warnings.recorded[0].contains("UNVERIFIED"))
        #expect(first.verification == .unverified)
        #expect(first.notice == warnings.recorded[0])
        // Re-localizing an existing pair is unverified too, and says so.
        let again = PathRecorder()
        let second = try await RemoteVectorLocalization.localize(
            serverID: "/srv/runs/2026-run/fear",
            workspaceRelativeID: "runs/2026-run/fear",
            workspaceRoot: root,
            warn: { again.record($0) }
        ) { _, _ in throw CocoaError(.fileNoSuchFile) }
        #expect(again.recorded.count == 1)
        #expect(again.recorded[0].contains("unverified"))
        #expect(second.verification == .unverified)
        #expect(second.notice == again.recorded[0])
    }

    @Test func verifiedLocalizationIsSilent() async throws {
        let root = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let warnings = PathRecorder()
        let fetched = PathRecorder()
        let outcome = try await RemoteVectorLocalization.localize(
            serverID: "/srv/runs/2026-run/fear",
            workspaceRelativeID: "runs/2026-run/fear",
            sidecarSha256: sha256("x"), tensorSha256: sha256("x"),
            workspaceRoot: root,
            warn: { warnings.record($0) },
            download: downloader(fetched))
        #expect(warnings.recorded.isEmpty)
        #expect(outcome.verification == .fullyVerified)
        #expect(outcome.verification.isFullyVerified)
        #expect(outcome.notice == nil)
    }

    // MARK: - Verification disclosure (2026-08-06 review round 2, P2)

    @Test func aSingleDigestRowLocalizesAndDisclosesTheOtherHalf() async throws {
        // The reviewer's case, resolved by DISCLOSURE rather than refusal
        // (Christian's call: requiring both digests would turn a pre-field
        // server into an unexplainable save failure). The advertised half is
        // still verified; the unadvertised half is named.
        let root = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let warnings = PathRecorder()
        let fetched = PathRecorder()
        let outcome = try await RemoteVectorLocalization.localize(
            serverID: "/srv/runs/2026-run/fear",
            workspaceRelativeID: "runs/2026-run/fear",
            tensorSha256: sha256("tensor-bytes"),
            workspaceRoot: root,
            warn: { warnings.record($0) },
            download: downloader(fetched, body: { _ in "tensor-bytes" }))
        // The pair landed — a partial digest never blocks.
        #expect(outcome.relativeID == "runs/2026-run/fear")
        #expect(RemoteVectorLocalization.isLocal(
            relativeID: "runs/2026-run/fear", workspaceRoot: root))
        #expect(outcome.verification == .partiallyVerified(missing: [".json"]))
        #expect(outcome.notice?.contains("PARTIALLY verified") == true)
        #expect(outcome.notice?.contains(".json") == true)
        // …and it reaches the caller's channel, not just the return value.
        #expect(warnings.recorded.count == 1)
        #expect(warnings.recorded.first == outcome.notice)
    }

    @Test func theAdvertisedHalfIsStillEnforcedOnASingleDigestRow() async throws {
        // Disclosure is not indulgence: what the catalog DOES advertise is
        // checked exactly as before, and a mismatch still publishes nothing.
        let root = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let fetched = PathRecorder()
        await #expect(throws: RemoteVectorLocalization.LocalizationError
            .downloadedBytesDiffer(
                path: "fear.safetensors",
                advertised: sha256("what-the-catalog-says"),
                downloaded: sha256("truncated"))) {
            _ = try await RemoteVectorLocalization.localize(
                serverID: "/srv/runs/2026-run/fear",
                workspaceRelativeID: "runs/2026-run/fear",
                tensorSha256: sha256("what-the-catalog-says"),
                workspaceRoot: root,
                download: downloader(fetched, body: { _ in "truncated" }))
        }
        #expect(!RemoteVectorLocalization.isLocal(
            relativeID: "runs/2026-run/fear", workspaceRoot: root))
    }

    @Test func aPartiallyVerifiedExistingPairDisclosesToo() async throws {
        // The kept-pair path discloses on the same rule as the fetch path —
        // the earlier code warned only when BOTH digests were absent, so a
        // single-digest row went silently unverified on half its bytes.
        let root = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let runDir = root.appending(components: "runs", "2026-run")
        try FileManager.default.createDirectory(
            at: runDir, withIntermediateDirectories: true)
        try Data("tensor-bytes".utf8).write(
            to: runDir.appending(component: "fear.safetensors"))
        try Data("whatever".utf8).write(
            to: runDir.appending(component: "fear.json"))
        let warnings = PathRecorder()
        let outcome = try await RemoteVectorLocalization.localize(
            serverID: "/srv/runs/2026-run/fear",
            workspaceRelativeID: "runs/2026-run/fear",
            tensorSha256: sha256("tensor-bytes"),
            workspaceRoot: root,
            warn: { warnings.record($0) }
        ) { _, _ in throw CocoaError(.fileNoSuchFile) }
        #expect(outcome.verification == .partiallyVerified(missing: [".json"]))
        #expect(warnings.recorded.count == 1)
        #expect(warnings.recorded[0].contains("PARTIALLY verified"))
        #expect(outcome.notice == warnings.recorded[0])
    }

    @Test func theSaveDisclosureReportsACompletedSaveWithACaveat() {
        // Never a failure message: the save happened, and refusing was the
        // option Christian rejected — so the line must not read as a stop.
        let one = RemoteVectorLocalization.saveDisclosure(concepts: ["fear"])
        #expect(one.hasPrefix("saved — but the vector fear"))
        #expect(one.contains("WITHOUT full hash verification"))
        #expect(one.contains("Update the server"))
        let many = RemoteVectorLocalization.saveDisclosure(
            concepts: ["fear", "sympathy"])
        #expect(many.contains("the vectors fear, sympathy"))
    }

    @Test func verificationStateIsDerivedFromWhatIsAdvertised() {
        // The pure rule, so the three states cannot drift apart in the two
        // notice paths that switch over them.
        #expect(RemoteVectorLocalization.verification(
            for: [".safetensors": "a", ".json": "b"]) == .fullyVerified)
        #expect(RemoteVectorLocalization.verification(
            for: [".safetensors": nil, ".json": nil]) == .unverified)
        #expect(RemoteVectorLocalization.verification(
            for: [".safetensors": "a", ".json": nil])
            == .partiallyVerified(missing: [".json"]))
        #expect(RemoteVectorLocalization.verification(
            for: [".safetensors": nil, ".json": "b"])
            == .partiallyVerified(missing: [".safetensors"]))
    }

    // MARK: - Publish serialization and rollback (round 2, P2)

    @Test func rollsBackTheFirstMoveWhenTheSecondFails() async throws {
        // Publication is two moves. The second one failing — or a concurrent
        // localization claiming the name between our check and our publish —
        // used to leave a TENSOR-ONLY workspace, contradicting the
        // "workspace exactly as it was" promise the staging exists for.
        //
        // The failure is injected the way the race would produce it: the
        // sidecar's destination is occupied (by a directory) after the
        // presence check has already run, so the second move cannot land.
        let root = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let runDir = root.appending(components: "runs", "2026-run")
        let recorder = PathRecorder()
        let claimed = runDir.appending(component: "fear.json")
        let download: @Sendable (String, URL) async throws -> URL = {
            path, directory in
            recorder.record(path)
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            if path.hasSuffix(".json") {
                // The concurrent writer, arriving after our presence check.
                try FileManager.default.createDirectory(
                    at: claimed, withIntermediateDirectories: true)
            }
            let destination = directory.appending(
                component: URL(filePath: path).lastPathComponent)
            try Data("bytes".utf8).write(to: destination)
            return destination
        }
        await #expect(throws: RemoteVectorLocalization.LocalizationError.self) {
            _ = try await RemoteVectorLocalization.localize(
                serverID: "/srv/runs/2026-run/fear",
                workspaceRelativeID: "runs/2026-run/fear",
                sidecarSha256: sha256("bytes"), tensorSha256: sha256("bytes"),
                workspaceRoot: root,
                download: download)
        }
        // The tensor this call published was taken back out: what remains is
        // exactly what the other writer put there, and no staging debris.
        let left = Set(try FileManager.default.contentsOfDirectory(
            atPath: runDir.path))
        #expect(left == ["fear.json"])
        #expect(!FileManager.default.fileExists(
            atPath: runDir.appending(component: "fear.safetensors").path))
    }

    @Test func thePublishFailureNamesTheRollback() {
        let error = RemoteVectorLocalization.LocalizationError.publishFailed(
            member: "fear.json", reason: "File exists")
        let message = error.errorDescription ?? ""
        #expect(message.contains("fear.json"))
        #expect(message.contains("rolled back"))
        #expect(message.contains("unchanged"))
    }

    @Test func concurrentLocalizationsOfOneStemDoNotInterleave() async throws {
        // Two saves picking the same server vector at once. Serialized per
        // stem, the second finds a complete, verified pair and fetches
        // nothing; unserialized, both would stage and their two-move
        // publishes could interleave into a half-pair.
        let root = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let fetched = PathRecorder()
        let slow: @Sendable (String, URL) async throws -> URL = {
            path, directory in
            fetched.record(path)
            try await Task.sleep(for: .milliseconds(20))
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            let destination = directory.appending(
                component: URL(filePath: path).lastPathComponent)
            try Data("bytes".utf8).write(to: destination)
            return destination
        }
        async let first = RemoteVectorLocalization.localize(
            serverID: "/srv/runs/2026-run/fear",
            workspaceRelativeID: "runs/2026-run/fear",
            sidecarSha256: sha256("bytes"), tensorSha256: sha256("bytes"),
            workspaceRoot: root, download: slow)
        async let second = RemoteVectorLocalization.localize(
            serverID: "/srv/runs/2026-run/fear",
            workspaceRelativeID: "runs/2026-run/fear",
            sidecarSha256: sha256("bytes"), tensorSha256: sha256("bytes"),
            workspaceRoot: root, download: slow)
        let outcomes = try await [first, second]
        #expect(outcomes.allSatisfy { $0.relativeID == "runs/2026-run/fear" })
        // Exactly one fetch of each member: the loser of the lock saw the
        // completed pair rather than re-downloading over it.
        #expect(fetched.recorded.count == 2)
        let left = Set(try FileManager.default.contentsOfDirectory(
            atPath: root.appending(components: "runs", "2026-run").path))
        #expect(left == ["fear.json", "fear.safetensors"])
    }

    // MARK: - Containment on the real path

    @Test func refusesARunDirectorySymlinkedOutOfTheWorkspace() async throws {
        // Lexically perfect, materially an escape: `runs/<run>` is a symlink
        // to somewhere else entirely, so the "localized" bytes would land
        // right back on the substrate we are localizing away FROM.
        let root = try temporaryWorkspace()
        let elsewhere = try temporaryWorkspace()
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: elsewhere)
        }
        let runs = root.appending(component: "runs")
        try FileManager.default.createDirectory(
            at: runs, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            at: runs.appending(component: "2026-run"), withDestinationURL: elsewhere)

        await #expect(throws: RemoteVectorLocalization.LocalizationError.self) {
            _ = try await RemoteVectorLocalization.localize(
                serverID: "/srv/runs/2026-run/fear",
                workspaceRelativeID: "runs/2026-run/fear",
                sidecarSha256: sha256("x"), tensorSha256: sha256("x"),
                workspaceRoot: root
            ) { _, _ in throw CocoaError(.fileNoSuchFile) }
        }
        let landed = (try? FileManager.default.contentsOfDirectory(
            atPath: elsewhere.path)) ?? []
        #expect(landed.isEmpty)
    }

    @Test func acceptsASymlinkedWorkspaceRoot() throws {
        // The workspace root itself is routinely a symlink (/tmp on macOS):
        // resolving both sides is what keeps that from reading as an escape.
        let root = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let stem = try RemoteVectorLocalization.resolvedStem(
            relativeID: "runs/2026-run/fear", workspaceRoot: root)
        #expect(stem.lastPathComponent == "fear")
        #expect(stem.path.hasPrefix(
            root.resolvingSymlinksInPath().appending(component: "runs").path))
    }

    @Test func refusesASiblingDirectoryThatMerelyStartsWithRuns() throws {
        // Component-wise containment, not string prefix: a `runs-scratch`
        // sibling must not pass a `…/runs` prefix test.
        let root = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let runs = root.appending(component: "runs")
        try FileManager.default.createDirectory(
            at: runs, withIntermediateDirectories: true)
        let sneaky = root.appending(component: "runs-scratch")
        try FileManager.default.createDirectory(
            at: sneaky, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            at: runs.appending(component: "2026-run"), withDestinationURL: sneaky)
        #expect(throws: RemoteVectorLocalization.LocalizationError.self) {
            _ = try RemoteVectorLocalization.resolvedStem(
                relativeID: "runs/2026-run/fear", workspaceRoot: root)
        }
    }

    // MARK: - Catalog row

    @Test func recordAnswersToBothReferenceForms() {
        var record = RemoteVectorRecord(
            id: "/srv/ws/runs/2026-run/fear",
            runDirectory: "/srv/ws/runs/2026-run",
            name: "fear", concept: "fear", modelID: "org/m",
            revision: nil, layerCount: 4, hiddenSize: 8,
            method: nil, reading: nil, residualNormSource: nil,
            hasResidualNorms: nil)
        #expect(record.canonicalStoredID == "/srv/ws/runs/2026-run/fear")
        record.workspaceRelativeID = "runs/2026-run/fear"
        #expect(record.canonicalStoredID == "runs/2026-run/fear")
        #expect(record.matches(reference: "runs/2026-run/fear"))
        #expect(record.matches(reference: "/srv/ws/runs/2026-run/fear"))
        #expect(!record.matches(reference: "runs/other/fear"))
        // Pre-field servers advertise no hashes; nil is the honest decode.
        #expect(record.sidecarSha256 == nil)
        #expect(record.tensorSha256 == nil)
    }

    @Test func recordDecodesTheAdvertisedArtifactHashes() throws {
        // The wire keys are a pinned cross-engine contract with
        // `catalog.VectorArtifact` — decode a catalog row verbatim.
        let json = """
            {"id": "/srv/ws/runs/2026-run/fear",
             "runDirectory": "/srv/ws/runs/2026-run", "name": "fear",
             "concept": "fear", "modelID": "org/m", "layerCount": 4,
             "hiddenSize": 8, "hasResidualNorms": false,
             "workspaceRelativeID": "runs/2026-run/fear",
             "sidecarSha256": "\(sha256("sidecar"))",
             "tensorSha256": "\(sha256("tensor"))"}
            """
        let record = try JSONDecoder().decode(
            RemoteVectorRecord.self, from: Data(json.utf8))
        #expect(record.sidecarSha256 == sha256("sidecar"))
        #expect(record.tensorSha256 == sha256("tensor"))
    }
}
