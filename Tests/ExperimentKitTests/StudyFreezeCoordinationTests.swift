import Foundation
import Testing

@testable import ExperimentKit

@MainActor
struct StudyFreezeCoordinationTests {
    private func withWorkspace(_ body: (ExperimentManifest) async throws -> Void) async throws {
        ExperimentRootOverrideLock.acquire()
        let root = FileManager.default.temporaryDirectory.appending(
            component: "freeze-owner-\(UUID())")
        let previous = WorkspaceRoot.programmaticOverride
        WorkspaceRoot.programmaticOverride = root
        defer {
            WorkspaceRoot.programmaticOverride = previous
            try? FileManager.default.removeItem(at: root)
            ExperimentRootOverrideLock.release()
        }
        let manifest = try ExperimentStore.create(
            name: "study", description: "original", modelID: "test/model")
        try await body(manifest)
    }

    private func request(_ manifest: ExperimentManifest, paired: Bool = false) -> StudyFreezeRequest
    {
        .init(
            name: manifest.name, localData: ExperimentStore.manifestData(name: manifest.name),
            localIsDraft: manifest.status == .draft, substrate: "test server",
            workspacePaired: paired)
    }

    private func transport(_ manifest: ExperimentManifest) -> StudyFreezeTransport {
        StudyFreezeTransport(
            status: { _ in "draft" },
            manifestBody: { _ in try JSONEncoder().encode(manifest) },
            freeze: { _ in
                Issue.record("freeze should not be submitted")
                throw CancellationError()
            },
            replace: { _, _ in
                Issue.record("manifest should not be pushed")
                throw CancellationError()
            })
    }

    @Test func mismatchedManifestOffersSyncWithoutSubmittingFreeze() async throws {
        try await withWorkspace { manifest in
            let owner = StudyFreezeController()
            var serverCopy = manifest
            serverCopy.maxTokens += 1
            await owner.freezeOnServer(request: request(manifest), transport: transport(serverCopy))
            #expect(owner.remoteFreezeIdentityWarning != nil)
            #expect(owner.remoteFreezeCanSyncDraft)
            #expect(!owner.isFreezingOnServer)
        }
    }

    @Test func serverResidencyAndLifecycleStopBeforeIdentityFetch() async throws {
        try await withWorkspace { manifest in
            let owner = StudyFreezeController()
            var resident: Bool?
            owner.presentation.residency = { _, value in resident = value }
            var io = transport(manifest)
            io.manifestBody = { _ in
                Issue.record("identity fetch must not run after residency/lifecycle refusal")
                throw CancellationError()
            }
            io.status = { _ in throw ClusterClient.ClientError.badResponse(404, "missing") }
            await owner.freezeOnServer(request: request(manifest), transport: io)
            #expect(resident == false)
            io.status = { _ in "frozen" }
            await owner.freezeOnServer(request: request(manifest), transport: io)
            #expect(resident == true)
            #expect(!owner.isFreezingOnServer)
        }
    }

    @Test func verifiedFreezeRetainsServerAdvisoriesAndRefreshes() async throws {
        try await withWorkspace { manifest in
            let owner = StudyFreezeController()
            var refreshes = 0
            var names: [String] = []
            owner.presentation.refresh = { refreshes += 1 }
            var io = transport(manifest)
            io.freeze = { name in
                names.append(name)
                var frozen = manifest
                frozen.status = .frozen
                frozen.freezeHash = "server-hash"
                return RemoteFreezeResult(manifest: frozen, advisories: ["server advisory"])
            }
            await owner.freezeOnServer(request: request(manifest), transport: io)
            #expect(names == [manifest.name])
            #expect(refreshes == 1)
            #expect(owner.remoteFreezeAdvisories == ["server advisory"])
            #expect(!owner.isFreezingOnServer)
            let persisted = try ExperimentStore.load(name: manifest.name)
            #expect(persisted.status == .draft)
        }
    }

    @Test func serverGateRefusalRemainsVerbatim() async throws {
        try await withWorkspace { manifest in
            let owner = StudyFreezeController()
            var io = transport(manifest)
            io.freeze = { _ in
                throw ClusterClient.ClientError.badResponse(400, "missing validation pin")
            }
            await owner.freezeOnServer(request: request(manifest), transport: io)
            #expect(owner.remoteFreezeGateFailure == "missing validation pin")
            #expect(!owner.isFreezingOnServer)
        }
    }

    @Test func syncAdoptsPreservedRevisionAndRechecksWithoutAutoFreezing() async throws {
        try await withWorkspace { manifest in
            let owner = StudyFreezeController()
            var io = transport(manifest)
            var pushed: Data?
            io.replace = { name, bytes in
                pushed = bytes
                return .init(
                    name: name, status: "draft",
                    preserved: .init(
                        modelRevision: "server-revision", conditions: nil, capabilityBattery: nil))
            }
            var bodyFetches = 0
            io.manifestBody = { name in
                bodyFetches += 1
                var serverCopy = manifest
                serverCopy.modelRevision = "server-revision"
                return try JSONEncoder().encode(serverCopy)
            }
            let original = request(manifest)
            await owner.pushManifest(request: original, transport: io)
            #expect(pushed == original.localData)
            let persisted = try ExperimentStore.load(name: manifest.name)
            #expect(persisted.modelRevision == "server-revision")
            #expect(bodyFetches == 1)
            #expect(owner.remoteFreezeIdentityWarning == nil)
            #expect(owner.remoteFreezeIdentityNote != nil)
            #expect(!owner.isSyncingServerDraft)
        }
    }

    @Test func supersededPreflightCannotSubmitOrReplaceNewerState() async throws {
        try await withWorkspace { manifest in
            let owner = StudyFreezeController()
            let ready = AsyncStream<Void>.makeStream()
            let release = AsyncStream<Void>.makeStream()
            var delayed = transport(manifest)
            delayed.status = { _ in
                ready.continuation.yield(())
                for await _ in release.stream { break }
                return "draft"
            }
            let oldRequest = request(manifest)
            let old = Task { await owner.freezeOnServer(request: oldRequest, transport: delayed) }
            for await _ in ready.stream { break }
            #expect(owner.isFreezingOnServer)
            owner.resetSelection()
            var different = manifest
            different.maxTokens += 1
            await owner.freezeOnServer(request: request(manifest), transport: transport(different))
            let warning = owner.remoteFreezeIdentityWarning
            release.continuation.yield(())
            await old.value
            ready.continuation.finish()
            release.continuation.finish()
            #expect(warning != nil)
            #expect(owner.remoteFreezeIdentityWarning == warning)
            #expect(owner.remoteFreezeCanSyncDraft)
            #expect(!owner.isFreezingOnServer)
        }
    }

    @Test func contextSwitchAndLocalEditsPreventSubmissionAfterIdentityFetch() async throws {
        try await withWorkspace { manifest in
            let owner = StudyFreezeController()
            var current = true
            var io = transport(manifest)
            io.manifestBody = { _ in
                current = false
                return try JSONEncoder().encode(manifest)
            }
            await owner.freezeOnServer(
                request: request(manifest), transport: io, isCurrent: { current })
            #expect(!owner.isFreezingOnServer)
            current = true
            io.manifestBody = { _ in
                var changed = manifest
                changed.maxTokens += 1
                try ExperimentStore.save(changed)
                return try JSONEncoder().encode(manifest)
            }
            await owner.freezeOnServer(
                request: request(manifest), transport: io, isCurrent: { current })
            #expect(owner.remoteFreezeIdentityWarning?.contains("changed during") == true)
        }
    }

    @Test func staleSubmittedResponseCannotRefreshOrOverwriteNewSelection() async throws {
        try await withWorkspace { manifest in
            let owner = StudyFreezeController()
            let ready = AsyncStream<Void>.makeStream()
            let release = AsyncStream<Void>.makeStream()
            var refreshes = 0
            owner.presentation.refresh = { refreshes += 1 }
            var io = transport(manifest)
            io.freeze = { _ in
                ready.continuation.yield(())
                for await _ in release.stream { break }
                return RemoteFreezeResult(manifest: manifest, advisories: ["stale advisory"])
            }
            let captured = request(manifest)
            let old = Task { await owner.freezeOnServer(request: captured, transport: io) }
            for await _ in ready.stream { break }
            owner.resetSelection()
            release.continuation.yield(())
            await old.value
            ready.continuation.finish()
            release.continuation.finish()
            #expect(owner.remoteFreezeAdvisories.isEmpty)
            #expect(refreshes == 0)
            #expect(!owner.isFreezingOnServer)
        }
    }

    @Test func staleSyncResponseCannotAdoptPinsIntoTheCurrentWorkspace() async throws {
        try await withWorkspace { manifest in
            let owner = StudyFreezeController()
            var current = true
            var io = transport(manifest)
            io.replace = { name, _ in
                current = false
                return .init(
                    name: name, status: "draft",
                    preserved: .init(
                        modelRevision: "wrong-workspace-revision", conditions: nil,
                        capabilityBattery: nil))
            }
            await owner.pushManifest(
                request: request(manifest), transport: io, isCurrent: { current })
            let persisted = try ExperimentStore.load(name: manifest.name)
            #expect(persisted.modelRevision == nil)
            #expect(owner.remoteFreezeIdentityNote == nil)
            #expect(!owner.isSyncingServerDraft)
        }
    }

    @Test func localFreezeRefusalKeepsTheDraftAndReportsItsGate() async throws {
        try await withWorkspace { manifest in
            let owner = StudyFreezeController()
            var message: String?
            owner.presentation.note = { text, _ in message = text }
            owner.refreshReadiness(
                manifest: manifest, violations: ExperimentStore.verify(manifest),
                runSubstrate: ExperimentStore.evidenceSubstrate, serverPaired: false)
            #expect(owner.freezeReadiness?.ready == false)
            owner.freeze(name: manifest.name, runSubstrate: ExperimentStore.evidenceSubstrate)
            #expect(message?.contains("Freeze did not complete") == true)
            let persisted = try ExperimentStore.load(name: manifest.name)
            #expect(persisted.status == .draft)
            owner.refreshReadiness(
                manifest: nil, violations: [],
                runSubstrate: ExperimentStore.evidenceSubstrate, serverPaired: false)
            #expect(owner.freezeReadiness == nil)
        }
    }
}
