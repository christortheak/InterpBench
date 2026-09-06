import Foundation
import Testing

@testable import ExperimentKit

@MainActor struct LocalModelPreparationTests {
    @Test func planUsesTheRequestedCachedRevisionWithoutLoadingOrWriting() throws {
        let root = FileManager.default.temporaryDirectory.appending(component: "model-plan-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let repo = root.appending(path: "hub/models--vendor--model")
        let revision = String(repeating: "a", count: 40)
        let snapshot = repo.appending(path: "snapshots/\(revision)")
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        let missing = try LocalModelPreparation.plan(modelID: "vendor/model", revision: revision, cacheRoot: root)
        #expect(!missing.cacheFileSetPresent)
        // File-presence fixtures only; no real weights or scientific evidence.
        for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"] {
            try Data("{}".utf8).write(to: snapshot.appending(component: name))
        }
        let present = try LocalModelPreparation.plan(modelID: "vendor/model", revision: revision, cacheRoot: root)
        #expect(present.cacheFileSetPresent)
        #expect(present.resolvedRevision == revision)
        #expect(present.memoryFit == "notChecked" && present.credentials == "notChecked")
        #expect(!FileManager.default.fileExists(atPath: repo.appending(component: "refs").path))
        #expect(try !LocalModelPreparation.plan(modelID: "vendor/model", revision: "missing", cacheRoot: root).cacheFileSetPresent)
        for (model, ref) in [("../model", "main"), ("vendor/model", "../outside"), ("vendor/model", "") , ("vendor\tname/model", "main")] {
            #expect(throws: LocalModelPreparationError.self) { try LocalModelPreparation.plan(modelID: model, revision: ref, cacheRoot: root) }
        }
    }

    @Test func cliUsesTheSharedInstallerAndPreservesRequestedRevision() async throws {
        let recorder = Recorder()
        let installer = LocalModelInstaller { model, revision, report in
            await recorder.record(model, revision)
            report(70)
        }
        let invocation = try ExperimentCLIParser.parse(namespace: "model", ["install", "vendor/model", "--revision", "reviewed-tag", "--json"])
        let result = try await LocalModelPreparationCLI.run(invocation, sink: .discarding, installer: installer)
        #expect(result.changed)
        #expect(result.payload["state"] == .string("finished"))
        #expect(await recorder.model == "vendor/model")
        #expect(await recorder.revision == "reviewed-tag")
        #expect(installer.request?.revision == "reviewed-tag")
        #expect(installer.phase == .finished(modelID: "vendor/model"))
    }

    @Test func httpStatusAndCancellationStayBoundToTheObservedInstallation() async throws {
        let installer = LocalModelInstaller { _, _, _ in try await Task.sleep(for: .seconds(60)) }
        func send(_ operation: LocalModelPreparationHTTP.Operation, _ fields: [String: Any]) throws -> StudyAuthoringHTTP.Response {
            LocalModelPreparationHTTP.perform(operation, body: try JSONSerialization.data(withJSONObject: fields), installer: installer)
        }
        #expect(try send(.install, ["modelID": "vendor/model", "target": "remote"]).status == "400 Bad Request")
        #expect(try send(.install, ["modelID": "vendor/model", "revision": "tag"]).status == "202 Accepted")
        let first = try #require(installer.request)
        #expect(try send(.install, ["modelID": "vendor/other"]).status == "409 Conflict")
        #expect(installer.request == first)
        let status = try send(.status, [:])
        #expect(status.status == "200 OK")
        #expect(try send(.cancel, [:]).status == "428 Precondition Required")
        #expect(try send(.cancel, ["requestID": UUID().uuidString]).status == "409 Conflict")
        #expect(installer.isInstalling)
        #expect(try send(.cancel, ["requestID": first.id.uuidString]).status == "200 OK")
        #expect(installer.phase == .cancelled(modelID: "vendor/model"))
        #expect(try send(.install, ["modelID": "vendor/model"]).status == "202 Accepted")
        let second = try #require(installer.request)
        #expect(first.id != second.id)
        #expect(try send(.cancel, ["requestID": first.id.uuidString]).status == "409 Conflict")
        #expect(installer.isInstalling)
        try installer.cancel(requestID: second.id)
        _ = try await installer.completion(requestID: second.id)
    }

    @Test func failedForegroundInstallKeepsTheFailureAndClearingInvalidatesItsRequest() async throws {
        struct Failure: LocalizedError { var errorDescription: String? { "Repository access denied" } }
        let installer = LocalModelInstaller { _, _, _ in throw Failure() }
        let invocation = try ExperimentCLIParser.parse(namespace: "model", ["install", "vendor/model", "--json"])
        do {
            _ = try await LocalModelPreparationCLI.run(invocation, sink: .discarding, installer: installer)
            Issue.record("failed download was reported as installed")
        } catch let stop as ExperimentCLIStop {
            #expect(stop.exitCode == 70 && stop.state == .failed)
            #expect(stop.reason == "Repository access denied")
            #expect(stop.payload["state"] == .string("failed"))
        }
        let old = try #require(installer.request)
        installer.clearStatus()
        #expect(throws: LocalModelPreparationError.self) { try installer.cancel(requestID: old.id) }
    }

    @Test func cancelledTailCannotRestoreClearedStatus() async throws {
        let installer = LocalModelInstaller { _, _, _ in try await Task.sleep(for: .seconds(60)) }
        installer.install("vendor/model")
        let request = try #require(installer.request)
        let completion = Task { try await installer.completion(requestID: request.id) }
        await Task.yield()
        installer.cancel()
        installer.clearStatus()
        _ = try? await completion.value
        #expect(installer.phase == .idle)
        #expect(installer.request == nil)
    }

    private actor Recorder {
        var model: String?
        var revision: String?
        func record(_ model: String, _ revision: String?) { self.model = model; self.revision = revision }
    }
}
