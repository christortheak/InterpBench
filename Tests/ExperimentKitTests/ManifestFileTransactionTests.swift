import Foundation
import Testing
@testable import ExperimentKit

struct ManifestFileTransactionTests {
    @Test func externalPreconditionLeavesFrozenBytesAndIdentityUntouched() throws {
        let root = FileManager.default.temporaryDirectory.appending(component: "file-precondition-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let url = root.appending(component: "experiment.json")
        let data = Data(#"{ "name":"example", "status":"frozen", "freezeHash":"existing" }"#.utf8)
        try data.write(to: url)
        let snapshot = try ManifestFileTransaction.snapshot(at: url)
        try ManifestFileTransaction.withLock(manifestURL: url, workspaceRoot: root) {
            try ManifestFileTransaction.withLock(manifestURL: url, workspaceRoot: root) {
                try ManifestFileTransaction.requireCurrent(.sha256(snapshot.sha256), at: url)
            }
        }
        #expect(try Data(contentsOf: url) == data)
        #expect(throws: ExperimentError.self) {
            try ManifestFileTransaction.withLock(manifestURL: url, workspaceRoot: root) {
                try ManifestFileTransaction.requireCurrent(.absent, at: url)
            }
        }
    }

    @Test func aPriorSnapshotCannotPublishOverAnInterveningEdit() throws {
        let root = FileManager.default.temporaryDirectory.appending(component: "stale-file-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let url = root.appending(component: "experiment.json")
        try Data("first".utf8).write(to: url)
        let old = try ManifestFileTransaction.snapshot(at: url)
        try Data("second".utf8).write(to: url)
        do {
            try ManifestFileTransaction.withLock(manifestURL: url, workspaceRoot: root) {
                try ManifestFileTransaction.requireCurrent(.sha256(old.sha256), at: url)
                try Data("stale replacement".utf8).write(to: url)
            }
            Issue.record("expected stale-write refusal")
        } catch let error as ExperimentError {
            #expect(error.lifecycleRefusal?.gate == .staleManifest)
        }
        #expect(try String(contentsOf: url, encoding: .utf8) == "second")
    }
    @Test func authoringSnapshotsRefuseLostUpdatesWithoutAddingManifestFields() throws {
        let root = FileManager.default.temporaryDirectory.appending(component: "draft-cas-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let manifest = ExperimentManifest(name: "study", description: "original", modelID: "test/model")
        try ExperimentStore.save(manifest, allowCreate: true, workspaceRoot: root, expectedFile: .absent)
        let first = try DraftAuthoringSnapshot(workspaceRoot: root, name: manifest.name)
        let second = try DraftAuthoringSnapshot(workspaceRoot: root, name: manifest.name)
        var edited = first.manifest
        edited.experimentDescription = "first edit"
        let saved = try DraftAuthoringTransaction.replace(edited, reviewed: first)
        var stale = second.manifest
        stale.maxTokens += 1
        #expect(throws: ExperimentError.self) {
            try DraftAuthoringTransaction.replace(stale, reviewed: second)
        }
        let current = try DraftAuthoringSnapshot(workspaceRoot: root, name: manifest.name)
        #expect(current.file.data == saved.file.data)
        let beforeKeys = Set((try JSONSerialization.jsonObject(with: first.file.data) as! [String: Any]).keys)
        let afterKeys = Set((try JSONSerialization.jsonObject(with: current.file.data) as! [String: Any]).keys)
        #expect(beforeKeys == afterKeys)
        #expect(current.manifest.experimentDescription == "first edit")
    }

}
