import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit

/// A manifest key this build does not model survives a Mac draft save.
///
/// The Mac decodes a typed manifest. It used to drop every key the typed
/// structure did not name, so an older app that re-saved a draft written by a
/// newer engine or client silently lost the newer fields. Now:
///
/// - load and save of a draft keeps each unknown top-level key verbatim, and
///   so do Copy Study JSON and pack export;
/// - the typed encoding, and therefore `manifestHash`, is unchanged;
/// - freeze refuses a draft carrying unknown keys, naming them, with a repair
///   to update the app, because this build's freeze hash could not cover them;
///   the file is not touched;
/// - a server-frozen manifest carrying such a key is compared with its
///   canonical bytes instead of reported as changed after freeze.
@Suite(.serialized)
struct ManifestUnknownFieldsTests {
    static let future: JSONValue = .object([
        "nested": .array([.number(1), .string("two"), .null]),
        "flag": .bool(true),
    ])

    private func withDraft(_ body: (URL, URL) throws -> Void) throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "manifest-unknown") { root in
            _ = try ExperimentStore.create(name: "study", description: "Before", modelID: "test/model")
            let url = ExperimentStore.manifestURL("study")
            var object = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
            object["futureSetting"] = ["nested": [1, "two", NSNull()], "flag": true]
            object["futureLabel"] = "kept"
            try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys]).write(to: url)
            try body(root, url)
        }
    }

    private func object(_ data: Data) throws -> [String: JSONValue] {
        guard case .object(let object) = try JSONDecoder().decode(JSONValue.self, from: data) else {
            throw CancellationError()
        }
        return object
    }

    @Test func anUnknownKeySurvivesADraftSaveAndTheHashDoesNot() throws {
        try withDraft { _, url in
            var manifest = try ExperimentStore.load(name: "study")
            #expect(manifest.unknownTopLevelFields == ["futureSetting": Self.future, "futureLabel": .string("kept")])
            let hashBefore = ExperimentStore.manifestHash(manifest)
            var stripped = manifest
            stripped.unknownTopLevelFields = [:]
            #expect(hashBefore == ExperimentStore.manifestHash(stripped),
                    "the content hash must not change because a key is now carried")
            #expect(try object(JSONEncoder().encode(manifest))["futureSetting"] == nil,
                    "the typed encoding stays the typed encoding")

            manifest.experimentDescription = "After"
            try ExperimentStore.save(manifest)
            let saved = try object(Data(contentsOf: url))
            #expect(saved["futureSetting"] == Self.future)
            #expect(saved["futureLabel"] == .string("kept"))
            #expect(saved["experimentDescription"] == .string("After"))

            // A setter that loads fresh and saves keeps them too.
            try ExperimentStore.updateDraft(name: "study") { $0.temperature = 0.5 }
            #expect(try object(Data(contentsOf: url))["futureSetting"] == Self.future)
            // So does the copy a person pastes elsewhere.
            let copied = try ExperimentStore.exportStudyJSON(try ExperimentStore.load(name: "study"))
            #expect(try object(Data(copied.utf8))["futureLabel"] == .string("kept"))
        }
    }

    @Test func freezeRefusesUnknownKeysAndLeavesTheFileAlone() throws {
        try withDraft { _, url in
            let before = try Data(contentsOf: url)
            do {
                _ = try ExperimentStore.freeze(name: "study", force: true)
                Issue.record("freeze must refuse a draft with keys this build does not know")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("'futureLabel', 'futureSetting'"))
                #expect(error.lifecycleRefusal?.gate == .missingPrerequisite)
                #expect(error.lifecycleRefusal?.repairAction.contains("Update SteerLab") == true)
            }
            #expect(try Data(contentsOf: url) == before)
        }
    }

    @Test func packExportAndApplyCarryTheKey() throws {
        try withDraft { root, _ in
            let exported = try StudyPackAuthoring.export(reviewed: DraftAuthoringSnapshot(workspaceRoot: root, name: "study"))
            var pack = try object(exported.data)
            guard case .object(var study) = pack["study"] else {
                Issue.record("expected a study")
                return
            }
            #expect(study["futureSetting"] == Self.future)
            study["name"] = .string("copy")
            pack["study"] = .object(study)
            let data = try JSONEncoder().encode(JSONValue.object(pack))
            let review = try StudyPackAuthoring.preview(data, workspaceRoot: root)
            _ = try StudyPackAuthoring.apply(data, workspaceRoot: root, expectedReviewSHA256: review.reviewSHA256)
            let copy = try object(Data(contentsOf: ExperimentStore.manifestURL("copy")))
            #expect(copy["futureSetting"] == Self.future)
        }
    }

    @Test func aServerFrozenManifestWithAnUnknownKeyIsNotReportedAsChanged() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "manifest-unknown-frozen") { root in
            _ = try ExperimentStore.create(name: "study", description: "Server frozen", modelID: "test/model")
            let url = ExperimentStore.manifestURL("study")
            var object = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
            object["futureSetting"] = ["flag": true]
            var canonical = object
            for key in ["status", "frozenAt", "freezeHash", "gitCommit", "frozenBy", "createdAt", "appVersion"] {
                canonical.removeValue(forKey: key)
            }
            let canonicalData = try JSONSerialization.data(withJSONObject: canonical, options: [.sortedKeys])
            let hash = SHA256.hash(data: canonicalData).map { String(format: "%02x", $0) }.joined()
            object["status"] = "frozen"
            object["frozenBy"] = "server"
            object["freezeHash"] = hash
            try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys]).write(to: url)
            try canonicalData.write(to: url.deletingLastPathComponent().appending(component: "freeze-canonical.json"))
            let manifest = try ExperimentStore.load(name: "study")
            let drift = ExperimentStore.verify(manifest).filter { $0.contains("after freeze") }
            #expect(drift.isEmpty, "\(drift)")
            // …and the key is really compared: editing it after freeze is drift.
            object["futureSetting"] = ["flag": false]
            try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys]).write(to: url)
            let edited = ExperimentStore.verify(try ExperimentStore.load(name: "study"))
            #expect(edited.contains { $0.contains("after freeze") && $0.contains("futureSetting") }, "\(edited)")
        }
    }
}
