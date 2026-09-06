import Foundation
import Testing

@testable import ExperimentKit

struct AgentAttachmentReviewTests {
    private func fixture(_ body: (URL, ModelVariantRecord) throws -> Void) throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "agent-attachment") { root in
            let url = root.appending(components: "runs", "model-variants", "agent", "model-variant.json")
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            let artifact = ModelVariantArtifact(name: "agent", baseModelID: "test/model",
                promptMode: "chatAssistant", qwenThinkingEnabled: false, temperature: 0, systemPrompt: "Original")
            try JSONEncoder().encode(artifact).write(to: url)
            try body(root, ModelVariantRecord(url: url, artifact: artifact))
        }
    }

    @Test func pinAndEmbeddedArtifactDescribeTheSameBytes() throws {
        try fixture { _, record in
            let condition = try ExperimentStore.agentCondition(for: record)
            let bytes = try Data(contentsOf: record.url)
            #expect(condition.artifactHash == ExperimentStore.sha256Hex(bytes))
            #expect(condition.artifact == (try JSONDecoder().decode(ModelVariantArtifact.self, from: bytes)))
            // Formatting changes preserve the reviewed intervention. Pin the
            // current encoding without inventing a semantic change.
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(record.artifact).write(to: record.url)
            let formatted = try ExperimentStore.agentCondition(for: record)
            #expect(formatted.artifact == condition.artifact)
            #expect(formatted.artifactHash != condition.artifactHash)
            #expect(formatted.artifactHash == ExperimentStore.sha256Hex(try Data(contentsOf: record.url)))
        }
    }

    @Test func stalePickerRecordCannotAttachNewHashToOldIntervention() throws {
        try fixture { _, record in
            var changed = record.artifact
            changed.systemPrompt = "Changed"
            try JSONEncoder().encode(changed).write(to: record.url)
            var study = try ExperimentStore.create(name: "study", description: "", modelID: "test/model")
            let original = study
            do {
                try ExperimentStore.attachAgent(record, into: &study)
                Issue.record("An outdated agent selection must refuse before mutating the study")
            } catch let error as ExperimentError {
                #expect(error.lifecycleRefusal?.gate == .artifactPin)
                #expect(error.lifecycleRefusal?.repairAction.contains("review") == true)
            }
            #expect(study == original)
            #expect(try ExperimentStore.load(name: study.name) == original)
            let fresh = ModelVariantRecord(url: record.url, artifact: changed)
            try ExperimentStore.attachAgent(fresh, into: &study)
            #expect(study.variantConditions.first?.artifact.systemPrompt == "Changed")
        }
    }

    @Test(arguments: [false, true])
    func missingOrMalformedArtifactIsARepairableRefusal(malformed: Bool) throws {
        try fixture { _, record in
            if malformed { try Data("invalid".utf8).write(to: record.url) }
            else { try FileManager.default.removeItem(at: record.url) }
            do {
                _ = try ExperimentStore.agentCondition(for: record)
                Issue.record("Unreadable evidence must refuse")
            } catch let error as ExperimentError {
                #expect(error.lifecycleRefusal?.gate == .artifactPin)
                #expect(error.lifecycleRefusal?.repairAction.isEmpty == false)
            }
        }
    }
}
