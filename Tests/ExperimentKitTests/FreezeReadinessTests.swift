import CryptoKit
import Foundation
import Testing
@testable import ExperimentKit

/// Read-only freeze readiness (the Experiments panel's "ready to freeze" /
/// first-unmet-gates line): the summarizer wraps the SAME gate checks
/// `ExperimentStore.freeze` enforces, so these tests assert parity with the
/// throwing path as well as the pure display formatting. Extends the
/// serialized `ExperimentStoreTests` suite for its `rootOverride` seam.
extension ExperimentStoreTests {

    // MARK: pure formatting

    @Test func readinessDisplayLineWhenReady() {
        let readiness = ExperimentStore.FreezeReadiness(unmetGates: [])
        #expect(readiness.ready)
        #expect(readiness.displayLine() == "ready to freeze")
    }

    @Test func readinessDisplayLineShowsFirstGatesPlusRemainder() {
        let readiness = ExperimentStore.FreezeReadiness(
            unmetGates: ["gate one", "gate two", "gate three", "gate four", "gate five"])
        #expect(!readiness.ready)
        #expect(
            readiness.displayLine()
                == "not ready to freeze: gate one · gate two · gate three · +2 more")
        #expect(readiness.displayLine(maxGates: 1) == "not ready to freeze: gate one · +4 more")
    }

    @Test func plainGateReasonStripsFreezePreambleOnly() {
        #expect(
            ExperimentStore.plainGateReason(
                "cannot freeze 'vs': variant 'x' has no pinned artifactHash",
                experimentName: "vs")
                == "variant 'x' has no pinned artifactHash")
        // Foreign strings pass through untouched.
        #expect(
            ExperimentStore.plainGateReason("some other error", experimentName: "vs")
                == "some other error")
    }

    // MARK: gate parity with freeze()

    @Test func plainDraftWithPinnedRevisionIsReadyAndFreezeAgrees() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "ready", description: "", modelID: "test/model",
                modelRevision: "abc123")
            manifest.concepts.append(
                .init(
                    name: "french", stimulusSetHash: try realFrenchHash(),
                    options: .init()))
            try ExperimentStore.save(manifest)
            try fabricateValidationEvidence(for: manifest)
            let readiness = ExperimentStore.freezeReadiness(for: manifest)
            #expect(readiness.ready, "unexpected gates: \(readiness.unmetGates)")
            let frozen = try ExperimentStore.freeze(name: "ready")
            #expect(frozen.status == .frozen)
        }
    }

    @Test func unpinnedRevisionReportsTheRevisionGate() throws {
        try withTempRoot {
            let manifest = try ExperimentStore.create(
                name: "norev", description: "", modelID: "test/never-cached-model")
            let readiness = ExperimentStore.freezeReadiness(for: manifest)
            #expect(!readiness.ready)
            #expect(readiness.unmetGates.contains { $0.contains("model revision is not pinned") })
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.freeze(name: "norev")
            }
        }
    }

    @Test func judgeEvaluatedStudyReportsRubricAndJudgeCountGates() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "judged", description: "", modelID: "test/model",
                modelRevision: "abc123")
            manifest.concepts.append(
                .init(
                    name: "french", stimulusSetHash: try realFrenchHash(),
                    options: .init()))
            manifest.judges = [.init(name: "judge-1", kind: "claude", model: nil)]
            try ExperimentStore.save(manifest)
            try fabricateValidationEvidence(for: manifest)
            let readiness = ExperimentStore.freezeReadiness(for: manifest)
            #expect(!readiness.ready)
            // The rubric-file gate fires first; readiness reports it in plain
            // words (no "cannot freeze" preamble).
            let gate = try #require(readiness.unmetGates.first)
            #expect(gate.contains("judge rubric"))
            #expect(!gate.contains("cannot freeze"))
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.freeze(name: "judged")
            }
        }
    }

    @Test func variantStudyWithoutValidateEvidenceIsNotReady() throws {
        try withTempRoot {
            // Same fixture shape as the firewall-closure tests: hashed
            // adapter, artifact under runs/, pinned artifactHash — every
            // variant pin is satisfied, so the reported gate is the missing
            // validate evidence (capability battery per condition).
            let root = try #require(ExperimentStore.rootOverride)
            let artifact = ModelVariantArtifact(
                name: "fear-lora",
                baseModelID: "test/model",
                adapters: [
                    .init(
                        name: "a1", artifactPath: "x", adapterDirectory: "y",
                        adapterHash: "deadbeef")
                ],
                injections: [],
                promptMode: "chatAssistant",
                qwenThinkingEnabled: false,
                temperature: 0,
                systemPrompt: "")
            let relativePath = "runs/model-variants/fear-lora/model-variant.json"
            let artifactURL = root.appending(path: relativePath)
            try FileManager.default.createDirectory(
                at: artifactURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            let data = try encoder.encode(artifact)
            try data.write(to: artifactURL)
            var manifest = try ExperimentStore.create(
                name: "vstudy", description: "", modelID: "test/model",
                modelRevision: "abc123")
            manifest.variantConditions = [
                .init(
                    name: "fear-lora", artifactPath: relativePath,
                    artifactHash: SHA256.hash(data: data)
                        .map { String(format: "%02x", $0) }.joined(),
                    artifact: artifact)
            ]
            try ExperimentStore.save(manifest)
            let readiness = ExperimentStore.freezeReadiness(for: manifest)
            #expect(!readiness.ready)
            #expect(readiness.unmetGates.contains { $0.contains("validate") })
        }
    }

    @Test func nonDraftReadinessSaysAlreadyFrozen() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "once", description: "", modelID: "test/model",
                modelRevision: "abc123")
            manifest.concepts.append(
                .init(
                    name: "french", stimulusSetHash: try realFrenchHash(),
                    options: .init()))
            try ExperimentStore.save(manifest)
            try fabricateValidationEvidence(for: manifest)
            let frozen = try ExperimentStore.freeze(name: "once")
            let readiness = ExperimentStore.freezeReadiness(for: frozen)
            #expect(!readiness.ready)
            #expect(readiness.unmetGates == ["'once' is already frozen"])
        }
    }
}

/// Git pin-cleanliness gate (pure seams; the Process/git wiring is skipped
/// under the test rootOverride by design, so freeze tests never depend on
/// the developer's working-tree state).
extension ExperimentStoreTests {

    @Test func porcelainParsingNamesModifiedAndUntrackedPaths() {
        let porcelain = """
             M prompts/emotions/fear/stories.jsonl
            ?? prompts/tasks/new-items.jsonl
            """
        #expect(
            ExperimentStore.offendingPorcelainPaths(porcelain) == [
                "prompts/emotions/fear/stories.jsonl",
                "prompts/tasks/new-items.jsonl",
            ])
        #expect(ExperimentStore.offendingPorcelainPaths("") == [])
        #expect(ExperimentStore.offendingPorcelainPaths("\n  \n") == [])
    }

    @Test func pinnedInputPathsCollectExistingPinsOnly() throws {
        var manifest = ExperimentManifest(
            name: "pin-paths", description: "", modelID: "test/model")
        manifest.concepts.append(
            .init(name: "french", stimulusSetHash: "x", options: .init()))
        manifest.judgeRubricFile = "prompts/rubrics/default-paired-v1.md"
        manifest.taskPromptsFile = "prompts/does-not-exist.jsonl"

        let paths = ExperimentStore.pinnedInputPaths(manifest).map(\.path)
        #expect(paths.contains { $0.hasSuffix("prompts/concepts/french") })
        #expect(paths.contains { $0.hasSuffix("prompts/rubrics/default-paired-v1.md") })
        // Nonexistent pins are filtered — verify() owns missing-file errors.
        #expect(!paths.contains { $0.contains("does-not-exist") })
    }
}
