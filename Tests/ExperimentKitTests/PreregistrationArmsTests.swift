import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit

/// The preregistration's Conditions section lists every arm the run executes.
///
/// Field report 2026-10-19: a compare-agents study whose arms were attached
/// as agents (variant conditions, beside the implicit baseline) froze with an
/// empty Conditions section, because the section printed declared injection
/// conditions only. It now names each arm a run executes, in run order, and
/// names declared arms the run never executes after the list instead of in
/// it.
///
/// The sections in `Tests/Fixtures/cross-engine/preregistration-arms.json`
/// were written by the PYTHON engine's freeze; this engine must render the
/// same manifests to the same bytes. Server twin:
/// `Server/tests/test_preregistration_arms.py`.
extension ExperimentStoreTests {

    private static let armShapes: Set<String> = [
        "compare-agents", "compare-agents-carrying-conditions",
        "agent-interventions", "concept-study-implicit-baseline",
        "concept-study-declared-baseline", "sae-latent-arms",
        "multi-agent-panel", "multi-agent-panel-without-baseline",
    ]

    private struct ArmCase {
        let label: String
        let manifest: ExperimentManifest
        let conditionsSection: String
    }

    private func armCases() throws -> [ArmCase] {
        let url = CodeResources.compiledCheckoutPath.appending(
            components: "Tests", "Fixtures", "cross-engine",
            "preregistration-arms.json")
        let cases = try #require(
            try JSONSerialization.jsonObject(with: Data(contentsOf: url))
                as? [[String: Any]])
        return try cases.map { entry in
            let manifest = try #require(entry["manifest"])
            return ArmCase(
                label: try #require(entry["label"] as? String),
                manifest: try JSONDecoder().decode(
                    ExperimentManifest.self,
                    from: JSONSerialization.data(withJSONObject: manifest)),
                conditionsSection: try #require(
                    entry["conditionsSection"] as? String))
        }
    }

    /// Everything after the Conditions heading, up to the next heading.
    private func conditionsSection(_ text: String) throws -> String {
        let heading = try #require(text.range(of: "## Conditions\n\n"))
        let end = try #require(
            text.range(of: "\n\n## ", range: heading.upperBound ..< text.endIndex))
        return String(text[heading.upperBound ..< end.lowerBound])
    }

    private func listedArms(_ section: String) -> [String] {
        section.split(separator: "\n").compactMap { line in
            guard line.hasPrefix("- **"),
                let close = line.dropFirst(4).range(of: "**")
            else { return nil }
            return String(line.dropFirst(4)[..<close.lowerBound])
        }
    }

    @Test func preregistrationConditionsMatchThePythonEngineByteForByte() throws {
        try withTempRoot {
            let cases = try armCases()
            #expect(Set(cases.map(\.label)).isSuperset(of: Self.armShapes))
            for armCase in cases {
                let rendered = try conditionsSection(
                    ExperimentStore.preregistrationMarkdown(armCase.manifest))
                #expect(
                    rendered == armCase.conditionsSection,
                    "\(armCase.label): this engine's Conditions section differs")
            }
        }
    }

    @Test func compareAgentsPreregistrationListsTheBaselineAndEveryAgent() throws {
        try withTempRoot {
            let manifest = try #require(
                try armCases().first { $0.label == "compare-agents" }).manifest
            #expect(manifest.conditions.isEmpty)
            let section = try conditionsSection(
                ExperimentStore.preregistrationMarkdown(manifest))
            let lines = section.split(separator: "\n").map(String.init)
            #expect(lines.first == "- **baseline**: none (baseline, run implicitly)")
            #expect(lines.count == 1 + manifest.variantConditions.count)
            for (line, agent) in zip(lines.dropFirst(), manifest.variantConditions) {
                let injection = try #require(agent.artifact.injections.first)
                #expect(
                    line == "- **\(agent.name)** (agent): \(injection.concept)"
                        + "@L\(injection.layer) α=\(injection.alpha); base model "
                        + "test/model @ `abc123`; pinned artifact "
                        + "`\(agent.artifactPath)` (SHA-256 `\(agent.artifactHash)`)")
            }
        }
    }

    /// The section and the run select arms by one rule, so the names it lists
    /// are the names records will carry. SAE latent arms run on the server
    /// only, so this engine's vocabulary omits them; a panel's order is each
    /// engine's own.
    @Test func preregistrationArmsAreTheRunsRecordVocabulary() throws {
        try withTempRoot {
            for armCase in try armCases() {
                var latentNames: Set<String> = []
                if case .array(let entries)? = armCase.manifest.saeLatentConditions {
                    for case .object(let entry) in entries {
                        if case .string(let name)? = entry["name"] {
                            latentNames.insert(name)
                        }
                    }
                }
                let listed = listedArms(armCase.conditionsSection)
                    .filter { !latentNames.contains($0) }
                let run = ExperimentTasks.effectiveConditionNames(for: armCase.manifest)
                if armCase.manifest.studyKind == .modelOutput {
                    #expect(listed == run, "\(armCase.label)")
                } else {
                    #expect(listed.sorted() == run.sorted(), "\(armCase.label)")
                }
            }
        }
    }

    @Test func aGeneratedSummaryListingAgentsIsStillProvablyOurs() throws {
        try withTempRoot {
            let manifest = try #require(
                try armCases().first {
                    $0.label == "compare-agents-carrying-conditions"
                }).manifest
            let text = ExperimentStore.preregistrationMarkdown(manifest)
            #expect(text.contains(
                "\n\nDeclared but not run by this study design: fear-mid.\n\n## "))
            let url = ExperimentStore.directory.appending(
                components: manifest.name, ExperimentStore.preregistrationFilename)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            var unstamped = manifest
            unstamped.preregistrationHash = nil
            unstamped.preregistrationGeneratedHash = nil
            try text.write(to: url, atomically: true, encoding: .utf8)
            #expect(ExperimentStore.preregistrationIsGenerated(at: url, manifest: unstamped))
            try (text + "\nWe commit to the paired estimator.\n").write(
                to: url, atomically: true, encoding: .utf8)
            #expect(!ExperimentStore.preregistrationIsGenerated(at: url, manifest: unstamped))
        }
    }

    // MARK: - The freeze

    /// A compare-agents study whose two arms are attached the way
    /// `experiment attach-agent` attaches them, with validation evidence
    /// scoring the baseline and both agents.
    private func makeAgentStudy(name: String) throws {
        let root = try #require(ExperimentStore.rootOverride)
        var manifest = try ExperimentStore.create(
            name: name, description: "", modelID: "test/model",
            modelRevision: "abc123")
        manifest.studyType = "agentComparison"
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        for (agent, concept, layer, alpha) in [
            ("french-l28-a15", "french", 28, 0.15), ("hungry-l17-a10", "hungry", 17, 0.1),
        ] {
            let artifact = ModelVariantArtifact(
                name: agent, baseModelID: "test/model", baseRevision: "abc123",
                injections: [
                    .init(
                        concept: concept, vectorArtifactID: "runs/s-sweep/\(concept)",
                        layer: layer, alpha: alpha)
                ],
                promptMode: "chatAssistant", qwenThinkingEnabled: false,
                temperature: 0, systemPrompt: "")
            let url = root.appending(path: "runs/model-variants/\(agent)/model-variant.json")
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try encoder.encode(artifact).write(to: url)
            try ExperimentStore.attachAgent(
                ModelVariantRecord(url: url, artifact: artifact), into: &manifest,
                workspaceRoot: root)
        }
        try ExperimentStore.save(manifest)
        try fabricateValidationEvidence(
            for: manifest,
            capabilityBattery: ["baseline", "french-l28-a15", "hungry-l17-a10"].map {
                CapabilityBatteryConditionResult(
                    condition: $0, batteryHash: "bh", total: 12, correct: 12, accuracy: 1)
            })
    }

    private func agentLines(_ frozen: ExperimentManifest) -> [String] {
        ["- **baseline**: none (baseline, run implicitly)"]
            + frozen.variantConditions.map { agent in
                let injection = agent.artifact.injections[0]
                return "- **\(agent.name)** (agent): \(injection.concept)"
                    + "@L\(injection.layer) α=\(injection.alpha); base model "
                    + "test/model @ `abc123`; pinned artifact "
                    + "`\(agent.artifactPath)` (SHA-256 `\(agent.artifactHash)`)"
            }
    }

    @Test func freezeWritesEveryAgentArmIntoThePreregistration() throws {
        try withTempRoot {
            try makeAgentStudy(name: "agents")
            let frozen = try ExperimentStore.freeze(name: "agents")
            #expect(frozen.variantConditions.count == 2)
            let written = try String(
                contentsOf: ExperimentStore.directory.appending(
                    components: "agents", ExperimentStore.preregistrationFilename),
                encoding: .utf8)
            #expect(
                try conditionsSection(written).split(separator: "\n").map(String.init)
                    == agentLines(frozen))
            #expect(frozen.preregistrationHash == nil)
            #expect(
                frozen.preregistrationGeneratedHash
                    == ExperimentStore.sha256Hex(Data(written.utf8)))
        }
    }

    @Test func freezeKeepsAnAuthoredPreregistrationAndListsAgentsBesideIt() throws {
        try withTempRoot {
            try makeAgentStudy(name: "authored")
            let directory = ExperimentStore.directory.appending(component: "authored")
            let authored = "# Analysis preregistration\n\nWe compare each agent "
                + "with the unsteered model on every prompt.\n"
            try authored.write(
                to: directory.appending(component: ExperimentStore.preregistrationFilename),
                atomically: true, encoding: .utf8)
            let frozen = try ExperimentStore.freeze(name: "authored")
            #expect(
                try String(
                    contentsOf: directory.appending(
                        component: ExperimentStore.preregistrationFilename),
                    encoding: .utf8) == authored)
            #expect(
                frozen.preregistrationHash
                    == ExperimentStore.sha256Hex(Data(authored.utf8)))
            let settings = try String(
                contentsOf: directory.appending(
                    component: ExperimentStore.preregistrationFrozenSettingsFilename),
                encoding: .utf8)
            #expect(
                try conditionsSection(settings).split(separator: "\n").map(String.init)
                    == agentLines(frozen))
        }
    }
}
