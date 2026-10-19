import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// The preregistration's Conditions section lists every arm the run executes.
///
/// Field report 2026-10-19: a compare-agents study whose arms were attached
/// as agents (variant conditions, beside the implicit baseline) froze with an
/// empty Conditions section, because the section printed declared injection
/// conditions only. It now names each arm a run executes, in run order, and
/// names declared arms the run never executes after the list instead of in
/// it.
///
/// An injection condition's slot line also says what the run does with the
/// slot: an ablating slot reads `<concept> ablated λ=<λ> at every layer` (so
/// λ=1 ablation no longer reads like α=1 steering at one layer), and a
/// steering slot in a condition whose band widens it says by how much. An
/// agent line says the same of each steering injection when the agent's band
/// widens it, and states the agent's α units once, when the agent steers:
/// `α in residual-norm units` or `raw α (not scaled by the residual norm)`.
///
/// The sections in `Tests/Fixtures/cross-engine/preregistration-arms.json`
/// were written by the PYTHON engine's freeze; this engine must render the
/// same manifests to the same bytes. Server twin:
/// `Server/tests/test_preregistration_arms.py`.
extension ExperimentStoreTests {

    private static let armShapes: Set<String> = [
        "compare-agents", "compare-agents-carrying-conditions",
        "agent-interventions", "agent-band-and-units",
        "concept-study-implicit-baseline",
        "concept-study-declared-baseline", "concept-study-ablation",
        "concept-study-band-width", "sae-latent-arms",
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
                        + "@L\(injection.layer) α=\(injection.alpha); α in "
                        + "residual-norm units; base model "
                        + "test/model @ `abc123`; pinned artifact "
                        + "`\(agent.artifactPath)` (SHA-256 `\(agent.artifactHash)`)")
            }
        }
    }

    /// A steering injection reads like a condition's steering slot, band
    /// included; an ablating one takes no band; the α units are stated once,
    /// only when the agent steers, and a missing key reads as raw α.
    @Test func anAgentLineStatesItsBandAndItsAlphaUnits() throws {
        try withTempRoot {
            let manifest = try #require(
                try armCases().first { $0.label == "agent-band-and-units" }).manifest
            let hashes = Dictionary(
                uniqueKeysWithValues: manifest.variantConditions.map {
                    ($0.name, $0.artifactHash)
                })
            func line(_ name: String, _ interventions: String) -> String {
                "- **\(name)** (agent): \(interventions); base model "
                    + "test/model @ `abc123`; pinned artifact "
                    + "`experiments/prereg/pinned/variant-\(name).json` "
                    + "(SHA-256 `\(hashes[name] ?? "")`)"
            }
            let norm = "α in residual-norm units"
            let raw = "raw α (not scaled by the residual norm)"
            #expect(
                try conditionsSection(ExperimentStore.preregistrationMarkdown(manifest))
                    .split(separator: "\n").map(String.init) == [
                        "- **baseline**: none (baseline, run implicitly)",
                        line("fear-band-3", "fear@L14 α=0.8 (band width 3: ±1 layer); \(norm)"),
                        line("fear-band-2", "fear@L14 α=0.8 (band width 2: ±1 layer); \(norm)"),
                        line(
                            "fear-calm-band-5",
                            "fear@L14 α=0.8 (band width 5: ±2 layers), "
                                + "calm@L10 α=0.25 (band width 5: ±2 layers); \(norm)"),
                        line(
                            "fear-band-3-calm-ablated",
                            "fear@L14 α=0.8 (band width 3: ±1 layer), "
                                + "calm ablated λ=0.5 at L9+L10+L11; \(norm)"),
                        line("fear-band-0", "fear@L14 α=0.8; \(norm)"),
                        line("fear-raw", "fear@L14 α=2.5; \(raw)"),
                        line(
                            "fear-raw-band-3",
                            "fear@L14 α=2.5 (band width 3: ±1 layer); \(raw)"),
                        line("fear-legacy", "fear@L14 α=2.5; \(raw)"),
                        line("calm-ablated-band-3", "calm ablated λ=1 at every layer"),
                    ])
        }
    }

    /// The agent wording is read off this engine's own agent cell builder:
    /// the layers each injection names are the layers
    /// `ExperimentTasks.injections(for:)` injects it at, and the stated α
    /// units are the scale it applies. A residual norm of 10 over a vector
    /// norm of 5 doubles an α in residual-norm units and leaves a raw α as
    /// given; λ is never scaled.
    @Test func eachAgentLineNamesTheLayersAndUnitsTheRunInjects() throws {
        let layerCount = 32
        let vectors = ConceptVectors(
            perLayer: (0 ..< layerCount).map { _ in [3, 4, 0] })
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "prereg-agent-cells-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        for concept in ["french", "hungry", "fear", "calm"] {
            try SteeringVectorStore.save(
                vectors: vectors,
                sidecar: SteeringVectorSidecar(
                    modelID: "test/model", concept: concept, stimulusSetHash: "h",
                    vectors: vectors,
                    residualNormPerLayer: Array(repeating: 10, count: layerCount)),
                to: directory, name: concept,
                neutralMeanPerLayer: Array(repeating: [0, 0, 1], count: layerCount))
        }
        let partPattern = /(\w+)(?: ablated λ=([^\s;,]+) at (every layer|L\d+(?:\+L\d+)*)(?: \(centering \w+\))?|@L(\d+) α=([^\s;,]+)(?: \(band width -?\d+: ±(\d+) layers?\))?)/
        let normClause = "; α in residual-norm units; "
        let rawClause = "; raw α (not scaled by the residual norm); "
        var checked = 0
        for label in ["compare-agents", "agent-interventions", "agent-band-and-units"] {
            let armCase = try #require(try armCases().first { $0.label == label })
            var lines: [String: String] = [:]
            for line in armCase.conditionsSection.split(separator: "\n") {
                if let name = listedArms(String(line)).first {
                    lines[name] = String(line)
                }
            }
            for agent in armCase.manifest.variantConditions
            where agent.fromPromotion == nil && !agent.artifact.injections.isEmpty {
                var artifact = agent.artifact
                artifact.injections = artifact.injections.map { injection in
                    var local = injection
                    local.vectorArtifactID =
                        directory.appending(component: injection.concept).path
                    return local
                }
                let cells = try ExperimentTasks.injections(for: artifact)
                let line = try #require(lines[agent.name])
                let interventions = try #require(
                    line.split(separator: ": ", maxSplits: 1).last?
                        .split(separator: ";").first)
                let stated = interventions.matches(of: partPattern)
                #expect(
                    stated.map { String($0.output.1) }
                        == artifact.injections.map(\.concept),
                    "\(agent.name)")
                let steers = artifact.injections.contains { $0.effectiveMode != .ablate }
                let norm = line.contains(normClause)
                let raw = line.contains(rawClause)
                #expect(!(norm && raw), "\(agent.name)")
                #expect((norm || raw) == steers, "\(agent.name)")
                for match in stated {
                    let concept = String(match.output.1)
                    let mine = cells.filter { $0.concept == concept }
                    let named: [Int]
                    let expected: Float
                    if let lambda = match.output.2 {
                        let covered = try #require(match.output.3)
                        named = covered == "every layer"
                            ? Array(0 ..< layerCount)
                            : covered.split(separator: "+").compactMap {
                                Int($0.dropFirst())
                            }
                        #expect(mine.allSatisfy { $0.mode == .ablate }, "\(agent.name)")
                        expected = try #require(Float(lambda))
                    } else {
                        let layer = try #require(match.output.4.flatMap { Int($0) })
                        let half = match.output.6.flatMap { Int($0) } ?? 0
                        named = Array((layer - half) ... (layer + half))
                        #expect(mine.allSatisfy { $0.mode == .add }, "\(agent.name)")
                        let alpha = try #require(match.output.5.flatMap { Float($0) })
                        expected = alpha * (norm ? 2 : 1)
                    }
                    #expect(
                        mine.map(\.layer).sorted() == named,
                        "\(label) \(agent.name) \(concept)")
                    #expect(
                        mine.allSatisfy { abs($0.alpha - expected) < 1e-5 },
                        "\(label) \(agent.name) \(concept)")
                }
                checked += 1
            }
        }
        #expect(checked == 12)
    }

    @Test func anAblatingSlotNeverReadsLikeASteeringSlot() throws {
        try withTempRoot {
            let manifest = try #require(
                try armCases().first { $0.label == "concept-study-ablation" }).manifest
            #expect(
                try conditionsSection(ExperimentStore.preregistrationMarkdown(manifest))
                    .split(separator: "\n").map(String.init) == [
                        "- **baseline**: none (baseline, run implicitly)",
                        "- **fear-mid**: fear@L14 α=0.8",
                        "- **fear-steer-1**: fear@L14 α=1",
                        "- **fear-ablate**: fear ablated λ=1 at every layer",
                        "- **fear-ablate-random** [randomDirectionAblation]: "
                            + "fear ablated λ=1 at every layer",
                    ])
        }
    }

    @Test func aBandWiderThanOneLayerIsStatedOnEachSteeringSlot() throws {
        try withTempRoot {
            let manifest = try #require(
                try armCases().first { $0.label == "concept-study-band-width" }).manifest
            #expect(
                try conditionsSection(ExperimentStore.preregistrationMarkdown(manifest))
                    .split(separator: "\n").map(String.init) == [
                        "- **baseline**: none (baseline, run implicitly)",
                        "- **fear-band-3**: fear@L14 α=0.8 (band width 3: ±1 layer)",
                        "- **fear-band-2**: fear@L14 α=0.8 (band width 2: ±1 layer)",
                        "- **fear-calm-band-5**: fear@L14 α=0.8 (band width 5: ±2 layers), "
                            + "calm@L10 α=0.25 (band width 5: ±2 layers)",
                        "- **fear-band-3-calm-ablated**: fear@L14 α=0.8 "
                            + "(band width 3: ±1 layer), calm ablated λ=0.5 at every layer",
                        "- **fear-band-0**: fear@L14 α=0.8",
                    ])
        }
    }

    /// The wording is read off this engine's own cell builder: the layers a
    /// slot line names are the layers `ExperimentTasks.injections(for:
    /// extractions:)` injects it at.
    @Test func eachSlotLineNamesTheLayersTheRunInjects() throws {
        let layerCount = 32
        let extraction = ExperimentTasks.ConceptExtraction(
            result: ExtractionResult(
                vectors: ConceptVectors(
                    perLayer: (0 ..< layerCount).map { _ in [3, 4, 0] }),
                residualNormPerLayer: Array(repeating: 1, count: layerCount),
                residualNormSource: "neutral-corpus",
                options: ExtractionOptions(method: .meanDifference)),
            stimuli: nil)
        let slotPattern = /(\w+)(?: ablated λ=\S+ at (every layer)|@L(\d+) α=\S+(?: \(band width -?\d+: ±(\d+) layers?\))?)/
        for label in ["concept-study-ablation", "concept-study-band-width"] {
            let armCase = try #require(try armCases().first { $0.label == label })
            var lines: [String: String] = [:]
            for line in armCase.conditionsSection.split(separator: "\n") {
                let name = try #require(listedArms(String(line)).first)
                lines[name] = String(line)
            }
            for condition in armCase.manifest.conditions {
                let cells = try ExperimentTasks.injections(
                    for: condition,
                    extractions: ["fear": extraction, "calm": extraction])
                let line = try #require(lines[condition.name])
                let slotText = try #require(line.split(separator: ": ", maxSplits: 1).last)
                let stated = slotText.matches(of: slotPattern)
                #expect(
                    stated.map { String($0.output.1) } == condition.slots.map(\.concept),
                    "\(condition.name)")
                for match in stated {
                    let concept = String(match.output.1)
                    let injected = cells.filter { $0.concept == concept }.map(\.layer).sorted()
                    let named: [Int]
                    if match.output.2 != nil {
                        named = Array(0 ..< layerCount)
                    } else {
                        let layer = try #require(match.output.3.flatMap { Int($0) })
                        let half = match.output.4.flatMap { Int($0) } ?? 0
                        named = Array((layer - half) ... (layer + half))
                    }
                    #expect(injected == named, "\(label) \(condition.name) \(concept)")
                }
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

    /// These artifacts are written by this engine's encoder, which states
    /// residual-norm units (its default for new work).
    private func agentLines(_ frozen: ExperimentManifest) -> [String] {
        ["- **baseline**: none (baseline, run implicitly)"]
            + frozen.variantConditions.map { agent in
                let injection = agent.artifact.injections[0]
                return "- **\(agent.name)** (agent): \(injection.concept)"
                    + "@L\(injection.layer) α=\(injection.alpha); α in "
                    + "residual-norm units; base model "
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
