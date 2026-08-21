import CryptoKit
import Foundation
import Testing
@testable import ExperimentKit
@testable import SteeringKit

/// Confirmation-study perturbation machinery: a declared policy expands
/// MECHANICALLY into ordinary hashed conditions on a draft manifest. Pure-CPU:
/// the expansion is a pure function, and the store-backed tests exercise the
/// authoring op's refusals against a temp workspace.
struct ConfirmationStudyTests {

    private static func policy(
        agent: String = "fear-agent",
        concept: String = "fear",
        layer: Int = 17,
        alpha: Double = 0.75,
        deltas: [Double] = [0.25, 0.5],
        control: Bool = true
    ) -> ExperimentManifest.PerturbationPolicy {
        .init(
            sourceAgent: .init(
                name: agent, artifactPath: "runs/model-variants/x/model-variant.json",
                artifactHash: String(repeating: "a", count: 64), promoted: true),
            concept: concept,
            cell: .init(layer: layer, alpha: alpha),
            alphaDeltas: deltas,
            includeMatchedNormControl: control,
            declaredAt: "2026-07-07T00:00:00Z")
    }

    // MARK: cross-engine parity fixture

    @Test func expansionMatchesTheCrossEngineContract() throws {
        // THE shared contract, pinned identically in the server suite
        // (Server/tests/test_confirmation.py): same policy inputs must
        // produce the same condition names, layers, alphas, and controlType
        // values on both engines. Dyadic alphas (0.75 ± 0.25/0.5) keep the
        // arithmetic exact so the pinned expected list is representable.
        let conditions = try ConfirmationStudy.expandedConditions(
            policy: Self.policy(), bandWidth: 1)
        let expected: [(String, Int, Double, String?)] = [
            ("fear-agent-anchor", 17, 0.75, nil),
            ("fear-agent-minus-0.25", 17, 0.5, nil),
            ("fear-agent-plus-0.25", 17, 1.0, nil),
            ("fear-agent-minus-0.5", 17, 0.25, nil),
            ("fear-agent-plus-0.5", 17, 1.25, nil),
            ("fear-agent-control", 17, 0.75, "randomMatchedNorm"),
        ]
        #expect(conditions.count == expected.count)
        for (condition, want) in zip(conditions, expected) {
            #expect(condition.name == want.0)
            #expect(condition.slots.count == 1)
            #expect(condition.slots.first?.concept == "fear")
            #expect(condition.slots.first?.layer == want.1)
            #expect(condition.slots.first?.alpha == want.2)
            #expect(condition.controlType == want.3)
            #expect(condition.alphaInNormUnits)
            #expect(condition.bandWidth == 1)
        }
    }

    @Test func expansionWithoutControlOmitsTheControlCondition() throws {
        let conditions = try ConfirmationStudy.expandedConditions(
            policy: Self.policy(deltas: [0.25], control: false), bandWidth: 3)
        #expect(conditions.map(\.name) == [
            "fear-agent-anchor", "fear-agent-minus-0.25", "fear-agent-plus-0.25",
        ])
        #expect(conditions.allSatisfy { $0.bandWidth == 3 })
        #expect(conditions.allSatisfy { $0.controlType == nil })
    }

    @Test func nonpositivePerturbedAlphaRefusesTheWholeExpansion() {
        // 0.75 − 0.75 = 0: zero and negative alphas are refused, never
        // clamped — a "perturbed" agent at α ≤ 0 is not the declared agent.
        #expect(throws: ExperimentError.self) {
            try ConfirmationStudy.expandedConditions(
                policy: Self.policy(deltas: [0.25, 0.75]), bandWidth: 1)
        }
    }

    @Test func deltaFormattingIsMinimal() {
        #expect(ConfirmationStudy.format(0.2) == "0.2")
        #expect(ConfirmationStudy.format(0.25) == "0.25")
        #expect(ConfirmationStudy.format(1.0) == "1")
        #expect(ConfirmationStudy.format(0.5) == "0.5")
    }

    @Test func generatedNameMatcherIsAgentScoped() {
        for name in [
            "fear-agent-anchor", "fear-agent-control",
            "fear-agent-minus-0.2", "fear-agent-plus-0.5",
        ] {
            #expect(ConfirmationStudy.isGeneratedName(name, agent: "fear-agent"))
        }
        for name in ["baseline", "fear-recommended", "fear-agent", "other-anchor"] {
            #expect(!ConfirmationStudy.isGeneratedName(name, agent: "fear-agent"))
        }
    }

    // MARK: schema stability

    @Test func legacyManifestEncodingGainsNoPerturbationKey() throws {
        // The one regression that must never happen: a manifest without the
        // new optional block must encode WITHOUT the key — that is what keeps
        // every existing content hash (and freeze hash) stable.
        let manifest = ExperimentManifest(
            name: "legacy", description: "", modelID: "test/model")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let json = String(decoding: try encoder.encode(manifest), as: UTF8.self)
        #expect(!json.contains("\"perturbationPolicy\""))
    }

    @Test func policyRoundTrips() throws {
        var manifest = ExperimentManifest(
            name: "rt", description: "", modelID: "test/model")
        manifest.perturbationPolicy = Self.policy()
        let decoded = try JSONDecoder().decode(
            ExperimentManifest.self, from: JSONEncoder().encode(manifest))
        #expect(decoded.perturbationPolicy == Self.policy())
    }

    @Test func policyParticipatesInTheContentHash() {
        var manifest = ExperimentManifest(
            name: "hash", description: "", modelID: "test/model")
        let without = ExperimentStore.manifestHash(manifest)
        manifest.perturbationPolicy = Self.policy()
        let with = ExperimentStore.manifestHash(manifest)
        #expect(without != with)
    }
}

/// Store-backed confirmation-authoring tests — extends the serialized
/// `ExperimentStoreTests` suite because they use process-global workspace
/// overrides.
extension ExperimentStoreTests {

    private func withConfirmWorkspace<T>(_ body: (URL) throws -> T) rethrows -> T {
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "confirm-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        // Shared cross-suite lock: the workspace root is process-global
        // (see ExperimentRootOverrideLock).
        ExperimentRootOverrideLock.acquire()
        WorkspaceRoot.programmaticOverride = temp
        defer {
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: temp)
        }
        return try body(temp)
    }

    private static let confirmFearHash = String(repeating: "f", count: 64)

    /// Save an agent artifact through the real store so name-based lookup
    /// (ModelVariantStore.scan) sees exactly what production writes.
    @discardableResult
    private func plantAgent(
        name: String = "fear-agent",
        concept: String = "fear",
        layer: Int = 17,
        alpha: Double = 0.75,
        adapters: [ModelVariantArtifact.AdapterRef] = [],
        injections: [ModelVariantArtifact.InjectionRef]? = nil,
        promoted: Bool = true
    ) throws -> ModelVariantRecord {
        let promotion: ModelVariantArtifact.Promotion? =
            promoted
            ? .init(
                experiment: "screen", experimentHash: "h",
                promotedAt: "2026-07-07T00:00:00Z", promotedBy: "criterion",
                winningCell: .init(layer: layer, alpha: alpha),
                substrate: "swift-mlx", appVersion: "test")
            : nil
        let artifact = ModelVariantArtifact(
            name: name, baseModelID: "test/model",
            adapters: adapters,
            injections: injections ?? [
                .init(
                    concept: concept, vectorArtifactID: "runs/x/\(concept)",
                    layer: layer, alpha: alpha)
            ],
            promptMode: "chatAssistant", qwenThinkingEnabled: false,
            temperature: 0, systemPrompt: "", promotion: promotion)
        return try ModelVariantStore.save(artifact)
    }

    private func confirmableExperiment(name: String) throws {
        var manifest = try ExperimentStore.create(
            name: name, description: "", modelID: "test/model",
            modelRevision: "abc123")
        manifest.concepts.append(
            .init(name: "fear", stimulusSetHash: Self.confirmFearHash, options: .init()))
        try ExperimentStore.save(manifest)
    }

    @Test func confirmAttachExpandsConditionsAndStampsThePolicy() throws {
        try withConfirmWorkspace { _ in
            try confirmableExperiment(name: "cf")
            let record = try plantAgent()
            let manifest = try ConfirmationStudy.attach(
                experimentName: "cf", agent: "fear-agent",
                deltas: [0.25], includeControl: true, log: { _ in })

            let policy = try #require(manifest.perturbationPolicy)
            #expect(policy.sourceAgent.name == "fear-agent")
            #expect(policy.sourceAgent.promoted)
            #expect(policy.sourceAgent.artifactHash
                == (try ModelVariantStore.hash(record.url)))
            #expect(policy.concept == "fear")
            #expect(policy.cell == .init(layer: 17, alpha: 0.75))
            #expect(policy.alphaDeltas == [0.25])
            #expect(policy.includeMatchedNormControl)
            #expect(manifest.conditions.map(\.name) == [
                "fear-agent-anchor", "fear-agent-minus-0.25",
                "fear-agent-plus-0.25", "fear-agent-control",
            ])
            #expect(manifest.conditions.last?.controlType == "randomMatchedNorm")
            // Persisted, not just returned — and the policy is in the
            // content hash (it is NOT a volatile lifecycle stamp).
            let loaded = try ExperimentStore.load(name: "cf")
            #expect(loaded.perturbationPolicy == policy)
            var stripped = loaded
            stripped.perturbationPolicy = nil
            #expect(
                ExperimentStore.manifestHash(loaded)
                    != ExperimentStore.manifestHash(stripped))
        }
    }

    @Test func confirmRerunReplacesPreviouslyGeneratedConditions() throws {
        try withConfirmWorkspace { _ in
            try confirmableExperiment(name: "rr")
            try plantAgent()
            _ = try ConfirmationStudy.attach(
                experimentName: "rr", agent: "fear-agent",
                deltas: [0.25], includeControl: true, log: { _ in })
            // Second run with different deltas and no control: the stale
            // minus/plus-0.25 and control conditions must be gone, and a
            // hand-authored condition must survive.
            var manifest = try ExperimentStore.load(name: "rr")
            manifest.conditions.append(
                .init(name: "hand-authored", slots: [
                    .init(concept: "fear", layer: 3, alpha: 0.1)
                ]))
            try ExperimentStore.save(manifest)
            let updated = try ConfirmationStudy.attach(
                experimentName: "rr", agent: "fear-agent",
                deltas: [0.5], includeControl: false, log: { _ in })
            #expect(updated.conditions.map(\.name) == [
                "hand-authored", "fear-agent-anchor",
                "fear-agent-minus-0.5", "fear-agent-plus-0.5",
            ])
            #expect(updated.perturbationPolicy?.alphaDeltas == [0.5])
            #expect(updated.perturbationPolicy?.includeMatchedNormControl == false)
        }
    }

    @Test func confirmRefusesFrozenManifests() throws {
        try withConfirmWorkspace { _ in
            try confirmableExperiment(name: "fz")
            try plantAgent()
            var manifest = try ExperimentStore.load(name: "fz")
            manifest.status = .frozen
            try ExperimentStore.save(manifest)
            #expect(throws: ExperimentError.self) {
                try ConfirmationStudy.attach(
                    experimentName: "fz", agent: "fear-agent", log: { _ in })
            }
        }
    }

    @Test func confirmRefusesAdapterBearingAgents() throws {
        try withConfirmWorkspace { _ in
            try confirmableExperiment(name: "ad")
            try plantAgent(
                name: "lora-agent",
                adapters: [
                    .init(
                        name: "a", artifactPath: "runs/x/a.json",
                        adapterDirectory: "runs/x/a", adapterHash: "h")
                ])
            #expect(throws: ExperimentError.self) {
                try ConfirmationStudy.attach(
                    experimentName: "ad", agent: "lora-agent", log: { _ in })
            }
        }
    }

    @Test func confirmRefusesZeroAndMultiInjectionAgents() throws {
        try withConfirmWorkspace { _ in
            try confirmableExperiment(name: "mi")
            try plantAgent(name: "empty-agent", injections: [])
            try plantAgent(
                name: "multi-agent",
                injections: [
                    .init(concept: "fear", vectorArtifactID: "x", layer: 3, alpha: 0.4),
                    .init(concept: "calm", vectorArtifactID: "y", layer: 5, alpha: 0.2),
                ])
            #expect(throws: ExperimentError.self) {
                try ConfirmationStudy.attach(
                    experimentName: "mi", agent: "empty-agent", log: { _ in })
            }
            #expect(throws: ExperimentError.self) {
                try ConfirmationStudy.attach(
                    experimentName: "mi", agent: "multi-agent", log: { _ in })
            }
        }
    }

    @Test func confirmRefusesUnattachedConcepts() throws {
        try withConfirmWorkspace { _ in
            // Experiment WITHOUT the agent's concept attached: refuse with
            // the attach-first message, never silently attach.
            _ = try ExperimentStore.create(
                name: "uc", description: "", modelID: "test/model",
                modelRevision: "abc123")
            try plantAgent()
            #expect {
                try ConfirmationStudy.attach(
                    experimentName: "uc", agent: "fear-agent", log: { _ in })
            } throws: { error in
                "\(error)".contains("attach concept 'fear' to 'uc' first")
            }
        }
    }

    @Test func confirmRefusesNonpositivePerturbedAlphaEndToEnd() throws {
        try withConfirmWorkspace { _ in
            try confirmableExperiment(name: "np")
            try plantAgent()
            #expect {
                try ConfirmationStudy.attach(
                    experimentName: "np", agent: "fear-agent",
                    deltas: [0.75], log: { _ in })
            } throws: { error in
                "\(error)".contains("nonpositive")
            }
            // Nothing was written: the whole operation refused.
            let manifest = try ExperimentStore.load(name: "np")
            #expect(manifest.perturbationPolicy == nil)
            #expect(manifest.conditions.isEmpty)
        }
    }

    @Test func confirmResolvesAgentsByExplicitPathAndRecordsPromotedFalse() throws {
        try withConfirmWorkspace { _ in
            try confirmableExperiment(name: "pp")
            let record = try plantAgent(name: "hand-agent", promoted: false)
            let relative = ModelVariantStore.relativePath(for: record)
            let manifest = try ConfirmationStudy.attach(
                experimentName: "pp", agent: relative, log: { _ in })
            let policy = try #require(manifest.perturbationPolicy)
            #expect(policy.sourceAgent.name == "hand-agent")
            #expect(policy.sourceAgent.promoted == false)
            #expect(policy.sourceAgent.artifactPath == relative)
        }
    }
}
