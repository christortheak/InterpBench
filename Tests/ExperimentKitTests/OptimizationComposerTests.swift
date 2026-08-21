import CryptoKit
import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// Optimization Composer tests — plan validation (mixed models, missing
/// stimulus data, hash drift, method mapping), the instrument scan, name
/// suggestion, and end-to-end declare against a temp workspace. Declared as
/// an extension of the serialized `ExperimentStoreTests` suite because these
/// tests use the process-global workspace override
/// (`WorkspaceRoot.programmaticOverride`), the same seam as
/// `SweepPromotionTests`.
extension ExperimentStoreTests {

    // MARK: - Fixtures

    /// Creates a temp workspace and points the process-global override at
    /// it. Pair with `defer { cleanupComposerWorkspace(root) }` (no closure
    /// helper — the end-to-end tests are @MainActor and call MainActor
    /// engine code inline).
    private func makeComposerWorkspace() throws -> URL {
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "composer-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        // Shared cross-suite lock: the workspace root is process-global
        // (see ExperimentRootOverrideLock).
        ExperimentRootOverrideLock.acquire()
        WorkspaceRoot.programmaticOverride = temp
        return temp
    }

    private func cleanupComposerWorkspace(_ root: URL) {
        WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
        try? FileManager.default.removeItem(at: root)
    }

    private func writeComposerPairedStimuli(concept: String) throws {
        let directory = VectorCatalog.conceptsDirectory.appending(component: concept)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try #"{"text": "a positive stimulus"}"#.write(
            to: directory.appending(component: "positive.jsonl"),
            atomically: true, encoding: .utf8)
        try #"{"text": "a negative stimulus"}"#.write(
            to: directory.appending(component: "negative.jsonl"),
            atomically: true, encoding: .utf8)
    }

    private func writeComposerStories(concept: String, texts: [String]) throws {
        let directory = ExperimentStore.emotionsDirectory.appending(component: concept)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let lines = texts.enumerated().map { index, text in
            #"{"concept": "\#(concept)", "text": "\#(text)", "id": "\#(concept)-\#(index)"}"#
        }
        try (lines.joined(separator: "\n") + "\n").write(
            to: directory.appending(component: "stories.jsonl"),
            atomically: true, encoding: .utf8)
    }

    private func writeComposerNeutralCorpus() throws {
        let directory = VectorCatalog.projectRoot.appending(
            components: "prompts", "neutral")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try #"{"text": "a neutral sentence"}"#.write(
            to: directory.appending(component: "corpus.jsonl"),
            atomically: true, encoding: .utf8)
    }

    private func writeComposerRubric(_ relativePath: String) throws {
        let url = VectorCatalog.projectRoot.appending(path: relativePath)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try "Compare the two responses.".write(
            to: url, atomically: true, encoding: .utf8)
    }

    /// Plants a vector artifact (safetensors stub + decodable sidecar) in
    /// the workspace's runs/ so `VectorCatalog.scan` and
    /// `OptimizationComposer.facts(for:)` see it.
    private func plantComposerArtifact(
        in root: URL,
        concept: String,
        modelID: String = "test/model",
        stimulusSetHash: String,
        extractionMethod: String? = "meanDifference",
        recipeMethod: String? = nil,
        readingPosition: String? = "last token",
        comparisonConcepts: [String]? = nil,
        designatedReference: [String: String]? = nil
    ) throws {
        let runDir = root.appending(
            components: "runs", "20260708T000001000-extract-\(concept)")
        try FileManager.default.createDirectory(
            at: runDir, withIntermediateDirectories: true)
        try Data("weights".utf8).write(
            to: runDir.appending(component: "\(concept).safetensors"))
        var sidecar: [String: Any] = [
            "modelID": modelID, "concept": concept,
            "stimulusSetHash": stimulusSetHash, "layerCount": 4, "hiddenSize": 2,
            "normsPerLayer": [1.0, 1.0, 1.0, 1.0],
            "extractionDate": "2026-07-08T00:00:00Z",
        ]
        if let extractionMethod { sidecar["extractionMethod"] = extractionMethod }
        if let recipeMethod { sidecar["recipeMethod"] = recipeMethod }
        if let readingPosition { sidecar["readingPosition"] = readingPosition }
        if let comparisonConcepts { sidecar["comparisonConcepts"] = comparisonConcepts }
        if let designatedReference {
            sidecar["designatedReference"] = designatedReference
        }
        try JSONSerialization.data(withJSONObject: sidecar)
            .write(to: runDir.appending(component: "\(concept).json"))
    }

    private static let composerHashA = String(repeating: "a", count: 64)
    private static let composerHashB = String(repeating: "b", count: 64)

    private func composerFacts(
        concept: String = "fear",
        modelID: String = "test/model",
        recorded: String = String(repeating: "a", count: 64),
        recipeMethodRaw: String? = nil,
        extractionMethodRaw: String? = "meanDifference",
        readingPositionLabel: String? = "last token",
        grandMeanCorpusConcepts: [String]? = nil,
        currentPairedStimulusHash: String? = String(repeating: "a", count: 64),
        currentStoriesHash: String? = nil,
        missingCorpusMembers: [String] = []
    ) -> OptimizationComposer.ArtifactFacts {
        OptimizationComposer.ArtifactFacts(
            artifactID: "runs/x/\(concept)",
            concept: concept,
            modelID: modelID,
            recordedStimulusHash: recorded,
            recipeMethodRaw: recipeMethodRaw,
            extractionMethodRaw: extractionMethodRaw,
            readingPositionLabel: readingPositionLabel,
            grandMeanCorpusConcepts: grandMeanCorpusConcepts,
            currentPairedStimulusHash: currentPairedStimulusHash,
            currentStoriesHash: currentStoriesHash,
            missingCorpusMembers: missingCorpusMembers)
    }

    private func expectPinnable(
        _ verdict: OptimizationComposer.ArtifactVerdict
    ) -> OptimizationComposer.ConceptPin? {
        guard case .pinnable(let pin) = verdict else {
            Issue.record("expected pinnable, got \(verdict)")
            return nil
        }
        return pin
    }

    /// Asserts a refusal whose user-facing MESSAGE contains `needle` and —
    /// when given — whose technical DETAIL (the tooltip half) contains
    /// `detailContaining`: wording fixes must keep the precise paths/field
    /// names somewhere the view can surface.
    private func expectRefusal(
        _ verdict: OptimizationComposer.ArtifactVerdict,
        containing needle: String,
        detailContaining detailNeedle: String? = nil
    ) {
        guard case .refused(let refusal) = verdict else {
            Issue.record("expected refusal containing '\(needle)', got \(verdict)")
            return
        }
        #expect(
            refusal.message.contains(needle),
            "refusal '\(refusal.message)' lacks '\(needle)'")
        if let detailNeedle {
            let detail = refusal.detail ?? ""
            #expect(
                detail.contains(detailNeedle),
                "refusal detail '\(detail)' lacks '\(detailNeedle)'")
        }
    }

    // MARK: - Method mapping

    @Test func composerMethodMapping() {
        // recipeMethod (newer stamp) wins and maps each family.
        #expect(
            OptimizationComposer.mapMethod(
                recipeMethodRaw: "caaMeanDifference", extractionMethodRaw: nil)
                == .mapped(.meanDifference))
        #expect(
            OptimizationComposer.mapMethod(
                recipeMethodRaw: "repeLAT", extractionMethodRaw: nil)
                == .mapped(.lat))
        #expect(
            OptimizationComposer.mapMethod(
                recipeMethodRaw: "emotionGrandMean", extractionMethodRaw: "meanDifference")
                == .mapped(.emotionGrandMean))
        // Raw ExtractionMethod values map via the fallback.
        #expect(
            OptimizationComposer.mapMethod(
                recipeMethodRaw: nil, extractionMethodRaw: "lat")
                == .mapped(.lat))
        // Reader-derived vectors are conversions, not recipes.
        if case .mapped = OptimizationComposer.mapMethod(
            recipeMethodRaw: "repeReaderLAT", extractionMethodRaw: nil)
        {
            Issue.record("repeReaderLAT recipeMethod must refuse")
        }
        if case .mapped = OptimizationComposer.mapMethod(
            recipeMethodRaw: nil, extractionMethodRaw: "repeReaderLAT")
        {
            Issue.record("repeReaderLAT extractionMethod must refuse")
        }
        // Unknown strings refuse rather than guess.
        if case .mapped = OptimizationComposer.mapMethod(
            recipeMethodRaw: "mysteryMethod", extractionMethodRaw: nil)
        {
            Issue.record("unknown recipe method must refuse")
        }
        // Legacy sidecar with NO recorded method refuses in user vocabulary
        // (no "sidecar" in the message); the exact missing fields live in
        // the tooltip detail.
        guard
            case .refused(let refusal) = OptimizationComposer.mapMethod(
                recipeMethodRaw: nil, extractionMethodRaw: nil)
        else {
            Issue.record("legacy no-method sidecar must refuse")
            return
        }
        #expect(refusal.message.contains("predates recipe recording"))
        #expect(refusal.message.contains("no extraction method saved"))
        #expect(refusal.detail?.contains("recipeMethod") == true)
    }

    // MARK: - Assessment (pure, over plain facts)

    @Test func composerAssessPairedPinnableAndDriftFlagged() throws {
        // Clean: current == recorded → pinnable, no caution.
        let clean = expectPinnable(OptimizationComposer.assess(composerFacts()))
        #expect(clean?.caution == nil)
        #expect(clean?.method == .meanDifference)
        #expect(clean?.readingPosition == .lastToken)
        #expect(clean?.currentStimulusHash == Self.composerHashA)

        // Drift: flagged loudly but STILL declarable (advisory, never a gate).
        let drifted = expectPinnable(
            OptimizationComposer.assess(
                composerFacts(
                    recorded: Self.composerHashB,
                    currentPairedStimulusHash: Self.composerHashA)))
        let caution = try #require(drifted?.caution)
        #expect(caution.contains("changed"))
        #expect(drifted?.currentStimulusHash == Self.composerHashA)
    }

    @Test func composerAssessRefusesMissingStimulusDirectory() {
        // An imported feature vector (e.g. Gemma Scope) has no stimulus
        // directory: not pinnable. Message in user vocabulary; the exact
        // prompts/concepts/<concept>/ path lives in the tooltip detail.
        expectRefusal(
            OptimizationComposer.assess(
                composerFacts(
                    concept: "sae:gemma-scope:L20:1234",
                    currentPairedStimulusHash: nil)),
            containing: "no stimulus data",
            detailContaining: "prompts/concepts/sae:gemma-scope:L20:1234")
    }

    @Test func composerAssessRefusesLegacyArtifacts() {
        // No recorded method → refuse naming the gap.
        expectRefusal(
            OptimizationComposer.assess(
                composerFacts(extractionMethodRaw: nil)),
            containing: "no extraction method saved")
        // No recorded reading position → refuse (never silently default a
        // recipe field selection provenance depends on).
        expectRefusal(
            OptimizationComposer.assess(
                composerFacts(readingPositionLabel: nil)),
            containing: "no reading position saved")
        // Unparseable reading position → refuse, not guess; the raw label
        // stays visible in the message.
        expectRefusal(
            OptimizationComposer.assess(
                composerFacts(readingPositionLabel: "somewhere in the middle")),
            containing: "somewhere in the middle")
    }

    @Test func composerAssessGrandMeanRules() throws {
        // Fully reconstructable grand-mean recipe → pinnable with corpus +
        // pooling token for the engine's own pin helper.
        let pin = expectPinnable(
            OptimizationComposer.assess(
                composerFacts(
                    recorded: Self.composerHashB,
                    recipeMethodRaw: "emotionGrandMean",
                    extractionMethodRaw: nil,
                    readingPositionLabel: "mean from token 50",
                    grandMeanCorpusConcepts: ["calm", "fear"],
                    currentPairedStimulusHash: nil,
                    currentStoriesHash: Self.composerHashA)))
        #expect(pin?.method == .emotionGrandMean)
        #expect(pin?.poolFromToken == 50)
        #expect(pin?.corpusConcepts == ["calm", "fear"])
        // Recorded hash covers the artifact's selected build rows — the
        // difference is surfaced loudly, never gated.
        #expect(pin?.caution != nil)

        // No stories.jsonl locally → nothing to pin. Message in user
        // vocabulary; the stories path lives in the tooltip detail.
        expectRefusal(
            OptimizationComposer.assess(
                composerFacts(
                    recipeMethodRaw: "emotionGrandMean",
                    extractionMethodRaw: nil,
                    readingPositionLabel: "mean from token 50",
                    grandMeanCorpusConcepts: ["calm", "fear"],
                    currentPairedStimulusHash: nil,
                    currentStoriesHash: nil)),
            containing: "no stimulus data",
            detailContaining: "stories.jsonl")

        // No recorded corpus membership → the population cannot be
        // defaulted; the sidecar field name stays in the tooltip detail.
        expectRefusal(
            OptimizationComposer.assess(
                composerFacts(
                    recipeMethodRaw: "emotionGrandMean",
                    extractionMethodRaw: nil,
                    readingPositionLabel: "mean from token 50",
                    currentPairedStimulusHash: nil,
                    currentStoriesHash: Self.composerHashA)),
            containing: "grand-mean corpus",
            detailContaining: "comparisonConcepts")

        // A recorded member with no local stories → refuse (the pinned
        // population must exist), naming the member.
        expectRefusal(
            OptimizationComposer.assess(
                composerFacts(
                    recipeMethodRaw: "emotionGrandMean",
                    extractionMethodRaw: nil,
                    readingPositionLabel: "mean from token 50",
                    grandMeanCorpusConcepts: ["calm", "fear"],
                    currentPairedStimulusHash: nil,
                    currentStoriesHash: Self.composerHashA,
                    missingCorpusMembers: ["calm"])),
            containing: "calm")

        // A reading position the engine's grand-mean pin helper cannot
        // express → refuse rather than mispin, steering to a pooled
        // re-extraction.
        expectRefusal(
            OptimizationComposer.assess(
                composerFacts(
                    recipeMethodRaw: "emotionGrandMean",
                    extractionMethodRaw: nil,
                    readingPositionLabel: "last token",
                    grandMeanCorpusConcepts: ["calm", "fear"],
                    currentPairedStimulusHash: nil,
                    currentStoriesHash: Self.composerHashA)),
            containing: "pooled reading")
    }

    // MARK: - Plan rules

    @Test func composerPlanRefusesMixedModelsAndDuplicates() throws {
        func pin(
            _ concept: String, model: String
        ) -> OptimizationComposer.ConceptPin {
            OptimizationComposer.ConceptPin(
                concept: concept, modelID: model, method: .meanDifference,
                readingPosition: .lastToken,
                currentStimulusHash: Self.composerHashA,
                recordedStimulusHash: Self.composerHashA)
        }
        // Empty selection.
        #expect(OptimizationComposer.planProblem([]) != nil)
        // Mixed models refuse with both models named.
        let mixed = try #require(
            OptimizationComposer.planProblem(
                [pin("fear", model: "model/a"), pin("calm", model: "model/b")]))
        #expect(mixed.contains("model/a"))
        #expect(mixed.contains("model/b"))
        // Two artifacts for one concept refuse (one recipe per concept name).
        let duplicate = try #require(
            OptimizationComposer.planProblem(
                [pin("fear", model: "model/a"), pin("fear", model: "model/a")]))
        #expect(duplicate.contains("fear"))
        // Sound selection → plan with the shared model and pins.
        let plan = try OptimizationComposer.makePlan(
            [pin("fear", model: "model/a"), pin("calm", model: "model/a")])
        #expect(plan.modelID == "model/a")
        #expect(plan.pins.count == 2)
        #expect(plan.cautions.isEmpty)
    }

    // MARK: - Model availability partition

    @Test func composerPartitionByModelAvailability() {
        struct Row: Equatable {
            let id: String
            let model: String
        }
        let rows = [
            Row(id: "a", model: "m1"), Row(id: "b", model: "m2"),
            Row(id: "c", model: "m1"), Row(id: "d", model: "m3"),
        ]
        // Installed models list normally; the rest partition out, order
        // preserved within each half.
        let split = OptimizationComposer.partitionByModelAvailability(
            rows, availableModels: ["m1", "m3"], modelID: \.model)
        #expect(split.available == [rows[0], rows[2], rows[3]])
        #expect(split.unavailable == [rows[1]])
        // Display scoping only — nothing is dropped.
        #expect(split.available.count + split.unavailable.count == rows.count)
        // An EMPTY inventory means "unknown" (e.g. a server that has not
        // reported models): scoping would be a dishonest guess, so
        // everything stays available.
        let unknown = OptimizationComposer.partitionByModelAvailability(
            rows, availableModels: [], modelID: \.model)
        #expect(unknown.available == rows)
        #expect(unknown.unavailable.isEmpty)
    }

    @Test func composerPartitionScopesCatalogArtifactsBySidecarModel() throws {
        let root = try makeComposerWorkspace()
        defer { cleanupComposerWorkspace(root) }
        try plantComposerArtifact(
            in: root, concept: "fear", modelID: "mlx/model-a",
            stimulusSetHash: Self.composerHashA)
        try plantComposerArtifact(
            in: root, concept: "calm", modelID: "hf/model-b",
            stimulusSetHash: Self.composerHashA)
        let artifacts = VectorCatalog.scan()
        #expect(artifacts.count == 2)
        let split = OptimizationComposer.partitionByModelAvailability(
            artifacts, availableModels: ["mlx/model-a"])
        #expect(split.available.map(\.sidecar.concept) == ["fear"])
        #expect(split.unavailable.map(\.sidecar.concept) == ["calm"])
    }

    // MARK: - Name suggestion

    @Test func composerSuggestedName() throws {
        var components = DateComponents()
        components.year = 2026
        components.month = 7
        components.day = 8
        components.hour = 12
        let date = try #require(Calendar.current.date(from: components))
        #expect(
            OptimizationComposer.suggestedName(firstConcept: "fear", date: date)
                == "optimize-fear-2026-07-08")
        // Sanitized like ExperimentStore.create sanitizes.
        #expect(
            OptimizationComposer.suggestedName(
                firstConcept: "Fear Of Heights!", date: date)
                == "optimize-fear-of-heights-2026-07-08")
        #expect(
            OptimizationComposer.suggestedName(firstConcept: nil, date: date)
                == "optimize-concept-2026-07-08")
    }

    // MARK: - Instrument scan

    @Test func composerInstrumentScanFindsJSONLUnderPrompts() throws {
        let root = try makeComposerWorkspace()
        defer { cleanupComposerWorkspace(root) }
        let fm = FileManager.default
        func plant(_ relative: String, contents: String = #"{"text": "x"}"#) throws {
            let url = root.appending(path: relative)
            try fm.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try contents.write(to: url, atomically: true, encoding: .utf8)
        }
        try plant("prompts/root.jsonl")
        try plant("prompts/dev/dev-prompts.jsonl")
        try plant("prompts/a/b/c/d/deep.jsonl")  // at the depth cap → included
        try plant("prompts/a/b/c/d/e/too-deep.jsonl")  // beyond the cap
        try plant("prompts/dev/readme.md", contents: "not an instrument")
        try plant("outside/prompts.jsonl")  // not under prompts/

        let found = OptimizationComposer.scanInstrumentFiles(root: root, maxDepth: 4)
        #expect(
            found == [
                "prompts/a/b/c/d/deep.jsonl",
                "prompts/dev/dev-prompts.jsonl",
                "prompts/root.jsonl",
            ])
    }

    // MARK: - End-to-end declare

    @Test @MainActor func composerDeclareCreatesJudgeScoreOptimization() throws {
        let root = try makeComposerWorkspace()
        defer { cleanupComposerWorkspace(root) }
        try writeComposerPairedStimuli(concept: "fear")
        try writeComposerNeutralCorpus()
        try writeComposerRubric("prompts/rubrics/paired-v1.md")
        // Recorded hash deliberately differs from the current stimuli —
        // drift is flagged but declarable, and the pin must be CURRENT.
        try plantComposerArtifact(
            in: root, concept: "fear",
            stimulusSetHash: Self.composerHashB,
            extractionMethod: "meanDifference",
            readingPosition: "last token")

        // Through the real catalog + disk-backed facts.
        let artifact = try #require(
            VectorCatalog.scan().first { $0.sidecar.concept == "fear" })
        let verdict = OptimizationComposer.assess(
            OptimizationComposer.facts(for: artifact))
        let pin = try #require(expectPinnable(verdict))
        #expect(pin.caution != nil)

        let plan = try OptimizationComposer.makePlan([pin])
        var spec = ExperimentManifest.SweepSpec()
        spec.selection = ExperimentManifest.SweepSelection(
            objective: .init(metric: "judgeScore"),
            constraints: .init(capabilityTolerance: 0.15, coherenceFloor: 0.45))
        let panel = ExperimentPanel()
        // An openrouter judge rides along: declare must carry its provider
        // PIN through verbatim (a declare that drops it invalidates the
        // judge — round-trip regression, engineer review 2026-07-19).
        let judges = [
            ExperimentManifest.JudgeRef(name: "primary", kind: "claude", model: nil),
            ExperimentManifest.JudgeRef(
                name: "or-judge", kind: "openrouter",
                model: "google/gemma-3-27b-it", provider: "DeepInfra"),
        ]
        let name = try OptimizationComposer.declare(
            name: "optimize-fear-e2e", description: "composer end-to-end",
            plan: plan, spec: spec,
            judgeRubricFile: "prompts/rubrics/paired-v1.md",
            judges: judges,
            panel: panel)
        #expect(name == "optimize-fear-e2e")

        let manifest = try ExperimentStore.load(name: name)
        // Concept pinned at the CURRENT hash (file truth), not the
        // artifact's recorded hash.
        let currentHash = try StimulusSet(
            directory: VectorCatalog.conceptsDirectory.appending(component: "fear")
        ).hash
        let ref = try #require(manifest.concepts.first)
        #expect(ref.name == "fear")
        #expect(ref.stimulusSetHash == currentHash)
        #expect(ref.stimulusSetHash != Self.composerHashB)
        #expect(ref.options.method == .meanDifference)
        #expect(ref.options.readingPosition == .lastToken)
        // Same pins/defaults the Studies attach path stamps.
        #expect(manifest.neutralCorpusHash != nil)
        #expect(manifest.studyKind == .modelOutput)
        #expect(manifest.temperature == 0)
        #expect(manifest.maxTokens == 2048)
        #expect(manifest.promptMode == .chatAssistant)
        // judgeScore pins landed BEFORE the spec so the engine's criterion
        // validation could pass.
        #expect(manifest.judgeRubricFile == "prompts/rubrics/paired-v1.md")
        #expect(manifest.judgeRubricHash?.isEmpty == false)
        // Equality covers the provider pin: JudgeRef is Equatable over ALL
        // fields, so a dropped provider fails here.
        #expect(manifest.judges == judges)
        #expect(manifest.judges?.last?.provider == "DeepInfra")
        let sweep = try #require(manifest.sweep)
        #expect(sweep.selection?.objective?.metric == "judgeScore")
        #expect(sweep.selection?.constraints?.capabilityTolerance == 0.15)
        // The manifest verifies clean — pins are exactly what the engine demands.
        #expect(ExperimentStore.verify(manifest).isEmpty)

        // A name collision surfaces the store's own error verbatim.
        do {
            _ = try OptimizationComposer.declare(
                name: "optimize-fear-e2e", description: "",
                plan: plan, spec: spec,
                judgeRubricFile: "prompts/rubrics/paired-v1.md",
                judges: judges, panel: panel)
            Issue.record("expected the name collision to refuse")
        } catch let error as ExperimentError {
            #expect(error.reason.contains("already exists"))
        }
    }

    @Test @MainActor func composerDeclarePinsGrandMeanViaEngineHelper() throws {
        let root = try makeComposerWorkspace()
        defer { cleanupComposerWorkspace(root) }
        try writeComposerStories(
            concept: "fear",
            texts: ["Her heart pounded in the dark.", "The floor creaked behind her."])
        try writeComposerStories(
            concept: "calm",
            texts: ["Waves lapped gently at the shore.", "Tea steamed in the kitchen."])
        try plantComposerArtifact(
            in: root, concept: "fear",
            stimulusSetHash: Self.composerHashB,  // selected-build-rows hash
            extractionMethod: nil,
            recipeMethod: "emotionGrandMean",
            readingPosition: "mean from token 50",
            comparisonConcepts: ["calm", "fear"])

        let artifact = try #require(
            VectorCatalog.scan().first { $0.sidecar.concept == "fear" })
        let verdict = OptimizationComposer.assess(
            OptimizationComposer.facts(for: artifact))
        let pin = try #require(expectPinnable(verdict))
        #expect(pin.method == .emotionGrandMean)
        #expect(pin.poolFromToken == 50)
        #expect(pin.corpusConcepts == ["calm", "fear"])

        let plan = try OptimizationComposer.makePlan([pin])
        var spec = ExperimentManifest.SweepSpec()
        spec.selection = ExperimentManifest.SweepSelection(
            objective: .init(metric: "markerDensity"),
            constraints: .init(capabilityTolerance: 0.15, coherenceFloor: 0.45))
        let panel = ExperimentPanel()
        let name = try OptimizationComposer.declare(
            name: "optimize-fear-gm", description: "",
            plan: plan, spec: spec,
            judgeRubricFile: nil, judges: [],
            panel: panel)

        let manifest = try ExperimentStore.load(name: name)
        // Pinned exactly the way the engine's own grand-mean helper pins:
        // target stories hash + full population membership and hashes.
        let ref = try #require(manifest.concepts.first)
        #expect(ref.options.method == .emotionGrandMean)
        #expect(ref.options.readingPosition == .meanFromToken(50))
        #expect(ref.stimulusSetHash == ExperimentStore.storiesHash(for: "fear"))
        let corpus = try #require(manifest.grandMeanCorpus)
        #expect(Set(corpus.concepts) == ["calm", "fear"])
        #expect(corpus.hashes["calm"] == ExperimentStore.storiesHash(for: "calm"))
        #expect(corpus.hashes["fear"] == ExperimentStore.storiesHash(for: "fear"))
        let sweep = try #require(manifest.sweep)
        #expect(sweep.selection?.objective?.metric == "markerDensity")
        // The engine's own verify accepts the pin set.
        #expect(ExperimentStore.verify(manifest).isEmpty)
    }

    @Test @MainActor func serverGrandMeanCatalogRecordSeedsOptimizationRecipe() throws {
        let root = try makeComposerWorkspace()
        defer { cleanupComposerWorkspace(root) }
        try writeComposerStories(
            concept: "fear",
            texts: ["Her heart pounded in the dark.", "The floor creaked behind her."])
        try writeComposerStories(
            concept: "calm",
            texts: ["Waves lapped gently at the shore.", "Tea steamed in the kitchen."])

        let record = RemoteVectorRecord(
            id: "/scratch/runs/grand-mean/fear",
            runDirectory: "/scratch/runs/grand-mean",
            name: "fear",
            concept: "fear",
            modelID: "google/gemma-3-27b-it",
            revision: "pinned-revision",
            layerCount: 62,
            hiddenSize: 5376,
            method: nil,
            reading: "mean from token 50",
            residualNormSource: "neutral-corpus",
            hasResidualNorms: true,
            extracted: "2026-07-23",
            stimulusSetHash: Self.composerHashB,
            recipeMethod: "emotionGrandMean",
            extractionMethod: nil,
            readingPosition: "mean from token 50",
            extractionDate: "2026-07-23T12:00:00Z",
            grandMeanPopulation: [
                "calm": "calm-stories-hash",
                "fear": "fear-stories-hash",
            ])

        let verdict = OptimizationComposer.assess(
            OptimizationComposer.facts(for: record))
        let pin = try #require(expectPinnable(verdict))
        #expect(pin.concept == "fear")
        #expect(pin.modelID == "google/gemma-3-27b-it")
        #expect(pin.method == .emotionGrandMean)
        #expect(pin.poolFromToken == 50)
        #expect(pin.corpusConcepts == ["calm", "fear"])
        #expect(pin.currentStimulusHash == ExperimentStore.storiesHash(for: "fear"))
    }

    /// Review 2026-08-02 round 4 (P1): the composer's duplicate collapse
    /// keyed on a hand-selected field subset that omitted the grand-mean
    /// population — two artifacts built from DIFFERENT corpora could be
    /// labeled interchangeable, one hidden. The identity is canonical on
    /// the pin now, and corpus membership (order-insensitive) is part of
    /// it.
    @Test func recipeIdentityIncludesTheGrandMeanPopulation() {
        func pin(corpus: [String], pool: Int? = 50) -> OptimizationComposer.ConceptPin {
            OptimizationComposer.ConceptPin(
                concept: "fear", modelID: "org/m", method: .emotionGrandMean,
                readingPosition: .meanFromToken(50),
                currentStimulusHash: "h", recordedStimulusHash: "h",
                corpusConcepts: corpus, poolFromToken: pool)
        }
        // Same population, different declaration order: interchangeable.
        #expect(pin(corpus: ["calm", "fear"]).recipeIdentity
            == pin(corpus: ["fear", "calm"]).recipeIdentity)
        // Different population: NOT interchangeable.
        #expect(pin(corpus: ["calm", "fear"]).recipeIdentity
            != pin(corpus: ["calm", "fear", "anger"]).recipeIdentity)
        // Different pooling token: NOT interchangeable.
        #expect(pin(corpus: ["calm", "fear"], pool: 50).recipeIdentity
            != pin(corpus: ["calm", "fear"], pool: 1).recipeIdentity)
        // Caution (display state) is NOT identity.
        var cautioned = pin(corpus: ["calm", "fear"])
        cautioned.caution = "stimulus drift"
        #expect(cautioned.recipeIdentity
            == pin(corpus: ["calm", "fear"]).recipeIdentity)
    }

    // MARK: - designatedReference recipes (field diagnosis 2026-08-02)

    /// The optimizer special-cased only grand-mean and fell through to the
    /// paired path for everything else, so every designated-reference
    /// vector — the stance recipe — was falsely refused as "no stimulus
    /// data" (it reads prompts/emotions/, not prompts/concepts/).
    @Test @MainActor func designatedReferenceArtifactsPinWithTheirReference() throws {
        let root = try makeComposerWorkspace()
        defer { cleanupComposerWorkspace(root) }
        try writeComposerStories(
            concept: "crit",
            texts: ["The doctrine masks a distributional choice.",
                    "Neutral principles arrive after the allocation."])
        try writeComposerStories(
            concept: "plain-exposition",
            texts: ["The statute has three sections.",
                    "The court issued its order on Tuesday."])
        let targetHash = try #require(ExperimentStore.storiesHash(for: "crit"))
        let referenceHash = try #require(
            ExperimentStore.storiesHash(for: "plain-exposition"))
        try plantComposerArtifact(
            in: root, concept: "crit",
            stimulusSetHash: targetHash,
            extractionMethod: "designatedReference",
            readingPosition: "mean from token 50",
            designatedReference: ["name": "plain-exposition",
                                  "hash": referenceHash])

        let artifact = try #require(
            VectorCatalog.scan().first { $0.sidecar.concept == "crit" })
        let pin = try #require(expectPinnable(
            OptimizationComposer.assess(
                OptimizationComposer.facts(for: artifact))))
        #expect(pin.method == .designatedReference)
        #expect(pin.currentStimulusHash == targetHash)
        #expect(pin.designatedReferenceName == "plain-exposition")
        #expect(pin.currentReferenceHash == referenceHash)
        #expect(pin.caution == nil)

        // Declare pins a REAL designated-reference concept: target stories
        // hash + the reference beside it (the paired fallthrough omitted
        // the reference entirely).
        try writeComposerNeutralCorpus()
        let plan = try OptimizationComposer.makePlan([pin])
        var spec = ExperimentManifest.SweepSpec()
        spec.selection = ExperimentManifest.SweepSelection(
            objective: .init(metric: "markerDensity"))
        let name = try OptimizationComposer.declare(
            name: "optimize-crit", description: "", plan: plan, spec: spec,
            judgeRubricFile: nil, judges: [], panel: ExperimentPanel())
        let manifest = try ExperimentStore.load(name: name)
        let ref = try #require(manifest.concepts.first)
        #expect(ref.options.method == .designatedReference)
        #expect(ref.stimulusSetHash == targetHash)
        #expect(ref.options.readingPosition == .meanFromToken(50))
        #expect(ref.designatedReference?.name == "plain-exposition")
        #expect(ref.designatedReference?.hash == referenceHash)
        #expect(ExperimentStore.verify(manifest).isEmpty)
    }

    @Test @MainActor func designatedReferenceRefusalsAndDriftCautions() throws {
        let root = try makeComposerWorkspace()
        defer { cleanupComposerWorkspace(root) }
        try writeComposerStories(concept: "crit", texts: ["A story."])
        let targetHash = try #require(ExperimentStore.storiesHash(for: "crit"))

        // No recorded reference (legacy sidecar / old server catalog).
        try plantComposerArtifact(
            in: root, concept: "crit", stimulusSetHash: targetHash,
            extractionMethod: "designatedReference",
            readingPosition: "mean from token 50")
        let bare = try #require(
            VectorCatalog.scan().first { $0.sidecar.concept == "crit" })
        expectRefusal(
            OptimizationComposer.assess(OptimizationComposer.facts(for: bare)),
            containing: "doesn't record its designated reference")

        // Reference recorded but its stories are absent here.
        try FileManager.default.removeItem(
            at: root.appending(components: "runs"))
        try plantComposerArtifact(
            in: root, concept: "crit", stimulusSetHash: targetHash,
            extractionMethod: "designatedReference",
            readingPosition: "mean from token 50",
            designatedReference: ["name": "plain-exposition", "hash": "aa"])
        let missingReference = try #require(
            VectorCatalog.scan().first { $0.sidecar.concept == "crit" })
        expectRefusal(
            OptimizationComposer.assess(
                OptimizationComposer.facts(for: missingReference)),
            containing: "'plain-exposition' has no stimulus data",
            detailContaining: "prompts/emotions/plain-exposition/")

        // Both present, both drifted: BOTH cautions, still pinnable.
        try writeComposerStories(concept: "plain-exposition", texts: ["Plain."])
        try FileManager.default.removeItem(
            at: root.appending(components: "runs"))
        try plantComposerArtifact(
            in: root, concept: "crit", stimulusSetHash: Self.composerHashB,
            extractionMethod: "designatedReference",
            readingPosition: "mean from token 50",
            designatedReference: ["name": "plain-exposition",
                                  "hash": Self.composerHashA])
        let drifted = try #require(
            VectorCatalog.scan().first { $0.sidecar.concept == "crit" })
        let pin = try #require(expectPinnable(
            OptimizationComposer.assess(
                OptimizationComposer.facts(for: drifted))))
        let caution = try #require(pin.caution)
        #expect(caution.contains("stimulus data changed"))
        #expect(caution.contains("reference 'plain-exposition' stories changed"))
        // Different references are DIFFERENT recipes.
        var other = pin
        other.designatedReferenceName = "neutral-stories"
        #expect(other.recipeIdentity != pin.recipeIdentity)
    }
}
