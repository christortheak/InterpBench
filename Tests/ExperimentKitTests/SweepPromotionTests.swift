import CryptoKit
import Foundation
import Testing
@testable import ExperimentKit
@testable import SteeringKit

/// Sweep selection criterion as manifest data + headless Promote — the
/// screening→confirmation lifecycle edge. Pure-CPU: the decision rule is a
/// pure function, promote never loads a model, and schema tests assert the
/// one regression that must never happen (legacy manifests changing content
/// hash because new optional fields leaked into their encoding).
struct SweepSelectionRuleTests {

    // MARK: criterion resolution

    @Test func absentSelectionResolvesToLegacyHardcodedRule() throws {
        let resolved = try SweepSelectionRule.resolve(nil)
        #expect(resolved.metric == "markerDensity")
        #expect(resolved.capabilityTolerance == 0.15)
        #expect(resolved.coherenceFloor == 0.45)
        #expect(resolved.matchedNormRandomMargin == nil)
    }

    @Test func explicitSelectionResolvesAndEmbedsVerbatim() throws {
        let resolved = try SweepSelectionRule.resolve(
            .init(
                objective: .init(metric: "markerDensity"),
                constraints: .init(capabilityTolerance: 0.2, coherenceFloor: 0.5),
                controls: .init(matchedNormRandomMargin: 0.05)))
        #expect(resolved.capabilityTolerance == 0.2)
        #expect(resolved.coherenceFloor == 0.5)
        #expect(resolved.matchedNormRandomMargin == 0.05)
        let criterion = resolved.asCriterion
        #expect(criterion.objective?.metric == "markerDensity")
        #expect(criterion.controls?.matchedNormRandomMargin == 0.05)
    }

    @Test func allDeclaredMetricsNowResolve() throws {
        // These were declared-ahead refusals until the instruments landed
        // (2026-07-08); resolve now accepts all three.
        for metric in ["markerDensity", "judgeScore", "logprobShift"] {
            let resolved = try SweepSelectionRule.resolve(
                .init(objective: .init(metric: metric)))
            #expect(resolved.metric == metric)
        }
    }

    @Test func unknownMetricIsAnError() {
        #expect(throws: ExperimentError.self) {
            try SweepSelectionRule.resolve(.init(objective: .init(metric: "vibes")))
        }
    }

    @Test func criterionRangeValidationFailsAtResolveTime() throws {
        let invalid: [ExperimentManifest.SweepSelection] = [
            // capabilityTolerance must be finite and in [0, 1].
            .init(constraints: .init(capabilityTolerance: 1.5)),
            .init(constraints: .init(capabilityTolerance: -0.01)),
            .init(constraints: .init(capabilityTolerance: .nan)),
            // coherenceFloor must be finite and in [0, 1].
            .init(constraints: .init(coherenceFloor: 1.5)),
            .init(constraints: .init(coherenceFloor: .infinity)),
            .init(constraints: .init(coherenceFloor: .nan)),
            // matchedNormRandomMargin must be finite and ≥ 0.
            .init(controls: .init(matchedNormRandomMargin: -0.1)),
            .init(controls: .init(matchedNormRandomMargin: .nan)),
        ]
        for spec in invalid {
            do {
                _ = try SweepSelectionRule.resolve(spec)
                Issue.record("resolve must refuse out-of-range spec \(spec)")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("must be a finite number"))
            }
        }
        // Boundary values are legal.
        let boundary = try SweepSelectionRule.resolve(
            .init(
                constraints: .init(capabilityTolerance: 0, coherenceFloor: 1),
                controls: .init(matchedNormRandomMargin: 0)))
        #expect(boundary.capabilityTolerance == 0)
        #expect(boundary.coherenceFloor == 1)
        #expect(boundary.matchedNormRandomMargin == 0)
    }

    // MARK: the pure decision rule

    private let baseline = SweepSelectionRule.Baseline(
        metric: 0.02, distinct2: 0.8, batteryAccuracy: 0.9)

    private func cell(
        _ layer: Int, _ alpha: Double, _ metric: Double,
        distinct2: Double = 0.8, accuracy: Double = 0.9
    ) -> SweepSelectionRule.Cell {
        .init(
            layer: layer, alpha: alpha, metric: metric,
            distinct2: distinct2, batteryAccuracy: accuracy)
    }

    @Test func selectsBestEligibleCellByObjective() throws {
        let best = SweepSelectionRule.select(
            cells: [cell(3, 0.1, 0.1), cell(3, 0.2, 0.4), cell(5, 0.1, 0.3)],
            baseline: baseline, criterion: try SweepSelectionRule.resolve(nil))
        #expect(best?.layer == 3)
        #expect(best?.alpha == 0.2)
    }

    @Test func constraintsFilterCells() throws {
        let best = SweepSelectionRule.select(
            cells: [
                cell(3, 0.2, 0.9, accuracy: 0.7),      // capability drop too big
                cell(3, 0.4, 0.8, distinct2: 0.3),     // below coherence floor
                cell(5, 0.1, 0.2),                     // only eligible cell
            ],
            baseline: baseline, criterion: try SweepSelectionRule.resolve(nil))
        #expect(best?.layer == 5)
        #expect(best?.alpha == 0.1)
    }

    @Test func winnerMustExceedBaselineMetric() throws {
        let best = SweepSelectionRule.select(
            cells: [cell(3, 0.1, 0.02)],
            baseline: baseline, criterion: try SweepSelectionRule.resolve(nil))
        #expect(best == nil)
    }

    @Test func controlMarginPassAndFail() {
        #expect(SweepSelectionRule.controlPasses(
            bestMetric: 0.4, controlMetric: 0.1, margin: 0.2))
        #expect(!SweepSelectionRule.controlPasses(
            bestMetric: 0.4, controlMetric: 0.35, margin: 0.2))
        #expect(
            SweepSelectionRule.controlFailureMessage(
                bestMetric: 0.4, controlMetric: 0.35, margin: 0.2)
                .contains("matched-norm control"))
    }

    @Test func controlVectorIsDeterministicAndNormMatched() throws {
        let a = try SweepSelectionRule.controlVector(
            seedText: "c-recommended|c|3", dimension: 16, norm: 2.5)
        let b = try SweepSelectionRule.controlVector(
            seedText: "c-recommended|c|3", dimension: 16, norm: 2.5)
        let other = try SweepSelectionRule.controlVector(
            seedText: "c-recommended|c|4", dimension: 16, norm: 2.5)
        #expect(a == b)
        #expect(a != other)
        #expect(abs(SteeringVectorMath.l2Norm(a) - 2.5) < 1e-4)
    }

    @Test func selectionControlRoundTripsRandomVectorAlgorithmStamp() throws {
        // The sweep stamps the random-control recipe on selection provenance
        // ("gaussian-isotropic-v1", same string on the server); the field
        // must survive a manifest JSON round trip, and a legacy control block
        // WITHOUT the key must decode as unstamped (= cube-uniform on Swift).
        let stamped = ExperimentManifest.SelectionProvenance.Control(
            type: "randomMatchedNorm", metricValue: 0.1, margin: 0.05,
            randomVectorAlgorithm: SteeringVectorMath.randomVectorAlgorithm)
        let decoded = try JSONDecoder().decode(
            ExperimentManifest.SelectionProvenance.Control.self,
            from: JSONEncoder().encode(stamped))
        #expect(decoded.randomVectorAlgorithm == "gaussian-isotropic-v1")

        let legacyJSON = Data(
            #"{"type":"randomMatchedNorm","metricValue":0.1,"margin":0.05}"#.utf8)
        let legacy = try JSONDecoder().decode(
            ExperimentManifest.SelectionProvenance.Control.self, from: legacyJSON)
        #expect(legacy.randomVectorAlgorithm == nil)
    }

    // MARK: objective resolution (judgeScore / logprobShift)

    @Test func baselineMetricPins() {
        // judgeScore's baseline is the 0.5 tie; logprobShift's is 0 — both
        // by construction, so `select` requires the winner to beat them.
        #expect(SweepSelectionRule.baselineMetric("judgeScore", baselineDensity: 0.123) == 0.5)
        #expect(SweepSelectionRule.baselineMetric("logprobShift", baselineDensity: 0.123) == 0)
        #expect(SweepSelectionRule.baselineMetric("markerDensity", baselineDensity: 0.123) == 0.123)
    }

    @Test func meanLogprobShiftIsThePairedMeanOverRows() {
        let base = ["c1": -1.0, "c2": -2.0]
        let cell = ["c1": -0.5, "c2": -1.0]
        #expect(abs(ExperimentTasks.meanLogprobShift(cell: cell, baseline: base) - 0.75) < 1e-12)
        #expect(ExperimentTasks.meanLogprobShift(cell: base, baseline: base) == 0)
        #expect(ExperimentTasks.meanLogprobShift(cell: [:], baseline: [:]) == 0)
    }

    private func tempChoiceRoot(_ jsonl: String?) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "choices-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: root.appending(components: "prompts", "dev"),
            withIntermediateDirectories: true)
        if let jsonl {
            try jsonl.write(
                to: root.appending(components: "prompts", "dev", "choices.jsonl"),
                atomically: true, encoding: .utf8)
        }
        return root
    }

    @Test func logprobShiftChoiceFileRefusalsFireAtResolve() throws {
        // No declared file.
        #expect(throws: ExperimentError.self) {
            try SweepSelectionRule.loadChoiceRows(
                file: nil, root: FileManager.default.temporaryDirectory)
        }
        // Missing file.
        let empty = try tempChoiceRoot(nil)
        #expect(throws: ExperimentError.self) {
            try SweepSelectionRule.loadChoiceRows(
                file: "prompts/dev/choices.jsonl", root: empty)
        }
        // No rows.
        let blank = try tempChoiceRoot("\n")
        #expect(throws: ExperimentError.self) {
            try SweepSelectionRule.loadChoiceRows(
                file: "prompts/dev/choices.jsonl", root: blank)
        }
        // A row without >= 2 options.
        let single = try tempChoiceRoot(
            #"{"id": "c1", "prompt": "p", "options": ["A"]}"# + "\n")
        #expect(throws: ExperimentError.self) {
            try SweepSelectionRule.loadChoiceRows(
                file: "prompts/dev/choices.jsonl", root: single)
        }
        // A target outside its own options.
        let foreign = try tempChoiceRoot(
            #"{"id": "c1", "prompt": "p", "options": ["A", "B"], "target": "C"}"# + "\n")
        #expect(throws: ExperimentError.self) {
            try SweepSelectionRule.loadChoiceRows(
                file: "prompts/dev/choices.jsonl", root: foreign)
        }
    }

    @Test func logprobShiftResolutionPinsFileHashAndTargets() throws {
        let jsonl =
            #"{"id": "c1", "prompt": "Rule?", "options": ["A", "B"], "target": "B"}"# + "\n"
            + #"{"id": "c2", "prompt": "Affirm?", "options": ["A", "B"]}"# + "\n"
        let root = try tempChoiceRoot(jsonl)
        let criterion = try SweepSelectionRule.resolve(
            .init(objective: .init(metric: "logprobShift")))
        let manifest = ExperimentManifest(
            name: "lp", description: "", modelID: "test/model")
        let spec = ExperimentManifest.SweepSelection(
            objective: .init(
                metric: "logprobShift",
                choicePromptsFile: "prompts/dev/choices.jsonl"))
        let objective = try SweepSelectionRule.resolveObjective(
            criterion: criterion, spec: spec, manifest: manifest, root: root)
        // Target designation is the study path's rule: explicit target,
        // else the first option.
        #expect(objective.choiceRows.map(\.target) == ["B", "A"])
        let expectedHash = SHA256.hash(data: Data(jsonl.utf8))
            .map { String(format: "%02x", $0) }.joined()
        #expect(objective.choicePromptsHash == expectedHash)
        let embedded = criterion.asCriterion(objective: objective)
        #expect(embedded.objective?.metric == "logprobShift")
        #expect(embedded.objective?.choicePromptsFile == "prompts/dev/choices.jsonl")
        #expect(embedded.objective?.choicePromptsHash == expectedHash)
        #expect(embedded.objective?.judges == nil)
    }

    @Test func judgeScoreObjectiveRequiresManifestPins() throws {
        let criterion = try SweepSelectionRule.resolve(
            .init(objective: .init(metric: "judgeScore")))
        var manifest = ExperimentManifest(
            name: "js", description: "", modelID: "test/model")
        // No rubric pins.
        do {
            _ = try SweepSelectionRule.resolveObjective(
                criterion: criterion, spec: nil, manifest: manifest)
            Issue.record("judgeScore without rubric pins must refuse")
        } catch let error as ExperimentError {
            #expect(error.reason.contains("pinned judge rubric"))
        }
        // Rubric pinned, no judges.
        manifest.judgeRubricFile = "prompts/rubrics/r.md"
        manifest.judgeRubricHash = String(repeating: "a", count: 64)
        do {
            _ = try SweepSelectionRule.resolveObjective(
                criterion: criterion, spec: nil, manifest: manifest)
            Issue.record("judgeScore without judges must refuse")
        } catch let error as ExperimentError {
            #expect(error.reason.contains("at least one judge"))
        }
    }

    @Test func claudeJudgeWithoutCredentialRefusesAtResolveNamingTheJudge() throws {
        let criterion = try SweepSelectionRule.resolve(
            .init(objective: .init(metric: "judgeScore")))
        var manifest = ExperimentManifest(
            name: "jc", description: "", modelID: "test/model")
        manifest.judgeRubricFile = "prompts/rubrics/r.md"
        manifest.judgeRubricHash = String(repeating: "a", count: 64)
        manifest.judges = [.init(name: "opus-judge", kind: "claude")]
        do {
            _ = try SweepSelectionRule.resolveObjective(
                criterion: criterion, spec: nil, manifest: manifest,
                hasClaudeCredential: false)
            Issue.record("claude judge without credential must refuse at resolve")
        } catch let error as ExperimentError {
            #expect(error.reason.contains("judge 'opus-judge' (claude)"))
            #expect(error.reason.contains("ANTHROPIC_API_KEY"))
        }
    }

    // MARK: the judge-preference objective (fake judges, no network)

    private func fakeJudge(
        name: String, preferring marker: String?
    ) -> ExperimentTasks.SweepJudge {
        .init(name: name) { _, responseA, responseB in
            let winner: String
            if let marker, responseA.contains(marker), !responseB.contains(marker) {
                winner = "A"
            } else if let marker, responseB.contains(marker), !responseA.contains(marker) {
                winner = "B"
            } else {
                winner = "tie"
            }
            let json = #"{"winner": "\#(winner)", "confidence": 1.0, "brief_reason": "t"}"#
            return try JSONDecoder().decode(
                PairedJudgeResponse.self, from: Data(json.utf8))
        }
    }

    @Test func judgePreferenceMapsToUnitIntervalThroughTheBlind() async throws {
        let prompts = ["p1", "p2", "p3"]
        let cell = ["dread one", "dread two", "dread three"]
        let base = ["calm one", "calm two", "calm three"]
        // A judge that recognizes the steered content scores 1.0 whatever
        // the blinded A/B assignment was; a baseline-preferring judge 0.0;
        // an undecided judge pins the 0.5 tie.
        let cellWins = try await ExperimentTasks.sweepJudgePreference(
            experiment: "e", conditionTag: "sweep:fear:L2:a0.4",
            prompts: prompts, cellTexts: cell, baselineTexts: base,
            judges: [fakeJudge(name: "j1", preferring: "dread")])
        #expect(cellWins == 1.0)
        let baseWins = try await ExperimentTasks.sweepJudgePreference(
            experiment: "e", conditionTag: "sweep:fear:L2:a0.4",
            prompts: prompts, cellTexts: cell, baselineTexts: base,
            judges: [fakeJudge(name: "j1", preferring: "calm")])
        #expect(baseWins == 0.0)
        let ties = try await ExperimentTasks.sweepJudgePreference(
            experiment: "e", conditionTag: "sweep:fear:L2:a0.4",
            prompts: prompts, cellTexts: cell, baselineTexts: base,
            judges: [fakeJudge(name: "j1", preferring: nil)])
        #expect(ties == 0.5)
        // No judges / no prompts degenerate to the tie, never a crash.
        let empty = try await ExperimentTasks.sweepJudgePreference(
            experiment: "e", conditionTag: "sweep:fear:L2:a0.4",
            prompts: [], cellTexts: [], baselineTexts: [], judges: [])
        #expect(empty == 0.5)
    }

    @Test func judgePreferenceUnblindsPositionally() async throws {
        // A judge that always answers "A" must score exactly the fraction of
        // items whose blinded assignment put the CELL in slot A — proving
        // the score is unblinded through the same deterministic flip the
        // judge saw.
        let count = 8
        let prompts = (0..<count).map { "p\($0)" }
        let cell = (0..<count).map { "cell \($0)" }
        let base = (0..<count).map { "base \($0)" }
        let alwaysA = ExperimentTasks.SweepJudge(name: "a") { _, _, _ in
            try JSONDecoder().decode(
                PairedJudgeResponse.self,
                from: Data(#"{"winner": "A", "confidence": 1, "brief_reason": "t"}"#.utf8))
        }
        let tag = "sweep:fear:L2:a0.4"
        let score = try await ExperimentTasks.sweepJudgePreference(
            experiment: "e", conditionTag: tag, prompts: prompts,
            cellTexts: cell, baselineTexts: base, judges: [alwaysA])
        let expected = (0..<count).reduce(0.0) { partial, index in
            let flip = ExperimentTasks.shouldFlip(
                experiment: "e", condition: tag, seed: 0,
                promptID: "dev-\(index + 1)")
            return partial + (flip ? 1 : 0)
        } / Double(count)
        #expect(abs(score - expected) < 1e-12)
        // Two judges average over judge × item.
        let two = try await ExperimentTasks.sweepJudgePreference(
            experiment: "e", conditionTag: tag, prompts: prompts,
            cellTexts: cell, baselineTexts: base,
            judges: [
                fakeJudge(name: "j1", preferring: "cell"),
                fakeJudge(name: "j2", preferring: nil),
            ])
        #expect(abs(two - 0.75) < 1e-12)
    }

    @Test func sweepMetricsBlockIsKeyedByObjective() {
        let best = SweepSelectionRule.Cell(
            layer: 3, alpha: 0.4, metric: 0.9, distinct2: 0.8, batteryAccuracy: 1.0)
        let judge = ExperimentTasks.sweepMetricsBlock(
            metric: "judgeScore", best: best, baselineMetric: 0.5,
            baselineDensity: 0.02, baselineAccuracy: 0.95)
        #expect(judge["judgeScore"] == 0.9)
        #expect(judge["baselineJudgeScore"] == 0.5)
        #expect(judge["markerDensity"] == nil)
        let shift = ExperimentTasks.sweepMetricsBlock(
            metric: "logprobShift", best: best, baselineMetric: 0,
            baselineDensity: 0.02, baselineAccuracy: 0.95)
        #expect(shift["logprobShift"] == 0.9)
        #expect(shift["baselineLogprobShift"] == 0)
        // markerDensity keeps its historical keys byte-for-byte.
        let marker = ExperimentTasks.sweepMetricsBlock(
            metric: "markerDensity", best: best, baselineMetric: 0.02,
            baselineDensity: 0.02, baselineAccuracy: 0.95)
        #expect(marker["markerDensity"] == 0.9)
        #expect(marker["baselineDensity"] == 0.02)
        #expect(marker["distinct2"] == 0.8)
        #expect(marker["batteryAccuracy"] == 1.0)
        #expect(marker["baselineBatteryAccuracy"] == 0.95)
    }

    // MARK: schema round-trips and the hash-stability regression

    @Test func legacyEncodingGainsNoNewKeys() throws {
        // The one regression that must never happen: a manifest without the
        // new optional fields must encode WITHOUT them, byte-identically to
        // the pre-change encoder — that is what keeps every existing content
        // hash (and freeze hash) stable.
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let condition = ExperimentManifest.Condition(
            name: "c", slots: [.init(concept: "x", layer: 3, alpha: 0.4)])
        let conditionJSON = String(decoding: try encoder.encode(condition), as: UTF8.self)
        #expect(!conditionJSON.contains("\"selection\""))
        let spec = ExperimentManifest.SweepSpec()
        let specJSON = String(decoding: try encoder.encode(spec), as: UTF8.self)
        #expect(!specJSON.contains("\"selection\""))
        // A metric-only objective (every pre-existing declared selection)
        // must not sprout the new per-metric keys.
        let selection = ExperimentManifest.SweepSelection(
            objective: .init(metric: "markerDensity"))
        let selectionJSON = String(
            decoding: try encoder.encode(selection), as: UTF8.self)
        #expect(!selectionJSON.contains("choicePromptsFile"))
        #expect(!selectionJSON.contains("choicePromptsHash"))
        #expect(!selectionJSON.contains("judgeRubricHash"))
        #expect(!selectionJSON.contains("judges"))
        let variant = ModelVariantArtifact(
            name: "v", baseModelID: "m", promptMode: "chatAssistant",
            qwenThinkingEnabled: false, temperature: 0, systemPrompt: "")
        let variantJSON = String(decoding: try encoder.encode(variant), as: UTF8.self)
        #expect(!variantJSON.contains("\"promotion\""))
    }

    @Test func selectionProvenanceRoundTrips() throws {
        let provenance = ExperimentManifest.SelectionProvenance(
            sweepRun: "20260707T000000000-exp-x-sweep",
            criterion: try SweepSelectionRule.resolve(nil).asCriterion,
            devPromptsHash: String(repeating: "d", count: 64),
            winningCell: .init(layer: 3, alpha: 0.4),
            metrics: ["markerDensity": 0.31, "baselineDensity": 0.02],
            control: .init(type: "randomMatchedNorm", metricValue: 0.05, margin: 0.0))
        let condition = ExperimentManifest.Condition(
            name: "x-recommended", slots: [.init(concept: "x", layer: 3, alpha: 0.4)],
            selection: provenance)
        let data = try JSONEncoder().encode(condition)
        let decoded = try JSONDecoder().decode(
            ExperimentManifest.Condition.self, from: data)
        #expect(decoded.selection == provenance)
    }

    @Test func promotionBlockRoundTripsAndLegacyArtifactsDecode() throws {
        let promotion = ModelVariantArtifact.Promotion(
            experiment: "e", experimentHash: "h", promotedAt: "2026-07-07T00:00:00Z",
            promotedBy: "criterion", sweepRun: "run",
            criterion: try SweepSelectionRule.resolve(nil).asCriterion,
            devPromptsHash: "d", winningCell: .init(layer: 3, alpha: 0.4),
            metrics: ["markerDensity": 0.3], control: nil,
            substrate: "swift-mlx", appVersion: "test")
        let variant = ModelVariantArtifact(
            name: "agent", baseModelID: "m", promptMode: "chatAssistant",
            qwenThinkingEnabled: false, temperature: 0, systemPrompt: "",
            promotion: promotion)
        let decoded = try JSONDecoder().decode(
            ModelVariantArtifact.self, from: JSONEncoder().encode(variant))
        #expect(decoded.promotion == promotion)

        // Legacy artifact JSON (no promotion key) decodes with nil.
        let legacy = """
            {"name": "old", "baseModelID": "m", "injections": [], \
            "promptMode": "chatAssistant"}
            """
        let old = try JSONDecoder().decode(
            ModelVariantArtifact.self, from: Data(legacy.utf8))
        #expect(old.promotion == nil)
    }
}

/// Store-backed promote tests — extends the serialized `ExperimentStoreTests`
/// suite because they use process-global workspace overrides.
extension ExperimentStoreTests {

    private func withTempWorkspace<T>(_ body: (URL) throws -> T) rethrows -> T {
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "promote-\(UUID().uuidString)")
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

    private static let fearHash = String(repeating: "f", count: 64)

    /// Path equality modulo the /var → /private/var symlink: the catalog
    /// walks the (standardized) workspace root, the fixture builds paths from
    /// `temporaryDirectory` — same file, two spellings. The leaf is an
    /// artifact BASE path (no file exists at exactly that path), and
    /// `resolvingSymlinksInPath` bails on nonexistent leaves, so resolve the
    /// existing parent directory and re-append the leaf.
    private func samePath(_ a: String?, _ b: String) -> Bool {
        guard let a else { return false }
        // Through ArtifactIdentity: promote serializes the WORKSPACE-RELATIVE
        // reference (the portable cross-engine shape) while the test helpers
        // hand back absolute catalog ids — the resolver exists precisely to
        // make those compare equal.
        return ArtifactIdentity.canonical(a) == ArtifactIdentity.canonical(b)
    }

    /// A FULL recipe sidecar (readingPosition + neutralProjection +
    /// residualNormSource included) — what the extraction writers stamp —
    /// so the artifact can prove its recipe identity to promote. Pass nils
    /// or `extras` to simulate legacy or divergent artifacts.
    private func plantVectorArtifact(
        in root: URL, run: String = "20260707T000001000-extract",
        stimulusHash: String = fearHash, substrate: String? = "swift-mlx",
        method: String? = "meanDifference", revision: String? = "abc123",
        readingPosition: String? = "last token",
        neutralProjection: String? = "none",
        residualNormSource: String? = "extraction-stimuli",
        extras: [String: Any] = [:]
    ) throws -> String {
        let runDir = root.appending(components: "runs", run)
        try FileManager.default.createDirectory(
            at: runDir, withIntermediateDirectories: true)
        try Data("weights".utf8).write(
            to: runDir.appending(component: "fear.safetensors"))
        var sidecar: [String: Any] = [
            "modelID": "test/model", "concept": "fear",
            "stimulusSetHash": stimulusHash, "layerCount": 4, "hiddenSize": 2,
            "normsPerLayer": [1.0, 1.0, 1.0, 1.0],
            "residualNormPerLayer": [1.0, 1.0, 1.0, 1.0],
            "extractionDate": "2026-07-07T00:00:00Z",
        ]
        if let method { sidecar["extractionMethod"] = method }
        if let substrate { sidecar["substrate"] = substrate }
        if let revision { sidecar["revision"] = revision }
        if let readingPosition { sidecar["readingPosition"] = readingPosition }
        if let neutralProjection { sidecar["neutralProjection"] = neutralProjection }
        if let residualNormSource {
            sidecar["residualNormSource"] = residualNormSource
        }
        for (key, value) in extras { sidecar[key] = value }
        try JSONSerialization.data(withJSONObject: sidecar)
            .write(to: runDir.appending(component: "fear.json"))
        return runDir.appending(component: "fear").path
    }

    private func promotableExperiment(
        name: String, withRecommendation: Bool = true,
        plantConditionRun: Bool = true
    ) throws -> ExperimentManifest.SelectionProvenance {
        var manifest = try ExperimentStore.create(
            name: name, description: "", modelID: "test/model",
            modelRevision: "abc123")
        manifest.concepts.append(
            .init(name: "fear", stimulusSetHash: Self.fearHash, options: .init()))
        let provenance = ExperimentManifest.SelectionProvenance(
            sweepRun: "20260707T000000000-exp-\(name)-sweep",
            criterion: try SweepSelectionRule.resolve(nil).asCriterion,
            devPromptsHash: String(repeating: "d", count: 64),
            winningCell: .init(layer: 3, alpha: 0.4),
            metrics: ["markerDensity": 0.31, "baselineDensity": 0.02])
        if withRecommendation {
            manifest.conditions.append(
                .init(
                    name: "fear-recommended",
                    slots: [.init(concept: "fear", layer: 3, alpha: 0.4)],
                    bandWidth: 1, alphaInNormUnits: true, selection: provenance))
            if plantConditionRun {
                // Promotion verifies the condition's named run completed
                // and still matches (review 2026-08-03 round 3, P2) — a
                // promotable fixture must carry real evidence.
                try plantCompletedConditionRun(provenance: provenance)
            }
        }
        try ExperimentStore.save(manifest)
        return provenance
    }

    /// The completed sweep run a `fear-recommended` condition names:
    /// directory + `recommendations.json` (the completion marker) whose
    /// entry matches — or deliberately mismatches — the condition.
    private func plantCompletedConditionRun(
        provenance: ExperimentManifest.SelectionProvenance,
        winning: ExperimentManifest.SelectionProvenance.Cell? = nil,
        entryStampedRun: String? = nil
    ) throws {
        let directory = ExperimentStore.runsDirectory
            .appending(component: provenance.sweepRun)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        var entry = provenance
        if let winning { entry.winningCell = winning }
        // entryStampedRun forges an entry claiming ANOTHER run — the
        // self-identity gate's test lever (round 5, P2).
        if let entryStampedRun { entry.sweepRun = entryStampedRun }
        let payload = try JSONEncoder().encode(["fear": entry])
        try payload.write(
            to: directory.appending(component: "recommendations.json"))
    }

    @Test func promoteMintsAgentWithBirthCertificate() throws {
        try withTempWorkspace { root in
            let provenance = try promotableExperiment(name: "pr")
            let artifactID = try plantVectorArtifact(in: root)
            let record = try AgentPromotion.promote(
                experimentName: "pr", concept: "fear", log: { _ in })
            let agent = record.artifact
            #expect(agent.name == "pr-fear-agent")
            #expect(agent.baseModelID == "test/model")
            #expect(agent.alphaInNormUnits)
            #expect(agent.injections.count == 1)
            let got = agent.injections.first?.vectorArtifactID
            #expect(samePath(got, artifactID), "got \(got ?? "nil") want \(artifactID)")
            // PORTABLE, like the Python twin's `os.path.relpath`: an absolute
            // catalog id serialized into the artifact named this Mac's
            // filesystem, and every cluster run died resolving it literally
            // (2026-08-04).
            #expect(got?.hasPrefix("/") == false,
                    "vectorArtifactID must be workspace-relative, got \(got ?? "nil")")
            #expect(got?.hasPrefix("runs/") == true)
            #expect(agent.injections.first?.layer == 3)
            #expect(agent.injections.first?.alpha == 0.4)
            let promotion = try #require(agent.promotion)
            #expect(promotion.promotedBy == "criterion")
            #expect(promotion.overrideReason == nil)
            #expect(promotion.experiment == "pr")
            #expect(promotion.sweepRun == provenance.sweepRun)
            #expect(promotion.criterion == provenance.criterion)
            #expect(promotion.metrics == provenance.metrics)
            #expect(promotion.substrate == "swift-mlx")
            let manifest = try ExperimentStore.load(name: "pr")
            #expect(promotion.experimentHash == ExperimentStore.manifestHash(manifest))
            // The certificate records WHICH full recipe the matched artifact
            // satisfied.
            let ref = try #require(manifest.concepts.first)
            let required = try RecipeIdentity.required(manifest: manifest, ref: ref)
            #expect(promotion.recipeIdentityHash == RecipeIdentity.hash(required))
            // The minted artifact is discoverable in the variant library.
            #expect(
                ModelVariantStore.scan().contains { $0.artifact.name == "pr-fear-agent" })
        }
    }

    /// Plants a sweep run directory (sweep.csv + recommendations.json) under
    /// the workspace's runs/, the manual-override path's evidence source.
    private func plantSweepRun(
        in root: URL, run: String, recommendations: String
    ) throws {
        let directory = root.appending(components: "runs", run)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let csv = SweepRunCatalog.csvHeader
            + "\nfear,-1,0,0.01,0.8,1\nfear,3,0.4,0.2,0.7,0.9"
        try csv.write(
            to: directory.appending(component: "sweep.csv"),
            atomically: true, encoding: .utf8)
        try recommendations.write(
            to: directory.appending(component: "recommendations.json"),
            atomically: true, encoding: .utf8)
    }

    @Test func promoteCellOverrideIsLoudAndStamped() throws {
        try withTempWorkspace { root in
            _ = try promotableExperiment(name: "ov")
            _ = try plantVectorArtifact(in: root)
            var messages: [String] = []
            let record = try AgentPromotion.promote(
                experimentName: "ov", concept: "fear",
                cell: (layer: 5, alpha: 0.2),
                overrideReason: "confirming the adjacent layer on purpose",
                log: { messages.append($0) })
            let promotion = try #require(record.artifact.promotion)
            #expect(promotion.promotedBy == "manualOverride")
            #expect(promotion.overrideReason == "confirming the adjacent layer on purpose")
            #expect(promotion.winningCell == .init(layer: 5, alpha: 0.2))
            // Declared criterion still travels (what the override deviated
            // from) but the winning cell's metrics do not.
            #expect(promotion.criterion != nil)
            #expect(promotion.metrics == nil)
            #expect(messages.contains { $0.contains("manualOverride") })
        }
    }

    // Review 2026-08-03 round 3, P2: a `<concept>-recommended` condition is
    // a PROJECTION of a sweep run, never evidence in itself — the criterion
    // path verifies the named run completed and still matches. Server twins:
    // test_promote_refuses_condition_whose_sweep_run_is_missing et al.

    @Test func promoteRefusesConditionWhoseSweepRunIsMissing() throws {
        try withTempWorkspace { root in
            _ = try promotableExperiment(name: "gm", plantConditionRun: false)
            _ = try plantVectorArtifact(in: root)
            do {
                _ = try AgentPromotion.promote(
                    experimentName: "gm", concept: "fear", log: { _ in })
                Issue.record("a condition naming a missing run must refuse")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("which is not in runs/"))
            }
        }
    }

    @Test func promoteRefusesConditionWhoseSweepNeverCompleted() throws {
        try withTempWorkspace { root in
            let provenance = try promotableExperiment(
                name: "gi", plantConditionRun: false)
            _ = try plantVectorArtifact(in: root)
            // The save-before-marker crash window: run directory exists,
            // completion marker never landed.
            try FileManager.default.createDirectory(
                at: ExperimentStore.runsDirectory
                    .appending(component: provenance.sweepRun),
                withIntermediateDirectories: true)
            do {
                _ = try AgentPromotion.promote(
                    experimentName: "gi", concept: "fear", log: { _ in })
                Issue.record("a marker-less run must refuse promotion")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("that sweep never completed"))
            }
        }
    }

    @Test func promoteRefusesConditionStaleAgainstItsRun() throws {
        try withTempWorkspace { root in
            let provenance = try promotableExperiment(
                name: "gs", plantConditionRun: false)
            _ = try plantVectorArtifact(in: root)
            try plantCompletedConditionRun(
                provenance: provenance,
                winning: .init(layer: 5, alpha: 0.1))
            do {
                _ = try AgentPromotion.promote(
                    experimentName: "gs", concept: "fear", log: { _ in })
                Issue.record("a stale condition must refuse promotion")
            } catch let error as ExperimentError {
                #expect(
                    error.reason.contains("the manifest condition is stale"))
            }
        }
    }

    @Test func parseCSVSplitsCRLFServerFiles() throws {
        // Field incident 2026-08-04: the server's csv module writes CRLF,
        // and Swift's "\r\n" is ONE grapheme — a literal "\n" split saw
        // the whole file as a single header line, so every server-executed
        // sweep rendered an EMPTY grid with no error anywhere.
        let crlf = "concept,layer,alpha,markerDensity,distinct2,words,batteryAccuracy,objective\r\n"
            + "fear,-1,0,0.0,0.98,58.0,0.9,0.0\r\n"
            + "fear,31,0.2,0.0,0.98,54.5,0.9,2.24\r\n"
        let rows = try SweepRunCatalog.parseCSV(crlf)
        #expect(rows.count == 2)
        #expect(rows.last?.layer == 31)
        #expect(rows.last?.objective == 2.24)
        // The Swift-written LF form stays equivalent.
        let lf = crlf.replacingOccurrences(of: "\r\n", with: "\n")
        #expect(try SweepRunCatalog.parseCSV(lf).count == 2)
    }

    @Test func promoteFailsClosedWithoutSweepRunStamp() throws {
        // Round 4, P2: both engines' schemas stamp sweepRun — a condition
        // without one is hand-written or legacy, not criterion evidence.
        try withTempWorkspace { root in
            var manifest = try ExperimentStore.create(
                name: "glx", description: "", modelID: "test/model",
                modelRevision: "abc123")
            manifest.concepts.append(
                .init(name: "fear", stimulusSetHash: Self.fearHash,
                      options: .init()))
            let unstamped = ExperimentManifest.SelectionProvenance(
                sweepRun: "",
                criterion: try SweepSelectionRule.resolve(nil).asCriterion,
                devPromptsHash: String(repeating: "d", count: 64),
                winningCell: .init(layer: 3, alpha: 0.4),
                metrics: ["markerDensity": 0.31])
            manifest.conditions.append(
                .init(name: "fear-recommended",
                      slots: [.init(concept: "fear", layer: 3, alpha: 0.4)],
                      bandWidth: 1, alphaInNormUnits: true,
                      selection: unstamped))
            try ExperimentStore.save(manifest)
            _ = try plantVectorArtifact(in: root)
            do {
                _ = try AgentPromotion.promote(
                    experimentName: "glx", concept: "fear", log: { _ in })
                Issue.record("an unstamped condition must fail closed")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("carries no sweepRun stamp"))
            }
        }
    }

    @Test func promoteRefusesSweepRunThatIsNotAPlainName() throws {
        // Round 4, P2: the run name joins onto runs/ — traversal refuses.
        try withTempWorkspace { root in
            var manifest = try ExperimentStore.create(
                name: "gtx", description: "", modelID: "test/model",
                modelRevision: "abc123")
            manifest.concepts.append(
                .init(name: "fear", stimulusSetHash: Self.fearHash,
                      options: .init()))
            let traversal = ExperimentManifest.SelectionProvenance(
                sweepRun: "../../etc",
                criterion: try SweepSelectionRule.resolve(nil).asCriterion,
                devPromptsHash: String(repeating: "d", count: 64),
                winningCell: .init(layer: 3, alpha: 0.4),
                metrics: ["markerDensity": 0.31])
            manifest.conditions.append(
                .init(name: "fear-recommended",
                      slots: [.init(concept: "fear", layer: 3, alpha: 0.4)],
                      bandWidth: 1, alphaInNormUnits: true,
                      selection: traversal))
            try ExperimentStore.save(manifest)
            _ = try plantVectorArtifact(in: root)
            do {
                _ = try AgentPromotion.promote(
                    experimentName: "gtx", concept: "fear", log: { _ in })
                Issue.record("a traversal run name must refuse")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("not a plain run name"))
            }
        }
    }

    @Test func promoteCertificateCopiesFromTheRunNotTheCondition() throws {
        // Round 4, P1: same cell, drifted provenance — the certificate must
        // carry the RUN's criterion/metrics, never the condition's claims,
        // and the drift is logged loudly.
        try withTempWorkspace { root in
            var manifest = try ExperimentStore.create(
                name: "gcx", description: "", modelID: "test/model",
                modelRevision: "abc123")
            manifest.concepts.append(
                .init(name: "fear", stimulusSetHash: Self.fearHash,
                      options: .init()))
            let trueProvenance = ExperimentManifest.SelectionProvenance(
                sweepRun: "20260707T000000000-exp-gcx-sweep",
                criterion: try SweepSelectionRule.resolve(nil).asCriterion,
                devPromptsHash: String(repeating: "d", count: 64),
                winningCell: .init(layer: 3, alpha: 0.4),
                metrics: ["markerDensity": 0.31, "baselineDensity": 0.02])
            var tampered = trueProvenance
            tampered.metrics = ["markerDensity": 999]
            manifest.conditions.append(
                .init(name: "fear-recommended",
                      slots: [.init(concept: "fear", layer: 3, alpha: 0.4)],
                      bandWidth: 1, alphaInNormUnits: true,
                      selection: tampered))
            try ExperimentStore.save(manifest)
            try plantCompletedConditionRun(provenance: trueProvenance)
            _ = try plantVectorArtifact(in: root)
            var logs: [String] = []
            let record = try AgentPromotion.promote(
                experimentName: "gcx", concept: "fear",
                log: { logs.append($0) })
            let promotion = try #require(record.artifact.promotion)
            #expect(promotion.metrics == trueProvenance.metrics)
            #expect(logs.contains { $0.contains("differs from sweep run") })
        }
    }

    @Test func promoteRefusesRecommendationNotNamingItsOwnRun() throws {
        // Self-identity (round 5, P2): a directory whose entry stamps a
        // DIFFERENT run is a copied or edited directory — never evidence.
        try withTempWorkspace { root in
            let provenance = try promotableExperiment(
                name: "gid", plantConditionRun: false)
            _ = try plantVectorArtifact(in: root)
            try plantCompletedConditionRun(
                provenance: provenance,
                entryStampedRun: "20260707T999999999-exp-other-sweep")
            do {
                _ = try AgentPromotion.promote(
                    experimentName: "gid", concept: "fear", log: { _ in })
                Issue.record("an entry naming another run must refuse")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("does not name itself"))
            }
        }
    }

    @Test func promotePinnedContractValidatesThePlainRunName() throws {
        // Round 5, P1: the pinned path appends its run name onto runs/ —
        // traversal refuses through the same shared helper.
        try withTempWorkspace { root in
            _ = try promotableExperiment(name: "gpt")
            _ = try plantVectorArtifact(in: root)
            do {
                _ = try AgentPromotion.promote(
                    experimentName: "gpt", concept: "fear",
                    pins: .init(sweepRun: "../../etc"), log: { _ in })
                Issue.record("a traversal pinned run name must refuse")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("not a plain run name"))
            }
        }
    }

    @Test func ambientFallbackRefusesRecommendationNotNamingItsRun() throws {
        // Round 6, P1: the ambient newest-run fallback (criterion path with
        // no projected condition — the frozen-manifest shape — and the
        // manual-override gate) applies the same self-identity check as
        // every other evidence path. Refusal is loud, never a silent skip.
        try withTempWorkspace { root in
            _ = try promotableExperiment(
                name: "gaf", withRecommendation: false)
            _ = try plantVectorArtifact(in: root)
            let forged = ExperimentManifest.SelectionProvenance(
                sweepRun: "20260101T000000000-exp-other-sweep",
                criterion: try SweepSelectionRule.resolve(nil).asCriterion,
                devPromptsHash: String(repeating: "d", count: 64),
                winningCell: .init(layer: 3, alpha: 0.4),
                metrics: [:])
            let payload = try JSONEncoder().encode(["fear": forged])
            try plantSweepRun(
                in: root, run: "20260707T000000000-exp-gaf-sweep",
                recommendations: String(decoding: payload, as: UTF8.self))
            do {
                _ = try AgentPromotion.promote(
                    experimentName: "gaf", concept: "fear", log: { _ in })
                Issue.record("the ambient fallback must refuse a forged entry")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("does not name itself"))
            }
            do {
                _ = try AgentPromotion.promote(
                    experimentName: "gaf", concept: "fear",
                    cell: (layer: 5, alpha: 0.2),
                    overrideReason: "deliberate deviation", log: { _ in })
                Issue.record("the override gate must refuse a forged entry")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("does not name itself"))
            }
        }
    }

    @Test func promoteCellOverrideWithoutAnySweepRefuses() throws {
        try withTempWorkspace { root in
            _ = try promotableExperiment(name: "nosweep", withRecommendation: false)
            _ = try plantVectorArtifact(in: root)
            do {
                _ = try AgentPromotion.promote(
                    experimentName: "nosweep", concept: "fear",
                    cell: (layer: 5, alpha: 0.2), log: { _ in })
                Issue.record("override with no sweep at all must refuse")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("no sweep has run for 'fear'"))
                #expect(error.reason.contains("cannot replace one"))
            }
        }
    }

    @Test func promoteCellOverrideAfterFailedSweepStampsOutcome() throws {
        try withTempWorkspace { root in
            _ = try promotableExperiment(name: "failsweep", withRecommendation: false)
            _ = try plantVectorArtifact(in: root)
            let runName = "20260707T000010000-exp-failsweep-sweep"
            try plantSweepRun(
                in: root, run: runName,
                recommendations:
                    #"{"fear": "no cell passed the capability/coherence gates"}"#)
            let record = try AgentPromotion.promote(
                experimentName: "failsweep", concept: "fear",
                cell: (layer: 5, alpha: 0.2),
                overrideReason: "confirming despite the failed gates",
                log: { _ in })
            let promotion = try #require(record.artifact.promotion)
            #expect(promotion.promotedBy == "manualOverride")
            #expect(promotion.sweepRun == runName)
            #expect(
                promotion.selectionOutcome
                    == "no cell passed the capability/coherence gates")
            #expect(promotion.criterion == nil)
            #expect(promotion.metrics == nil)
        }
    }

    @Test func promoteCellOverrideCopiesProvenanceFromSweepRunEntry() throws {
        try withTempWorkspace { root in
            _ = try promotableExperiment(name: "runprov", withRecommendation: false)
            _ = try plantVectorArtifact(in: root)
            let runName = "20260707T000011000-exp-runprov-sweep"
            let devHash = String(repeating: "d", count: 64)
            try plantSweepRun(
                in: root, run: runName,
                recommendations: """
                    {"fear": {"sweepRun": "\(runName)",
                              "criterion": {"objective": {"metric": "markerDensity"},
                                            "constraints": {"capabilityTolerance": 0.15,
                                                            "coherenceFloor": 0.45}},
                              "devPromptsHash": "\(devHash)",
                              "winningCell": {"layer": 3, "alpha": 0.4},
                              "metrics": {"markerDensity": 0.31}}}
                    """)
            // Same cell as the sweep's winner: the winner's metrics travel.
            let same = try AgentPromotion.promote(
                experimentName: "runprov", concept: "fear",
                cell: (layer: 3, alpha: 0.4), log: { _ in })
            let samePromotion = try #require(same.artifact.promotion)
            #expect(samePromotion.sweepRun == runName)
            #expect(samePromotion.devPromptsHash == devHash)
            #expect(samePromotion.criterion != nil)
            #expect(samePromotion.metrics == ["markerDensity": 0.31])
            #expect(samePromotion.selectionOutcome == nil)
            // A DIFFERENT cell: criterion/context travel, metrics do not.
            let other = try AgentPromotion.promote(
                experimentName: "runprov", concept: "fear",
                agentName: "runprov-other", cell: (layer: 5, alpha: 0.2),
                log: { _ in })
            let otherPromotion = try #require(other.artifact.promotion)
            #expect(otherPromotion.sweepRun == runName)
            #expect(otherPromotion.criterion != nil)
            #expect(otherPromotion.metrics == nil)
        }
    }

    /// A sweep on a FROZEN manifest cannot stamp `<concept>-recommended` —
    /// its selection lives only in the run's recommendations.json. Criterion
    /// promotion (no --cell) must fall back to that evidence and promote it
    /// as `promotedBy: "criterion"` with the full provenance copied.
    /// (Simulated the way it manifests: no stamped condition, run evidence
    /// present — the mechanism is the missing condition, not the status.)
    @Test func promoteCriterionFallsBackToSweepRunEvidence() throws {
        try withTempWorkspace { root in
            _ = try promotableExperiment(name: "frozenrec", withRecommendation: false)
            _ = try plantVectorArtifact(in: root)
            let runName = "20260707T000012000-exp-frozenrec-sweep"
            let devHash = String(repeating: "d", count: 64)
            try plantSweepRun(
                in: root, run: runName,
                recommendations: """
                    {"fear": {"sweepRun": "\(runName)",
                              "criterion": {"objective": {"metric": "markerDensity"},
                                            "constraints": {"capabilityTolerance": 0.15,
                                                            "coherenceFloor": 0.45}},
                              "devPromptsHash": "\(devHash)",
                              "winningCell": {"layer": 3, "alpha": 0.4},
                              "metrics": {"markerDensity": 0.31},
                              "control": {"type": "randomMatchedNorm",
                                          "metricValue": 0.05, "margin": 0.0}}}
                    """)
            let record = try AgentPromotion.promote(
                experimentName: "frozenrec", concept: "fear", log: { _ in })
            let agent = record.artifact
            // The sweep writes norm-unit cells: run-evidence defaults.
            #expect(agent.injections.first?.layer == 3)
            #expect(agent.injections.first?.alpha == 0.4)
            #expect(agent.bandWidth == 1)
            #expect(agent.alphaInNormUnits)
            let promotion = try #require(agent.promotion)
            #expect(promotion.promotedBy == "criterion")
            #expect(promotion.overrideReason == nil)
            #expect(promotion.sweepRun == runName)
            #expect(promotion.devPromptsHash == devHash)
            #expect(promotion.criterion?.objective?.metric == "markerDensity")
            #expect(promotion.winningCell == .init(layer: 3, alpha: 0.4))
            #expect(promotion.metrics == ["markerDensity": 0.31])
            #expect(promotion.control?.type == "randomMatchedNorm")
            #expect(promotion.selectionOutcome == nil)
        }
    }

    /// A failure entry is a sweep that RAN and selected nothing: criterion
    /// promotion must refuse with the sweep's own conclusion — only a loud
    /// manual override can promote past it.
    @Test func promoteCriterionRefusesOnFailedSweepEvidence() throws {
        try withTempWorkspace { root in
            _ = try promotableExperiment(name: "failcrit", withRecommendation: false)
            _ = try plantVectorArtifact(in: root)
            try plantSweepRun(
                in: root, run: "20260707T000013000-exp-failcrit-sweep",
                recommendations:
                    #"{"fear": "no cell passed the capability/coherence gates"}"#)
            do {
                _ = try AgentPromotion.promote(
                    experimentName: "failcrit", concept: "fear", log: { _ in })
                Issue.record("criterion promotion after a failed selection must refuse")
            } catch let error as ExperimentError {
                #expect(error.reason.contains(
                    "the sweep selected no cell for 'fear'"))
                #expect(error.reason.contains(
                    "no cell passed the capability/coherence gates"))
                #expect(error.reason.contains("manual override"))
            }
        }
    }

    /// When BOTH a stamped manifest condition and newer run evidence exist,
    /// the manifest condition wins — it is the declared, saved state.
    @Test func promoteCriterionPrefersManifestConditionOverRunEvidence() throws {
        try withTempWorkspace { root in
            let provenance = try promotableExperiment(name: "prec")
            _ = try plantVectorArtifact(in: root)
            let newerRun = "20260708T000000000-exp-prec-sweep"
            try plantSweepRun(
                in: root, run: newerRun,
                recommendations: """
                    {"fear": {"sweepRun": "\(newerRun)",
                              "criterion": {"objective": {"metric": "markerDensity"}},
                              "devPromptsHash": "\(String(repeating: "e", count: 64))",
                              "winningCell": {"layer": 2, "alpha": 0.1},
                              "metrics": {"markerDensity": 0.99}}}
                    """)
            let record = try AgentPromotion.promote(
                experimentName: "prec", concept: "fear", log: { _ in })
            let promotion = try #require(record.artifact.promotion)
            #expect(promotion.sweepRun == provenance.sweepRun)
            #expect(promotion.sweepRun != newerRun)
            #expect(promotion.winningCell == .init(layer: 3, alpha: 0.4))
            #expect(promotion.metrics == provenance.metrics)
            #expect(record.artifact.injections.first?.layer == 3)
            #expect(record.artifact.injections.first?.alpha == 0.4)
        }
    }

    @Test func promoteRequiresASweepRecommendation() throws {
        try withTempWorkspace { root in
            _ = try promotableExperiment(name: "norec", withRecommendation: false)
            _ = try plantVectorArtifact(in: root)
            #expect(throws: ExperimentError.self) {
                try AgentPromotion.promote(
                    experimentName: "norec", concept: "fear", log: { _ in })
            }
        }
    }

    @Test func promoteRequiresAMatchingArtifact() throws {
        try withTempWorkspace { root in
            _ = try promotableExperiment(name: "noart")
            // Wrong stimulus hash: recipe identity fails, never a fallback.
            _ = try plantVectorArtifact(
                in: root, stimulusHash: String(repeating: "0", count: 64))
            #expect(throws: ExperimentError.self) {
                try AgentPromotion.promote(
                    experimentName: "noart", concept: "fear", log: { _ in })
            }
        }
    }

    /// The live bug (2026-07-08): a successful sweep derived vectors in
    /// memory but never persisted them, so Create Agent refused with "no
    /// extraction artifact … matches this experiment's recipe". `sweep` now
    /// persists through the SAME sidecar writer extract/validate use
    /// (`extractAll(into: runDirectory)`), so a sweep run directory carrying
    /// the artifact must satisfy promote with NO separate extract run.
    @Test func promoteMatchesArtifactPersistedInSweepRunDirectory() throws {
        try withTempWorkspace { root in
            _ = try promotableExperiment(name: "swpro", withRecommendation: false)
            let runName = "20260708T000020000-exp-swpro-sweep"
            let devHash = String(repeating: "d", count: 64)
            try plantSweepRun(
                in: root, run: runName,
                recommendations: """
                    {"fear": {"sweepRun": "\(runName)",
                              "criterion": {"objective": {"metric": "judgeScore"}},
                              "devPromptsHash": "\(devHash)",
                              "winningCell": {"layer": 3, "alpha": 0.4},
                              "metrics": {"judgeScore": 0.83,
                                          "baselineJudgeScore": 0.5}}}
                    """)
            // The vector artifact lives INSIDE the sweep run directory — the
            // post-fix layout — and is the workspace's ONLY artifact.
            let artifactID = try plantVectorArtifact(in: root, run: runName)
            let record = try AgentPromotion.promote(
                experimentName: "swpro", concept: "fear", log: { _ in })
            let got = record.artifact.injections.first?.vectorArtifactID
            #expect(samePath(got, artifactID), "got \(got ?? "nil") want \(artifactID)")
            #expect(got?.contains(runName) == true)
            let promotion = try #require(record.artifact.promotion)
            #expect(promotion.promotedBy == "criterion")
            #expect(promotion.sweepRun == runName)
            #expect(promotion.metrics?["judgeScore"] == 0.83)
        }
    }

    @Test func promoteSkipsForeignSubstrateArtifactsAndPrefersNewest() throws {
        try withTempWorkspace { root in
            _ = try promotableExperiment(name: "pick")
            // A foreign-substrate artifact never matches…
            _ = try plantVectorArtifact(
                in: root, run: "20260707T000003000-extract",
                substrate: "python-hf-transformers")
            // …and among native matches, the newest run wins.
            _ = try plantVectorArtifact(in: root, run: "20260707T000001000-extract")
            let newest = try plantVectorArtifact(
                in: root, run: "20260707T000002000-extract")
            let record = try AgentPromotion.promote(
                experimentName: "pick", concept: "fear", log: { _ in })
            #expect(samePath(record.artifact.injections.first?.vectorArtifactID, newest))
        }
    }

    /// A legacy artifact (six-field sidecar: no reading position, projection,
    /// or norm source) can no longer be silently matched: promote refuses,
    /// NAMES every recipe field the artifact cannot prove, and says how to
    /// fix it — never a fallback to the old partial-field match.
    @Test func promoteRefusesLegacyArtifactNamingUnprovableFields() throws {
        try withTempWorkspace { root in
            _ = try promotableExperiment(name: "legacyart")
            _ = try plantVectorArtifact(
                in: root, readingPosition: nil, neutralProjection: nil,
                residualNormSource: nil)
            do {
                _ = try AgentPromotion.promote(
                    experimentName: "legacyart", concept: "fear", log: { _ in })
                Issue.record("legacy artifact must refuse, never partially match")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("cannot prove recipe fields"))
                #expect(error.reason.contains(
                    "[neutralProjection, readingPosition, residualNormSource]"))
                #expect(error.reason.contains("re-extract to stamp the full recipe"))
            }
        }
    }

    /// A near-miss candidate must be actionable: the refusal compares the
    /// canonical identity components and NAMES every differing field with
    /// both values — "revision (manifest: …, artifact: …)" — never just "a
    /// DIFFERENT recipe identity" (the 2026-07-14 lesson: an opaque refusal
    /// hid a one-field revision mismatch on the server). Same wording shape
    /// as the server's refusal.
    @Test func promoteRefusalNamesDifferingRecipeFieldsPerCandidate() throws {
        try withTempWorkspace { root in
            _ = try promotableExperiment(name: "fdiff")  // revision "abc123"
            // Candidate 1: only the revision differs (the server bug's shape).
            _ = try plantVectorArtifact(
                in: root, run: "20260707T000005000-extract",
                revision: "def456789012345678")
            // Candidate 2: right revision, divergent projection recipe.
            _ = try plantVectorArtifact(
                in: root, run: "20260707T000006000-extract",
                neutralProjection: "legacy-pooled top-3 neutral PCs")
            do {
                _ = try AgentPromotion.promote(
                    experimentName: "fdiff", concept: "fear", log: { _ in })
                Issue.record("divergent candidates must refuse")
            } catch let error as ExperimentError {
                let message = error.reason
                // Artifact ids are absolute; assert on the run-relative tail
                // (the catalog standardizes /var → /private/var).
                #expect(message.contains(
                    "20260707T000005000-extract/fear' carries a DIFFERENT "
                        + "recipe identity (never matched by recency) — "
                        + "revision (manifest: abc123, artifact: def456789012…)"))
                #expect(message.contains("20260707T000006000-extract/fear'"))
                #expect(message.contains(
                    "neutralProjection.mode (manifest: none, artifact: legacyPooled)"))
                #expect(message.contains(
                    "neutralProjection.count (manifest: null, artifact: 3)"))
                // The old bare counter phrasing is gone.
                #expect(!message.contains(
                    "candidate artifact(s) carry a DIFFERENT recipe identity"))
            }
        }
    }

    /// A stamped-but-mismatched candidate (path (a)) still gets a field
    /// diff, computed from its recorded sidecar fields.
    @Test func promoteRefusalDiffsAStampedButMismatchedCandidate() throws {
        try withTempWorkspace { root in
            _ = try promotableExperiment(name: "fstamp")
            let sidecarComponents = RecipeIdentity.Components(
                concept: "fear", modelID: "test/model",
                revision: "def456789012345678",
                extractionMethod: "meanDifference",
                stimulusSetHash: Self.fearHash,
                readingPositionMode: "lastToken", readingPositionParameter: nil,
                projectionMode: "none", projectionCount: nil,
                projectionExplainedVariance: nil, projectionBasisHash: nil,
                residualNormSource: "extraction-stimuli", normCorpusHash: nil,
                grandMeanPopulation: nil)
            _ = try plantVectorArtifact(
                in: root, run: "20260707T000007000-extract",
                revision: "def456789012345678",
                extras: ["recipeIdentityHash": RecipeIdentity.hash(sidecarComponents)])
            do {
                _ = try AgentPromotion.promote(
                    experimentName: "fstamp", concept: "fear", log: { _ in })
                Issue.record("stamped mismatch must refuse with a field diff")
            } catch let error as ExperimentError {
                // Run-relative tail: the catalog standardizes /var →
                // /private/var, so the absolute fixture path may differ.
                #expect(error.reason.contains(
                    "20260707T000007000-extract/fear' carries a DIFFERENT "
                        + "recipe identity"))
                #expect(error.reason.contains(
                    "revision (manifest: abc123, artifact: def456789012…)"))
            }
        }
    }

    /// Swift never had the server's revision asymmetry —
    /// `loadContainer(pinning:)` writes the resolved revision into a draft
    /// manifest before extraction, and `saveVectorSidecar` stamps the
    /// MANIFEST's revision — so a revision-less draft and its own artifacts
    /// agree (null == null) at the promotion-matching level.
    @Test func promoteMatchesWhenManifestAndArtifactBothLackARevision() throws {
        try withTempWorkspace { root in
            var manifest = try ExperimentStore.create(
                name: "norev", description: "", modelID: "test/model")
            manifest.concepts.append(
                .init(name: "fear", stimulusSetHash: Self.fearHash, options: .init()))
            manifest.conditions.append(
                .init(
                    name: "fear-recommended",
                    slots: [.init(concept: "fear", layer: 3, alpha: 0.4)],
                    bandWidth: 1, alphaInNormUnits: true,
                    selection: .init(
                        sweepRun: "20260707T000000000-exp-norev-sweep",
                        criterion: try SweepSelectionRule.resolve(nil).asCriterion,
                        devPromptsHash: String(repeating: "d", count: 64),
                        winningCell: .init(layer: 3, alpha: 0.4),
                        metrics: [:])))
            try ExperimentStore.save(manifest)
            try plantCompletedConditionRun(
                provenance: #require(
                    manifest.conditions.first?.selection))
            let artifactID = try plantVectorArtifact(in: root, revision: nil)
            let record = try AgentPromotion.promote(
                experimentName: "norev", concept: "fear", log: { _ in })
            #expect(samePath(record.artifact.injections.first?.vectorArtifactID,
                             artifactID))
        }
    }

    /// Newest-wins must never break a tie between DIFFERENT recipes: a newer
    /// artifact whose reading position (or projection, or norm source)
    /// differs is excluded, and the older exact-identity match wins.
    @Test func promoteNeverLetsANewerDifferentRecipeWin() throws {
        try withTempWorkspace { root in
            _ = try promotableExperiment(name: "tiebreak")
            let matching = try plantVectorArtifact(
                in: root, run: "20260707T000001000-extract")
            // Newer, but a different recipe on each newly covered axis.
            _ = try plantVectorArtifact(
                in: root, run: "20260707T000002000-extract",
                readingPosition: "mean from token 50")
            _ = try plantVectorArtifact(
                in: root, run: "20260707T000003000-extract",
                neutralProjection: "legacy-pooled top-3 neutral PCs")
            _ = try plantVectorArtifact(
                in: root, run: "20260707T000004000-extract",
                residualNormSource: "neutral-corpus",
                extras: ["neutralCorpusHash": String(repeating: "0", count: 64)])
            let record = try AgentPromotion.promote(
                experimentName: "tiebreak", concept: "fear", log: { _ in })
            #expect(samePath(record.artifact.injections.first?.vectorArtifactID, matching))
        }
    }

    /// Path (a): an artifact stamped with the exact required
    /// `recipeIdentityHash` matches on the stamp — the normal case for every
    /// artifact the extraction writers produce going forward.
    @Test func promoteMatchesStampedRecipeIdentityHash() throws {
        try withTempWorkspace { root in
            _ = try promotableExperiment(name: "stamped")
            let manifest = try ExperimentStore.load(name: "stamped")
            let ref = try #require(manifest.concepts.first)
            let requiredHash = RecipeIdentity.hash(
                try RecipeIdentity.required(manifest: manifest, ref: ref))
            // Deliberately sparse otherwise (legacy-shaped): the stamp alone
            // carries the proof.
            let artifactID = try plantVectorArtifact(
                in: root, readingPosition: nil, neutralProjection: nil,
                residualNormSource: nil,
                extras: ["recipeIdentityHash": requiredHash])
            let record = try AgentPromotion.promote(
                experimentName: "stamped", concept: "fear", log: { _ in })
            #expect(samePath(record.artifact.injections.first?.vectorArtifactID, artifactID))
            #expect(record.artifact.promotion?.recipeIdentityHash == requiredHash)
            // A WRONG stamp is a different identity, not a fallback candidate.
            _ = try plantVectorArtifact(
                in: root, run: "20260707T000009000-extract",
                readingPosition: nil, neutralProjection: nil,
                residualNormSource: nil,
                extras: ["recipeIdentityHash": String(repeating: "e", count: 64)])
            let again = try AgentPromotion.promote(
                experimentName: "stamped", concept: "fear",
                agentName: "stamped-2", log: { _ in })
            #expect(samePath(again.artifact.injections.first?.vectorArtifactID, artifactID))
        }
    }

    /// Grand-mean promotion requires the artifact to prove the FULL
    /// comparison population (every member + stories hash); an artifact
    /// without the stamp refuses, one with a matching population promotes —
    /// including across the Swift/server norm-source label unification.
    @Test func promoteGrandMeanMatchesOnTheFullPopulation() throws {
        try withTempWorkspace { root in
            var manifest = try ExperimentStore.create(
                name: "gm", description: "", modelID: "test/model",
                modelRevision: "abc123")
            manifest.concepts.append(
                .init(
                    name: "fear", stimulusSetHash: Self.fearHash,
                    options: .init(method: .emotionGrandMean)))
            let population = [
                "fear": Self.fearHash,
                "joy": String(repeating: "a", count: 64),
                "anger": String(repeating: "b", count: 64),
            ]
            manifest.grandMeanCorpus = .init(
                concepts: ["fear", "joy", "anger"], hashes: population)
            manifest.conditions.append(
                .init(
                    name: "fear-recommended",
                    slots: [.init(concept: "fear", layer: 3, alpha: 0.4)],
                    bandWidth: 1, alphaInNormUnits: true,
                    selection: .init(
                        sweepRun: "20260707T000000000-exp-gm-sweep",
                        criterion: try SweepSelectionRule.resolve(nil).asCriterion,
                        devPromptsHash: String(repeating: "d", count: 64),
                        winningCell: .init(layer: 3, alpha: 0.4),
                        metrics: [:])))
            try ExperimentStore.save(manifest)
            try plantCompletedConditionRun(
                provenance: #require(
                    manifest.conditions.first?.selection))

            // No population stamp: unprovable, named.
            _ = try plantVectorArtifact(
                in: root, method: "emotionGrandMean",
                residualNormSource: "multi-concept-corpus")
            do {
                _ = try AgentPromotion.promote(
                    experimentName: "gm", concept: "fear", log: { _ in })
                Issue.record("grand-mean artifact without population must refuse")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("[grandMeanPopulation]"))
            }

            // A DIFFERENT population (one member's stories drifted) refuses.
            var drifted = population
            drifted["joy"] = String(repeating: "c", count: 64)
            _ = try plantVectorArtifact(
                in: root, run: "20260707T000002000-extract",
                method: "emotionGrandMean",
                residualNormSource: "multi-concept-corpus",
                extras: ["grandMeanPopulation": drifted])
            #expect(throws: ExperimentError.self) {
                try AgentPromotion.promote(
                    experimentName: "gm", concept: "fear", log: { _ in })
            }

            // The full pinned population matches — with the Swift-recorded
            // "multi-concept-corpus" label unifying to the canonical source.
            let artifactID = try plantVectorArtifact(
                in: root, run: "20260707T000003000-extract",
                method: "emotionGrandMean",
                residualNormSource: "multi-concept-corpus",
                extras: ["grandMeanPopulation": population])
            let record = try AgentPromotion.promote(
                experimentName: "gm", concept: "fear", log: { _ in })
            #expect(samePath(record.artifact.injections.first?.vectorArtifactID, artifactID))
        }
    }

    @Test func judgeScoreObjectiveResolvesFromWorkspacePins() throws {
        try withTempWorkspace { root in
            // A real pinned rubric file + a local judge panel: resolution
            // embeds the PINNED hash and the judges verbatim in the resolved
            // criterion (no credential needed for local judges).
            let rubricDirectory = root.appending(components: "prompts", "rubrics")
            try FileManager.default.createDirectory(
                at: rubricDirectory, withIntermediateDirectories: true)
            let text = "Which response expresses more dread?\n"
            try text.write(
                to: rubricDirectory.appending(component: "r.md"),
                atomically: true, encoding: .utf8)
            let hash = SHA256.hash(data: Data(text.utf8))
                .map { String(format: "%02x", $0) }.joined()
            var manifest = ExperimentManifest(
                name: "jsr", description: "", modelID: "test/model")
            manifest.judgeRubricFile = "prompts/rubrics/r.md"
            manifest.judgeRubricHash = hash
            // The local judge names the STUDY model — a different local
            // model refuses at sweep start (one loaded model; covered in
            // SweepJudgeResolutionTests).
            manifest.judges = [.init(name: "j1", kind: "local", model: "test/model")]
            let criterion = try SweepSelectionRule.resolve(
                .init(objective: .init(metric: "judgeScore")))
            let objective = try SweepSelectionRule.resolveObjective(
                criterion: criterion, spec: nil, manifest: manifest,
                hasClaudeCredential: false)
            let embedded = criterion.asCriterion(objective: objective)
            #expect(embedded.objective?.metric == "judgeScore")
            #expect(embedded.objective?.judgeRubricHash == hash)
            #expect(embedded.objective?.judges == manifest.judges)
            #expect(embedded.objective?.choicePromptsFile == nil)
        }
    }

    @Test func freezeAdvisoriesFlagUndeclaredSelections() throws {
        let handMade = ModelVariantArtifact(
            name: "hand", baseModelID: "m", promptMode: "chatAssistant",
            qwenThinkingEnabled: false, temperature: 0, systemPrompt: "")
        let promoted = ModelVariantArtifact(
            name: "born", baseModelID: "m", promptMode: "chatAssistant",
            qwenThinkingEnabled: false, temperature: 0, systemPrompt: "",
            promotion: .init(
                experiment: "e", experimentHash: "h",
                promotedAt: "2026-07-07T00:00:00Z", promotedBy: "criterion",
                substrate: "swift-mlx", appVersion: "test"))
        let overridden = ModelVariantArtifact(
            name: "ovr", baseModelID: "m", promptMode: "chatAssistant",
            qwenThinkingEnabled: false, temperature: 0, systemPrompt: "",
            promotion: .init(
                experiment: "e", experimentHash: "h",
                promotedAt: "2026-07-07T00:00:00Z", promotedBy: "manualOverride",
                overrideReason: "adjacent layer on purpose",
                substrate: "swift-mlx", appVersion: "test"))
        try withTempWorkspace { _ in
            var manifest = try ExperimentStore.create(
                name: "adv", description: "", modelID: "test/model",
                modelRevision: "abc123")
            manifest.variantConditions = [
                .init(name: "hand", artifactPath: "x", artifactHash: "h", artifact: handMade)
            ]
            let advisories = ExperimentStore.freezeAdvisories(for: manifest)
            #expect(advisories.count == 1)
            #expect(advisories.first?.contains("hand-created") == true)
            manifest.variantConditions = [
                .init(name: "born", artifactPath: "x", artifactHash: "h", artifact: promoted)
            ]
            #expect(ExperimentStore.freezeAdvisories(for: manifest).isEmpty)

            // Override-promoted agents get their own advisory, carrying the
            // documented reason.
            manifest.variantConditions = [
                .init(name: "ovr", artifactPath: "x", artifactHash: "h", artifact: overridden)
            ]
            let overrideAdvisories = ExperimentStore.freezeAdvisories(for: manifest)
            #expect(overrideAdvisories.count == 1)
            #expect(overrideAdvisories.first?.contains("manual override") == true)
            #expect(
                overrideAdvisories.first?.contains("adjacent layer on purpose") == true)

            // A confirmation policy whose source agent is hand-created is
            // surfaced even though confirmation studies attach PLAIN
            // conditions (the variant-condition advisories never fire).
            manifest.variantConditions = []
            manifest.perturbationPolicy = .init(
                sourceAgent: .init(
                    name: "handAgent", artifactPath: "x", artifactHash: "h",
                    promoted: false),
                concept: "fear", cell: .init(layer: 3, alpha: 0.4),
                alphaDeltas: [0.1], includeMatchedNormControl: false,
                declaredAt: "2026-07-07T00:00:00Z")
            let policyAdvisories = ExperimentStore.freezeAdvisories(for: manifest)
            #expect(policyAdvisories.count == 1)
            #expect(policyAdvisories.first?.contains("handAgent") == true)
            #expect(policyAdvisories.first?.contains("exploratory") == true)

            // …and a sweep-promoted source agent is silent.
            manifest.perturbationPolicy?.sourceAgent.promoted = true
            #expect(ExperimentStore.freezeAdvisories(for: manifest).isEmpty)
        }
    }
}
