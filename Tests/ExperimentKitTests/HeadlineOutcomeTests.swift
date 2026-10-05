import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// The headline outcome: which outcome leads a results summary, which rule
/// chose it, and the primary outcome a study declares.
///
/// The rule, in order: the outcome the researcher declared; else a judged
/// outcome; else a declared choice or numeric outcome; else a reader or
/// probe score; else a reasoning-style feature; else marker density; else a
/// surface measure such as word count. Every summary says which rule chose
/// the headline.
///
/// The cases are the shared fixture
/// `Tests/Fixtures/cross-engine/headline-outcome.json`. The Python engine
/// (`Server/tests/test_headline_outcome.py`) and the results explorer
/// (`results-explorer/test/headline.test.ts`) read the same file, so a rule
/// that moves in one codebase and not the others fails where it did not
/// move.
///
/// Serialized: the store and command-line tests hold
/// `ExperimentRootOverrideLock` around a temporary workspace.
@Suite(.serialized) struct HeadlineOutcomeTests {

    // MARK: - The shared fixture

    private struct Fixture: Decodable {
        let selection: [SelectionCase]
        let tiers: [TierCase]
        let phrases: [PhraseCase]
        let producible: [ProducibleCase]
        let cannotProduceReasons: [ReasonCase]
        let settingsSummary: [SummaryCase]
    }

    private struct SelectionCase: Decodable {
        struct Expected: Decodable {
            let outcome: String?
            let tier: String?
            let rule: String?
            let source: String?
            let declaredAbsent: Bool
            let chosenBy: String
        }
        let label: String
        let declared: String?
        let analysisOutcomes: [String]
        let evaluationOutcomes: [String]
        let expected: Expected
    }

    private struct TierCase: Decodable {
        let name: String
        let tier: String
    }

    private struct PhraseCase: Decodable {
        let name: String
        let plain: String?
    }

    private struct ProducibleCase: Decodable {
        struct Study: Decodable {
            let outcomeInstruments: [String]?
            let concepts: [String]?
            let numericParser: String?
            let implicitNumericEndpoint: Bool?
            let readerConcepts: [String]?
            let reasoningStyleTaxonomy: Bool?
            let judging: String?
        }
        struct Expected: Decodable {
            let names: [String]
            let patterns: [String]
        }
        let label: String
        let study: Study
        let expected: Expected
        let accepts: [String]
        let refuses: [String]
    }

    private struct SummaryCase: Decodable {
        let declared: String?
        let line: String
    }

    private struct ReasonCase: Decodable {
        let outcome: String
        let reason: String
    }

    private func fixture() throws -> Fixture {
        let url = CodeResources.compiledCheckoutPath.appending(
            components: "Tests", "Fixtures", "cross-engine",
            "headline-outcome.json")
        return try JSONDecoder().decode(
            Fixture.self, from: try Data(contentsOf: url))
    }

    @Test func selectionMatchesTheSharedFixture() throws {
        let cases = try fixture().selection
        #expect(cases.count >= 6)
        for value in cases {
            let headline = HeadlineOutcome.select(
                declared: value.declared,
                analysisOutcomes: value.analysisOutcomes,
                evaluationOutcomes: value.evaluationOutcomes)
            #expect(headline.outcome == value.expected.outcome, "\(value.label)")
            #expect(headline.tier == value.expected.tier, "\(value.label)")
            #expect(headline.rule?.rawValue == value.expected.rule, "\(value.label)")
            #expect(
                headline.source?.rawValue == value.expected.source,
                "\(value.label)")
            #expect(
                headline.declaredAbsent == value.expected.declaredAbsent,
                "\(value.label)")
            #expect(headline.chosenBy == value.expected.chosenBy, "\(value.label)")
        }
    }

    @Test func everyOutcomeNameFallsInItsTier() throws {
        for value in try fixture().tiers {
            #expect(HeadlineOutcome.tier(of: value.name) == value.tier, "\(value.name)")
        }
    }

    @Test func plainPhrasesMatchTheSharedFixture() throws {
        for value in try fixture().phrases {
            #expect(
                HeadlineOutcome.plainPhrase(value.name) == value.plain,
                "\(value.name)")
        }
    }

    /// A real manifest from the fixture's flat study description, in the
    /// fields this engine stores.
    private func manifest(_ study: ProducibleCase.Study) -> ExperimentManifest {
        var manifest = ExperimentManifest(
            name: "demo", description: "", modelID: "test/model",
            createdAt: Date(timeIntervalSince1970: 0))
        manifest.outcomeInstruments = study.outcomeInstruments
        manifest.concepts = (study.concepts ?? []).map {
            .init(
                name: $0, stimulusSetHash: String(repeating: "0", count: 64),
                options: ExtractionOptions())
        }
        manifest.numericParser = study.numericParser
        if study.implicitNumericEndpoint == true {
            manifest.caseFamily = ExperimentManifest.implicitEndpointCaseFamily
        }
        if let readers = study.readerConcepts {
            manifest.readerRefs = readers.map {
                .init(
                    path: "readers/\($0).json",
                    hash: String(repeating: "0", count: 64), concept: $0)
            }
        }
        if study.reasoningStyleTaxonomy == true {
            manifest.reasoningStyleTaxonomyPath = "prompts/taxonomies/style.json"
            manifest.reasoningStyleTaxonomyHash = String(repeating: "0", count: 64)
        }
        func pinJudges() {
            manifest.judges = [.init(name: "j1", kind: "local")]
            manifest.judgeRubricFile = "prompts/rubrics/r.md"
        }
        switch study.judging ?? "none" {
        case "pinnedRubric":
            pinJudges()
        case "evaluationBlock":
            manifest.evaluation = .init(
                kind: .pairedJudge, judgeModel: "",
                judgePrompt: "which is better?")
        case "explicitNone":
            pinJudges()
            manifest.evaluation = .init(kind: .none, judgeModel: "", judgePrompt: "")
        default:
            break
        }
        return manifest
    }

    @Test func producibleOutcomesMatchTheSharedFixture() throws {
        let cases = try fixture().producible
        #expect(cases.count >= 6)
        for value in cases {
            let manifest = manifest(value.study)
            let listing = HeadlineOutcome.producible(manifest)
            #expect(listing.names == value.expected.names, "\(value.label)")
            #expect(listing.patterns == value.expected.patterns, "\(value.label)")
            for outcome in value.accepts {
                #expect(
                    HeadlineOutcome.canProduce(manifest, outcome: outcome),
                    "\(value.label): \(outcome)")
            }
            for outcome in value.refuses {
                #expect(
                    !HeadlineOutcome.canProduce(manifest, outcome: outcome),
                    "\(value.label): \(outcome)")
            }
        }
    }

    @Test func theRefusalSaysWhatAnOutcomeNeeds() throws {
        let cases = try fixture().cannotProduceReasons
        #expect(cases.count >= 8)
        for value in cases {
            #expect(
                HeadlineOutcome.cannotProduceReason(value.outcome) == value.reason,
                "\(value.outcome)")
        }
    }

    @Test func theSettingsSummaryLineMatchesTheSharedFixture() throws {
        for value in try fixture().settingsSummary {
            var manifest = ExperimentManifest(
                name: "demo", description: "", modelID: "test/model")
            manifest.primaryOutcome = value.declared
            #expect(HeadlineOutcome.settingsSummaryLine(manifest) == value.line)
            // …and the generated settings summary carries that line.
            #expect(
                ExperimentStore.preregistrationMarkdown(manifest)
                    .contains(value.line))
        }
    }

    // MARK: - The mapping itself

    /// The compiled copy IS the shipped file. The generator checks the same
    /// thing; this is the check that runs with the suite.
    @Test func theCompiledMappingIsTheShippedFile() throws {
        let url = CodeResources.compiledCheckoutPath.appending(
            components: "Server", "steerlab_server", "client", "resources",
            "headline-outcomes.json")
        let shipped = try JSONSerialization.jsonObject(
            with: Data(contentsOf: url)) as? NSDictionary
        let compiled = try JSONSerialization.jsonObject(
            with: Data(HeadlineOutcomeData.json.utf8)) as? NSDictionary
        #expect(shipped != nil)
        #expect(shipped == compiled)
    }

    @Test func theMappingIsWellFormed() {
        let mapping = HeadlineOutcome.mapping
        #expect(mapping.schemaVersion == 1)
        #expect(mapping.manifestKey == HeadlineOutcome.manifestKey)
        #expect(
            mapping.tiers.map(\.id) == [
                "judged", "choiceOrNumeric", "readerOrProbe", "reasoningStyle",
                "markerDensity", "surface", "unlisted",
            ])
        #expect(mapping.unlistedTier == "unlisted")
        #expect(Set(mapping.outcomes.map(\.name)).count == mapping.outcomes.count)
        let tierIDs = mapping.tiers.map(\.id)
        let positions = mapping.outcomes.compactMap { tierIDs.firstIndex(of: $0.tier) }
        #expect(positions.count == mapping.outcomes.count)
        #expect(positions == positions.sorted())
        #expect(mapping.rules["declared"] == "declared by the researcher")
        #expect(mapping.rules["defaultOrder"] == "chosen by default order")
    }

    /// Every outcome this engine's analysis writes today is in the order,
    /// so none of them can lead by accident of being unlisted.
    @Test func everyOutcomeThisEngineEmitsIsInTheOrder() {
        for name in [
            "wordCount", "distinct2", "warmthMarkerDensity", "rs_hedging",
            "ordinalPosition",
        ] {
            #expect(HeadlineOutcome.tier(of: name) != "unlisted", "\(name)")
        }
    }

    // MARK: - The manifest field

    @Test func aManifestWithoutTheFieldDecodesAndHashesAsBefore() throws {
        var manifest = ExperimentManifest(
            name: "demo", description: "", modelID: "test/model",
            createdAt: Date(timeIntervalSince1970: 0))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let before = try encoder.encode(manifest)
        // Absent when nil: the bytes carry no key at all.
        #expect(!String(decoding: before, as: UTF8.self).contains("primaryOutcome"))
        let hashBefore = ExperimentStore.manifestHash(manifest)
        let decoded = try JSONDecoder().decode(ExperimentManifest.self, from: before)
        #expect(decoded.primaryOutcome == nil)
        #expect(ExperimentStore.manifestHash(decoded) == hashBefore)

        // Declared: the key round-trips, and it is part of the content hash
        // (frozen with the study).
        manifest.primaryOutcome = "choiceRate"
        let declared = try JSONDecoder().decode(
            ExperimentManifest.self, from: try encoder.encode(manifest))
        #expect(declared.primaryOutcome == "choiceRate")
        #expect(ExperimentStore.manifestHash(declared) != hashBefore)

        // Cleared: the manifest is the one it was.
        manifest.primaryOutcome = nil
        #expect(ExperimentStore.manifestHash(manifest) == hashBefore)
    }

    /// The Python engine writes the same key; a manifest it authored decodes
    /// here with the declaration intact, and re-encodes with it.
    @Test func aPythonAuthoredDeclarationIsPreserved() throws {
        let document = """
            {"name": "demo", "experimentDescription": "", "modelID": "org/m",
             "createdAt": "2026-01-01T00:00:00Z", "status": "draft",
             "concepts": [], "conditions": [], "primaryOutcome": "judged"}
            """
        let manifest = try JSONDecoder().decode(
            ExperimentManifest.self, from: Data(document.utf8))
        #expect(manifest.primaryOutcome == "judged")
        let again = try JSONSerialization.jsonObject(
            with: try JSONEncoder().encode(manifest)) as? [String: Any]
        #expect(again?["primaryOutcome"] as? String == "judged")
    }

    @Test func declaringThePrimaryOutcomeIsMeasurementSideDrift() {
        #expect(RunEpoch.measurementFields.contains("primaryOutcome"))
        let snapshot = ExperimentManifest(
            name: "demo", description: "", modelID: "test/model",
            createdAt: Date(timeIntervalSince1970: 0))
        var live = snapshot
        live.primaryOutcome = "wordCount"
        #expect(
            ExperimentStore.manifestHash(live)
                != ExperimentStore.manifestHash(snapshot))
        let drift = RunEpoch.measurementDrift(live: live, snapshot: snapshot)
        #expect(drift?.contains("primaryOutcome") == true)
    }

    // MARK: - The declaration, the verb, and freeze

    private func plantStories(_ name: String, root: URL) throws {
        let directory = root.appending(components: "prompts", "emotions", name)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try """
        {"concept": "\(name)", "text": "a story about \(name)"}
        {"concept": "\(name)", "text": "another story about \(name)"}

        """.write(
            to: directory.appending(component: "stories.jsonl"),
            atomically: true, encoding: .utf8)
    }

    @Test func theDeclarationSurvivesFreezeAndIsInTheSettingsSummary() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "headline") { root in
            _ = try ExperimentStore.create(
                name: "demo", description: "", modelID: "test/model")
            try plantStories("warmth", root: root)
            _ = try ExperimentStore.attachConcept(
                "warmth", method: .emotionGrandMean, experimentName: "demo")

            // An outcome these settings cannot produce: refused with the
            // list, and nothing written.
            do {
                try ExperimentStore.setPrimaryOutcome(
                    "choiceLogOdds", experimentName: "demo")
                Issue.record("an outcome the study cannot produce was accepted")
            } catch let error as ExperimentError {
                #expect(
                    error.reason
                        == "this study's settings cannot produce the outcome "
                        + "'choiceLogOdds': it needs the answerTokenLogprob "
                        + "outcome instrument. The outcomes they can produce: "
                        + "choiceRate | warmthMarkerDensity | wordCount | "
                        + "distinct2")
                #expect(
                    error.malformedInvocation?.repairAction
                        == "steerlab-cli experiment set-primary-outcome demo "
                        + "<choiceRate | warmthMarkerDensity | wordCount | "
                        + "distinct2>  (\"\" clears the declaration)")
            }
            #expect(try ExperimentStore.load(name: "demo").primaryOutcome == nil)

            let declared = try ExperimentStore.setPrimaryOutcome(
                "warmthMarkerDensity", experimentName: "demo")
            #expect(declared.primaryOutcome == "warmthMarkerDensity")

            let frozen = try ExperimentStore.freeze(name: "demo", force: true)
            #expect(frozen.status == .frozen)
            #expect(frozen.primaryOutcome == "warmthMarkerDensity")
            #expect(frozen.freezeHash == ExperimentStore.manifestHash(frozen))
            let summary = try String(
                contentsOf: ExperimentStore.directory.appending(
                    components: "demo", "preregistration.md"),
                encoding: .utf8)
            #expect(
                summary.contains(
                    "- **Primary outcome:** warmthMarkerDensity ('warmth' "
                        + "marker density), declared by the researcher"))

            // Frozen with the study: it can no longer be changed.
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.setPrimaryOutcome(
                    "wordCount", experimentName: "demo")
            }
            #expect(
                try ExperimentStore.load(name: "demo").primaryOutcome
                    == "warmthMarkerDensity")
        }
    }

    private func withAsyncTempRoot<T>(_ body: (URL) async throws -> T) async rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "headline-verb-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        WorkspaceRoot.programmaticOverride = temp
        ExperimentStore.rootOverride = temp
        defer {
            ExperimentStore.rootOverride = nil
            WorkspaceRoot.programmaticOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try await body(temp)
    }

    @Test func theVerbDeclaresRefusesAndClears() async throws {
        try await withAsyncTempRoot { _ in
            _ = try ExperimentStore.create(
                name: "demo", description: "", modelID: "test/model")
            let runner = ExperimentCLIRunner(sink: .discarding)

            let declared = await runner.run(
                namespace: "experiment",
                ["set-primary-outcome", "demo", "wordCount"])
            #expect(declared.envelope.state == .ready)
            #expect(declared.envelope.exitCode == 0)
            #expect(declared.envelope.result?["experiment"] == .string("demo"))
            #expect(
                declared.envelope.result?["primaryOutcome"] == .string("wordCount"))
            #expect(
                declared.envelope.result?["plain"]
                    == .string("response length in words"))
            #expect(
                declared.envelope.result?["producibleOutcomes"]
                    == .object([
                        "names": .array([
                            .string("choiceRate"), .string("wordCount"),
                            .string("distinct2"),
                        ]),
                        "patterns": .array([]),
                    ]))
            #expect(try ExperimentStore.load(name: "demo").primaryOutcome == "wordCount")

            // An outcome this study cannot produce: blocked (64), with the
            // list to retype, and nothing written.
            let refused = await runner.run(
                namespace: "experiment",
                ["set-primary-outcome", "demo", "judged"])
            #expect(refused.envelope.state == .blocked)
            #expect(refused.envelope.exitCode == 64)
            #expect(
                refused.envelope.error?.reason.contains(
                    "cannot produce the outcome 'judged'") == true)
            #expect(
                refused.envelope.error?.repairAction.hasPrefix(
                    "steerlab-cli experiment set-primary-outcome demo <choiceRate | ")
                    == true)
            #expect(try ExperimentStore.load(name: "demo").primaryOutcome == "wordCount")

            let cleared = await runner.run(
                namespace: "experiment", ["set-primary-outcome", "demo", ""])
            #expect(cleared.envelope.state == .ready)
            #expect(cleared.envelope.result?["primaryOutcome"] == .null)
            #expect(try ExperimentStore.load(name: "demo").primaryOutcome == nil)
        }
    }

    // MARK: - A real analysis, and the envelope

    private struct Record: Encodable {
        let condition: String
        let seed: UInt64
        let sampleIndex: Int
        let promptIndex: Int
        let promptID: String
        let wordCount: Int
        let distinct2: Double
        let markerDensity: [String: Double]
    }

    /// Four items, one response each, under a baseline and one condition.
    private func records() -> [Record] {
        var records: [Record] = []
        for (index, item) in ["item-1", "item-2", "item-3", "item-4"].enumerated() {
            records.append(
                Record(
                    condition: "baseline", seed: 1, sampleIndex: 0,
                    promptIndex: index, promptID: item, wordCount: 20,
                    distinct2: 0.5, markerDensity: ["warmth": 0.01]))
            records.append(
                Record(
                    condition: "steered", seed: 1, sampleIndex: 0,
                    promptIndex: index, promptID: item, wordCount: 24 + index,
                    distinct2: 0.6 + Double(index) * 0.01,
                    markerDensity: ["warmth": 0.05 + Double(index) * 0.01]))
        }
        return records
    }

    /// Writes the records as a completed run in a fresh temporary
    /// workspace, runs the real analysis over it, and returns the analysis
    /// directory (inside `root`, which the caller removes).
    private func analyze(
        _ manifest: ExperimentManifest, root: URL
    ) throws -> URL {
        let run = root.appending(path: "runs/20261003T000000000Z-exp-demo-run")
        try FileManager.default.createDirectory(
            at: run, withIntermediateDirectories: true)
        try JSONEncoder().encode(manifest).write(
            to: run.appending(component: "experiment.json"))
        try Data(ExperimentStore.manifestHash(manifest).utf8).write(
            to: run.appending(component: "experiment-hash.txt"))
        let encoder = JSONEncoder()
        let lines = try records().map {
            String(decoding: try encoder.encode($0), as: UTF8.self)
        }
        try Data((lines.joined(separator: "\n") + "\n").utf8).write(
            to: run.appending(component: "generations.jsonl"))
        try Data("{}".utf8).write(to: run.appending(component: "report.json"))
        return try StudyAnalysisWorkflow.analyze(
            manifest: manifest,
            repository: StudyAnalysisRepository(workspaceRoot: root, promptRoot: root),
            allowUnverifiedEpoch: false)
    }

    private func study(declared: String? = nil) -> ExperimentManifest {
        var manifest = ExperimentManifest(
            name: "demo", description: "", modelID: "test/model",
            createdAt: Date(timeIntervalSince1970: 0))
        manifest.concepts = [
            .init(
                name: "warmth", stimulusSetHash: String(repeating: "0", count: 64),
                options: ExtractionOptions())
        ]
        manifest.primaryOutcome = declared
        return manifest
    }

    private func headlineBlock(_ payload: [String: JSONValue]) -> [String: JSONValue] {
        guard case .object(let block)? = payload["headline"] else { return [:] }
        return block
    }

    @Test func aRealAnalysisNoLongerLeadsWithWordCount() throws {
        let root = FileManager.default.temporaryDirectory.appending(
            component: "headline-analysis-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let out = try analyze(study(), root: root)

        // The engine's own row order is untouched: word count is still the
        // first row it wrote.
        let report = try JSONDecoder().decode(
            ExperimentTasks.AnalyzeReport.self,
            from: Data(contentsOf: out.appending(component: "analysis.json")))
        #expect(report.effectSizes.first?.metric == "wordCount")

        let payload = ExperimentCLIRunner.analysisPayload(inRunAt: out)
        let headline = headlineBlock(payload)
        #expect(headline["outcome"] == .string("warmthMarkerDensity"))
        #expect(headline["tier"] == .string("markerDensity"))
        #expect(headline["rule"] == .string("defaultOrder"))
        #expect(headline["source"] == .string("analysisRows"))
        #expect(headline["chosenBy"] == .string("chosen by default order"))
        #expect(headline["declaredAbsent"] == .bool(false))
        #expect(headline["plain"] == .string("'warmth' marker density"))
        // Nothing was dropped: the surface measures are still listed.
        #expect(
            payload["metrics"]
                == .array([
                    .string("distinct2"), .string("warmthMarkerDensity"),
                    .string("wordCount"),
                ]))

        // The app's model of the same directory leads with the same outcome,
        // and its charts open on it.
        let model = RunResults.load(runDirectory: out)
        #expect(model.headline.outcome == "warmthMarkerDensity")
        #expect(model.headline.rule == .defaultOrder)
        let metrics = (model.effectSizes ?? []).map(\.metric)
        #expect(metrics.first == "wordCount")
        #expect(
            HeadlineOutcome.chartLead(model.headline, among: metrics).outcome
                == "warmthMarkerDensity")
        let header = try #require(
            EffectNarrative.headline(model.headline, rows: model.effectSizes ?? []))
        #expect(
            header.title
                == "Headline outcome: 'warmth' marker density "
                + "(warmthMarkerDensity), chosen by default order.")
        #expect(header.sentences.count == 1)
        #expect(header.sentences.first?.contains("warmthMarkerDensity") == true)
    }

    @Test func aRealAnalysisLeadsWithTheDeclaredOutcome() throws {
        let root = FileManager.default.temporaryDirectory.appending(
            component: "headline-analysis-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let out = try analyze(study(declared: "distinct2"), root: root)
        let headline = headlineBlock(ExperimentCLIRunner.analysisPayload(inRunAt: out))
        #expect(headline["outcome"] == .string("distinct2"))
        #expect(headline["rule"] == .string("declared"))
        #expect(headline["chosenBy"] == .string("declared by the researcher"))
        #expect(headline["declaredOutcome"] == .string("distinct2"))

        let model = RunResults.load(runDirectory: out)
        #expect(model.headline.outcome == "distinct2")
        #expect(model.headline.rule == .declared)
    }

    @Test func aDeclaredOutcomeTheRunLacksFallsBackAndSaysSo() throws {
        let root = FileManager.default.temporaryDirectory.appending(
            component: "headline-analysis-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let out = try analyze(study(declared: "choiceLogOdds"), root: root)
        let headline = headlineBlock(ExperimentCLIRunner.analysisPayload(inRunAt: out))
        #expect(headline["outcome"] == .string("warmthMarkerDensity"))
        #expect(headline["rule"] == .string("defaultOrder"))
        #expect(headline["declaredAbsent"] == .bool(true))
        #expect(
            headline["chosenBy"]
                == .string(
                    "chosen by default order; the declared primary outcome "
                        + "'choiceLogOdds' is not in this run"))
    }

    /// A judged outcome is never an analysis row: it comes from the
    /// evaluation report for the same source run, whichever engine wrote it.
    @Test(arguments: [
        ["sourceRun", "20261003T000000000Z-exp-demo-run", "judge-report.json"],
        ["sourceRunDirectory", "/w/runs/20261003T000000000Z-exp-demo-run", "judge-report.json"],
        ["sourceRun", "20261003T000000000Z-exp-demo-run", "coding-report.json"],
    ])
    func aJudgedOutcomeComesFromTheEvaluationReport(stamp: [String]) throws {
        let root = FileManager.default.temporaryDirectory.appending(
            component: "headline-analysis-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let out = try analyze(study(), root: root)

        func evaluation(_ name: String, key: String, source: String, file: String) throws {
            let directory = root.appending(components: "runs", name)
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            try JSONSerialization.data(withJSONObject: [
                "experiment": "demo", key: source, "conditions": [String: Any](),
            ]).write(to: directory.appending(component: file))
        }

        // An evaluation of a DIFFERENT run is not this run's judged outcome.
        try evaluation(
            "20261005T000000000Z-exp-demo-evaluate", key: "sourceRun",
            source: "20260101T000000000Z-exp-demo-run", file: "judge-report.json")
        #expect(HeadlineOutcome.evaluationReport(forRunAt: out) == nil)
        #expect(
            headlineBlock(ExperimentCLIRunner.analysisPayload(inRunAt: out))["outcome"]
                == .string("warmthMarkerDensity"))

        try evaluation(
            "20261004T120000000Z-exp-demo-evaluate", key: stamp[0],
            source: stamp[1], file: stamp[2])
        let headline = headlineBlock(ExperimentCLIRunner.analysisPayload(inRunAt: out))
        #expect(headline["outcome"] == .string("judged"))
        #expect(headline["tier"] == .string("judged"))
        #expect(headline["source"] == .string("evaluationReport"))
        #expect(headline["chosenBy"] == .string("chosen by default order"))
        #expect(
            HeadlineOutcome.evaluationReport(forRunAt: out)?.lastPathComponent
                == stamp[2])

        // The app says where the judged outcome is, and its charts open on
        // the best of the measures they do have.
        let model = RunResults.load(runDirectory: out)
        #expect(model.headline.outcome == "judged")
        let header = try #require(
            EffectNarrative.headline(model.headline, rows: model.effectSizes ?? []))
        #expect(
            header.title
                == "Headline outcome: the judged outcome (judged), chosen by "
                + "default order.")
        #expect(header.sentences.isEmpty)
        #expect(header.note?.contains("evaluation report") == true)
        let lead = HeadlineOutcome.chartLead(
            model.headline, among: (model.effectSizes ?? []).map(\.metric))
        #expect(lead.outcome == "warmthMarkerDensity")
        #expect(lead.chosenBy == "chosen by default order")
    }

    @Test func aRunWithNothingToLeadWithHasNoHeader() {
        #expect(EffectNarrative.headline(.none, rows: []) == nil)
        let declaredOnly = HeadlineOutcome.select(
            declared: "choiceRate", analysisOutcomes: [])
        #expect(
            EffectNarrative.headline(declaredOnly, rows: [])?.title
                == "Headline outcome: no outcome to lead with; the declared "
                + "primary outcome 'choiceRate' is not in this run.")
    }
}
