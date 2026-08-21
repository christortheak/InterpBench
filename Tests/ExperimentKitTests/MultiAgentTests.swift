import Foundation
import Testing

@testable import ExperimentKit

@Suite("MultiAgentTests")
struct MultiAgentTests {
    @Test("scenario store round-trips agents, turns, and routing")
    func scenarioStoreRoundTrip() throws {
        let root = URL(filePath: NSTemporaryDirectory())
            .appending(component: "steerlab-multi-agent-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }

        let agentA = MultiAgentScenario.Agent(
            id: "a",
            name: "Agent A",
            baseModelID: "test/model")
        let agentB = MultiAgentScenario.Agent(
            id: "b",
            name: "Agent B",
            baseModelID: "test/model")
        let scenario = MultiAgentScenario(
            name: "panel",
            baseModelID: "test/model",
            sharedMaterials: "Shared record.",
            agents: [agentA, agentB],
            turns: [
                MultiAgentScenario.Turn(
                    id: "turn-1",
                    title: "Private notes",
                    speakerAgentID: "a",
                    promptTemplate: "Read {{scenario.materials}}.",
                    outputLabel: "notes_a",
                    routing: .speakerOnly),
                MultiAgentScenario.Turn(
                    id: "turn-2",
                    title: "Share memo",
                    speakerAgentID: "b",
                    promptTemplate: "Visible: {{agent.context}}",
                    outputLabel: "memo_b",
                    routing: .selected,
                    routedAgentIDs: ["a"]),
            ])

        // Canonical layout (B1): one flat prompts/panels/<slug>.json.
        let localURL = root.appending(component: "panel.json")
        _ = try MultiAgentScenarioStore.update(scenario, at: localURL)

        let scanned = MultiAgentScenarioStore.scan(directory: root)
        #expect(scanned.count == 1)
        #expect(scanned[0].scenario.agents.map(\.name) == ["Agent A", "Agent B"])
        #expect(scanned[0].scenario.turns[1].routing == .selected)
        #expect(scanned[0].scenario.turns[1].routedAgentIDs == ["a"])
        try MultiAgentRunner.validate(scanned[0].scenario)
    }

    @Test("scenario validation rejects missing speakers")
    func validationRejectsMissingSpeaker() throws {
        let scenario = MultiAgentScenario(
            name: "bad",
            baseModelID: "test/model",
            agents: [
                MultiAgentScenario.Agent(id: "a", name: "Agent A", baseModelID: "test/model")
            ],
            turns: [
                MultiAgentScenario.Turn(
                    title: "Broken",
                    speakerAgentID: "missing",
                    promptTemplate: "Speak.")
            ])

        #expect(throws: ExperimentError.self) {
            try MultiAgentRunner.validate(scenario)
        }
    }

    // MARK: - Prompt rendering
    //
    // The fallback blocks ("Visible prior context:", "Shared scenario
    // materials:") exist for templates that never ASKED for that material.
    // Testing the SUBSTITUTED prompt for the placeholder made both guards
    // unconditionally true — substitution has just removed it — so every
    // built-in template, all of which interpolate `{{agent.context}}`, got a
    // second copy. These pin the exactly-once contract on both branches, and
    // mirror the server's `_render_prompt` behaviour. Their PLACEMENT (record
    // first, transcript second, instruction last, since 2026-08-17) is pinned
    // in `PanelTurnContractTests`.

    private func occurrences(of needle: String, in haystack: String) -> Int {
        haystack.components(separatedBy: needle).count - 1
    }

    private func renderScenario(materials: String) -> MultiAgentScenario {
        MultiAgentScenario(
            name: "panel",
            description: "A scripted panel.",
            baseModelID: "test/model",
            sharedMaterials: materials,
            agents: [
                MultiAgentScenario.Agent(id: "a", name: "Judge A", baseModelID: "test/model")
            ],
            turns: [])
    }

    @Test("an interpolated {{agent.context}} is not also appended")
    func interpolatedContextIsNotDuplicated() {
        let turn = MultiAgentScenario.Turn(
            title: "Memo",
            speakerAgentID: "a",
            promptTemplate: "You are {{agent.name}}.\n\nSo far:\n{{agent.context}}\n\nWrite a memo.")

        let rendered = MultiAgentRunner.renderPrompt(
            scenario: renderScenario(materials: ""), turn: turn,
            speakerName: "Judge A", speakerContext: "PRIOR-OUTPUT",
            outputsByLabel: [:])

        #expect(occurrences(of: "PRIOR-OUTPUT", in: rendered) == 1)
        #expect(!rendered.contains("Visible prior context:"))
        #expect(rendered.contains("You are Judge A."))
    }

    @Test("a template that omits {{agent.context}} gets the fallback block once")
    func omittedContextIsAppendedOnce() {
        let turn = MultiAgentScenario.Turn(
            title: "Notes",
            speakerAgentID: "a",
            promptTemplate: "You are {{agent.name}}. Write private notes.")

        let rendered = MultiAgentRunner.renderPrompt(
            scenario: renderScenario(materials: ""), turn: turn,
            speakerName: "Judge A", speakerContext: "PRIOR-OUTPUT",
            outputsByLabel: [:])

        #expect(occurrences(of: "PRIOR-OUTPUT", in: rendered) == 1)
        #expect(occurrences(of: "Visible prior context:", in: rendered) == 1)
    }

    @Test("an interpolated {{scenario.materials}} is not also appended")
    func interpolatedMaterialsAreNotDuplicated() {
        let turn = MultiAgentScenario.Turn(
            title: "Notes",
            speakerAgentID: "a",
            promptTemplate: "Read:\n{{scenario.materials}}\n\nWrite notes.")

        let rendered = MultiAgentRunner.renderPrompt(
            scenario: renderScenario(materials: "CASE-RECORD"), turn: turn,
            speakerName: "Judge A", speakerContext: "",
            outputsByLabel: [:])

        #expect(occurrences(of: "CASE-RECORD", in: rendered) == 1)
        #expect(!rendered.contains("Shared scenario materials:"))
    }

    @Test("a template that omits {{scenario.materials}} gets the fallback block once")
    func omittedMaterialsAreAppendedOnce() {
        let turn = MultiAgentScenario.Turn(
            title: "Notes",
            speakerAgentID: "a",
            promptTemplate: "Write notes.")

        let rendered = MultiAgentRunner.renderPrompt(
            scenario: renderScenario(materials: "CASE-RECORD"), turn: turn,
            speakerName: "Judge A", speakerContext: "",
            outputsByLabel: [:])

        #expect(occurrences(of: "CASE-RECORD", in: rendered) == 1)
        #expect(occurrences(of: "Shared scenario materials:", in: rendered) == 1)
    }

    @Test("turn visibility switches suppress material entirely, not just the placeholder")
    func visibilitySwitchesSuppressBothPaths() {
        let turn = MultiAgentScenario.Turn(
            title: "Isolated",
            speakerAgentID: "a",
            promptTemplate: "Context:\n{{agent.context}}\nMaterials:\n{{scenario.materials}}",
            includeScenarioMaterials: false,
            includeSpeakerContext: false)

        let rendered = MultiAgentRunner.renderPrompt(
            scenario: renderScenario(materials: "CASE-RECORD"), turn: turn,
            speakerName: "Judge A", speakerContext: "PRIOR-OUTPUT",
            outputsByLabel: [:])

        #expect(!rendered.contains("PRIOR-OUTPUT"))
        #expect(!rendered.contains("CASE-RECORD"))
        #expect(!rendered.contains("Visible prior context:"))
        #expect(!rendered.contains("Shared scenario materials:"))
    }

    // MARK: - Transcript is the unit of analysis (D1)

    private func turnRow(
        condition: String, replicate: Int, turn: String, wordCount: Int
    ) -> ExperimentTasks.MetricRow {
        ExperimentTasks.MetricRow(
            condition: condition, seed: 1, promptIndex: 0, promptID: turn,
            wordCount: wordCount, distinct2: 0, markerDensity: [:],
            replicate: replicate)
    }

    /// `transcripts` play-throughs per arm, `turns` turns each; every
    /// configured turn beats its baseline twin by that transcript's `lift`.
    private func panelRows(lifts: [Int], turns: Int) -> [ExperimentTasks.MetricRow] {
        var rows: [ExperimentTasks.MetricRow] = []
        for (replicate, lift) in lifts.enumerated() {
            for turn in 0 ..< turns {
                rows.append(
                    turnRow(
                        condition: "baseline", replicate: replicate,
                        turn: "t\(turn)", wordCount: 10))
                rows.append(
                    turnRow(
                        condition: "configured", replicate: replicate,
                        turn: "t\(turn)", wordCount: 10 + lift))
            }
        }
        return rows
    }

    @Test("n counts transcripts, not turns")
    func effectSizeNIsTranscripts() throws {
        // 3 transcripts x 8 turns = 24 turn pairs, but only 3 independent units.
        let entries = try #require(
            ExperimentTasks.transcriptEffectSizes(
                rows: panelRows(lifts: [2, 5, 8], turns: 8),
                styleFeatureIDs: [], phase: nil))

        let wordCount = try #require(entries.first { $0.metric == "wordCount" })
        #expect(wordCount.n == 3)
        #expect(abs(wordCount.meanDiff - 5.0) < 1e-9)
    }

    @Test("a single transcript per arm yields no effect sizes")
    func singleTranscriptYieldsNoInterval() {
        // A point estimate is available, but no interval — and a zero-width CI
        // would read as certainty rather than as absent replication.
        #expect(
            ExperimentTasks.transcriptEffectSizes(
                rows: panelRows(lifts: [4], turns: 8),
                styleFeatureIDs: [], phase: nil) == nil)
    }

    @Test("clustering does not understate uncertainty")
    func clusteringWidensTheInterval() throws {
        // Turns within a transcript agree closely; the two transcripts do not.
        // Treating the 12 turns as independent draws would report an interval
        // the design has not earned.
        let rows = panelRows(lifts: [1, 9], turns: 6)
        let clustered = try #require(
            ExperimentTasks.transcriptEffectSizes(
                rows: rows, styleFeatureIDs: [], phase: nil))
        let clusteredWordCount = try #require(clustered.first { $0.metric == "wordCount" })

        // The naive view: every turn its own unit (replicate stripped).
        let flattened = rows.map { row in
            ExperimentTasks.MetricRow(
                condition: row.condition, seed: row.seed, promptIndex: row.promptIndex,
                promptID: "\(row.promptID)-\(row.replicate ?? 0)",
                wordCount: row.wordCount, distinct2: row.distinct2,
                markerDensity: [:])
        }
        let naive = ExperimentTasks.effectSizes(rows: flattened, concepts: [])
        let naiveWordCount = try #require(naive.first { $0.metric == "wordCount" })

        #expect(naiveWordCount.n == 12)
        #expect(clusteredWordCount.n == 2)
        // Same centre, honest width.
        #expect(abs(naiveWordCount.meanDiff - clusteredWordCount.meanDiff) < 1e-9)
        #expect(
            (clusteredWordCount.ciUpper - clusteredWordCount.ciLower)
                > (naiveWordCount.ciUpper - naiveWordCount.ciLower))
    }

    @Test("turns pair within their own play-through, never across replicates")
    func pairingIsWithinTranscript() throws {
        // Baseline exists only for replicate 0. Replicate 1's configured turns
        // must find no partner rather than pairing against another
        // transcript's baseline — which would invent a difference.
        var rows = panelRows(lifts: [3], turns: 4)
        for turn in 0 ..< 4 {
            rows.append(
                turnRow(
                    condition: "configured", replicate: 1, turn: "t\(turn)",
                    wordCount: 100))
        }

        // Only one transcript pairs, so there is no interval to report.
        #expect(
            ExperimentTasks.transcriptEffectSizes(
                rows: rows, styleFeatureIDs: [], phase: nil) == nil)
    }

    // MARK: - Authoring advisories (F1)

    private func labelScenario(
        _ turns: [MultiAgentScenario.Turn]
    ) -> MultiAgentScenario {
        MultiAgentScenario(
            name: "p", baseModelID: "m",
            agents: [.init(id: "a", name: "A", baseModelID: "m")],
            turns: turns)
    }

    @Test("silent authoring failures are advised, not enforced")
    func advisoriesCatchSilentFailures() throws {
        let scenario = labelScenario([
            .init(id: "t1", title: "One", speakerAgentID: "a",
                  promptTemplate: "go", outputLabel: "dup"),
            .init(id: "t2", title: "Two", speakerAgentID: "a",
                  promptTemplate: "see {{outputs.nope}}", outputLabel: "dup"),
        ])

        let notes = MultiAgentRunner.advisories(scenario)
        #expect(notes.contains { $0.contains("no earlier turn produces the label 'nope'") })
        #expect(notes.contains { $0.contains("reuses the output label 'dup'") })
        // Advisory only — a draft is never blocked.
        try MultiAgentRunner.validate(scenario)
    }

    @Test("a clean panel draws no advisories")
    func cleanPanelIsQuiet() {
        let scenario = labelScenario([
            .init(id: "t1", title: "One", speakerAgentID: "a",
                  promptTemplate: "go", outputLabel: "first"),
            .init(id: "t2", title: "Two", speakerAgentID: "a",
                  promptTemplate: "see {{outputs.first}}", outputLabel: "second"),
        ])
        #expect(MultiAgentRunner.advisories(scenario).isEmpty)
    }

    @Test("a collision with the turn_<n> default is caught too")
    func defaultLabelCollisionIsCaught() {
        let scenario = labelScenario([
            .init(id: "t1", title: "One", speakerAgentID: "a", promptTemplate: "go"),
            .init(id: "t2", title: "Two", speakerAgentID: "a",
                  promptTemplate: "go", outputLabel: "turn_1"),
        ])
        #expect(MultiAgentRunner.advisories(scenario).contains { $0.contains("turn_1") })
    }

    @Test("a forward reference is flagged; a backward one is not")
    func onlyForwardReferencesAreFlagged() {
        // Order is the whole point: {{outputs.X}} resolves only from turns
        // that have ALREADY run.
        let backward = labelScenario([
            .init(id: "t1", title: "One", speakerAgentID: "a",
                  promptTemplate: "go", outputLabel: "draft"),
            .init(id: "t2", title: "Two", speakerAgentID: "a",
                  promptTemplate: "revise {{outputs.draft}}", outputLabel: "final"),
        ])
        let forward = labelScenario([
            .init(id: "t1", title: "One", speakerAgentID: "a",
                  promptTemplate: "revise {{outputs.draft}}", outputLabel: "final"),
            .init(id: "t2", title: "Two", speakerAgentID: "a",
                  promptTemplate: "go", outputLabel: "draft"),
        ])

        #expect(MultiAgentRunner.advisories(backward).isEmpty)
        #expect(MultiAgentRunner.advisories(forward).count == 1)
    }

    @Test("output references parse out of a template")
    func outputReferencesParse() {
        #expect(
            MultiAgentRunner.outputReferences(
                in: "a {{outputs.one}} b {{outputs.two}} c") == ["one", "two"])
        #expect(MultiAgentRunner.outputReferences(in: "no refs here").isEmpty)
    }

    // MARK: - One slug rule across engines (B3)

    @Test("panel slugs match the cross-engine fixture")
    func panelSlugMatchesCrossEngineFixture() throws {
        struct Fixture: Decodable {
            struct Case: Decodable {
                let name: String
                let slug: String
            }
            let cases: [Case]
        }
        // Same file the Python suite reads. Both engines write into
        // prompts/panels/, so a divergence here means two files for one panel
        // on the cluster and a silent overwrite on a case-insensitive Mac.
        let url = VectorCatalog.bundledSeedRoot
            .appending(components: "prompts", "fixtures", "panel-slug", "slugs.json")
        let fixture = try JSONDecoder().decode(Fixture.self, from: try Data(contentsOf: url))

        #expect(!fixture.cases.isEmpty)
        for testCase in fixture.cases {
            #expect(
                ExperimentStore.canonicalSlug(testCase.name) == testCase.slug,
                "slug for \(testCase.name.debugDescription)")
        }
    }

    @Test("a new panel never overwrites a differently-named neighbour")
    func newPanelDisambiguatesAgainstNeighbour() throws {
        let root = URL(filePath: NSTemporaryDirectory())
            .appending(component: "steerlab-panels-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

        // Two distinct panels that slugify identically.
        let taken = root.appending(component: "judicial-panel.json")
        _ = try MultiAgentScenarioStore.update(
            MultiAgentScenario(name: "Judicial Panel", baseModelID: "m"), at: taken)
        let before = try MultiAgentScenarioStore.hash(taken)

        let second = root.appending(component: "judicial-panel-2.json")
        _ = try MultiAgentScenarioStore.update(
            MultiAgentScenario(name: "Judicial: Panel", baseModelID: "m"), at: second)

        // The neighbour is untouched, and both are discoverable.
        #expect(try MultiAgentScenarioStore.hash(taken) == before)
        let scanned = MultiAgentScenarioStore.scan(directory: root, legacyDirectory: nil)
        #expect(Set(scanned.map(\.scenario.name)) == ["Judicial Panel", "Judicial: Panel"])
    }

    // MARK: - Canonical location and dialect (B1/B2)

    @Test("legacy runs/multi-agent-scenarios layout still opens, read-only")
    func legacyScenarioLayoutStillReadable() throws {
        let root = URL(filePath: NSTemporaryDirectory())
            .appending(component: "steerlab-panels-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let canonical = root.appending(component: "panels")
        let legacy = root.appending(component: "legacy")

        // Only the legacy tree has anything in it.
        let scenario = MultiAgentScenario(
            name: "old-panel", baseModelID: "test/model",
            agents: [.init(id: "a", name: "A", baseModelID: "test/model")],
            turns: [.init(id: "t", title: "T", speakerAgentID: "a", promptTemplate: "go")])
        _ = try MultiAgentScenarioStore.update(
            scenario, at: legacy.appending(components: "old-panel", "scenario.json"))

        let scanned = MultiAgentScenarioStore.scan(
            directory: canonical, legacyDirectory: legacy)

        // Frozen manifests pin paths in the old tree; they must keep resolving.
        #expect(scanned.count == 1)
        #expect(scanned[0].scenario.name == "old-panel")
    }

    @Test("the canonical file carries the recipe and no timestamps")
    func canonicalFileHasNoVolatileMetadata() throws {
        let root = URL(filePath: NSTemporaryDirectory())
            .appending(component: "steerlab-panels-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appending(component: "panel.json")
        _ = try MultiAgentScenarioStore.update(
            MultiAgentScenario(name: "panel", baseModelID: "test/model"), at: url)

        let text = try String(contentsOf: url, encoding: .utf8)

        // A timestamp inside a content-hashed pinned input turns a no-op save
        // into freeze drift. Git holds the history instead.
        #expect(!text.contains("createdAt"))
        #expect(!text.contains("updatedAt"))
        #expect(text.contains("\"schemaVersion\""))
    }

    @Test("re-saving an unedited panel is byte-identical")
    func resavingIsAByteForByteNoOp() throws {
        let root = URL(filePath: NSTemporaryDirectory())
            .appending(component: "steerlab-panels-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appending(component: "panel.json")
        let scenario = MultiAgentScenario(
            name: "panel", baseModelID: "test/model",
            agents: [.init(id: "a", name: "A", baseModelID: "test/model")],
            turns: [.init(id: "t", title: "T", speakerAgentID: "a", promptTemplate: "go")])

        _ = try MultiAgentScenarioStore.update(scenario, at: url)
        let first = try MultiAgentScenarioStore.hash(url)
        _ = try MultiAgentScenarioStore.update(scenario, at: url)
        let second = try MultiAgentScenarioStore.hash(url)

        // This is the whole point of dropping the timestamps: the pinned-input
        // hash only moves when the recipe does.
        #expect(first == second)
    }

    @Test("a Python-engine panel file decodes here")
    func serverDialectDecodes() throws {
        // Exactly what `multi_agent._scenario_to_dict` emits — note the null
        // variant fields and the absence of any Swift-only key. Before B2 the
        // missing schemaVersion made this throw, and `scan`'s `try?` swallowed
        // it, so a server-authored panel simply never appeared in the picker.
        let json = """
            {"schemaVersion":1,"name":"server-panel","baseModelID":"google/gemma-3-27b-it",
             "description":"","sharedMaterials":"m","temperature":0.0,"maxTokens":512,
             "agents":[{"id":"a","name":"A","baseModelID":"google/gemma-3-27b-it",
                        "systemPrompt":"","variantArtifactPath":null,
                        "variantArtifactHash":null}],
             "turns":[{"id":"t","title":"T","speakerAgentID":"a","promptTemplate":"go",
                       "outputLabel":"","routing":"all","routedAgentIDs":[],
                       "includeScenarioMaterials":true,"includeSpeakerContext":true,
                       "maxTokens":null}]}
            """
        let scenario = try JSONDecoder().decode(
            MultiAgentScenario.self, from: Data(json.utf8))

        #expect(scenario.name == "server-panel")
        #expect(scenario.agents.count == 1)
        #expect(scenario.turns.first?.routing == .all)
        try MultiAgentRunner.validate(scenario)
    }

    @Test("a panel file with no schemaVersion still decodes")
    func absentSchemaVersionDefaults() throws {
        let json = """
            {"name":"p","baseModelID":"m","agents":[],"turns":[]}
            """
        let scenario = try JSONDecoder().decode(
            MultiAgentScenario.self, from: Data(json.utf8))

        // Read off the turn script (no contracts ⇒ 1), not assumed to be
        // whatever this build writes today — a file that never used contracts
        // must not start claiming schema 2 because the engine learned one.
        #expect(scenario.schemaVersion == 1)
        #expect(scenario.maxTokens == 2048)
    }

    // MARK: - Artifact back-compatibility (A5)
    //
    // A5 added replicateIndex/temperature to turn records and
    // temperature/replicateIndex/seedPolicy to the run report. Both were made
    // optional so runs written before A5 still open; these pin that claim
    // rather than leaving it as a comment.

    @Test("a pre-A5 turn record still decodes")
    func preA5TurnRecordDecodes() throws {
        let json = """
            {"turnID":"t1","turnIndex":1,"title":"Memo","speakerAgentID":"a",
             "speakerName":"Judge A","prompt":"p","output":"o",
             "outputLabel":"memo_a","routedAgentIDs":["a","b"]}
            """
        let turn = try JSONDecoder().decode(
            MultiAgentTurnResult.self, from: Data(json.utf8))

        #expect(turn.turnID == "t1")
        #expect(turn.routedAgentIDs == ["a", "b"])
        // Absent, not defaulted to a value that would misreport the run.
        #expect(turn.replicateIndex == nil)
        #expect(turn.temperature == nil)
        #expect(turn.modelRevision == nil)
    }

    @Test("a pre-A5 run report still decodes")
    func preA5RunReportDecodes() throws {
        let json = """
            {"scenarioName":"panel","conditionName":"configured",
             "strippedInterventions":false,"scenarioHash":"h",
             "baseModelID":"test/model","runDirectory":"/tmp/r",
             "startedAt":"2026-07-01T00:00:00Z","completedAt":"2026-07-01T00:01:00Z",
             "turnCount":2,"warnings":[]}
            """
        let report = try JSONDecoder().decode(
            MultiAgentRunReport.self, from: Data(json.utf8))

        #expect(report.turnCount == 2)
        #expect(report.temperature == nil)
        #expect(report.replicateIndex == nil)
        #expect(report.seedPolicy == nil)
    }

    @Test("a warm local turn records its temperature and carries no seed policy claim")
    func warmTurnRecordRoundTrips() throws {
        let turn = MultiAgentTurnResult(
            turnID: "t1", turnIndex: 1, title: "Memo", speakerAgentID: "a",
            speakerName: "Judge A", modelRevision: "rev", prompt: "p", output: "o",
            outputLabel: "memo_a", routedAgentIDs: ["a"],
            replicateIndex: 3, temperature: 0.8)

        let data = try JSONEncoder().encode(turn)
        let decoded = try JSONDecoder().decode(MultiAgentTurnResult.self, from: data)

        #expect(decoded.replicateIndex == 3)
        #expect(decoded.temperature == 0.8)
        #expect(decoded == turn)
    }

    @Test("{{outputs.<label>}} interpolates a prior turn's output")
    func outputLabelsInterpolate() {
        let turn = MultiAgentScenario.Turn(
            title: "Revise",
            speakerAgentID: "a",
            promptTemplate: "Draft:\n{{outputs.draft_v1}}\n\nRevise it.")

        let rendered = MultiAgentRunner.renderPrompt(
            scenario: renderScenario(materials: ""), turn: turn,
            speakerName: "Judge A", speakerContext: "",
            outputsByLabel: ["draft_v1": "THE-DRAFT"])

        #expect(occurrences(of: "THE-DRAFT", in: rendered) == 1)
        #expect(!rendered.contains("{{outputs.draft_v1}}"))
    }
}
