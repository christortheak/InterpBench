import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit

/// The factorial/counterbalancing generator (Usability Plan Phase 4, items
/// 20–21) — pure functions: full-crossing counts and ids, substitution
/// completeness, options/target substitution, counterbalancing, metadata
/// round-trips through the existing task-prompt machinery, and byte
/// determinism. The write→pin transaction tests live in the serialized
/// suite below (process-global workspace override).
struct FactorialDesignTests {

    // MARK: Fixtures

    private func anchorFrameDesign(
        counterbalance: Bool = false
    ) -> FactorialDesign {
        FactorialDesign(
            factors: [
                .init(
                    name: "anchor",
                    levels: [
                        .init(name: "low", substitutions: ["DEMAND": "12 months"]),
                        .init(name: "high", substitutions: ["DEMAND": "9 years"]),
                    ]),
                .init(
                    name: "frame",
                    levels: [
                        .init(name: "gain", substitutions: ["FRAME": "spared"]),
                        .init(name: "loss", substitutions: ["FRAME": "denied"]),
                    ]),
            ],
            templates: [
                .init(
                    id: "t1",
                    text: "The prosecutor demands {{DEMAND}}; relief is {{FRAME}}.",
                    options: ["grant {{DEMAND}}", "deny"],
                    target: "grant {{DEMAND}}"),
                .init(
                    id: "t2",
                    text: "Sentence sought: {{DEMAND}} ({{FRAME}}).",
                    options: ["yes", "no"],
                    target: "no"),
            ],
            counterbalanceOptionOrder: counterbalance)
    }

    // MARK: Full crossing — counts and ids

    @Test func fullCrossingEmitsEveryCellTimesEveryTemplate() throws {
        let design = anchorFrameDesign()
        let items = try design.generate()
        #expect(items.count == 8)  // 2 × 2 × 2 templates
        #expect(design.itemCount == 8)
        // Deterministic ids encode the cell: template outermost, factors in
        // declared order, LAST factor varying fastest.
        #expect(items.map(\.id) == [
            "t1-anchor_low-frame_gain",
            "t1-anchor_low-frame_loss",
            "t1-anchor_high-frame_gain",
            "t1-anchor_high-frame_loss",
            "t2-anchor_low-frame_gain",
            "t2-anchor_low-frame_loss",
            "t2-anchor_high-frame_gain",
            "t2-anchor_high-frame_loss",
        ])
        // Factor metadata names the cell exactly.
        #expect(items[3].factors == ["anchor": "high", "frame": "loss"])
    }

    @Test func cellCountLineNamesEveryDimension() {
        let design = anchorFrameDesign(counterbalance: true)
        #expect(
            design.cellCountLine
                == "2 anchor × 2 frame × 2 templates × 2 orders = 16 items")
        #expect(design.itemCount == 16)
    }

    @Test func noFactorsEmitsTemplatesVerbatim() throws {
        let design = FactorialDesign(
            factors: [],
            templates: [.init(id: "t1", text: "no placeholders here")])
        let items = try design.generate()
        #expect(items.count == 1)
        #expect(items[0].id == "t1")
        #expect(items[0].text == "no placeholders here")
        #expect(items[0].factors.isEmpty)
    }

    // MARK: Substitution — text, options, target, completeness

    @Test func substitutionFillsTextOptionsAndTarget() throws {
        let items = try anchorFrameDesign().generate()
        let cell = items.first { $0.id == "t1-anchor_high-frame_loss" }
        #expect(cell?.text == "The prosecutor demands 9 years; relief is denied.")
        #expect(cell?.options == ["grant 9 years", "deny"])
        #expect(cell?.target == "grant 9 years")
    }

    @Test func unsubstitutedPlaceholderIsAHardErrorNamingVariableAndCell() {
        var design = anchorFrameDesign()
        design.templates[0].text += " Filed under {{DOCKET}}."
        do {
            _ = try design.generate()
            Issue.record("expected a completeness error")
        } catch let problem as FactorialDesign.Problem {
            #expect(problem.message.contains("{{DOCKET}}"))
            #expect(problem.message.contains("'DOCKET'"))
            #expect(problem.message.contains("template 't1'"))
            #expect(problem.message.contains("anchor=low, frame=gain"))
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }

    @Test func unsubstitutedPlaceholderInOptionsOrTargetAlsoRefuses() {
        var design = anchorFrameDesign()
        design.templates[1].options = ["yes", "appeal to {{COURT}}"]
        do {
            _ = try design.generate()
            Issue.record("expected a completeness error for the option")
        } catch let problem as FactorialDesign.Problem {
            #expect(problem.message.contains("'COURT'"))
            #expect(problem.message.contains("option 2"))
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
        var targetDesign = anchorFrameDesign()
        targetDesign.templates[1].target = "{{VERDICT}}"
        do {
            _ = try targetDesign.generate()
            Issue.record("expected a completeness error for the target")
        } catch let problem as FactorialDesign.Problem {
            #expect(problem.message.contains("'VERDICT'"))
            #expect(problem.message.contains("target"))
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }

    @Test func ambiguousVariableAcrossFactorsRefuses() {
        let design = FactorialDesign(
            factors: [
                .init(name: "a", levels: [.init(name: "x", substitutions: ["V": "1"])]),
                .init(name: "b", levels: [.init(name: "y", substitutions: ["V": "2"])]),
            ],
            templates: [.init(id: "t1", text: "{{V}}")])
        #expect(throws: FactorialDesign.Problem.self) { try design.generate() }
    }

    @Test func levelsOfOneFactorMustFillTheSameVariables() {
        let design = FactorialDesign(
            factors: [
                .init(
                    name: "a",
                    levels: [
                        .init(name: "x", substitutions: ["V": "1"]),
                        .init(name: "y", substitutions: ["W": "2"]),
                    ])
            ],
            templates: [.init(id: "t1", text: "{{V}}")])
        #expect(throws: FactorialDesign.Problem.self) { try design.generate() }
    }

    // MARK: Counterbalancing

    @Test func counterbalanceEmitsEachCellTwiceWithReversedOptions() throws {
        let items = try anchorFrameDesign(counterbalance: true).generate()
        #expect(items.count == 16)
        let original = items.first { $0.id == "t1-anchor_high-frame_loss-orderFlipped_false" }
        let flipped = items.first { $0.id == "t1-anchor_high-frame_loss-orderFlipped_true" }
        #expect(original?.options == ["grant 9 years", "deny"])
        #expect(flipped?.options == ["deny", "grant 9 years"])
        #expect(original?.factors["orderFlipped"] == "false")
        #expect(flipped?.factors["orderFlipped"] == "true")
        // The target names its option by value, so it FOLLOWS the option
        // through the reversal unchanged — and stays a member of options.
        #expect(flipped?.target == "grant 9 years")
        #expect(flipped?.options?.contains("grant 9 years") == true)
        // Unflipped copy always precedes the flipped one (determinism).
        let originalIndex = items.firstIndex { $0.id == original?.id }
        let flippedIndex = items.firstIndex { $0.id == flipped?.id }
        #expect(originalIndex != nil && flippedIndex != nil)
        if let originalIndex, let flippedIndex {
            #expect(flippedIndex == originalIndex + 1)
        }
    }

    @Test func counterbalanceRefusesTemplatesWithoutTwoOptions() {
        var design = anchorFrameDesign(counterbalance: true)
        design.templates[1].options = nil
        do {
            _ = try design.generate()
            Issue.record("expected a counterbalance refusal")
        } catch let problem as FactorialDesign.Problem {
            #expect(problem.message.contains("t2"))
            #expect(problem.message.contains("counterbalanc"))
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }

    @Test func reservedOrderFactorNameRefuses() {
        let design = FactorialDesign(
            factors: [
                .init(
                    name: "orderFlipped",
                    levels: [.init(name: "x", substitutions: [:])])
            ],
            templates: [.init(id: "t1", text: "t")])
        #expect(throws: FactorialDesign.Problem.self) { try design.generate() }
    }

    // MARK: Emitted JSONL — shape, round-trips, determinism

    @Test func jsonlIsStandardTaskPromptShapeAndParses() throws {
        let data = try anchorFrameDesign(counterbalance: true).generatedJSONL()
        // The run loop's own parser accepts the emitted records AND carries
        // the `factors` cell metadata into the parsed items (2026-07-20 —
        // it used to be dropped here, so factor metadata never reached run
        // evidence).
        let prompts = try ExperimentTasks.parseTaskPrompts(data)
        #expect(prompts.count == 16)
        #expect(prompts[0].id == "t1-anchor_low-frame_gain-orderFlipped_false")
        #expect(prompts[0].options == ["grant 12 months", "deny"])
        #expect(prompts[0].target == "grant 12 months")
        #expect(prompts[0].text.contains("12 months"))
        #expect(
            prompts[0].factors == [
                "anchor": "low", "frame": "gain", "orderFlipped": "false",
            ])
    }

    // MARK: Factors reach run evidence (review finding 3)

    @Test func parserCarriesFactorsAndValidatesTheirShape() throws {
        let prompts = try ExperimentTasks.parseTaskPrompts(
            Data("""
                {"id": "f1", "prompt": "a", "factors": {"anchor": "low"}}
                {"id": "plain", "prompt": "b"}
                {"id": "empty", "prompt": "c", "factors": {}}
                """.utf8))
        #expect(prompts[0].factors == ["anchor": "low"])
        // No factors — and an EMPTY factors object — parse as absent, so
        // non-factorial items keep producing byte-identical records.
        #expect(prompts[1].factors == nil)
        #expect(prompts[2].factors == nil)

        // A non-string level refuses with the pinned cross-engine message
        // (server twin: tasks._load_prompts).
        #expect {
            _ = try ExperimentTasks.parseTaskPrompts(
                Data("""
                    {"id": "bad", "prompt": "x", "factors": {"anchor": 3}}
                    """.utf8))
        } throws: { error in
            (error as? ExperimentError)?.reason
                == "task prompts: item 'bad' has a 'factors' value that is "
                + "not a flat string-to-string object — factor names and "
                + "level names must both be strings"
        }
        // A non-object factors value refuses too.
        #expect(throws: ExperimentError.self) {
            _ = try ExperimentTasks.parseTaskPrompts(
                Data("""
                    {"id": "bad2", "prompt": "x", "factors": "anchor=low"}
                    """.utf8))
        }
    }

    private func factorPrompt(_ factors: [String: String]?) -> ExperimentTasks.StudyPrompt {
        ExperimentTasks.StudyPrompt(
            id: "p1", text: "decide", options: ["yes", "no"], target: "yes",
            anchorMonths: nil, severity: nil, arm: nil, caseID: nil,
            factors: factors)
    }

    private func recordJSONKeys(of value: some Encodable) throws -> Set<String> {
        let object = try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(value))
        guard let dictionary = object as? [String: Any] else { return [] }
        return Set(dictionary.keys)
    }

    @Test func recordsCarryFactorsForBothRecordKinds() throws {
        let manifest = ExperimentManifest(
            name: "fact-rec", description: "", modelID: "test/model")
        let factors = ["anchor": "low", "frame": "gain"]
        let row = ExperimentTasks.MetricRow(
            condition: "baseline", seed: 0, promptIndex: 1, promptID: "p1",
            wordCount: 3, distinct2: 0.9, markerDensity: [:])

        // Sampled generation record: `factors` stamped verbatim.
        let sampled = ExperimentTasks.sampledGenerationRecord(
            manifest: manifest, experimentHash: "hh",
            taskPromptsFile: "f.jsonl", taskPromptsHash: "ph",
            promptMode: .chatAssistant, systemPrompt: nil,
            qwenThinkingEnabled: false, condition: "baseline", seed: 0,
            promptIndex: 1, prompt: factorPrompt(factors), output: "yes then",
            row: row)
        #expect(sampled.factors == factors)
        #expect(try recordJSONKeys(of: sampled).contains("factors"))

        // Choice-instrument record: same key, same value (instrument
        // readouts must be stratifiable too).
        let choice = ExperimentTasks.choiceRecord(
            manifest: manifest, experimentHash: "hh",
            taskPromptsFile: "f.jsonl", taskPromptsHash: "ph",
            promptMode: .chatAssistant, systemPrompt: nil,
            qwenThinkingEnabled: false, condition: "baseline",
            promptIndex: 1, prompt: factorPrompt(factors),
            choice: ChoiceResult(options: [
                OptionScore(option: "yes", tokenIDs: [1], tokenLogprobs: [-0.2]),
                OptionScore(option: "no", tokenIDs: [2], tokenLogprobs: [-2.0]),
            ]))
        #expect(choice.factors == factors)
        #expect(try recordJSONKeys(of: choice).contains("factors"))

        // A factor-less item produces records WITHOUT the key — byte-level
        // schema churn for non-factorial studies is zero.
        let plainSampled = ExperimentTasks.sampledGenerationRecord(
            manifest: manifest, experimentHash: "hh",
            taskPromptsFile: "f.jsonl", taskPromptsHash: "ph",
            promptMode: .chatAssistant, systemPrompt: nil,
            qwenThinkingEnabled: false, condition: "baseline", seed: 0,
            promptIndex: 1, prompt: factorPrompt(nil), output: "yes then",
            row: row)
        #expect(!(try recordJSONKeys(of: plainSampled).contains("factors")))
        let plainChoice = ExperimentTasks.choiceRecord(
            manifest: manifest, experimentHash: "hh",
            taskPromptsFile: "f.jsonl", taskPromptsHash: "ph",
            promptMode: .chatAssistant, systemPrompt: nil,
            qwenThinkingEnabled: false, condition: "baseline",
            promptIndex: 1, prompt: factorPrompt(nil),
            choice: ChoiceResult(options: [
                OptionScore(option: "yes", tokenIDs: [1], tokenLogprobs: [-0.2]),
                OptionScore(option: "no", tokenIDs: [2], tokenLogprobs: [-2.0]),
            ]))
        #expect(!(try recordJSONKeys(of: plainChoice).contains("factors")))
    }

    @Test func metricsCSVAddsFactorColumnsOnlyWhenDeclared() {
        func metricRow(
            condition: String, factors: [String: String] = [:]
        ) -> ExperimentTasks.MetricRow {
            ExperimentTasks.MetricRow(
                condition: condition, seed: 0, promptIndex: 1, promptID: "p1",
                wordCount: 5, distinct2: 0.5, markerDensity: [:],
                factors: factors)
        }
        // Factor-bearing rows: one `factor_<name>` column per name (sorted
        // union, appended last); rows missing a factor leave the cell empty.
        let csv = ExperimentTasks.metricsCSV(
            rows: [
                metricRow(
                    condition: "baseline",
                    factors: ["frame": "gain", "anchor": "low"]),
                metricRow(condition: "steered"),
            ], concepts: [])
        let lines = csv.split(separator: "\n")
        #expect(
            lines[0]
                == "condition,seed,promptIndex,promptID,wordCount,distinct2,"
                + "factor_anchor,factor_frame")
        #expect(lines[1] == "baseline,0,1,p1,5,0.5,low,gain")
        #expect(lines[2] == "steered,0,1,p1,5,0.5,,")

        // Factor-less rows: header and bytes identical to the historical
        // shape — no schema churn for non-factorial studies.
        let plain = ExperimentTasks.metricsCSV(
            rows: [metricRow(condition: "baseline")], concepts: [])
        #expect(
            plain.split(separator: "\n")[0]
                == "condition,seed,promptIndex,promptID,wordCount,distinct2")
        #expect(!plain.contains("factor_"))
    }

    /// The mission's explicit verification: the `factors` metadata key
    /// survives the EXISTING round-trip machinery — `TaskPromptsDocument`
    /// preserves it byte-faithfully on load/edit/save, and the run-loop
    /// parser accepts the re-serialized file.
    @Test func factorsMetadataSurvivesDocumentRoundTripAndReparse() throws {
        let data = try anchorFrameDesign().generatedJSONL()
        let document = try TaskPromptsDocument.load(data)
        #expect(document.count == 8)
        // Untouched save: byte-identical, factors intact.
        #expect(document.serialized() == data)
        // Edit ONE item's text through the editor path; its other keys —
        // factors included — must survive the rewrite.
        var texts = document.texts
        texts[0] = "edited prompt text"
        let edited = document.applyingEditedTexts(texts).serialized()
        let firstLine = String(decoding: edited, as: UTF8.self)
            .split(separator: "\n")[0]
        #expect(firstLine.contains("\"factors\""))
        #expect(firstLine.contains("\"anchor\":\"low\""))
        #expect(firstLine.contains("edited prompt text"))
        // And the run loop still parses the edited file.
        let prompts = try ExperimentTasks.parseTaskPrompts(edited)
        #expect(prompts[0].text == "edited prompt text")
    }

    @Test func regenerationIsByteIdentical() throws {
        let first = try anchorFrameDesign(counterbalance: true).generatedJSONL()
        let second = try anchorFrameDesign(counterbalance: true).generatedJSONL()
        #expect(first == second)
        let firstHash = SHA256.hash(data: first)
            .map { String(format: "%02x", $0) }.joined()
        let secondHash = SHA256.hash(data: second)
            .map { String(format: "%02x", $0) }.joined()
        #expect(firstHash == secondHash)
    }

    // MARK: Design JSON round-trip (Load/Save design…, study-pack files)

    @Test func designJSONRoundTripsExactly() throws {
        let design = anchorFrameDesign(counterbalance: true)
        let encoded = try design.encoded()
        let decoded = try FactorialDesign.decode(encoded)
        #expect(decoded == design)
        // Canonical bytes: re-encoding the decoded spec is byte-identical.
        #expect(try decoded.encoded() == encoded)
    }

    @Test func missingCounterbalanceKeyDecodesAsFalse() throws {
        let json = """
            {"templates": [{"id": "t1", "text": "plain"}]}
            """
        let decoded = try FactorialDesign.decode(Data(json.utf8))
        #expect(decoded.counterbalanceOptionOrder == false)
        #expect(decoded.factors.isEmpty)
    }

    @Test func garbageDesignJSONRefusesPlainly() {
        do {
            _ = try FactorialDesign.decode(Data("not json".utf8))
            Issue.record("expected a decode refusal")
        } catch let problem as FactorialDesign.Problem {
            #expect(problem.message.contains("not a factorial design"))
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }

    // MARK: Structural validation

    @Test func duplicateAndUnsafeNamesRefuse() {
        // Duplicate template ids.
        #expect(throws: FactorialDesign.Problem.self) {
            try FactorialDesign(
                factors: [],
                templates: [
                    .init(id: "t1", text: "a"), .init(id: "t1", text: "b"),
                ]
            ).generate()
        }
        // A level name that would corrupt ids.
        #expect(throws: FactorialDesign.Problem.self) {
            try FactorialDesign(
                factors: [
                    .init(
                        name: "anchor",
                        levels: [.init(name: "hi there", substitutions: [:])])
                ],
                templates: [.init(id: "t1", text: "a")]
            ).generate()
        }
        // No templates at all.
        #expect(throws: FactorialDesign.Problem.self) {
            try FactorialDesign(factors: [], templates: []).generate()
        }
        // A substitution VALUE containing a placeholder (never recursive).
        #expect(throws: FactorialDesign.Problem.self) {
            try FactorialDesign(
                factors: [
                    .init(
                        name: "a",
                        levels: [.init(name: "x", substitutions: ["V": "{{W}}"])])
                ],
                templates: [.init(id: "t1", text: "{{V}}")]
            ).generate()
        }
    }
}

/// Write + pin transaction (temp workspace) — its own serialized suite
/// because it uses the process-global workspace override, guarded by the
/// shared `ExperimentRootOverrideLock` (the same precedent as the
/// task-prompts import tests).
@Suite(.serialized) struct FactorialImportTests {

    private func withWorkspace<T>(_ body: (URL) throws -> T) rethrows -> T {
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "factorial-import-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        ExperimentRootOverrideLock.acquire()
        WorkspaceRoot.programmaticOverride = temp
        defer {
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: temp)
        }
        return try body(temp)
    }

    private func smallDesign(text: String = "Demand: {{DEMAND}}.") -> FactorialDesign {
        FactorialDesign(
            factors: [
                .init(
                    name: "anchor",
                    levels: [
                        .init(name: "low", substitutions: ["DEMAND": "12 months"]),
                        .init(name: "high", substitutions: ["DEMAND": "9 years"]),
                    ])
            ],
            templates: [.init(id: "t1", text: text)])
    }

    @Test func generateLandsAtScaffoldDestinationPinsAndSavesDesign() throws {
        try withWorkspace { root in
            var manifest = try ExperimentStore.create(
                name: "fact1", description: "", modelID: "org/m")
            let result = try FactorialImport.generateIntoStudy(
                design: smallDesign(), manifest: &manifest,
                persist: { try ExperimentStore.save($0) })
            #expect(result.file
                == DataTemplates.taskPromptsDestination(experiment: "fact1"))
            #expect(result.designFile
                == "prompts/tasks/fact1-prompts-design.json")
            #expect(result.itemCount == 2)
            // The emitted file is pinned exactly like a hand-authored one.
            let written = try Data(contentsOf: root.appending(path: result.file))
            let expectedHash = SHA256.hash(data: written)
                .map { String(format: "%02x", $0) }.joined()
            #expect(manifest.taskPromptsFile == result.file)
            #expect(manifest.taskPromptsHash == expectedHash)
            #expect(result.hash == expectedHash)
            // The provenance spec sits next to the output and round-trips.
            let designData = try Data(
                contentsOf: root.appending(path: result.designFile))
            #expect(try FactorialDesign.decode(designData) == smallDesign())
            // Persisted through the transaction's save.
            #expect(try ExperimentStore.load(name: "fact1").taskPromptsHash
                == expectedHash)
        }
    }

    @Test func differingExistingFileRefusesAndNothingMoves() throws {
        try withWorkspace { root in
            var manifest = try ExperimentStore.create(
                name: "fact2", description: "", modelID: "org/m")
            let first = try FactorialImport.generateIntoStudy(
                design: smallDesign(), manifest: &manifest,
                persist: { try ExperimentStore.save($0) })
            // Identical bytes: idempotent re-generation, same pin.
            let again = try FactorialImport.generateIntoStudy(
                design: smallDesign(), manifest: &manifest,
                persist: { try ExperimentStore.save($0) })
            #expect(again.hash == first.hash)
            // A changed design refuses without the replace affordance, and
            // neither the file nor the pin moves.
            do {
                _ = try FactorialImport.generateIntoStudy(
                    design: smallDesign(text: "Changed: {{DEMAND}}."),
                    manifest: &manifest,
                    persist: { try ExperimentStore.save($0) })
                Issue.record("differing existing file should refuse")
            } catch let problem as TabularImport.Problem {
                #expect(problem.message.contains("never overwrite"))
                #expect(problem.message.contains("Replace the existing file"))
            }
            let written = try Data(contentsOf: root.appending(path: first.file))
            #expect(String(decoding: written, as: UTF8.self).contains("Demand:"))
            #expect(manifest.taskPromptsHash == first.hash)
        }
    }

    @Test func replaceAffordanceReplacesBothFilesAndRePins() throws {
        try withWorkspace { root in
            var manifest = try ExperimentStore.create(
                name: "fact3", description: "", modelID: "org/m")
            let first = try FactorialImport.generateIntoStudy(
                design: smallDesign(), manifest: &manifest,
                persist: { try ExperimentStore.save($0) })
            let changed = smallDesign(text: "Changed: {{DEMAND}}.")
            let replaced = try FactorialImport.generateIntoStudy(
                design: changed, manifest: &manifest,
                replacingExisting: true,
                persist: { try ExperimentStore.save($0) })
            #expect(replaced.hash != first.hash)
            let written = try Data(contentsOf: root.appending(path: first.file))
            #expect(String(decoding: written, as: UTF8.self).contains("Changed:"))
            let designData = try Data(
                contentsOf: root.appending(path: first.designFile))
            #expect(try FactorialDesign.decode(designData) == changed)
            #expect(try ExperimentStore.load(name: "fact3").taskPromptsHash
                == replaced.hash)
        }
    }

    /// A failure AFTER the writes (the caller's persist step) rolls BOTH
    /// files back: a first generation leaves no orphans; a replace restores
    /// the previous bytes of prompts AND design.
    @Test func persistFailureRollsBackBothWrites() throws {
        try withWorkspace { root in
            var manifest = try ExperimentStore.create(
                name: "fact4", description: "", modelID: "org/m")
            struct PersistRefused: Error {}
            do {
                _ = try FactorialImport.generateIntoStudy(
                    design: smallDesign(), manifest: &manifest,
                    persist: { _ in throw PersistRefused() })
                Issue.record("persist failure should throw")
            } catch is PersistRefused {}
            let prompts = root.appending(
                path: DataTemplates.taskPromptsDestination(experiment: "fact4"))
            let design = root.appending(
                path: FactorialImport.designDestination(experiment: "fact4"))
            #expect(!FileManager.default.fileExists(atPath: prompts.path))
            #expect(!FileManager.default.fileExists(atPath: design.path))

            // Now the replace case: a later failure restores PREVIOUS bytes.
            let first = try FactorialImport.generateIntoStudy(
                design: smallDesign(), manifest: &manifest,
                persist: { try ExperimentStore.save($0) })
            do {
                _ = try FactorialImport.generateIntoStudy(
                    design: smallDesign(text: "Changed: {{DEMAND}}."),
                    manifest: &manifest, replacingExisting: true,
                    persist: { _ in throw PersistRefused() })
                Issue.record("persist failure should throw")
            } catch is PersistRefused {}
            let written = try Data(contentsOf: root.appending(path: first.file))
            #expect(String(decoding: written, as: UTF8.self).contains("Demand:"))
            let designData = try Data(
                contentsOf: root.appending(path: first.designFile))
            #expect(try FactorialDesign.decode(designData) == smallDesign())
            #expect(try ExperimentStore.load(name: "fact4").taskPromptsHash
                == first.hash)
        }
    }

    /// An invalid design refuses BEFORE anything lands — generation
    /// validates ahead of the first write.
    @Test func invalidDesignWritesNothing() throws {
        try withWorkspace { root in
            var manifest = try ExperimentStore.create(
                name: "fact5", description: "", modelID: "org/m")
            let before = manifest
            #expect(throws: FactorialDesign.Problem.self) {
                try FactorialImport.generateIntoStudy(
                    design: smallDesign(text: "Unfilled {{HOLE}}"),
                    manifest: &manifest)
            }
            #expect(manifest == before)
            let prompts = root.appending(
                path: DataTemplates.taskPromptsDestination(experiment: "fact5"))
            #expect(!FileManager.default.fileExists(atPath: prompts.path))
        }
    }
}
