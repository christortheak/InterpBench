import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// The chat-format concept importer (`concept import-chat`, and the New
/// Dataset sheet's "Import chat pairs").
///
/// The first test is the cross-engine contract: the Python engine wrote
/// `Tests/Fixtures/cross-engine/chat-concept-import.json`
/// (`scripts/regenerate-cross-engine-fixtures.py`), and this engine must write
/// the same bytes for every success case and give the same code, sentence,
/// and repair for every refusal. The rest drive the verb and the app's plan
/// against a temporary workspace.
///
/// Serialized, and holding `ExperimentRootOverrideLock`: the CLI tests point
/// the process-global `rootOverride` at a temporary workspace.
@Suite(.serialized) struct ChatConceptImportTests {

    // MARK: The cross-engine fixture

    struct Fixture: Decodable {
        struct ExpectedFile: Decodable {
            let name: String
            let text: String
        }

        struct ExpectedRefusal: Decodable {
            let code: String
            let reason: String
            let repairAction: String
            let state: String
        }

        struct ExpectedSide: Decodable, Equatable {
            let keptRepeats: Int
            let validationInExtraction: Int
        }

        struct ExpectedAdvisory: Decodable, Equatable {
            let code: String
            let detail: String
        }

        struct Case: Decodable {
            let label: String
            let name: String
            let positiveFile: String
            let negativeFile: String
            let positive: String?
            let positiveBase64: String?
            let negative: String
            let validationPairs: Int
            let seed: UInt64?
            let files: [ExpectedFile]?
            let stimulusSetHash: String?
            let validationHash: String?
            let repeatedTexts: [String: ExpectedSide]?
            let advisories: [ExpectedAdvisory]?
            let summaryLines: [String]?
            let refusal: ExpectedRefusal?
        }

        let createdAt: String
        let cases: [Case]
    }

    static func fixture() throws -> Fixture {
        let url = CodeResources.compiledCheckoutPath.appending(
            components: "Tests", "Fixtures", "cross-engine", "chat-concept-import.json")
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }

    @Test func everyCaseWritesThePythonEnginesBytesOrItsRefusal() throws {
        let fixture = try Self.fixture()
        #expect(fixture.cases.count >= 20, "the fixture's coverage shrank")
        var successes = 0
        for item in fixture.cases {
            let positive =
                try item.positiveBase64.map { try #require(Data(base64Encoded: $0)) }
                ?? Data(try #require(item.positive).utf8)
            let outcome = Result {
                try ChatConceptImport.prepare(
                    name: item.name,
                    positive: .init(file: item.positiveFile, data: positive),
                    negative: .init(file: item.negativeFile, data: Data(item.negative.utf8)),
                    validationPairs: item.validationPairs, seed: item.seed,
                    createdAt: fixture.createdAt)
            }
            switch outcome {
            case .success(let prepared):
                let expected = try #require(item.files, "\(item.label): Python refused")
                successes += 1
                #expect(prepared.files.map(\.name) == expected.map(\.name), "\(item.label)")
                for (written, wanted) in zip(prepared.files, expected) {
                    #expect(
                        written.data == Data(wanted.text.utf8),
                        "\(item.label): \(written.name) differs from the Python engine's bytes")
                }
                #expect(prepared.stimulusSetHash == item.stimulusSetHash, "\(item.label)")
                #expect(prepared.validationHash == item.validationHash, "\(item.label)")
                // The repeated-text counts, the advisory, and the human
                // lines: the same numbers and sentences as the Python engine.
                let sides = try #require(item.repeatedTexts, "\(item.label): stale fixture")
                #expect(
                    sides["positive"]
                        == .init(
                            keptRepeats: prepared.positiveRepeats.keptRepeats,
                            validationInExtraction: prepared.positiveRepeats.validationInExtraction),
                    "\(item.label)")
                #expect(
                    sides["negative"]
                        == .init(
                            keptRepeats: prepared.negativeRepeats.keptRepeats,
                            validationInExtraction: prepared.negativeRepeats.validationInExtraction),
                    "\(item.label)")
                let outcome = ChatConceptImport.Outcome(
                    prepared: prepared, directory: URL(fileURLWithPath: "/Volumes/Lab"),
                    relativeDirectory: "prompts/concepts/\(prepared.name)")
                let advisories = try #require(item.advisories, "\(item.label): stale fixture")
                #expect(
                    outcome.advisories
                        == advisories.map {
                            SteerLabCLIEnvelope.Advisory(code: $0.code, detail: $0.detail)
                        },
                    "\(item.label)")
                #expect(outcome.summaryLines == item.summaryLines, "\(item.label)")
                if case .object(let payload)? = outcome.payload["repeatedTexts"] {
                    #expect(Set(payload.keys) == ["negative", "positive"], "\(item.label)")
                } else {
                    Issue.record("\(item.label): no repeatedTexts in the payload")
                }
            case .failure(let error):
                let wanted = try #require(item.refusal, "\(item.label): Swift refused, Python did not — \(error)")
                let refusal = try #require(error as? ChatConceptImport.Refusal, "\(item.label): \(error)")
                #expect(refusal.code == wanted.code, "\(item.label)")
                #expect(refusal.reason == wanted.reason, "\(item.label)")
                #expect(refusal.repairAction == wanted.repairAction, "\(item.label)")
                #expect(refusal.state.rawValue == wanted.state, "\(item.label)")
            }
        }
        #expect(successes >= 2)
    }

    /// The rule's two subtle halves, said directly rather than only through
    /// the fixture: prompts are compared code point by code point (Swift's
    /// `==` would call NFC and NFD equal), and Foundation's own reader is not
    /// what decides a row (it accepts a trailing comma).
    @Test func promptsCompareByScalarsAndTheReaderIsStrict() throws {
        #expect("caf\u{e9}" == "cafe\u{301}")
        #expect(!ChatConceptImport.same("caf\u{e9}", "cafe\u{301}"))
        let trailing = Data(#"{"messages": [],}"#.utf8)
        #expect((try? JSONSerialization.jsonObject(with: trailing)) != nil)
        #expect(throws: ChatConceptImport.Refusal.self) {
            try ChatConceptImport.readRows(trailing, file: "f.jsonl")
        }
    }

    // MARK: The verb

    static func row(_ prompt: String, _ reply: String) -> String {
        let row: [String: Any] = [
            "messages": [
                ["role": "user", "content": prompt],
                ["role": "assistant", "content": reply],
            ]
        ]
        let data = try! JSONSerialization.data(withJSONObject: row, options: [.sortedKeys])
        return String(decoding: data, as: UTF8.self)
    }

    /// Two row-aligned chat files with `count` rows, in `directory`.
    static func writeSources(in directory: URL, count: Int) throws -> (URL, URL) {
        let positive = directory.appending(component: "claims.jsonl")
        let negative = directory.appending(component: "denials.jsonl")
        try ((1 ... count).map { row("Prompt \($0)", "I feel \($0).") }
            .joined(separator: "\n") + "\n").write(to: positive, atomically: true, encoding: .utf8)
        try ((1 ... count).map { row("Prompt \($0)", "I compute \($0).") }
            .joined(separator: "\n") + "\n").write(to: negative, atomically: true, encoding: .utf8)
        return (positive, negative)
    }

    func withTempRoot<T>(_ body: (URL) async throws -> T) async throws -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "chat-import-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: temp.appending(path: "prompts/concepts"), withIntermediateDirectories: true)
        ExperimentStore.rootOverride = temp
        defer {
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try await body(temp)
    }

    static let clock = Date(timeIntervalSince1970: 1_760_832_000)

    func invoke(_ args: [String], recorder: ExperimentCLIRecorder = .init()) async
        -> ExperimentCLIOutcome
    {
        await ExperimentCLIRunner(sink: recorder.sink, now: { Self.clock })
            .run(namespace: "concept", args)
    }

    @Test func theVerbWritesAConceptTheStimulusLoadersRead() async throws {
        try await withTempRoot { root in
            let (positive, negative) = try Self.writeSources(in: root, count: 6)
            let recorder = ExperimentCLIRecorder()
            let outcome = await invoke(
                [
                    "import-chat", "machine-experience", "--positive", positive.path,
                    "--negative", negative.path, "--validation-pairs", "2", "--seed", "7",
                    "--json",
                ], recorder: recorder)
            #expect(outcome.exitCode == 0)
            #expect(outcome.envelope.state == .ready)
            #expect(outcome.envelope.verb == "concept import-chat")
            #expect(outcome.envelope.changed)
            #expect(outcome.envelope.nextAction?.verb == "experiment attach <name> machine-experience")
            // Six different pairs: nothing repeats, so nothing is advised.
            #expect((outcome.envelope.advisories ?? []).isEmpty)
            let zero: JSONValue = .object([
                "keptRepeats": .number(0), "validationInExtraction": .number(0),
            ])
            #expect(
                outcome.envelope.result?["repeatedTexts"]
                    == .object(["negative": zero, "positive": zero]))

            let directory = root.appending(path: "prompts/concepts/machine-experience")
            let names = try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted()
            #expect(
                names == [
                    "concept.json", "negative.jsonl", "positive.jsonl", "provenance.json",
                    "validation.jsonl",
                ])
            // The loaders extraction and validation use accept what was
            // written, and the reported hash is the one `attach` will pin.
            let set = try StimulusSet(directory: directory)
            #expect(set.positive == ["I feel 1.", "I feel 2.", "I feel 3.", "I feel 5."])
            #expect(set.negative == ["I compute 1.", "I compute 2.", "I compute 3.", "I compute 5."])
            #expect(outcome.envelope.result?["stimulusSetHash"] == .string(set.hash))
            let validation = try #require(try StimulusSet.loadValidation(directory: directory))
            #expect(validation.map(\.text) == ["I feel 4.", "I compute 4.", "I feel 6.", "I compute 6."])
            #expect(validation.map(\.expresses) == [true, false, true, false])
            // The draw is the token bank's, and its rows are recorded.
            #expect(TokenBankDownsampler.selectedIndices(count: 6, cap: 2, seed: 7) == [3, 5])
            let provenance = try JSONSerialization.jsonObject(
                with: Data(contentsOf: directory.appending(component: "provenance.json")))
                as? [String: Any]
            let held = provenance?["validation"] as? [String: Any]
            #expect(held?["rows"] as? [Int] == [4, 6])
            #expect(held?["seed"] as? Int == 7)
            let sources = provenance?["sources"] as? [String: [String: Any]]
            #expect(sources?["positive"]?["file"] as? String == "claims.jsonl")
            #expect(
                sources?["positive"]?["sha256"] as? String
                    == ChatConceptImport.sha256(try Data(contentsOf: positive)))
            #expect(
                try String(contentsOf: directory.appending(component: "concept.json"), encoding: .utf8)
                    .contains(#""createdAt" : "2025-10-19T00:00:00Z""#))
            // Nothing staged is left beside the concept.
            let siblings = try FileManager.default.contentsOfDirectory(
                atPath: root.appending(path: "prompts/concepts").path)
            #expect(siblings == ["machine-experience"])
        }
    }

    @Test func refusalsAreTypedAndWriteNothing() async throws {
        try await withTempRoot { root in
            let (positive, negative) = try Self.writeSources(in: root, count: 3)
            let concepts = root.appending(path: "prompts/concepts")

            // A concept that exists is never added to or replaced.
            try FileManager.default.createDirectory(
                at: concepts.appending(component: "taken"), withIntermediateDirectories: true)
            let taken = await invoke([
                "import-chat", "taken", "--positive", positive.path, "--negative", negative.path,
                "--json",
            ])
            #expect(taken.exitCode == 65)
            #expect(taken.envelope.state == .refused)
            #expect(taken.envelope.error?.code == "conceptExists")
            #expect(try FileManager.default.contentsOfDirectory(
                atPath: concepts.appending(component: "taken").path).isEmpty)

            // A missing source is 66 in both modes.
            let missing = await invoke([
                "import-chat", "fresh", "--positive", positive.path + "x", "--negative",
                negative.path,
            ])
            #expect(missing.exitCode == 66)
            #expect(missing.envelope.error?.code == "notFound")
            #expect(missing.failure?.repairAction == "check the path to the positive chat file")

            // Malformed requests are 64.
            for args in [
                ["import-chat", "Fresh Name", "--positive", positive.path, "--negative", negative.path],
                ["import-chat", "fresh", "--positive", positive.path],
                ["import-chat", "fresh", "--positive", positive.path, "--negative", negative.path,
                 "--validation-pairs", "1"],
                ["import-chat", "fresh", "--positive", positive.path, "--negative", negative.path,
                 "--validation-pairs", "1", "--seed", "9007199254740992"],
            ] {
                let outcome = await invoke(args)
                #expect(outcome.exitCode == 64, "\(args)")
                #expect(outcome.envelope.state == .blocked, "\(args)")
                let repair = outcome.envelope.error?.repairAction ?? ""
                #expect(repair.isEmpty == false, "\(args)")
            }

            // Mismatched files refuse with 65, and leave nothing behind.
            try (Self.row("Prompt 1", "I feel 1.") + "\n").write(
                to: positive, atomically: true, encoding: .utf8)
            let mismatch = await invoke([
                "import-chat", "fresh", "--positive", positive.path, "--negative", negative.path,
            ])
            #expect(mismatch.exitCode == 65)
            #expect(mismatch.envelope.error?.code == "rowCountMismatch")
            #expect(try FileManager.default.contentsOfDirectory(atPath: concepts.path) == ["taken"])
        }
    }

    // MARK: Repeated texts: a row hold-out is not a text hold-out

    // Every count below is worked out by hand from the rows; the fixture
    // test above holds these numbers and sentences to the Python engine's.

    /// The advisory, written out here rather than read from the importer, so
    /// a change to the sentence has to be made on purpose.
    static func overlapAdvice(total: Int, texts: Int, positive: Int, negative: Int) -> String {
        "positive.jsonl and negative.jsonl also contain \(total) of the \(texts) "
            + "texts in validation.jsonl (\(positive) positive, \(negative) negative), so "
            + "validation can read better than it would on unseen text. Pairs are held "
            + "out by row, not by text. To hold out unseen texts, remove the repeated "
            + "rows from the source files and import again under a new name, or import "
            + "with no hold-out and write validation.jsonl from texts the extraction "
            + "files do not contain."
    }

    /// `prepare` over `(prompt, positive reply, negative reply)` rows.
    static func prepare(
        _ rows: [(String, String, String)], validationPairs: Int = 0, seed: UInt64? = nil
    ) throws -> ChatConceptImport.Outcome {
        let positive = rows.map { row($0.0, $0.1) + "\n" }.joined()
        let negative = rows.map { row($0.0, $0.2) + "\n" }.joined()
        let prepared = try ChatConceptImport.prepare(
            name: "repeats",
            positive: .init(file: "claims.jsonl", data: Data(positive.utf8)),
            negative: .init(file: "denials.jsonl", data: Data(negative.utf8)),
            validationPairs: validationPairs, seed: seed, createdAt: "2026-10-19T00:00:00Z")
        return ChatConceptImport.Outcome(
            prepared: prepared, directory: URL(fileURLWithPath: "/Volumes/Lab/repeats"),
            relativeDirectory: "prompts/concepts/repeats")
    }

    /// (positive keptRepeats, negative keptRepeats, positive
    /// validationInExtraction, negative validationInExtraction).
    static func counts(_ outcome: ChatConceptImport.Outcome) -> [Int] {
        let p = outcome.prepared
        return [
            p.positiveRepeats.keptRepeats, p.negativeRepeats.keptRepeats,
            p.positiveRepeats.validationInExtraction, p.negativeRepeats.validationInExtraction,
        ]
    }

    static func heldRows(_ outcome: ChatConceptImport.Outcome) -> [Int] {
        guard case .object(let provenance) = outcome.prepared.provenance,
            case .object(let validation)? = provenance["validation"],
            case .array(let rows)? = validation["rows"]
        else { return [] }
        return rows.compactMap { (value: ChatConceptImport.Value) -> Int? in
            guard case .integer(let row) = value else { return nil }
            return row
        }
    }

    static func written(_ outcome: ChatConceptImport.Outcome, _ name: String) -> String {
        String(decoding: outcome.prepared.files.first { $0.name == name }?.data ?? Data(), as: UTF8.self)
    }

    @Test func noRepeatedTextsCountZeroAndAdviseNothing() throws {
        let outcome = try Self.prepare(
            (1 ... 6).map { ("Prompt \($0)", "I feel \($0).", "I compute \($0).") },
            validationPairs: 2, seed: 7)
        #expect(Self.counts(outcome) == [0, 0, 0, 0])
        #expect(outcome.prepared.overlapAdvisory == nil)
        #expect(outcome.advisories.isEmpty)
        // Only the two lines every import prints: what was imported, and where.
        #expect(outcome.summaryLines.map { $0.split(separator: " ")[0] } == ["imported", "wrote"])
    }

    @Test func repeatsInsideExtractionAreCountedButNotAdvised() throws {
        // Positive "Warm." twice (rows 2, 3); negative "Dim." twice (rows 4,
        // 5). Seed 0 holds out row 1, whose texts appear nowhere else.
        let outcome = try Self.prepare(
            [
                ("Prompt 1", "Even.", "Odd."),
                ("Prompt 2", "Warm.", "Cold."),
                ("Prompt 3", "Warm.", "Chilly."),
                ("Prompt 4", "Bright.", "Dim."),
                ("Prompt 5", "Vivid.", "Dim."),
            ], validationPairs: 1, seed: 0)
        #expect(Self.heldRows(outcome) == [1])
        #expect(Self.counts(outcome) == [1, 1, 0, 0])
        #expect(outcome.prepared.overlapAdvisory == nil)
        #expect(outcome.advisories.isEmpty)
        #expect(
            outcome.summaryLines.contains(
                "1 of the 5 kept positive replies and 1 of the 5 kept negative "
                    + "replies repeat an earlier reply on the same side, and every copy "
                    + "is kept"))
    }

    /// The review's case: one pair four times over, one pair held out. Both
    /// validation texts are also extraction texts, and the import still
    /// succeeds, writes every copy, and keeps its provenance shape.
    @Test func onePairFourTimesWithOneHeldOutIsAdvisedNotRefused() throws {
        let outcome = try Self.prepare(
            Array(repeating: ("Prompt", "I feel.", "I compute."), count: 4),
            validationPairs: 1, seed: 1)
        #expect(Self.heldRows(outcome) == [2])
        #expect(Self.counts(outcome) == [3, 3, 1, 1])
        let advice = Self.overlapAdvice(total: 2, texts: 2, positive: 1, negative: 1)
        #expect(outcome.prepared.overlapAdvisory == advice)
        #expect(outcome.advisories == [.init(code: "validationOverlapsExtraction", detail: advice)])
        #expect(outcome.summaryLines.last == advice)
        // The claim itself, read off the written files.
        #expect(Self.written(outcome, "positive.jsonl") == String(repeating: #"{"text": "I feel."}"# + "\n", count: 3))
        #expect(Self.written(outcome, "negative.jsonl") == String(repeating: #"{"text": "I compute."}"# + "\n", count: 3))
        #expect(
            Self.written(outcome, "validation.jsonl")
                == #"{"text": "I feel.", "expresses": true}"# + "\n"
                + #"{"text": "I compute.", "expresses": false}"# + "\n")
        // Reported, never written: provenance.json keeps the chat-pairs-v1 keys.
        let provenance = try #require(
            try JSONSerialization.jsonObject(with: Data(Self.written(outcome, "provenance.json").utf8))
                as? [String: Any])
        #expect(
            Set(provenance.keys) == [
                "artifactType", "concept", "dropped", "importedAt", "importer", "pairs", "rule",
                "schemaVersion", "sources", "validation",
            ])
        #expect(provenance["importer"] as? String == "chat-pairs-v1")
    }

    @Test func repeatsOnBothSidesOfTheSplitAreCountedByScalar() throws {
        // Seed 36 holds out rows 2, 4, and 8. Extraction keeps 1, 3, 5, 6, 7.
        //   positive: "Warm." (row 2) is row 1's and "Bright." (row 4) is row
        //     3's, so 2 held-out texts are extraction texts; NFD "café" (row
        //     8) is not NFC "café" (row 7), though Swift's `==` says it is.
        //     Repeats among the 8 kept: Warm., Bright. -> 2.
        //   negative: "Cold." (row 2) is row 1's; "Grey." and "Juice." are
        //     not, so 1. Repeats among the 8 kept: Cold., and "Quiet." (rows
        //     5, 6, both extraction) -> 2.
        #expect("caf\u{e9}" == "cafe\u{301}")
        let outcome = try Self.prepare(
            [
                ("Prompt 1", "Warm.", "Cold."),
                ("Prompt 2", "Warm.", "Cold."),
                ("Prompt 3", "Bright.", "Dim."),
                ("Prompt 4", "Bright.", "Grey."),
                ("Prompt 5", "Loud.", "Quiet."),
                ("Prompt 6", "Soft.", "Quiet."),
                ("Prompt 7", "caf\u{e9}", "Tea."),
                ("Prompt 8", "cafe\u{301}", "Juice."),
            ], validationPairs: 3, seed: 36)
        #expect(Self.heldRows(outcome) == [2, 4, 8])
        #expect(Self.counts(outcome) == [2, 2, 2, 1])
        let advice = Self.overlapAdvice(total: 3, texts: 6, positive: 2, negative: 1)
        #expect(outcome.advisories == [.init(code: "validationOverlapsExtraction", detail: advice)])
        #expect(
            Array(outcome.summaryLines.suffix(2)) == [
                "2 of the 8 kept positive replies and 2 of the 8 kept negative replies "
                    + "repeat an earlier reply on the same side, and every copy is kept",
                advice,
            ])
    }

    @Test func theVerbReportsAnOverlappingHoldOutAsAnAdvisory() async throws {
        try await withTempRoot { root in
            let positive = root.appending(component: "claims.jsonl")
            let negative = root.appending(component: "denials.jsonl")
            try String(repeating: Self.row("Prompt", "I feel.") + "\n", count: 4)
                .write(to: positive, atomically: true, encoding: .utf8)
            try String(repeating: Self.row("Prompt", "I compute.") + "\n", count: 4)
                .write(to: negative, atomically: true, encoding: .utf8)
            let recorder = ExperimentCLIRecorder()
            let outcome = await invoke(
                [
                    "import-chat", "one-pair", "--positive", positive.path,
                    "--negative", negative.path, "--validation-pairs", "1", "--seed", "1",
                    "--json",
                ], recorder: recorder)
            let advice = Self.overlapAdvice(total: 2, texts: 2, positive: 1, negative: 1)
            #expect(outcome.exitCode == 0)
            #expect(outcome.envelope.state == .okWithAdvisories)
            #expect(outcome.envelope.changed)
            #expect(
                outcome.envelope.advisories == [
                    .init(code: "validationOverlapsExtraction", detail: advice)
                ])
            let side: JSONValue = .object([
                "keptRepeats": .number(3), "validationInExtraction": .number(1),
            ])
            #expect(
                outcome.envelope.result?["repeatedTexts"]
                    == .object(["negative": side, "positive": side]))
            #expect(
                FileManager.default.fileExists(
                    atPath: root.appending(path: "prompts/concepts/one-pair/validation.jsonl").path))
        }
    }

    // MARK: The app's route

    @Test func theNewDatasetPlanImportsChatPairsIntoItsDestination() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "chat-import-plan-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let (positive, negative) = try Self.writeSources(in: root, count: 4)

        let plan = DatasetCreationPlanner.plan(
            DatasetCreationRequest(role: .conceptStimuli, rawName: "Machine Experience"),
            root: root)
        #expect(plan.name == "machine-experience")
        let outcome = try plan.applyChatImport(
            positive: positive, negative: negative, validationPairs: 1, seed: 3,
            now: Self.clock)
        #expect(outcome.directory.standardizedFileURL == plan.directory.standardizedFileURL)
        #expect(outcome.prepared.extractionPairs == 3)
        #expect(try StimulusSet(directory: plan.directory).positive.count == 3)

        // The same plan again: the destination now exists, and the import says so.
        let again = DatasetCreationPlanner.plan(
            DatasetCreationRequest(role: .conceptStimuli, rawName: "machine-experience"),
            root: root)
        #expect(throws: ChatConceptImport.Refusal.self) {
            try again.applyChatImport(
                positive: positive, negative: negative, validationPairs: 0, seed: nil)
        }
        // Only a concept stimulus set takes chat pairs.
        let battery = DatasetCreationPlanner.plan(
            DatasetCreationRequest(role: .capabilityBattery, rawName: "x"), root: root)
        #expect(throws: DatasetCreationError.self) {
            try battery.applyChatImport(
                positive: positive, negative: negative, validationPairs: 0, seed: nil)
        }
    }

    /// The app's card has a hold-out count and a seed, not flags: a refusal
    /// that names a flag on the command line is said in the card's terms.
    @Test func theAppWordsFlagRefusalsWithoutFlags() throws {
        func refusal(_ body: () throws -> Void) -> ChatConceptImport.Refusal? {
            do { try body() } catch let refusal as ChatConceptImport.Refusal { return refusal } catch {}
            return nil
        }
        let row = { (prompt: String, reply: String) in
            #"{"messages": [{"role": "user", "content": "\#(prompt)"}, "#
                + #"{"role": "assistant", "content": "\#(reply)"}]}"#
        }
        let positive = ChatConceptImport.Source(
            file: "claims.jsonl", data: Data((row("P1", "I feel 1.") + "\n" + row("P2", "I feel 2.") + "\n").utf8))
        let negative = ChatConceptImport.Source(
            file: "denials.jsonl", data: Data((row("P1", "I compute 1.") + "\n" + row("P2", "I compute 2.") + "\n").utf8))
        let refusals = [
            refusal { _ = try ChatConceptImport.parseSeed("soon") },
            refusal { _ = try ChatConceptImport.parseCount("many") },
            refusal { _ = try ChatConceptImport.prepare(
                name: "probe", positive: positive, negative: negative,
                validationPairs: 2, seed: 1, createdAt: "2026-10-19T00:00:00Z") },
            refusal { _ = try ChatConceptImport.prepare(
                name: "probe", positive: positive, negative: negative,
                validationPairs: 1, seed: nil, createdAt: "2026-10-19T00:00:00Z") },
        ]
        for case let found? in refusals {
            #expect(found.description.contains("--"))
            #expect(!found.appDescription.contains("--"), "\(found.code): \(found.appDescription)")
        }
        #expect(refusals.compactMap { $0 }.count == 4)
        #expect(refusals[2]?.code == "validationTooLarge")
        #expect(refusals[2]?.appDescription.contains("fewer than 2") == true)
    }
}
