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
}
