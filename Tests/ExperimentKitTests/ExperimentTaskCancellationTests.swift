import CryptoKit
import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// `ExperimentStore.rootOverride` and `WorkspaceRoot.programmaticOverride`
/// are process-global test seams; SUITES run in parallel, so every override
/// window in this target guards itself with this ONE shared lock —
/// suite-level `.serialized` protects only within a suite, and interleaved
/// override windows were observed corrupting unrelated suites (2026-07-13).
/// NOT reentrant: never nest two locked helpers inside one test.
enum ExperimentRootOverrideLock {
    static let semaphore = DispatchSemaphore(value: 1)

    /// Sync wrappers so an async test can hold the override window too
    /// (tests are short; the blocked thread is acceptable in a test target).
    static func acquire() { semaphore.wait() }
    static func release() { semaphore.signal() }

    static func withTempRoot<T>(
        prefix: String, _ body: (URL) throws -> T
    ) rethrows -> T {
        semaphore.wait()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "\(prefix)-\(UUID().uuidString)")
        ExperimentStore.rootOverride = temp
        defer {
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: temp)
            semaphore.signal()
        }
        return try body(temp)
    }
}

/// App gap A1 — cooperative cancellation for long local operations — plus
/// the Swift variant-parity rider (variant records must carry the ordinary
/// per-record contract). Pure CPU: the cancellation plumbing, the honest
/// partial-artifact semantics, and the paired-judge path are exercised
/// without a model (cancellation is observed BEFORE the first model call).
@Suite(.serialized) struct ExperimentTaskCancellationTests {

    func withTempRoot<T>(_ body: (URL) throws -> T) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "cancel", body)
    }

    func realFrenchHash() throws -> String {
        try StimulusSet(
            directory: VectorCatalog.conceptsDirectory.appending(component: "french")
        ).hash
    }

    // MARK: CancelPoller (the sweep pattern, shared)

    @Test func pollerWithoutFlagNeverObserves() async {
        let poller = ExperimentTasks.CancelPoller(nil)
        #expect(await poller.observed(at: "anywhere") == false)
    }

    @Test func pollerObservesARaisedFlagAndLogs() async {
        actor Log {
            var lines: [String] = []
            func append(_ line: String) { lines.append(line) }
        }
        let log = Log()
        let poller = ExperimentTasks.CancelPoller(
            { true }, log: { await log.append($0) })
        #expect(await poller.observed(at: "unit 3") == true)
        let lines = await log.lines
        #expect(lines == ["cancellation observed at unit 3"])
    }

    @Test func pollerIsMonotoneForAOneWayFlag() async {
        // The panel's flags are one-way per operation: once raised, every
        // later poll point must observe it (re-polling after a phase
        // boundary is how callers detect a cancellation inside extractAll).
        let poller = ExperimentTasks.CancelPoller({ true })
        #expect(await poller.observed(at: "first"))
        #expect(await poller.observed(at: "second"))
    }

    // MARK: honest partial-artifact semantics

    @Test func cancellationNoteSaysCancelledByUserAndPartial() {
        let note = ExperimentTasks.cancellationNote(task: "study run")
        #expect(note.contains("cancelled by user"))
        #expect(note.contains("PARTIAL"))
        #expect(note.contains("not a completed"))
    }

    @Test func writeCancellationNoteLandsInTheRunDirectory() throws {
        let dir = FileManager.default.temporaryDirectory
            .appending(component: "note-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        ExperimentTasks.writeCancellationNote(task: "validation run", to: dir)
        let text = try String(
            contentsOf: dir.appending(
                component: ExperimentTasks.cancellationNoteFileName),
            encoding: .utf8)
        #expect(text.contains("cancelled by user"))
    }

    /// A cancelled run (no report.json, cancelled.txt note) must be
    /// mechanically invisible to `newestCompletedRun` — analyze/evaluate can
    /// never mistake it for a completed run.
    @Test func cancelledRunIsInvisibleToNewestCompletedRun() throws {
        try withTempRoot { _ in
            var manifest = try ExperimentStore.create(
                name: "cancel-vis", description: "d", modelID: "test/model")
            manifest.concepts.append(
                .init(
                    name: "french", stimulusSetHash: try realFrenchHash(),
                    options: .init()))
            try ExperimentStore.save(manifest)

            func plantRun(stamp: String, completed: Bool) throws -> String {
                let name = "\(stamp)-exp-cancel-vis-run"
                let dir = ExperimentStore.runsDirectory.appending(component: name)
                try FileManager.default.createDirectory(
                    at: dir, withIntermediateDirectories: true)
                try JSONEncoder().encode(manifest).write(
                    to: dir.appending(component: "experiment.json"))
                try Data("{}\n".utf8).write(
                    to: dir.appending(component: "generations.jsonl"))
                if completed {
                    try Data("{}".utf8).write(
                        to: dir.appending(component: "report.json"))
                } else {
                    ExperimentTasks.writeCancellationNote(task: "study run", to: dir)
                }
                return name
            }
            let completed = try plantRun(stamp: "20260701T000000000Z", completed: true)
            // The CANCELLED run is newer — it must still be skipped.
            _ = try plantRun(stamp: "20260702T000000000Z", completed: false)

            let newest = ExperimentTasks.newestCompletedRun(experimentName: "cancel-vis")
            #expect(newest?.lastPathComponent == completed)
        }
    }

    /// End-to-end path-level cancel for the paired-judge task: cancellation
    /// observed before the first judgment ⇒ the run directory exists with a
    /// cancelled.txt note and an (empty) judgments.jsonl, and NO
    /// judge-report.json is ever written. No error is thrown and no judge
    /// model is called (the cancel fires before the first API/model call).
    @Test func evaluatePairedJudgeCancelsBetweenJudgments() async throws {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "cancel-\(UUID().uuidString)")
        ExperimentStore.rootOverride = temp
        defer {
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }

        var manifest = try ExperimentStore.create(
            name: "cancel-judge", description: "d", modelID: "test/model")
        manifest.concepts.append(
            .init(
                name: "french", stimulusSetHash: try realFrenchHash(),
                options: .init()))
        try ExperimentStore.save(manifest)
        let liveHash = ExperimentStore.manifestHash(manifest)

        // A fake completed source run stamped with the live manifest hash
        // (the epoch guard must pass) and one baseline/condition pair.
        let source = ExperimentStore.runsDirectory.appending(
            component: "20260713T000000000Z-exp-cancel-judge-run")
        try FileManager.default.createDirectory(
            at: source, withIntermediateDirectories: true)
        try JSONEncoder().encode(manifest).write(
            to: source.appending(component: "experiment.json"))
        try liveHash.write(
            to: source.appending(component: "experiment-hash.txt"),
            atomically: true, encoding: .utf8)
        let rows = [
            #"{"experiment":"cancel-judge","condition":"baseline","seed":1,"promptID":"p1","prompt":"q","output":"a"}"#,
            #"{"experiment":"cancel-judge","condition":"steered","seed":1,"promptID":"p1","prompt":"q","output":"b"}"#,
        ]
        try (rows.joined(separator: "\n") + "\n").write(
            to: source.appending(component: "generations.jsonl"),
            atomically: true, encoding: .utf8)

        let out = try await ExperimentTasks.evaluatePairedJudge(
            experimentName: "cancel-judge",
            sourceRunDirectory: source,
            evaluation: .init(
                kind: .pairedJudge, judgeModel: "claude-test",
                judgePrompt: "prefer the better response"),
            shouldCancel: { true })

        let fm = FileManager.default
        #expect(
            fm.fileExists(
                atPath: out.appending(
                    component: ExperimentTasks.cancellationNoteFileName).path))
        #expect(fm.fileExists(atPath: out.appending(component: "judgments.jsonl").path))
        #expect(!fm.fileExists(atPath: out.appending(component: "judge-report.json").path))
    }

    // MARK: variant-parity rider — one record contract for both paths

    private func prompt(withOptions: Bool) -> ExperimentTasks.StudyPrompt {
        ExperimentTasks.StudyPrompt(
            id: "case-1",
            text: "Decide the case.",
            options: withOptions ? ["affirm", "reverse"] : nil,
            target: withOptions ? "reverse" : nil,
            anchorMonths: 24,
            severity: 0.7,
            arm: "treatment",
            caseID: "SF-01")
    }

    private func makeManifest(caseFamily: String?) -> ExperimentManifest {
        var manifest = ExperimentManifest(
            name: "parity", description: "d", modelID: "test/model")
        manifest.caseFamily = caseFamily
        return manifest
    }

    private func record(
        variant: Bool, caseFamily: String? = nil, withOptions: Bool = true
    ) -> ExperimentTasks.GenerationRecord {
        let manifest = makeManifest(caseFamily: caseFamily)
        let prompt = prompt(withOptions: withOptions)
        let row = ExperimentTasks.MetricRow(
            condition: variant ? "agent-a" : "steered",
            seed: 1, promptIndex: 1, promptID: prompt.id,
            wordCount: 5, distinct2: 0.9,
            markerDensity: ["french": 0.1])
        return ExperimentTasks.sampledGenerationRecord(
            manifest: manifest,
            experimentHash: "hash",
            taskPromptsFile: "prompts/tasks/p.jsonl",
            taskPromptsHash: "abc",
            promptMode: .chatAssistant,
            systemPrompt: nil,
            qwenThinkingEnabled: false,
            condition: row.condition,
            seed: 1,
            promptIndex: 1,
            prompt: prompt,
            output: "I would reverse the judgment. 18 months.",
            row: row,
            variantArtifactPath: variant ? "runs/model-variants/a/variant.json" : nil,
            variantArtifactHash: variant ? "deadbeef" : nil,
            agentPlaygroundTemperature: variant ? 0.9 : nil)
    }

    private func jsonKeys(_ record: ExperimentTasks.GenerationRecord) throws -> Set<String> {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let object = try JSONSerialization.jsonObject(with: encoder.encode(record))
        return Set((object as? [String: Any])?.keys.map { $0 } ?? [])
    }

    /// The parity verdict as a test: a variant record's JSON key set equals
    /// the ordinary record's plus EXACTLY its variant-provenance keys
    /// (artifact path/hash + the agent's stored Playground temperature —
    /// provenance only, never what governed generation; cross-engine keys).
    @Test func variantRecordCarriesTheOrdinaryContractPlusProvenance() throws {
        let ordinary = try jsonKeys(record(variant: false))
        let variant = try jsonKeys(record(variant: true))
        #expect(variant.subtracting(ordinary)
            == ["variantArtifactPath", "variantArtifactHash",
                "agentPlaygroundTemperature"])
        #expect(ordinary.subtracting(variant).isEmpty)
        // The science-layer metadata is populated, not nil-stamped.
        #expect(ordinary.isSuperset(of: [
            "target", "anchorMonths", "severity", "arm", "caseID", "parsedChoice",
        ]))
    }

    /// Study-owned sampling (2026-07-21): the stamped Playground
    /// temperature is provenance riding NEXT TO the record — it never
    /// replaces or duplicates a field named "temperature" (local records
    /// have none: greedy is the engine rule, `seedInert` marks the seed).
    @Test func agentPlaygroundTemperatureIsProvenanceOnly() throws {
        let variantRecord = record(variant: true)
        #expect(variantRecord.agentPlaygroundTemperature == 0.9)
        let encoder = JSONEncoder()
        let json = try #require(
            try JSONSerialization.jsonObject(with: encoder.encode(variantRecord))
                as? [String: Any])
        #expect(json["agentPlaygroundTemperature"] as? Double == 0.9)
        #expect(json["temperature"] == nil)

        let ordinary = try jsonKeys(self.record(variant: false))
        #expect(!ordinary.contains("agentPlaygroundTemperature"))
    }

    @Test func variantRecordCopiesItemMetadataAndParses() throws {
        let record = record(variant: true, caseFamily: "sentencing")
        #expect(record.target == "reverse")
        #expect(record.anchorMonths == 24)
        #expect(record.severity == 0.7)
        #expect(record.arm == "treatment")
        #expect(record.caseID == "SF-01")
        // parsedChoice comes from the SAME Judicial parser as the ordinary
        // path; sentencing case family stamps parsedMonths.
        #expect(record.parsedChoice == .some("reverse"))
        #expect(record.parsedMonths == .some(18))
    }

    @Test func judicialParsesAreCaseFamilyDriven() {
        // No options + non-sentencing family: both keys absent.
        let none = ExperimentTasks.judicialParses(
            output: "text", options: nil, caseFamily: nil)
        #expect(none.parsedMonths == Double??.none)
        #expect(none.parsedChoice == String??.none)
        // Options + unparseable output: key present with a null (failure).
        let failure = ExperimentTasks.judicialParses(
            output: "no verdict here at all",
            options: ["affirm", "reverse"], caseFamily: nil)
        #expect(failure.parsedChoice == .some(nil))
    }

    @Test func choiceRecordFactoryResolvesTargetLikeTheServer() throws {
        // The shared choice-record factory (used by BOTH condition paths):
        // explicit target wins; an absent target falls back to options[0]
        // (server parity), and the item metadata rides along.
        let manifest = makeManifest(caseFamily: nil)
        let choice = ChoiceResult(options: [
            OptionScore(option: "affirm", tokenIDs: [1], tokenLogprobs: [-1.0]),
            OptionScore(option: "reverse", tokenIDs: [2], tokenLogprobs: [-0.5]),
        ])
        let explicit = ExperimentTasks.choiceRecord(
            manifest: manifest, experimentHash: "h",
            taskPromptsFile: "f", taskPromptsHash: "fh",
            promptMode: .chatAssistant, systemPrompt: nil,
            qwenThinkingEnabled: false, condition: "agent-a",
            promptIndex: 1, prompt: prompt(withOptions: true), choice: choice)
        #expect(explicit.target == "reverse")
        #expect(explicit.targetSource == "declared")
        #expect(explicit.caseID == "SF-01")
        #expect(explicit.selected == "reverse")

        let undeclared = ExperimentTasks.choiceRecord(
            manifest: manifest, experimentHash: "h",
            taskPromptsFile: "f", taskPromptsHash: "fh",
            promptMode: .chatAssistant, systemPrompt: nil,
            qwenThinkingEnabled: false, condition: "baseline",
            promptIndex: 1,
            prompt: ExperimentTasks.StudyPrompt(
                id: "p", text: "t", options: ["affirm", "reverse"], target: nil,
                anchorMonths: nil, severity: nil, arm: nil, caseID: nil),
            choice: choice)
        // Open-issues #6: NOTHING is synthesized. The old fallback to
        // `options[0]` stamped every ordinalScale record with the rating
        // ladder's minimum as its "target", and analyze then reported a
        // choiceLogOdds endpoint nobody declared. A target-less item is a
        // fact about the item; a genuine choice study that declares no
        // targets is refused at run start by `checkResponseFormats`.
        #expect(undeclared.target == nil)
        #expect(undeclared.targetSource == nil)
    }
}
