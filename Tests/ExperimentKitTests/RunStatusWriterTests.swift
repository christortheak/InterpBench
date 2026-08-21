import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// The Swift WRITER half of the cross-engine run-status contract
/// (2026-07-27). `RunStatusFile`'s header claimed "both engines write and
/// read exactly this" while no Swift writer existed — so a failed LOCAL
/// run/evaluate left real artifacts in a directory with no
/// `run-status.json`, which `isPartial` read as legacy and therefore
/// trusted. These tests pin the tracker itself and the evaluate path
/// end-to-end (run's failure window opens only after a model load, so the
/// shared tracker is exercised through evaluate; the wrapper wiring is
/// identical).
@Suite(.serialized) struct RunStatusWriterTests {

    private func tempDirectory(_ name: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(components: "steerlab-tests-runstatus-writer",
                       "\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: url, withIntermediateDirectories: true)
        return url
    }

    // MARK: - tracker unit behavior

    @Test func aFailureLeavesTheAnnotatedStatusAndFailureNote() async throws {
        let directory = try tempDirectory("fail")
        defer { try? FileManager.default.removeItem(at: directory) }
        try "{\"a\":1}\n{\"a\":2}\n".write(
            to: directory.appending(component: "judgments.jsonl"),
            atomically: true, encoding: .utf8)

        let tracker = RunStatusFile.Tracker(
            stage: "evaluate", experiment: "s", sourceRun: "src-run",
            itemLabel: "judgment", itemsFile: "judgments.jsonl")
        await tracker.begin(directoryPath: directory.path)
        await tracker.fail(
            ExperimentError(reason: "judge transport exploded"))

        #expect(RunStatusFile.isPartial(at: directory))
        let status = try #require(RunStatusFile.read(at: directory))
        #expect(status.status == "failed")
        #expect(status.stage == "evaluate")
        #expect(status.error == "judge transport exploded")
        #expect(status.errorType == "ExperimentError")
        #expect(status.itemsWritten == 2)
        #expect(status.itemLabel == "judgment")
        #expect(status.evidenceComplete == false)
        let note = try String(
            contentsOf: directory.appending(
                component: RunStatusFile.failureNoteFilename),
            encoding: .utf8)
        #expect(note.contains("# evaluate FAILED"))
        #expect(note.contains("failure record"))
        #expect(note.contains("judge transport exploded"))
        #expect(note.contains("- **Source run:** src-run"))
        #expect(note.contains("Judgments written before the failure:** 2"))
    }

    @Test func aCompletionLeavesTheCompletedStatus() async throws {
        let directory = try tempDirectory("complete")
        defer { try? FileManager.default.removeItem(at: directory) }
        try "{\"a\":1}\n".write(
            to: directory.appending(component: "generations.jsonl"),
            atomically: true, encoding: .utf8)

        let tracker = RunStatusFile.Tracker(
            stage: "run", experiment: "s",
            itemLabel: "record", itemsFile: "generations.jsonl")
        await tracker.begin(directoryPath: directory.path)
        await tracker.finish()

        #expect(!RunStatusFile.isPartial(at: directory))
        let status = try #require(RunStatusFile.read(at: directory))
        #expect(status.status == "completed")
        #expect(status.evidenceComplete == true)
        #expect(status.itemsWritten == 1)
        #expect(
            !FileManager.default.fileExists(
                atPath: directory.appending(
                    component: RunStatusFile.failureNoteFilename).path))
    }

    @Test func beginAloneLeavesAnInProgressPartial() async throws {
        // What a SIGKILL leaves behind: the in-progress stamp written when
        // the directory was created, never finalized. It must read as
        // partial — the crash that interrupts a run must not also make it
        // look complete.
        let directory = try tempDirectory("killed")
        defer { try? FileManager.default.removeItem(at: directory) }
        let tracker = RunStatusFile.Tracker(
            stage: "run", experiment: "s",
            itemLabel: "record", itemsFile: "generations.jsonl")
        await tracker.begin(directoryPath: directory.path)

        #expect(RunStatusFile.isPartial(at: directory))
        #expect(RunStatusFile.read(at: directory)?.status == "inProgress")
    }

    @Test func aCancelledTaskIsPartialButGetsNoFailureNote() async throws {
        // Cooperative cancellation returns NORMALLY with a note in the
        // directory; stamping that "completed" would make the partial it
        // left look citable, while a FAILED.md for a deliberate stop would
        // train the researcher to ignore the marker that matters (the
        // server's checkpoint rule). Partial yes, failure prose no.
        let directory = try tempDirectory("cancelled")
        defer { try? FileManager.default.removeItem(at: directory) }
        ExperimentTasks.writeCancellationNote(
            task: "study run", to: directory)

        let tracker = RunStatusFile.Tracker(
            stage: "run", experiment: "s",
            itemLabel: "record", itemsFile: "generations.jsonl")
        await tracker.begin(directoryPath: directory.path)
        await tracker.finish()

        #expect(RunStatusFile.isPartial(at: directory))
        let status = try #require(RunStatusFile.read(at: directory))
        #expect(status.status == "failed")
        #expect(status.errorType == "Cancelled")
        #expect(status.error?.contains("cancelled by user") == true)
        #expect(
            !FileManager.default.fileExists(
                atPath: directory.appending(
                    component: RunStatusFile.failureNoteFilename).path))
    }

    // MARK: - the two asymmetries with the Python writer (2026-07-27)

    /// `itemsWritten` must be current ON DISK while the stage runs.
    ///
    /// The writer originally wrote only at begin/finish/fail, so the last
    /// state a HARD stop left behind (preemption, power loss, force-quit) was
    /// `inProgress, itemsWritten: 0` beside a directory holding real records.
    /// `isPartial` was still right, but the count a reader consults to learn
    /// how much survived said zero in the one situation the file exists for.
    /// Python's `note_item` rewrites per item; so does `noteItem`.
    @Test func itemsWrittenIsRefreshedPerItemNotOnlyAtTheEnd() async throws {
        let directory = try tempDirectory("refresh")
        defer { try? FileManager.default.removeItem(at: directory) }
        let items = directory.appending(component: "generations.jsonl")

        let tracker = RunStatusFile.Tracker(
            stage: "run", experiment: "s",
            itemLabel: "record", itemsFile: "generations.jsonl")
        await tracker.begin(directoryPath: directory.path)
        #expect(RunStatusFile.read(at: directory)?.itemsWritten == 0)

        // Three records land, each announced — and NOTHING terminal is ever
        // called: this is the killed-process shape.
        try "{\"a\":1}\n".write(to: items, atomically: true, encoding: .utf8)
        await tracker.noteItem()
        #expect(RunStatusFile.read(at: directory)?.itemsWritten == 1)
        try "{\"a\":1}\n{\"a\":2}\n{\"a\":3}\n".write(
            to: items, atomically: true, encoding: .utf8)
        await tracker.noteItem()

        let status = try #require(RunStatusFile.read(at: directory))
        #expect(status.itemsWritten == 3)
        // Still honestly partial, and still not citable.
        #expect(status.status == "inProgress")
        #expect(status.evidenceComplete == false)
        #expect(RunStatusFile.isPartial(at: directory))
    }

    /// `invalidResponses` must be a measurement, not a hardcoded zero.
    ///
    /// A stamped `0` is worse than an omitted key: it is indistinguishable
    /// from "none occurred", so a reader cannot tell that this engine simply
    /// was not counting. The malformed attempts also land verbatim in
    /// `judge-failures.jsonl` — the Python filename and record keys — because
    /// "the judge needed two tries" is exactly what a quiet retry erases.
    @Test func invalidJudgeResponsesAreCountedAndKeptVerbatim() async throws {
        let directory = try tempDirectory("invalid")
        defer { try? FileManager.default.removeItem(at: directory) }

        let tracker = RunStatusFile.Tracker(
            stage: "evaluate", experiment: "s",
            itemLabel: "judgment", itemsFile: "judgments.jsonl")
        await tracker.begin(directoryPath: directory.path)
        #expect(RunStatusFile.read(at: directory)?.invalidResponses == 0)

        await tracker.noteInvalidResponse([
            "attempt": "1", "error": "winner 'maybe' is not A, B, or tie",
            "verdict": "maybe", "judge": "judge-1", "item": "pair x/p1",
        ])
        await tracker.noteInvalidResponse([
            "attempt": "2", "error": "winner 'unclear' is not A, B, or tie",
            "verdict": "unclear", "judge": "judge-1", "item": "pair x/p1",
        ])
        await tracker.finish()

        #expect(RunStatusFile.read(at: directory)?.invalidResponses == 2)

        // Appended, not rewritten: BOTH attempts survive.
        let failures = try String(
            contentsOf: directory.appending(
                component: RunStatusFile.invalidResponsesFilename),
            encoding: .utf8)
        let lines = failures.split(separator: "\n").filter { !$0.isEmpty }
        #expect(lines.count == 2)
        #expect(failures.contains("maybe"))
        #expect(failures.contains("unclear"))
        #expect(failures.contains("judge-1"))
        // Same sidecar name the server writes, so either engine's file reads
        // the same way.
        #expect(RunStatusFile.invalidResponsesFilename == "judge-failures.jsonl")
    }

    /// The retry helper is the SOURCE of that count: it reports every
    /// malformed attempt, and still refuses after the second.
    @Test func theJudgeRetryHelperReportsEveryMalformedAttempt() async throws {
        actor Reports {
            var all: [[String: String]] = []
            func add(_ record: [String: String]) { all.append(record) }
        }
        let reports = Reports()

        func verdict(_ winner: String) -> PairedJudgeResponse {
            PairedJudgeResponse(
                aScores: nil, bScores: nil, structuredFields: nil,
                winner: winner, confidence: 0.5, briefReason: "raw \(winner)",
                reasoningTruncated: nil)
        }

        // Malformed once, then valid: one report, no refusal.
        var call = 0
        let recovered = try await ExperimentTasks.judgmentWithValidWinner(
            judgeName: "judge-1", item: "pair x/p1",
            onInvalid: { await reports.add($0) }
        ) {
            call += 1
            return call == 1 ? verdict("maybe") : verdict("A")
        }
        #expect(recovered.winner == "A")
        var all = await reports.all
        #expect(all.count == 1)
        #expect(all.first?["attempt"] == "1")
        #expect(all.first?["judge"] == "judge-1")
        #expect(all.first?["verdict"] == "maybe")
        #expect(all.first?["rawResponse"] == "raw maybe")

        // Malformed twice: both reported, and the phase still refuses rather
        // than recording an invented tie.
        await #expect(throws: (any Error).self) {
            _ = try await ExperimentTasks.judgmentWithValidWinner(
                judgeName: "judge-1", item: "pair x/p2",
                onInvalid: { await reports.add($0) }
            ) { verdict("unclear") }
        }
        all = await reports.all
        #expect(all.count == 3)
        #expect(all.last?["attempt"] == "2")
    }

    @Test func aFailureBeforeAnyDirectoryExistsWritesNothing() async throws {
        // A preflight refusal (bad manifest, missing prompts) throws before
        // any run directory exists — there is nothing to annotate, and the
        // tracker must not invent a location.
        let tracker = RunStatusFile.Tracker(
            stage: "run", experiment: "s",
            itemLabel: "record", itemsFile: "generations.jsonl")
        await tracker.fail(ExperimentError(reason: "no prompts"))
        await tracker.finish()
        // Nothing to assert on disk — the guarantee is that neither call
        // trapped nor created files anywhere; reaching here is the test.
    }

    // MARK: - the evaluate path end-to-end

    private func realFrenchHash() throws -> String {
        try StimulusSet(
            directory: VectorCatalog.conceptsDirectory
                .appending(component: "french")
        ).hash
    }

    /// A verified draft plus a source run of raw JSONL rows stamped with the
    /// live manifest hash (the `PairedJudgePairingTests` harness, reduced).
    private func plantStudy(name: String, rows: [String]) throws -> URL {
        var manifest = try ExperimentStore.create(
            name: name, description: "d", modelID: "test/model")
        manifest.concepts.append(
            .init(
                name: "french", stimulusSetHash: try realFrenchHash(),
                options: .init()))
        try ExperimentStore.save(manifest)
        let liveHash = ExperimentStore.manifestHash(manifest)
        let source = ExperimentStore.runsDirectory.appending(
            component: "20260727T000000000Z-exp-\(name)-run")
        try FileManager.default.createDirectory(
            at: source, withIntermediateDirectories: true)
        try JSONEncoder().encode(manifest).write(
            to: source.appending(component: "experiment.json"))
        try liveHash.write(
            to: source.appending(component: "experiment-hash.txt"),
            atomically: true, encoding: .utf8)
        try (rows.joined(separator: "\n") + "\n").write(
            to: source.appending(component: "generations.jsonl"),
            atomically: true, encoding: .utf8)
        return source
    }

    private func row(
        study: String, condition: String, seed: UInt64,
        promptID: String, output: String
    ) -> String {
        "{\"experiment\":\"\(study)\",\"condition\":\"\(condition)\","
            + "\"seed\":\(seed),\"promptID\":\"\(promptID)\","
            + "\"prompt\":\"q-\(promptID)\",\"output\":\"\(output)\"}"
    }

    private func plantPairedStudy(name: String) throws -> URL {
        var rows: [String] = []
        var seed: UInt64 = 0
        for condition in ["baseline", "steered"] {
            for promptID in ["p0", "p1"] {
                seed += 1
                rows.append(
                    row(
                        study: name, condition: condition, seed: seed,
                        promptID: promptID,
                        output: "\(condition)-\(promptID)"))
            }
        }
        return try plantStudy(name: name, rows: rows)
    }

    private static let evaluation = ExperimentManifest.EvaluationSpec(
        kind: .pairedJudge, judgeModel: "claude-test",
        judgePrompt: "prefer the better response")

    private func withEvaluateSeams<T>(
        judge: @escaping @Sendable (
            String, String, String, String
        ) async throws -> PairedJudgeResponse,
        _ body: () async throws -> T
    ) async throws -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "status-writer-\(UUID().uuidString)")
        ExperimentStore.rootOverride = temp
        ExperimentTasks.judgeOverrideForTesting = judge
        defer {
            ExperimentTasks.judgeOverrideForTesting = nil
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try await body()
    }

    private func evaluateDirectory(study: String) throws -> URL {
        let runs = try FileManager.default.contentsOfDirectory(
            at: ExperimentStore.runsDirectory,
            includingPropertiesForKeys: nil)
        return try #require(
            runs.first { $0.lastPathComponent.hasSuffix("-evaluate") },
            "no evaluate run directory found for \(study)")
    }

    private actor Counter {
        var value = 0
        func next() -> Int {
            value += 1
            return value
        }
    }

    @Test func aFailedLocalEvaluateLeavesTheStatusContract() async throws {
        let calls = Counter()
        try await withEvaluateSeams(judge: { _, _, _, _ in
            if await calls.next() > 1 {
                throw ExperimentError(reason: "judge transport exploded")
            }
            return PairedJudgeResponse(
                aScores: nil, bScores: nil, structuredFields: nil,
                winner: "A", confidence: 0.9, briefReason: "fake")
        }) {
            let study = "status-fail"
            let source = try plantPairedStudy(name: study)
            await #expect(throws: (any Error).self) {
                try await ExperimentTasks.evaluatePairedJudge(
                    experimentName: study,
                    sourceRunDirectory: source,
                    evaluation: Self.evaluation)
            }
            let directory = try evaluateDirectory(study: study)
            #expect(RunStatusFile.isPartial(at: directory))
            let status = try #require(RunStatusFile.read(at: directory))
            #expect(status.status == "failed")
            #expect(status.stage == "evaluate")
            #expect(status.error?.contains("judge transport exploded") == true)
            // The judgment that landed before the failure is counted and
            // kept; NO judge-report.json summarizes the partial panel.
            #expect(status.itemsWritten == 1)
            #expect(
                FileManager.default.fileExists(
                    atPath: directory.appending(
                        component: RunStatusFile.failureNoteFilename).path))
            #expect(
                !FileManager.default.fileExists(
                    atPath: directory.appending(
                        component: "judge-report.json").path))
        }
    }

    @Test func aCompletedLocalEvaluateLeavesTheCompletedStatus() async throws {
        try await withEvaluateSeams(judge: { _, _, _, _ in
            PairedJudgeResponse(
                aScores: nil, bScores: nil, structuredFields: nil,
                winner: "A", confidence: 0.9, briefReason: "fake")
        }) {
            let study = "status-done"
            let source = try plantPairedStudy(name: study)
            let directory = try await ExperimentTasks.evaluatePairedJudge(
                experimentName: study,
                sourceRunDirectory: source,
                evaluation: Self.evaluation)
            #expect(!RunStatusFile.isPartial(at: directory))
            let status = try #require(RunStatusFile.read(at: directory))
            #expect(status.status == "completed")
            #expect(status.evidenceComplete == true)
            #expect(status.itemsWritten == 2)
            #expect(status.stage == "evaluate")
            #expect(
                FileManager.default.fileExists(
                    atPath: directory.appending(
                        component: "judge-report.json").path))
        }
    }

    @Test func aCancelledLocalEvaluateIsPartialNotCompleted() async throws {
        try await withEvaluateSeams(judge: { _, _, _, _ in
            PairedJudgeResponse(
                aScores: nil, bScores: nil, structuredFields: nil,
                winner: "A", confidence: 0.9, briefReason: "fake")
        }) {
            let study = "status-cancel"
            let source = try plantPairedStudy(name: study)
            let directory = try await ExperimentTasks.evaluatePairedJudge(
                experimentName: study,
                sourceRunDirectory: source,
                evaluation: Self.evaluation,
                shouldCancel: { true })
            #expect(RunStatusFile.isPartial(at: directory))
            let status = try #require(RunStatusFile.read(at: directory))
            #expect(status.status == "failed")
            #expect(status.errorType == "Cancelled")
            #expect(
                !FileManager.default.fileExists(
                    atPath: directory.appending(
                        component: RunStatusFile.failureNoteFilename).path))
        }
    }

    @Test func legacyDirectoriesRemainUnannotatedAndTrusted() throws {
        // The writer must not change the READER's classification of
        // directories nothing wrote a status into — every run produced
        // before this contract stays legacy/trusted.
        let directory = try tempDirectory("legacy")
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(RunStatusFile.reading(at: directory) == .absent)
        #expect(!RunStatusFile.isPartial(at: directory))
    }
}
