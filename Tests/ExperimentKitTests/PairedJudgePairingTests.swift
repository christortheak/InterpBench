import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// The paired-judge JOIN rule (external review 2026-07-22, P0): pairs join
/// on (promptID, sampleIndex) — never the seed, which under the server's
/// derivedSHA256 policy includes condition identity and therefore differs
/// between baseline and variant by design. Every pair/judgment artifact
/// stamps both sides' seeds (`baselineSeed`/`variantSeed`, cross-engine
/// keys), and zero surviving pairs REFUSE with the shared message instead
/// of writing a successful-looking empty report. Server twin:
/// `Server/tests/test_paired_judge_sampled.py`.
@Suite(.serialized) struct PairedJudgePairingTests {

    func realFrenchHash() throws -> String {
        try StimulusSet(
            directory: VectorCatalog.conceptsDirectory.appending(component: "french")
        ).hash
    }

    /// Plants a verified draft plus a source run whose generations are the
    /// caller's raw JSONL rows, stamped with the live manifest hash so the
    /// epoch guard passes. Returns the source run directory.
    private func plantStudy(
        name: String, rows: [String]
    ) throws -> URL {
        var manifest = try ExperimentStore.create(
            name: name, description: "d", modelID: "test/model")
        manifest.concepts.append(
            .init(
                name: "french", stimulusSetHash: try realFrenchHash(),
                options: .init()))
        try ExperimentStore.save(manifest)
        let liveHash = ExperimentStore.manifestHash(manifest)
        let source = ExperimentStore.runsDirectory.appending(
            component: "20260722T000000000Z-exp-\(name)-run")
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
        sampleIndex: UInt64?, promptID: String, output: String
    ) -> String {
        let sample = sampleIndex.map { ",\"sampleIndex\":\($0)" } ?? ""
        return "{\"experiment\":\"\(study)\",\"condition\":\"\(condition)\","
            + "\"seed\":\(seed)\(sample),\"promptID\":\"\(promptID)\","
            + "\"prompt\":\"q-\(promptID)\",\"output\":\"\(output)\"}"
    }

    private func withEvaluateSeams<T>(
        _ body: () async throws -> T
    ) async throws -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "pairing-\(UUID().uuidString)")
        ExperimentStore.rootOverride = temp
        ExperimentTasks.judgeOverrideForTesting = { _, _, _, _ in
            PairedJudgeResponse(
                aScores: nil, bScores: nil, structuredFields: nil,
                winner: "A", confidence: 0.9, briefReason: "fake")
        }
        defer {
            ExperimentTasks.judgeOverrideForTesting = nil
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try await body()
    }

    private static let evaluation = ExperimentManifest.EvaluationSpec(
        kind: .pairedJudge, judgeModel: "claude-test",
        judgePrompt: "prefer the better response")

    /// Server-shaped sampled records (derived per-condition seeds, so NO
    /// seed is ever shared across conditions) pair by sample cell — one
    /// pair per (promptID, sampleIndex) — and every judgment stamps
    /// sampleIndex + both sides' distinct seeds, with no "seed" key.
    @Test func sampledRecordsPairBySampleCellAndStampBothSeeds() async throws {
        try await withEvaluateSeams {
            let study = "pair-sampled"
            var rows: [String] = []
            var seed: UInt64 = 100
            for condition in ["baseline", "steered"] {
                for promptID in ["p0", "p1"] {
                    for sample in UInt64(0)..<2 {
                        seed += 1  // all seeds distinct, as derivedSHA256 makes them
                        rows.append(
                            row(
                                study: study, condition: condition, seed: seed,
                                sampleIndex: sample, promptID: promptID,
                                output: "\(condition)-\(promptID)-s\(sample)"))
                    }
                }
            }
            let source = try plantStudy(name: study, rows: rows)
            let out = try await ExperimentTasks.evaluatePairedJudge(
                experimentName: study,
                sourceRunDirectory: source,
                evaluation: Self.evaluation)
            let lines = try String(
                contentsOf: out.appending(component: "judgments.jsonl"),
                encoding: .utf8
            ).split(separator: "\n")
            // prompts × samples pairs, one judge.
            #expect(lines.count == 4)
            for line in lines {
                let object = try #require(
                    try JSONSerialization.jsonObject(with: Data(line.utf8))
                        as? [String: Any])
                #expect(object["seed"] == nil, "no field named seed on a pair")
                let sampleIndex = try #require(object["sampleIndex"] as? UInt64)
                #expect(sampleIndex == 0 || sampleIndex == 1)
                let baselineSeed = try #require(object["baselineSeed"] as? UInt64)
                let variantSeed = try #require(object["variantSeed"] as? UInt64)
                #expect(baselineSeed != variantSeed)
            }
        }
    }

    /// Local greedy single-sample records (no sampleIndex key at all)
    /// normalize to cell 0 and keep pairing exactly as before.
    @Test func greedySingleSampleRecordsStillPairPerPrompt() async throws {
        try await withEvaluateSeams {
            let study = "pair-greedy"
            let rows = [
                row(study: study, condition: "baseline", seed: 1,
                    sampleIndex: nil, promptID: "p0", output: "a"),
                row(study: study, condition: "baseline", seed: 1,
                    sampleIndex: nil, promptID: "p1", output: "b"),
                row(study: study, condition: "steered", seed: 1,
                    sampleIndex: nil, promptID: "p0", output: "c"),
                row(study: study, condition: "steered", seed: 1,
                    sampleIndex: nil, promptID: "p1", output: "d"),
            ]
            let source = try plantStudy(name: study, rows: rows)
            let out = try await ExperimentTasks.evaluatePairedJudge(
                experimentName: study,
                sourceRunDirectory: source,
                evaluation: Self.evaluation)
            let lines = try String(
                contentsOf: out.appending(component: "judgments.jsonl"),
                encoding: .utf8
            ).split(separator: "\n")
            #expect(lines.count == 2)
            for line in lines {
                let object = try #require(
                    try JSONSerialization.jsonObject(with: Data(line.utf8))
                        as? [String: Any])
                #expect(object["sampleIndex"] as? UInt64 == 0)
                // Greedy runs: both sides ran the same recorded seed.
                #expect(object["baselineSeed"] as? UInt64 == 1)
                #expect(object["variantSeed"] as? UInt64 == 1)
            }
        }
    }

    /// Both conditions present but no shared sample cell: refuse with the
    /// shared cross-engine message — never a quiet empty judged report.
    @Test func zeroPairsRefusesWithTheSharedMessage() async throws {
        try await withEvaluateSeams {
            let study = "pair-none"
            let rows = [
                row(study: study, condition: "baseline", seed: 1,
                    sampleIndex: 0, promptID: "p0", output: "a"),
                row(study: study, condition: "steered", seed: 2,
                    sampleIndex: 5, promptID: "p0", output: "b"),
            ]
            let source = try plantStudy(name: study, rows: rows)
            do {
                _ = try await ExperimentTasks.evaluatePairedJudge(
                    experimentName: study,
                    sourceRunDirectory: source,
                    evaluation: Self.evaluation)
                Issue.record("expected the zero-pairs refusal")
            } catch let error as ExperimentError {
                #expect(error.reason == ExperimentTasks.noPairsMessage)
            }
        }
    }

    /// The refusal copy is the cross-engine contract: it names the join key
    /// and the likely causes in plain language (value-pinned against
    /// `paired_judge.NO_PAIRS_MESSAGE` on the server).
    @Test func noPairsMessageNamesJoinKeyAndLikelyCause() {
        let message = ExperimentTasks.noPairsMessage
        #expect(message.contains("(promptID, sampleIndex)"))
        #expect(message.contains("baseline"))
        #expect(message.contains("instrument readouts and error records"))
        #expect(message.contains("Refusing"))
    }
}
