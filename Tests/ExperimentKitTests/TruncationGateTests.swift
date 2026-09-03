import Foundation
import MLXLMCommon
import Testing

@testable import ExperimentKit

/// Why a generation stopped, and the declared per-cell gate on it.
///
/// The 2026-08-30 incident: a run capped at a token budget produced outputs
/// where a large fraction never reached their required final line. The
/// generation record carried 28 fields and not one said whether the model had
/// finished or been cut off, so the loss was invisible until a human read the
/// text — and it was structural, not random: almost entirely on one arm of an
/// order manipulation, dropping the longer class of output roughly 16:1.
///
/// This engine already KNEW the answer and threw it away:
/// `GenerateCompletionInfo.stopReason` says so directly, and `generateMeasured`
/// reduced it to one Bool that only the choice parser ever saw.
///
/// Server twin: `Server/tests/test_truncation_gate.py`.
@Suite struct TruncationGateTests {

    // MARK: - The field

    @Test func stopReasonMapsToTheClosedVocabulary() {
        #expect(ExperimentTasks.FinishReason.of(.stop) == "stop")
        #expect(ExperimentTasks.FinishReason.of(.length) == "length")
        // A third spelling because this engine's stream genuinely has a third
        // ending — folding it into "stop" would be a lie about a generation
        // that neither finished nor filled its budget. The server's
        // record-writing paths cannot produce it.
        #expect(ExperimentTasks.FinishReason.of(.cancelled) == "cancelled")
        // The fourth spelling (2026-09-03) is never produced by the stream's
        // own stop reason — only the budgeted decode can end there — and it
        // counts as cut off exactly as `length` does.
        #expect(ExperimentTasks.FinishReason.vocabulary
            == ["stop", "length", "lengthInReasoning", "cancelled"])
        #expect(ExperimentTasks.FinishReason.isCutOff("lengthInReasoning"))
        #expect(!ExperimentTasks.FinishReason.isCutOff("cancelled"))
    }

    @Test func truncationIsDerivedFromTheReasonNotStoredTwice() {
        // One notion of truncation, one place: the stamped field and the
        // choice parser cannot disagree about the same generation.
        #expect(
            ExperimentTasks.MeasuredGeneration(text: "x", finishReason: "length")
                .hitTokenCap)
        #expect(
            !ExperimentTasks.MeasuredGeneration(text: "x", finishReason: "stop")
                .hitTokenCap)
        #expect(
            !ExperimentTasks.MeasuredGeneration(text: "x", finishReason: "cancelled")
                .hitTokenCap)
        // Cut off inside the reasoning block: no answer was ever started, so
        // the choice parser must read it as truncated too.
        #expect(
            ExperimentTasks.MeasuredGeneration(
                text: "x", finishReason: "lengthInReasoning"
            ).hitTokenCap)
    }

    private func manifest(maxLengthStoppedFraction: Double? = nil)
        -> ExperimentManifest
    {
        var m = ExperimentManifest(
            name: "s", description: "", modelID: "org/m")
        m.maxLengthStoppedFraction = maxLengthStoppedFraction
        return m
    }

    private func record(
        finishReason: String, options: [String]? = nil,
        output: String = "Apply Kansas law"
    ) -> ExperimentTasks.GenerationRecord {
        let prompt = ExperimentTasks.StudyPrompt(
            id: "p1", text: "Decide.", options: options, target: nil,
            anchorMonths: nil, severity: nil, arm: nil, caseID: nil)
        return ExperimentTasks.sampledGenerationRecord(
            manifest: manifest(),
            experimentHash: "h",
            taskPromptsFile: "f",
            taskPromptsHash: "ph",
            promptMode: .chatAssistant,
            systemPrompt: nil,
            qwenThinkingEnabled: false,
            condition: "baseline",
            seed: 0,
            promptIndex: 1,
            prompt: prompt,
            output: output,
            row: ExperimentTasks.MetricRow(
                condition: "baseline", seed: 0, promptIndex: 1, promptID: "p1",
                wordCount: 3, distinct2: 1, finishReason: finishReason,
                markerDensity: [:]),
            finishReason: finishReason)
    }

    @Test func everySampledRecordCarriesTheReason() throws {
        let capped = record(finishReason: "length")
        #expect(capped.finishReason == "length")
        #expect(record(finishReason: "stop").finishReason == "stop")

        // The key is present in the ENCODED record, not merely on the struct —
        // a reader of generations.jsonl is what the field exists for.
        let encoded = try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(capped)) as? [String: Any]
        #expect(encoded?["finishReason"] as? String == "length")
    }

    @Test func aCappedGenerationStillParsesAsAChoiceFailure() {
        // The rule that already existed, now driven by the stamped reason:
        // a truncated output must never be coerced to its first-enumerated
        // option.
        let options = ["Apply Kansas law", "Apply Nebraska law"]
        // Deliberately a MENTION mid-sentence, not a whole-response match: an
        // exact answer (or a complete embedded JSON one) is still honored on a
        // truncated output — answering and then running out of room while
        // elaborating is a decision, not a coerced one. It is the fallback
        // scan that must not fire.
        let elaborating = "The court should Apply Kansas law because the"
        // `.some(nil)` is a parse FAILURE (JSON null); `.none` means the key
        // does not apply. Matched rather than `==`-compared so the two levels
        // of optionality stay legible.
        guard case .some(let cappedParse) =
            record(
                finishReason: "length", options: options, output: elaborating
            ).parsedChoice
        else {
            Issue.record("a capped option-set record must carry parsedChoice")
            return
        }
        #expect(cappedParse == nil)
        guard case .some(let finishedParse) =
            record(
                finishReason: "stop", options: options, output: elaborating
            ).parsedChoice
        else {
            Issue.record("a finished option-set record must carry parsedChoice")
            return
        }
        #expect(finishedParse == "Apply Kansas law")
    }

    // MARK: - The report

    private func rows(_ spec: [(String, String, String?)]) -> [ExperimentTasks.MetricRow] {
        spec.enumerated().map { index, entry in
            ExperimentTasks.MetricRow(
                condition: entry.0, seed: UInt64(index), promptIndex: index,
                promptID: entry.1, wordCount: 1, distinct2: 1,
                finishReason: entry.2, markerDensity: [:])
        }
    }

    @Test func theReportCountsPerCellNotPooled() {
        let report = ExperimentTasks.truncationReport(
            rows: rows([
                ("baseline", "p0", "stop"), ("baseline", "p0", "stop"),
                ("baseline", "p1", "length"), ("baseline", "p1", "length"),
            ]),
            threshold: nil)
        #expect(report.threshold == nil)
        #expect(report.classified == 4)
        #expect(report.lengthStopped == 2)
        // The pooled number is reported for orientation and gated on by
        // nobody: at 50% it looks unremarkable while p1 is entirely truncated.
        #expect(report.lengthStoppedFraction == 0.5)
        #expect(report.cells.count == 2)
        #expect(report.cells[0].lengthStoppedFraction == 0)
        #expect(report.cells[1].lengthStoppedFraction == 1)
    }

    @Test func aRowNobodyClassifiedIsCountedAsNeitherEnding() {
        // A row from an engine that predates the field is not a row that
        // finished. It contributes no denominator.
        let report = ExperimentTasks.truncationReport(
            rows: rows([("baseline", "p0", nil), ("baseline", "p0", nil)]),
            threshold: 0.5)
        #expect(report.cells.isEmpty)
        #expect(report.classified == 0)
        #expect(report.lengthStoppedFraction == 0)
        #expect(report.threshold == 0.5)
    }

    // MARK: - The gate

    @Test func theRefusalNamesTheCellTheCountsAndTheBudget() throws {
        let text = try #require(
            ExperimentTasks.lengthStoppedRefusal(
                classified: 8, lengthStopped: 7, threshold: 0.25,
                condition: "order-late", promptID: "case-3", maxTokens: 512))
        #expect(text.contains("condition 'order-late' item 'case-3'"))
        #expect(text.contains("7 of 8 generation(s) stopped at the 512-token cap"))
        #expect(text.contains("(87.5%)"))
        #expect(text.contains("maxLengthStoppedFraction of 25.0%"))
    }

    @Test func theCeilingIsTheLargestAcceptedFractionNotOneShortOfIt() {
        // STRICTLY over: exactly at the declared number passes.
        #expect(
            ExperimentTasks.lengthStoppedRefusal(
                classified: 4, lengthStopped: 1, threshold: 0.25,
                condition: "c", promptID: "p", maxTokens: 512) == nil)
        #expect(
            ExperimentTasks.lengthStoppedRefusal(
                classified: 4, lengthStopped: 2, threshold: 0.25,
                condition: "c", promptID: "p", maxTokens: 512) != nil)
        // An empty cell has nothing to refuse.
        #expect(
            ExperimentTasks.lengthStoppedRefusal(
                classified: 0, lengthStopped: 0, threshold: 0,
                condition: "c", promptID: "p", maxTokens: 512) == nil)
    }

    @Test func theCellHelperCountsOnlyItsOwnCell() {
        let all = rows([
            ("baseline", "p0", "length"), ("baseline", "p1", "stop"),
            ("steered", "p0", "stop"),
        ])
        let cell = ExperimentTasks.truncationCell(
            rows: all, condition: "baseline", promptID: "p0")
        #expect(cell.classified == 1 && cell.lengthStopped == 1)
    }

    @Test func theGateIsInTheClosedVocabularyAndClaimedByASite() throws {
        #expect(LifecycleGate.vocabulary.contains("lengthStopped"))
        let site = try #require(RefusalSiteRegistry.site(for: .lengthStopped))
        #expect(site.verbs == ["experiment run"])
        #expect(site.repairAction.contains("experiment set-sampling"))
    }

    @Test func theRepairIsACommandNotAdvice() {
        let repair = ExperimentTasks.lengthStoppedRepair(
            experiment: "memo", maxTokens: 512)
        #expect(repair.contains("steerlab-cli experiment set-sampling memo --max-tokens"))
        #expect(repair.contains("steerlab-cli experiment duplicate memo memo-v2"))
    }

    // MARK: - The manifest key

    @Test func theKeyIsOffByDefaultAndSurvivesTheMacRoundTrip() throws {
        #expect(manifest().maxLengthStoppedFraction == nil)

        // Decoded from a document, re-encoded, and still there — the failure
        // this test exists for is a key that is absent from `CodingKeys` and
        // is therefore destroyed by `duplicate`'s load/save.
        var document = try #require(
            try JSONSerialization.jsonObject(
                with: JSONEncoder().encode(manifest())) as? [String: Any])
        document["maxLengthStoppedFraction"] = 0.25
        let decoded = try JSONDecoder().decode(
            ExperimentManifest.self,
            from: JSONSerialization.data(withJSONObject: document))
        #expect(decoded.maxLengthStoppedFraction == 0.25)
        let round = try JSONDecoder().decode(
            ExperimentManifest.self, from: JSONEncoder().encode(decoded))
        #expect(round.maxLengthStoppedFraction == 0.25)
    }

    @Test func anAbsentKeyLeavesTheContentHashByteIdentical() {
        // Optional and omit-when-nil (the `exclusionRules` template), so every
        // manifest written before this key existed hashes exactly as it did.
        var declared = manifest()
        let bare = ExperimentStore.manifestHash(declared)
        declared.maxLengthStoppedFraction = nil
        #expect(ExperimentStore.manifestHash(declared) == bare)
        declared.maxLengthStoppedFraction = 0.25
        // …and a DECLARED ceiling is manifest data, so it enters the hash the
        // ordinary way and freeze pins it.
        #expect(ExperimentStore.manifestHash(declared) != bare)
    }
}
