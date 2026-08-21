import Foundation
import Testing

@testable import ExperimentKit

/// The Results semantic layer (A3/A6/A13, P3/P4): lenient cross-engine
/// record decoding, choice-matrix/arm/agreement/completion math, the
/// categorical N/A rule, the evidence-status classification rule, analyze
/// routing, and the analysis/validation artifact parsers. Pure CPU on
/// hand-authored fixtures — no model, no live run output.
struct RunResultsTests {

    // MARK: - Record decoding (A6), both engines' shapes

    /// A Swift-engine sampled generation record (seedInert stamp, judicial
    /// parse key present with a VALUE).
    private let swiftGenerationLine = """
        {"experiment":"pilot","experimentHash":"aaa","modelID":"m",\
        "modelRevision":"deadbeefcafe","taskPromptsFile":"t.jsonl",\
        "taskPromptsHash":"h","promptMode":"chatAssistant","systemPrompt":null,\
        "qwenThinkingEnabled":false,"condition":"baseline","seed":1,\
        "seedInert":true,"promptIndex":0,"promptID":"case-1","prompt":"P?",\
        "output":"guilty","wordCount":1,"distinct2":0.0,"markerDensity":{},\
        "variantArtifactPath":null,"variantArtifactHash":null,\
        "target":"guilty","arm":"treatment","caseID":"c1",\
        "options":["guilty","not guilty"],"parsedChoice":"guilty"}
        """

    /// A server-engine record: engine extras (interventionState,
    /// sampleIndex, sampling block) plus a parse FAILURE (JSON null).
    private let serverFailureLine = """
        {"experiment":"pilot","experimentHash":"aaa","modelID":"m",\
        "modelRevision":"deadbeefcafe","condition":"fear-0.8","seed":7,\
        "seedPolicy":"derivedSHA256","sampleIndex":0,"promptIndex":0,\
        "promptID":"case-1","prompt":"P?","interventionState":"fear@0.8",\
        "output":"I cannot say","wordCount":3,"distinct2":1.0,\
        "options":["guilty","not guilty"],"target":"guilty",\
        "parsedChoice":null,"temperature":0.7,"doSample":true}
        """

    /// A choice record (answerTokenLogprob) — the shared cross-engine
    /// envelope from `ChoiceRecord` / `as_record_fields`.
    private let choiceLine = """
        {"experiment":"pilot","experimentHash":"aaa","modelID":"m",\
        "modelRevision":"deadbeefcafe","condition":"fear-0.8","promptIndex":0,\
        "promptID":"case-1","prompt":"P?","target":"guilty",\
        "instrument":"answerTokenLogprob","options":["guilty","not guilty"],\
        "optionTokenCounts":{"guilty":1,"not guilty":2},\
        "optionLengthRatio":2.0,\
        "optionLogprobs":{"guilty":-0.3,"not guilty":-1.5},\
        "choiceProbability":{"guilty":0.77,"not guilty":0.23},\
        "logOdds":{"guilty":1.21,"not guilty":-1.21},\
        "selected":"guilty","margin":1.2}
        """

    /// A variant-condition record with artifact provenance.
    private let variantLine = """
        {"condition":"agent-frozen","promptID":"case-2","promptIndex":1,\
        "prompt":"P2?","output":"words flow freely here","wordCount":4,\
        "distinct2":1.0,"variantArtifactPath":"runs/model-variants/a.json",\
        "variantArtifactHash":"vh"}
        """

    @Test func decodesSwiftGenerationRecord() throws {
        let parsed = RunResults.records(fromJSONL: swiftGenerationLine)
        #expect(parsed.skippedLines == 0)
        let record = try #require(parsed.records.first)
        #expect(record.condition == "baseline")
        #expect(record.promptID == "case-1")
        #expect(record.wordCount == 1)
        #expect(record.parsedChoice == .value("guilty"))
        #expect(record.arm == "treatment")
        #expect(record.modelRevision == "deadbeefcafe")
        #expect(record.isCategorical)
        #expect(!record.isChoiceRecord)
    }

    @Test func decodesServerRecordWithEngineExtrasAndParseFailure() throws {
        let parsed = RunResults.records(fromJSONL: serverFailureLine)
        let record = try #require(parsed.records.first)
        #expect(record.sampleIndex == 0)
        #expect(record.parsedChoice == .failure)
        #expect(record.parsedChoice.isFailure)
        #expect(record.isCategorical)
    }

    @Test func decodesChoiceRecord() throws {
        let parsed = RunResults.records(fromJSONL: choiceLine)
        let record = try #require(parsed.records.first)
        #expect(record.isChoiceRecord)
        #expect(record.selected == "guilty")
        #expect(record.choiceProbability?["guilty"] == 0.77)
        #expect(record.logOdds?["not guilty"] == -1.21)
        #expect(record.margin == 1.2)
        #expect(record.parsedChoice == .absent)
    }

    @Test func decodesVariantRecordProvenance() throws {
        let parsed = RunResults.records(fromJSONL: variantLine)
        let record = try #require(parsed.records.first)
        #expect(record.variantArtifactPath == "runs/model-variants/a.json")
        #expect(record.variantArtifactHash == "vh")
        #expect(!record.isCategorical)
    }

    @Test func undecodableLinesAreCountedNeverDroppedSilently() {
        let text = swiftGenerationLine + "\nnot json at all\n{\"no\":\"condition\"}"
        let parsed = RunResults.records(fromJSONL: text)
        #expect(parsed.records.count == 1)
        #expect(parsed.skippedLines == 2)
    }

    // MARK: - report.json decoding (cross-engine, agreement key tolerated)

    @Test func decodesReportWithAgreementFractionAndErrorCondition() throws {
        let json = """
            {"experiment":"pilot","experimentHash":"aaa","conditionCount":3,
             "conditions":{
               "baseline":{"generations":12,"meanWordCount":1.0,"meanDistinct2":0.0},
               "fear-0.8":{"generations":12,"choiceReadouts":12,
                           "agreementWithBaseline":0.75,
                           "capabilityBattery":{"accuracy":0.9,"itemCount":20,
                                                "batteryHash":"bh"}},
               "broken-variant":{"generations":0,"error":"artifact hash drifted"}}}
            """
        let report = try #require(RunResults.report(fromJSON: Data(json.utf8)))
        #expect(report.conditions.count == 3)
        #expect(report.conditions["fear-0.8"]?.agreementWithBaseline == 0.75)
        #expect(report.conditions["fear-0.8"]?.capabilityBatteryAccuracy == 0.9)
        #expect(report.conditions["baseline"]?.agreementWithBaseline == nil)
        #expect(report.conditions["broken-variant"]?.error == "artifact hash drifted")
    }

    @Test func decodesAgreementObjectShape() throws {
        let json = """
            {"conditions":{"v":{"generations":4,
              "agreementWithBaseline":{"matches":3,"total":4}}}}
            """
        let report = try #require(RunResults.report(fromJSON: Data(json.utf8)))
        let summary = try #require(report.conditions["v"])
        #expect(summary.agreementMatches == 3)
        #expect(summary.agreementTotal == 4)
        #expect(summary.agreementWithBaseline == 0.75)
    }

    @Test func decodesServerAgreementShapeAndChoiceRate() throws {
        let json = """
            {"conditions":{"v":{"generations":4,"choiceReadouts":2,
              "choiceRate":0.5,
              "agreementWithBaseline":{"n":4,"agreement":0.75}}}}
            """
        let report = try #require(RunResults.report(fromJSON: Data(json.utf8)))
        let summary = try #require(report.conditions["v"])
        #expect(summary.choiceReadouts == 2)
        #expect(summary.choiceRate == 0.5)
        // No decode-time synthesis: the {n, agreement} shape carries no
        // explicit match count, so none is invented here — the one consumer
        // that needs a count derives it from the fraction itself.
        #expect(summary.agreementMatches == nil)
        #expect(summary.agreementTotal == 4)
        #expect(summary.agreementWithBaseline == 0.75)
    }

    @Test func decodesPerConditionMeansIncludingMarkerDensity() throws {
        // F2: the Swift engine's whole-run per-condition means — marker
        // density is the manipulation-check diagnostic and must survive
        // decoding (the app renders it for EVERY study run).
        let json = """
            {"conditions":{
               "baseline":{"generations":12,"meanWordCount":180.5,
                           "meanDistinct2":0.91,
                           "meanMarkerDensity":{"fear":0.004,"calm":0.001}},
               "fear-0.8":{"generations":12,"meanWordCount":175.0,
                           "meanDistinct2":0.88,
                           "meanMarkerDensity":{"fear":0.031,"calm":0.002}}}}
            """
        let report = try #require(RunResults.report(fromJSON: Data(json.utf8)))
        let steered = try #require(report.conditions["fear-0.8"])
        #expect(steered.meanWordCount == 175.0)
        #expect(steered.meanDistinct2 == 0.88)
        #expect(steered.meanMarkerDensity?["fear"] == 0.031)
        #expect(steered.meanMarkerDensity?["calm"] == 0.002)
        #expect(report.conditions["baseline"]?.meanMarkerDensity?["fear"] == 0.004)
    }

    @Test func decodesInlineEffectSizesFromSwiftReport() throws {
        let json = """
            {"conditions":{},"effectSizes":[
              {"condition":"fear-0.8","metric":"wordCount","n":10,
               "meanDiff":-2.5,"ciLower":-4.0,"ciUpper":-1.0,
               "wilcoxonW":3.0,"wilcoxonP":0.02}]}
            """
        let report = try #require(RunResults.report(fromJSON: Data(json.utf8)))
        let row = try #require(report.effectSizes.first)
        #expect(row.metric == "wordCount")
        #expect(row.meanDiff == -2.5)
        #expect(row.wilcoxonP == 0.02)
        #expect(row.adjustedP == nil)
    }

    /// Swift-local runs stamp the multiple-comparison correction inline in
    /// report.json (2026-07-19): the reader surfaces adjustedP/correction,
    /// the significance rule prefers the corrected p, and the narrative's
    /// "survives correction" path fires from a LOCAL run's report row — no
    /// server analyze artifact involved.
    @Test func decodesInlineCorrectedEffectSizesFromSwiftReport() throws {
        let json = """
            {"conditions":{},"effectSizes":[
              {"condition":"fear-0.8","metric":"wordCount","n":10,
               "meanDiff":-2.5,"ciLower":-4.0,"ciUpper":-1.0,
               "wilcoxonW":3.0,"wilcoxonP":0.02,
               "adjustedP":0.04,"correction":"bh"},
              {"condition":"fear-0.8","metric":"distinct2","n":10,
               "meanDiff":0.0,"ciLower":0.0,"ciUpper":0.0,
               "correction":"bh"}]}
            """
        let report = try #require(RunResults.report(fromJSON: Data(json.utf8)))
        let row = try #require(report.effectSizes.first)
        #expect(row.adjustedP == 0.04)
        #expect(row.correction == "bh")
        #expect(row.significantAfterCorrection == true)
        let sentence = EffectNarrative.sentence(for: row)
        #expect(sentence.contains("survives multiple-comparison correction"))
        #expect(sentence.contains("corrected p = 0.04, bh"))
        // A family-stamped row with an undefined p keeps the honest absence:
        // correction present, no adjusted p, no invented significance.
        let undefinedP = try #require(
            report.effectSizes.first { $0.metric == "distinct2" })
        #expect(undefinedP.correction == "bh")
        #expect(undefinedP.adjustedP == nil)
        #expect(undefinedP.significantAfterCorrection == nil)
    }

    // MARK: - Choice matrix / arms / agreement / completion (P3)

    /// Two items × (baseline + steered + no-op variant), instrumented cells
    /// for the steered condition, a parse failure on one baseline sample.
    private func fixtureRecords() -> [RunResults.Record] {
        var records: [RunResults.Record] = []
        func sampled(
            _ condition: String, _ item: String, index: Int, choice: String?,
            arm: String, failed: Bool = false
        ) {
            var record = RunResults.Record(condition: condition, promptID: item)
            record.promptIndex = index
            record.options = ["guilty", "not guilty"]
            record.target = "guilty"
            record.arm = arm
            record.output = choice ?? "unclear"
            record.wordCount = 1
            record.distinct2 = 0
            record.parsedChoice = failed ? .failure : .value(choice ?? "")
            records.append(record)
        }
        sampled("baseline", "case-1", index: 0, choice: "guilty", arm: "relevant")
        sampled("baseline", "case-2", index: 1, choice: nil, arm: "irrelevant", failed: true)
        sampled("no-op", "case-1", index: 0, choice: "guilty", arm: "relevant")
        sampled("no-op", "case-2", index: 1, choice: "not guilty", arm: "irrelevant")
        // Steered condition measured by the instrument only.
        for (item, index, selected, p, lo) in [
            ("case-1", 0, "not guilty", 0.61, 0.45),
            ("case-2", 1, "guilty", 0.81, 1.45),
        ] {
            var record = RunResults.Record(condition: "fear-0.8", promptID: item)
            record.promptIndex = index
            record.instrument = "answerTokenLogprob"
            record.options = ["guilty", "not guilty"]
            record.target = "guilty"
            record.arm = index == 0 ? "relevant" : "irrelevant"
            record.selected = selected
            record.choiceProbability = ["guilty": p]
            record.logOdds = ["guilty": lo]
            record.margin = 0.5
            records.append(record)
        }
        return records
    }

    @Test func choiceMatrixOrdersBaselineFirstAndFillsCells() throws {
        let matrix = RunResults.choiceMatrix(records: fixtureRecords())
        #expect(matrix.conditions == ["baseline", "fear-0.8", "no-op"])
        #expect(matrix.items.map(\.promptID) == ["case-1", "case-2"])
        let item1 = matrix.items[0]
        #expect(item1.target == "guilty")
        #expect(item1.cells["baseline"]?.choice == "guilty")
        #expect(item1.cells["baseline"]?.matchesTarget == true)
        let steered = try #require(item1.cells["fear-0.8"])
        #expect(steered.instrumented)
        #expect(steered.choice == "not guilty")
        #expect(steered.matchesTarget == false)
        #expect(steered.targetProbability == 0.61)
        #expect(steered.targetLogOdds == 0.45)
        // The failed baseline parse on case-2: no choice, one failure.
        let failedCell = try #require(matrix.items[1].cells["baseline"])
        #expect(failedCell.choice == nil)
        #expect(failedCell.parseFailures == 1)
    }

    @Test func armAggregationGroupsByArmMetadata() throws {
        let matrix = RunResults.choiceMatrix(records: fixtureRecords())
        let arms = RunResults.armAggregation(matrix: matrix)
        #expect(Set(arms.map(\.arm)) == ["relevant", "irrelevant"])
        let relevantSteered = try #require(
            arms.first { $0.arm == "relevant" && $0.condition == "fear-0.8" })
        #expect(relevantSteered.items == 1)
        #expect(relevantSteered.targetMatchRate == 0)
        #expect(relevantSteered.meanTargetProbability == 0.61)
    }

    @Test func parseFailuresListEveryNullParseKey() {
        let failures = RunResults.parseFailures(records: fixtureRecords())
        #expect(failures.count == 1)
        #expect(failures.first?.condition == "baseline")
        #expect(failures.first?.promptID == "case-2")
        #expect(failures.first?.endpoint == "choice")
        #expect(failures.first?.outputExcerpt == "unclear")
    }

    @Test func baselineAgreementComputedLocally() throws {
        let matrix = RunResults.choiceMatrix(records: fixtureRecords())
        let agreements = RunResults.baselineAgreement(matrix: matrix)
        // case-2 baseline failed to parse → only case-1 is comparable.
        let noOp = try #require(agreements.first { $0.condition == "no-op" })
        #expect(noOp.matches == 1)
        #expect(noOp.comparable == 1)
        #expect(!noOp.fromServerStamp)
        let steered = try #require(agreements.first { $0.condition == "fear-0.8" })
        #expect(steered.matches == 0)
        #expect(steered.comparable == 1)
    }

    @Test func serverStampedAgreementIsPreferred() throws {
        let matrix = RunResults.choiceMatrix(records: fixtureRecords())
        var report = RunResults.Report()
        var summary = RunResults.ConditionSummary(generations: 2)
        summary.agreementWithBaseline = 1.0
        summary.agreementMatches = 12
        summary.agreementTotal = 12
        report.conditions["no-op"] = summary
        let agreements = RunResults.baselineAgreement(matrix: matrix, report: report)
        let noOp = try #require(agreements.first { $0.condition == "no-op" })
        #expect(noOp.fromServerStamp)
        #expect(noOp.matches == 12)
        #expect(noOp.comparable == 12)
        // The un-stamped condition still gets the local count.
        let steered = try #require(agreements.first { $0.condition == "fear-0.8" })
        #expect(!steered.fromServerStamp)
    }

    @Test func agreementRequiresABaselineColumn() {
        var record = RunResults.Record(condition: "solo", promptID: "x")
        record.options = ["a", "b"]
        record.parsedChoice = .value("a")
        let matrix = RunResults.choiceMatrix(records: [record])
        #expect(RunResults.baselineAgreement(matrix: matrix).isEmpty)
    }

    @Test func completionDerivesUniformPlanAcrossConditions() {
        // A uniform 2-condition × 2-item sampled matrix with one record
        // missing: 3 of 4 slots complete, and the short condition shows it.
        var records: [RunResults.Record] = []
        for (condition, item) in [
            ("baseline", "case-1"), ("baseline", "case-2"), ("fear-0.8", "case-1"),
        ] {
            var record = RunResults.Record(condition: condition, promptID: item)
            record.output = "text"
            records.append(record)
        }
        let completion = RunResults.completion(records: records)
        #expect(completion.planned == 4)
        #expect(completion.completed == 3)
        let steered = completion.perCondition.first { $0.condition == "fear-0.8" }
        #expect(steered?.completed == 1)
        #expect(steered?.planned == 2)
        #expect(completion.summaryLine == "3 of 4 measurement records completed")
    }

    @Test func completionCountsInstrumentAndSampledSlotsSeparately() {
        // A condition measured by BOTH the choice instrument and a sampled
        // pass (the server's wants_choice + wants_sampled layout) plans two
        // slots per item, not one.
        var choice = RunResults.Record(condition: "baseline", promptID: "case-1")
        choice.instrument = "answerTokenLogprob"
        var sampledRecord = RunResults.Record(condition: "baseline", promptID: "case-1")
        sampledRecord.output = "guilty"
        let completion = RunResults.completion(records: [choice, sampledRecord])
        #expect(completion.planned == 2)
        #expect(completion.completed == 2)
        #expect(completion.summaryLine == "2 of 2 measurement records completed")
    }

    @Test func completionSurfacesReportErrorConditions() {
        var report = RunResults.Report()
        var summary = RunResults.ConditionSummary(generations: 0)
        summary.error = "variant artifact hash drifted"
        report.conditions["broken"] = summary
        let completion = RunResults.completion(
            records: fixtureRecords(), report: report)
        let broken = completion.perCondition.first { $0.condition == "broken" }
        #expect(broken?.completed == 0)
        #expect(broken?.error == "variant artifact hash drifted")
    }

    // MARK: - Categorical N/A rule (acceptance 7)

    @Test func categoricalConditionsDriveTheNARule() {
        let categorical = RunResults.categoricalConditions(
            records: fixtureRecords())
        #expect(categorical == ["baseline", "fear-0.8", "no-op"])
        #expect(
            RunResults.textMetricDisplay(0.0, categorical: true)
                == "N/A (categorical output)")
        #expect(
            RunResults.textMetricDisplay(1.0, categorical: true)
                == "N/A (categorical output)")
        #expect(RunResults.textMetricDisplay(0.375, categorical: false) == "0.38")
        #expect(RunResults.textMetricDisplay(nil, categorical: false) == "—")
    }

    @Test func freeTextConditionsAreNotCategorical() {
        let parsed = RunResults.records(fromJSONL: variantLine)
        #expect(RunResults.categoricalConditions(records: parsed.records).isEmpty)
    }

    @Test func reportChoiceMetricsProveCategoricalWithoutGenerationRows() {
        // A run whose generations.jsonl is missing/truncated away but whose
        // report stamps choice metrics is still a categorical run — the
        // categorical presentation must derive from report evidence too,
        // never hide stamped choiceRate behind zero decoded records.
        let reportJSON = """
            {"experiment": "choice-study", "conditions": {
               "baseline": {"generations": 12, "choiceReadouts": 12,
                            "choiceRate": 0.25},
               "steered": {"generations": 12, "choiceReadouts": 12,
                           "choiceRate": 0.75,
                           "agreementWithBaseline": {"n": 12, "agreement": 0.5}},
               "prose-only": {"generations": 12, "meanWordCount": 140.0}}}
            """
        var artifacts = RunResults.ArtifactBytes()
        artifacts.reportData = Data(reportJSON.utf8)
        let model = RunResults.remoteModel(runID: "r1", artifacts: artifacts)

        #expect(model.records.isEmpty)  // no generation rows at all
        #expect(model.isCategorical)  // …yet the run is provably categorical
        #expect(model.categoricalConditions == ["baseline", "steered"])
        // A condition with only text metrics stays non-categorical.
        #expect(!model.categoricalConditions.contains("prose-only"))

        // The pure helper agrees when called directly.
        let derived = RunResults.categoricalConditions(
            records: [], report: model.report)
        #expect(derived == ["baseline", "steered"])
    }

    // MARK: - Intervention summaries (P3 "conditions and interventions")

    @Test func interventionSummariesReadFromTheManifestSnapshot() {
        var manifest = ExperimentManifest(
            name: "pilot", description: "d", modelID: "m")
        manifest.conditions = [
            .init(
                name: "fear-0.8",
                slots: [.init(concept: "fear", layer: 14, alpha: 0.8)]),
            .init(
                name: "fear-random",
                slots: [.init(concept: "fear", layer: 14, alpha: 0.8)],
                controlType: "randomMatchedNorm"),
            .init(
                name: "mix",
                slots: [
                    .init(concept: "fear", layer: 14, alpha: 0.8),
                    .init(concept: "calm", layer: 12, alpha: -1),
                ]),
        ]
        let summaries = RunResults.interventionSummaries(manifest: manifest)
        #expect(summaries["baseline"] == "no intervention")
        #expect(summaries["fear-0.8"] == "fear L14 α0.8")
        #expect(
            summaries["fear-random"]
                == "matched-norm random control · fear L14 α0.8")
        #expect(summaries["mix"] == "fear L14 α0.8 + calm L12 α-1")
    }

    // MARK: - Classification rule (P4)

    @Test func classifiesAllFourStates() {
        // Frozen + epoch verified.
        let frozen = RunResults.classify(
            manifestStatus: "frozen", freezeForced: nil,
            hasMatchingValidateEvidence: true, stampedHashMatchesLive: true,
            manifestRevision: "abc", recordRevisions: ["abc"])
        #expect(frozen.runClass == .frozenEvidenceGrade)
        #expect(frozen.epoch == .verified)
        #expect(frozen.label == "Frozen (evidence-grade)")
        #expect(frozen.revisionWarning == nil)

        // Forced freeze wins over frozen — non-citable by stamp.
        let forced = RunResults.classify(
            manifestStatus: "frozen", freezeForced: true,
            hasMatchingValidateEvidence: true, stampedHashMatchesLive: true,
            manifestRevision: "abc", recordRevisions: [])
        #expect(forced.runClass == .forcedFreeze)
        #expect(forced.label == "Forced freeze — non-citable")

        // Draft with matching validate evidence.
        let validated = RunResults.classify(
            manifestStatus: "draft", freezeForced: nil,
            hasMatchingValidateEvidence: true, stampedHashMatchesLive: nil,
            manifestRevision: "abc", recordRevisions: [])
        #expect(validated.runClass == .validatedDraft)

        // Nothing: a pilot.
        let pilot = RunResults.classify(
            manifestStatus: "draft", freezeForced: nil,
            hasMatchingValidateEvidence: false, stampedHashMatchesLive: nil,
            manifestRevision: nil, recordRevisions: [])
        #expect(pilot.runClass == .draftPilot)
        #expect(pilot.label == "Draft/pilot")
    }

    @Test func completeStatusReadsAsFrozenEvidence() {
        let complete = RunResults.classify(
            manifestStatus: "complete", freezeForced: nil,
            hasMatchingValidateEvidence: nil, stampedHashMatchesLive: true,
            manifestRevision: "abc", recordRevisions: [])
        #expect(complete.runClass == .frozenEvidenceGrade)
    }

    @Test func epochMismatchAndUnknownAreDistinct() {
        let mismatch = RunResults.classify(
            manifestStatus: "frozen", freezeForced: nil,
            hasMatchingValidateEvidence: nil, stampedHashMatchesLive: false,
            manifestRevision: "abc", recordRevisions: [])
        #expect(mismatch.epoch == .failed)
        #expect(mismatch.runClass == .frozenEpochMismatch)
        #expect(mismatch.detail.contains("epoch mismatch"))
        let unknown = RunResults.classify(
            manifestStatus: "frozen", freezeForced: nil,
            hasMatchingValidateEvidence: nil, stampedHashMatchesLive: nil,
            manifestRevision: "abc", recordRevisions: [])
        #expect(unknown.epoch == .unknown)
        #expect(unknown.runClass == .frozenUnverified)
    }

    @Test func revisionChipFiresOnlyWhenUnpinnedButResolved() {
        // P4's exact scenario: manifest revision unset, records resolved one.
        let warned = RunResults.classify(
            manifestStatus: "draft", freezeForced: nil,
            hasMatchingValidateEvidence: false, stampedHashMatchesLive: nil,
            manifestRevision: nil, recordRevisions: ["deadbeefcafe0123"])
        #expect(warned.unpinnedResolvedRevision == "deadbeefcafe0123")
        #expect(warned.revisionWarning?.contains("never pinned") == true)

        // Pinned manifest: no chip regardless of records.
        let pinned = RunResults.classify(
            manifestStatus: "draft", freezeForced: nil,
            hasMatchingValidateEvidence: false, stampedHashMatchesLive: nil,
            manifestRevision: "deadbeefcafe0123",
            recordRevisions: ["deadbeefcafe0123"])
        #expect(pinned.unpinnedResolvedRevision == nil)

        // No revision anywhere: nothing resolved, nothing to warn about.
        let silent = RunResults.classify(
            manifestStatus: "draft", freezeForced: nil,
            hasMatchingValidateEvidence: false, stampedHashMatchesLive: nil,
            manifestRevision: nil, recordRevisions: [])
        #expect(silent.unpinnedResolvedRevision == nil)

        // Mixed resolved revisions still warn (deterministically).
        let mixed = RunResults.classify(
            manifestStatus: "draft", freezeForced: nil,
            hasMatchingValidateEvidence: false, stampedHashMatchesLive: nil,
            manifestRevision: nil, recordRevisions: ["bbb", "aaa"])
        #expect(mixed.unpinnedResolvedRevision == "aaa")
    }

    // MARK: - Analyze routing (A3)

    @Test func analyzeRoutesBySubstrateStamp() {
        #expect(
            RunResults.analyzeRoute(
                runType: "run", substrate: "swift-mlx", experiment: "pilot",
                serverAvailable: false)
                == .local(experiment: "pilot"))
        // Unstamped substrate defaults to local.
        #expect(
            RunResults.analyzeRoute(
                runType: "run", substrate: nil, experiment: "pilot",
                serverAvailable: true)
                == .local(experiment: "pilot"))
        // A server-produced run must analyze on the server (epoch guard is
        // per-engine).
        #expect(
            RunResults.analyzeRoute(
                runType: "run", substrate: "python-hf", experiment: "pilot",
                serverAvailable: true)
                == .server(experiment: "pilot"))
        if case .unavailable(let reason) = RunResults.analyzeRoute(
            runType: "run", substrate: "python-hf", experiment: "pilot",
            serverAvailable: false)
        {
            #expect(reason.contains("python-hf"))
        } else {
            Issue.record("expected unavailable without a connected server")
        }
    }

    @Test func analyzeRefusesNonStudyRunsAndMissingExperiment() {
        if case .unavailable(let reason) = RunResults.analyzeRoute(
            runType: "validate", substrate: "swift-mlx", experiment: "pilot",
            serverAvailable: true)
        {
            #expect(reason.contains("validate"))
        } else {
            Issue.record("expected unavailable for a validate run")
        }
        if case .unavailable = RunResults.analyzeRoute(
            runType: "run", substrate: "swift-mlx", experiment: nil,
            serverAvailable: true)
        {
        } else {
            Issue.record("expected unavailable without an experiment stamp")
        }
    }

    @Test func remoteBrowsingAlwaysRoutesToTheServer() {
        #expect(
            RunResults.analyzeRoute(
                runType: "run", substrate: "python-hf", experiment: "pilot",
                serverAvailable: true, browsingRemote: true)
                == .server(experiment: "pilot"))
        if case .unavailable = RunResults.analyzeRoute(
            runType: "run", substrate: "python-hf", experiment: "pilot",
            serverAvailable: false, browsingRemote: true)
        {
        } else {
            Issue.record("expected unavailable when disconnected")
        }
    }

    // MARK: - effect-sizes.csv (both engines' dialects)

    @Test func parsesSwiftEffectSizesDialect() throws {
        let csv = """
            condition,metric,n,meanDiff,ciLower,ciUpper,wilcoxonW,wilcoxonP
            fear-0.8,wordCount,10,-2.5,-4.0,-1.0,3.0,0.02
            fear-0.8,markerDensity:fear,10,0.1,0.05,0.2,,
            """
        let rows = try #require(RunResults.effectSizes(fromCSV: csv))
        #expect(rows.count == 2)
        #expect(rows[0].metric == "wordCount")
        #expect(rows[0].meanDiff == -2.5)
        #expect(rows[0].wilcoxonP == 0.02)
        #expect(rows[0].adjustedP == nil)
        #expect(rows[0].ciExcludesZero)
        #expect(rows[0].significantAfterCorrection == true)
        #expect(rows[1].wilcoxonP == nil)
        #expect(rows[1].significantAfterCorrection == nil)
    }

    @Test func parsesServerEffectSizesDialect() throws {
        let csv = """
            condition,endpoint,n,deltaMean,ciLower,ciUpper,wilcoxonW,wilcoxonP,adjustedP,correction,modality
            fear-0.8,choiceLogOdds,12,0.42,0.1,0.7,5,0.01,0.04,bh,injection
            calm-0.4,choiceLogOdds,12,0.02,-0.2,0.3,30,0.6,0.8,bh,injection
            """
        let rows = try #require(RunResults.effectSizes(fromCSV: csv))
        #expect(rows[0].metric == "choiceLogOdds")
        #expect(rows[0].meanDiff == 0.42)
        #expect(rows[0].adjustedP == 0.04)
        #expect(rows[0].correction == "bh")
        #expect(rows[0].modality == "injection")
        #expect(rows[0].significantAfterCorrection == true)
        // Corrected p wins over the raw p for significance.
        #expect(rows[1].significantAfterCorrection == false)
        #expect(!rows[1].ciExcludesZero)
    }

    @Test func effectSizesRefusesForeignCSV() {
        #expect(RunResults.effectSizes(fromCSV: "a,b\n1,2") == nil)
    }

    // MARK: - alien-residuals.csv / promoted-movers.json

    @Test func parsesAlienResiduals() throws {
        let csv = """
            condition,endpoint,deltaModel,ciModelLower,ciModelUpper,deltaHuman,ciHumanLower,ciHumanUpper,R,ciRLower,ciRUpper,region
            fear-0.8,meanMonths,4.2,3.0,5.4,1.1,0.5,1.7,3.1,1.3,4.9,hyperHuman
            calm-0.4,meanMonths,0.5,-0.2,1.2,0,-0.4,0.4,0.5,-0.6,1.6,inertBoth
            """
        let rows = try #require(RunResults.alienResiduals(fromCSV: csv))
        #expect(rows.count == 2)
        #expect(rows[0].r == 3.1)
        #expect(rows[0].region == "hyperHuman")
        #expect(rows[0].ciRLower == 1.3)
        #expect(rows[1].deltaHuman == 0)
    }

    @Test func parsesPromotedMovers() throws {
        let json = """
            {"experiment":"screen-1","experimentHash":"aaa",
             "promotionRule":{"fdrThreshold":0.05,"doseMonotone":true,
                              "exceedsRandomFloor":true,"capabilityGate":true},
             "promoted":[{"concept":"fear","condition":"fear-0.8",
                          "endpoint":"choiceLogOdds","effectEstimate":0.42,
                          "adjustedP":0.01,"promoted":true,"reasons":[]}],
             "rejected":[{"concept":"calm","condition":"calm-0.4",
                          "endpoint":"choiceLogOdds","effectEstimate":0.02,
                          "adjustedP":0.8,"promoted":false,
                          "reasons":["adjustedP 0.8 above threshold"]}]}
            """
        let movers = try #require(RunResults.promotedMovers(fromJSON: Data(json.utf8)))
        #expect(movers.promoted.count == 1)
        #expect(movers.promoted[0].concept == "fear")
        #expect(movers.promoted[0].promoted)
        #expect(movers.rejected[0].reasons == ["adjustedP 0.8 above threshold"])
    }

    // MARK: - cosine-matrix.csv (both engines' headers)

    @Test func parsesSwiftCosineMatrixWithLayerColumn() throws {
        let csv = """
            concept,layer,calm,fear
            calm,12,1.0000,0.6100
            fear,14,0.6100,1.0000
            """
        let matrix = try #require(RunResults.cosineMatrix(fromCSV: csv))
        #expect(matrix.concepts == ["calm", "fear"])
        #expect(matrix.values[0][1] == 0.61)
        #expect(matrix.isFlagged(row: 0, column: 1))
        #expect(!matrix.isFlagged(row: 0, column: 0))  // diagonal never flags
    }

    @Test func parsesServerCosineMatrixWithoutLayerColumn() throws {
        let csv = """
            concept,calm,fear
            calm,1.0000,0.2000
            fear,0.2000,nan
            """
        let matrix = try #require(RunResults.cosineMatrix(fromCSV: csv))
        #expect(matrix.concepts == ["calm", "fear"])
        #expect(!matrix.isFlagged(row: 0, column: 1))
        #expect(matrix.values[1][1] == nil)  // "nan" cell
    }

    // MARK: - Validation report (A13, both engines' dialects)

    @Test func parsesSwiftValidationReport() throws {
        let json = """
            {"experiment":"pilot",
             "validation":{
               "fear":{"scenarios":24,"layer":14,"accuracy":0.83},
               "calm":"no validation.jsonl — convergent gate NOT run"},
             "logitLens":{
               "fear":{"layer":14,
                 "topPositive":[{"tokenID":1,"token":"fear","logit":5.0},
                                {"tokenID":2,"token":"dread","logit":4.0}],
                 "topNegative":[{"tokenID":3,"token":"calm","logit":-4.0}]}},
             "worstCosinePair":"calm × fear = 0.61",
             "capabilityBattery":[{"condition":"baseline","batteryHash":"bh",
                                   "total":20,"correct":19,"accuracy":0.95}]}
            """
        let report = try #require(
            RunResults.validationReport(fromJSON: Data(json.utf8)))
        #expect(report.concepts.count == 2)
        let calm = try #require(report.concepts.first { $0.concept == "calm" })
        #expect(calm.accuracy == nil)
        #expect(calm.note?.contains("NOT run") == true)
        let fear = try #require(report.concepts.first { $0.concept == "fear" })
        #expect(fear.accuracy == 0.83)
        #expect(fear.beatsChance == true)
        #expect(fear.scenarios == 24)
        #expect(report.logitLens.first?.topPositive == ["fear", "dread"])
        #expect(report.logitLens.first?.topNegative == ["calm"])
        #expect(report.worstCosinePair == "calm × fear = 0.61")
        #expect(report.capabilityBattery.first?.correct == 19)
    }

    @Test func parsesServerValidationReport() throws {
        let json = """
            {"experiment":"pilot","concepts":{
               "fear":{"layer":14,"scenarioCount":24,"labeled":true,
                       "scenarioAccuracy":0.79},
               "calm":{"layer":12,"scenarioCount":10,"labeled":false,
                       "fractionAboveMidpoint":0.6,
                       "note":"validation.jsonl is unlabeled; add 'expresses' for true accuracy"}}}
            """
        let report = try #require(
            RunResults.validationReport(fromJSON: Data(json.utf8)))
        let fear = try #require(report.concepts.first { $0.concept == "fear" })
        #expect(fear.accuracy == 0.79)
        #expect(fear.scenarios == 24)
        let calm = try #require(report.concepts.first { $0.concept == "calm" })
        #expect(calm.accuracy == nil)
        #expect(calm.fractionAboveMidpoint == 0.6)
        #expect(calm.note?.contains("unlabeled") == true)
        #expect(report.logitLens.isEmpty)
    }

    /// Review 2026-08-01: the diagnostics existed only in the JSON — the
    /// app rendered transfer accuracy alone, the one number the `fair`
    /// incident proved can indict the threshold rather than the vector.
    @Test func decodesValidationDiagnosticsIntoTheRow() throws {
        let json = """
            {"concepts":{"fair":{"layer":31,"scenarioCount":40,"labeled":true,
               "scenarioAccuracy":0.5,
               "diagnostics":{"auc":0.855,"oneSidedPredictions":true,
                 "heldOutCalibration":{"threshold":5.5,"accuracy":0.85,
                   "balancedAccuracy":0.85,
                   "confusion":{"tp":17,"fp":3,"tn":17,"fn":3}}}}}}
            """
        let report = try #require(
            RunResults.validationReport(fromJSON: Data(json.utf8)))
        let fair = try #require(report.concepts.first)
        #expect(fair.accuracy == 0.5)
        #expect(fair.auc == 0.855)
        #expect(fair.calibratedAccuracy == 0.85)
        #expect(fair.oneSided == true)
        // The headline number prefers the calibration when present.
        #expect(fair.headlineAccuracy == 0.85)
        // Diagnostics-free reports keep nil fields (older engines).
        let legacy = try #require(RunResults.validationReport(
            fromJSON: Data(#"{"concepts":{"c":{"scenarioAccuracy":0.7}}}"#.utf8)))
        #expect(legacy.concepts.first?.auc == nil)
        #expect(legacy.concepts.first?.headlineAccuracy == 0.7)
    }

    @Test func multiDepthReportsDecodeOneRowPerDeclaredDepth() throws {
        // validate-at-the-sweep-layers (2026-08-01): the server writes
        // per-depth sub-entries under `depths` and NO flat mirror; the
        // table shows one row per concept×depth, layer-qualified ids.
        let json = """
            {"concepts":{"fair":{"scenarioCount":40,"labeled":true,
               "depths":[
                 {"layer":31,"scenarioAccuracy":0.5,
                  "diagnostics":{"auc":0.85,"oneSidedPredictions":true,
                    "heldOutCalibration":{"accuracy":0.85,
                      "balancedAccuracy":0.85}}},
                 {"layer":40,"scenarioAccuracy":0.78,
                  "diagnostics":{"auc":0.89,
                    "heldOutCalibration":{"accuracy":0.78,
                      "balancedAccuracy":0.78}}}]}},
             "cosineMatrixLayer":31,"cosineMatrixLayers":[31,40],
             "logitLens":{"fair":[
               {"layer":31,"topPositive":[{"token":"just"}],"topNegative":[]},
               {"layer":40,"topPositive":[{"token":"fair"}],"topNegative":[]}]}}
            """
        let report = try #require(
            RunResults.validationReport(fromJSON: Data(json.utf8)))
        #expect(report.concepts.count == 2)
        #expect(report.concepts.map(\.layer) == [31, 40])
        #expect(report.concepts.map(\.id) == ["fair@L31", "fair@L40"])
        // Shared scenario count reaches every depth row.
        #expect(report.concepts.map(\.scenarios) == [40, 40])
        #expect(report.concepts.map(\.auc) == [0.85, 0.89])
        #expect(report.concepts.first?.oneSided == true)
        #expect(report.cosineMatrixLayers == [31, 40])
        // The lens list decodes one row per depth as well.
        #expect(report.logitLens.map(\.layer) == [31, 40])
        #expect(report.logitLens.map(\.topPositive) == [["just"], ["fair"]])
    }

    @Test func validationReportRefusesStudyReports() {
        let studyReport = """
            {"experiment":"pilot","conditions":{"baseline":{"generations":2}}}
            """
        #expect(RunResults.validationReport(fromJSON: Data(studyReport.utf8)) == nil)
    }

    // MARK: - A6 excerpt line (generic JSONL preview path)

    @Test func choiceRecordsGetAReadableExcerptLine() throws {
        let parsed = RunBrowser.jsonlRecords(choiceLine)
        let record = try #require(parsed.records.first)
        #expect(record.fallback == nil)  // no longer raw JSON
        let summary = try #require(record.choiceSummary)
        #expect(summary.contains("answerTokenLogprob"))
        #expect(summary.contains("selected \"guilty\""))
        #expect(summary.contains("P(target) 0.770"))
        #expect(summary.contains("log-odds +1.210"))
        #expect(record.condition == "fear-0.8")
    }

    @Test func nonInstrumentRecordsHaveNoChoiceSummary() {
        #expect(RunBrowser.choiceSummaryLine(["condition": "baseline"]) == nil)
    }

    // MARK: - Model.load end to end (temp run directory)

    @Test func loadsARunDirectoryEndToEnd() throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "run-results-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let generations = [swiftGenerationLine, serverFailureLine, choiceLine]
            .joined(separator: "\n")
        try generations.write(
            to: directory.appending(component: "generations.jsonl"),
            atomically: true, encoding: .utf8)
        let report = """
            {"experiment":"pilot","conditions":{
               "baseline":{"generations":1,"meanWordCount":1.0,"meanDistinct2":0.0},
               "fear-0.8":{"generations":1,"choiceReadouts":1}}}
            """
        try report.write(
            to: directory.appending(component: "report.json"),
            atomically: true, encoding: .utf8)
        try """
            condition,promptID,samples,targetProbability
            baseline,case-1,1,
            fear-0.8,case-1,1,0.77
            """.write(
            to: directory.appending(component: "summaries.csv"),
            atomically: true, encoding: .utf8)

        let model = RunResults.load(runDirectory: directory)
        #expect(model.records.count == 3)
        #expect(model.isCategorical)
        #expect(model.categoricalConditions.contains("baseline"))
        #expect(!model.choiceMatrix.isEmpty)
        #expect(model.parseFailures.count == 1)
        #expect(model.summaries?.rows.count == 2)
        #expect(model.report?.conditions.count == 2)
        #expect(model.validationReport == nil)  // a study report, not validation
        // No manifest snapshot on disk: honest degradation, never an upgrade.
        #expect(model.classification.runClass == .draftPilot)
        #expect(model.classification.epoch == .unknown)
    }

    @Test func classifiesFromAManifestSnapshotOnDisk() throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "run-results-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        // Unique name so no live workspace experiment can match (epoch stays
        // honest-unknown in the test environment).
        var manifest = ExperimentManifest(
            name: "clsf-\(UUID().uuidString.prefix(8))",
            description: "fixture", modelID: "m")
        manifest.status = .frozen
        let encoder = JSONEncoder()
        try encoder.encode(manifest).write(
            to: directory.appending(component: "experiment.json"))
        try "aaa\n".write(
            to: directory.appending(component: "experiment-hash.txt"),
            atomically: true, encoding: .utf8)

        var record = RunResults.Record(condition: "baseline", promptID: "x")
        record.modelRevision = "resolvedrev123"
        let frozen = RunResults.classification(
            runDirectory: directory, records: [record])
        #expect(frozen.runClass == .frozenUnverified)
        #expect(frozen.epoch == .unknown)  // no live manifest to compare
        // Manifest revision unset + records resolved one → the P4 chip.
        #expect(frozen.unpinnedResolvedRevision == "resolvedrev123")

        // Forced freeze wins and reads non-citable.
        manifest.freezeForced = true
        try encoder.encode(manifest).write(
            to: directory.appending(component: "experiment.json"))
        let forced = RunResults.classification(
            runDirectory: directory, records: [])
        #expect(forced.runClass == .forcedFreeze)
    }

    @Test func remoteSemanticModelNeverClaimsAnUnverifiedFreezeIsEvidenceGrade() throws {
        var manifest = ExperimentManifest(
            name: "remote", description: "fixture", modelID: "m")
        manifest.status = .frozen
        let report = Data(
            #"{"conditions":{"fear-0.8":{"generations":0,"choiceReadouts":1}}}"#
                .utf8)
        let model = RunResults.remoteModel(
            runID: "remote-run",
            generationsText: choiceLine,
            generationsTruncated: false,
            reportData: report,
            snapshotData: try JSONEncoder().encode(manifest))
        #expect(model.isCategorical)
        #expect(model.choiceMatrix.items.count == 1)
        #expect(model.classification.runClass == .frozenUnverified)
        #expect(model.classification.epoch == .unknown)
    }

    @Test func loadsValidateRunArtifacts() throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "run-results-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try """
            {"experiment":"pilot",
             "validation":{"fear":{"scenarios":4,"layer":10,"accuracy":0.75}}}
            """.write(
            to: directory.appending(component: "validation-report.json"),
            atomically: true, encoding: .utf8)
        try """
            concept,layer,fear
            fear,10,1.0000
            """.write(
            to: directory.appending(component: "cosine-matrix.csv"),
            atomically: true, encoding: .utf8)

        let model = RunResults.load(runDirectory: directory)
        #expect(model.validationReport?.concepts.first?.accuracy == 0.75)
        #expect(model.cosineMatrix?.concepts == ["fear"])
        #expect(!model.isCategorical)
        #expect(!model.hasGenerations)
    }

    @Test func loadsAnalyzeRunArtifacts() throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "run-results-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try """
            condition,endpoint,n,deltaMean,ciLower,ciUpper,wilcoxonW,wilcoxonP,adjustedP,correction,modality
            fear-0.8,choiceLogOdds,12,0.42,0.1,0.7,5,0.01,0.04,bh,injection
            """.write(
            to: directory.appending(component: "effect-sizes.csv"),
            atomically: true, encoding: .utf8)
        try """
            condition,endpoint,deltaModel,ciModelLower,ciModelUpper,deltaHuman,ciHumanLower,ciHumanUpper,R,ciRLower,ciRUpper,region
            fear-0.8,choiceLogOdds,0.42,0.1,0.7,0.1,0.0,0.2,0.32,-0.1,0.7,alien
            """.write(
            to: directory.appending(component: "alien-residuals.csv"),
            atomically: true, encoding: .utf8)
        try """
            {"promoted":[],"rejected":[{"concept":"calm","promoted":false,
              "reasons":["no dose response"]}]}
            """.write(
            to: directory.appending(component: "promoted-movers.json"),
            atomically: true, encoding: .utf8)

        let model = RunResults.load(runDirectory: directory)
        #expect(model.hasAnalysisArtifacts)
        #expect(model.effectSizes?.count == 1)
        #expect(model.alienResiduals?.first?.region == "alien")
        #expect(model.promotedMovers?.rejected.first?.concept == "calm")
    }

    /// F2, acceptance 2's model half: a local greedy PROSE study (no
    /// option-bearing items) still carries every per-condition mean —
    /// including marker density — so the always-rendered condition table
    /// has its rows.
    @Test func nonCategoricalRunRetainsPerConditionMeansForTheConditionTable() throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "run-results-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        // Prose records: no options, no instrument, no parsedChoice.
        let generations = """
            {"condition":"baseline","promptID":"p1","output":"calm words flow","wordCount":3,"distinct2":1.0}
            {"condition":"fear-0.8","promptID":"p1","output":"dread dread dread","wordCount":3,"distinct2":0.5}
            """
        try generations.write(
            to: directory.appending(component: "generations.jsonl"),
            atomically: true, encoding: .utf8)
        try """
            {"experiment":"prose","promptCount":1,"conditionCount":2,
             "conditions":{
               "baseline":{"generations":1,"meanWordCount":3.0,
                           "meanDistinct2":1.0,
                           "meanMarkerDensity":{"fear":0.001}},
               "fear-0.8":{"generations":1,"meanWordCount":3.0,
                           "meanDistinct2":0.5,
                           "meanMarkerDensity":{"fear":0.042}}}}
            """.write(
            to: directory.appending(component: "report.json"),
            atomically: true, encoding: .utf8)

        let model = RunResults.load(runDirectory: directory)
        #expect(!model.isCategorical)  // the table must not depend on this
        let steered = try #require(model.report?.conditions["fear-0.8"])
        #expect(steered.meanWordCount == 3.0)
        #expect(steered.meanDistinct2 == 0.5)
        #expect(steered.meanMarkerDensity?["fear"] == 0.042)
        #expect(model.report?.conditions["baseline"]?.meanMarkerDensity?["fear"] == 0.001)
    }

    // MARK: - One derivation pipeline (F9): local and remote parity

    @Test func remoteModelMatchesLocalLoadFromTheSameBytes() throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "run-results-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let generations = [swiftGenerationLine, serverFailureLine, choiceLine]
            .joined(separator: "\n")
        // Report with inline effect sizes (the Swift dialect) and marker
        // means — the remote model must LIFT the inline effect sizes exactly
        // like the local load does, and carry panel effects when the run has
        // them (the historically drifted fields).
        let report = """
            {"experiment":"pilot","conditions":{
               "baseline":{"generations":1,"meanWordCount":1.0,"meanDistinct2":0.0,
                           "meanMarkerDensity":{"fear":0.02}},
               "fear-0.8":{"generations":1,"choiceReadouts":1,
                           "agreementWithBaseline":{"n":1,"agreement":0.0}}},
             "effectSizes":[
               {"condition":"fear-0.8","metric":"wordCount","n":10,
                "meanDiff":-2.5,"ciLower":-4.0,"ciUpper":-1.0,
                "wilcoxonW":3.0,"wilcoxonP":0.02}]}
            """
        let panelEffects = """
            endpoint,direct,directN,spillover,spilloverN,group,groupN,transmissionRatio,amplification,droppedTurns
            months,1.5,4,0.5,4,2.0,2,0.333,1.333,0
            """
        let summaries = """
            condition,promptID,samples,targetProbability
            baseline,case-1,1,
            fear-0.8,case-1,1,0.77
            """
        var manifest = ExperimentManifest(
            name: "parity-\(UUID().uuidString.prefix(8))",
            description: "fixture", modelID: "m")
        manifest.status = .frozen
        let snapshotData = try JSONEncoder().encode(manifest)

        try generations.write(
            to: directory.appending(component: "generations.jsonl"),
            atomically: true, encoding: .utf8)
        try report.write(
            to: directory.appending(component: "report.json"),
            atomically: true, encoding: .utf8)
        try panelEffects.write(
            to: directory.appending(component: "panel-effects.csv"),
            atomically: true, encoding: .utf8)
        try summaries.write(
            to: directory.appending(component: "summaries.csv"),
            atomically: true, encoding: .utf8)
        try snapshotData.write(
            to: directory.appending(component: "experiment.json"))

        let local = RunResults.load(runDirectory: directory)

        var artifacts = RunResults.ArtifactBytes()
        artifacts.assign(
            name: "generations.jsonl", data: Data(generations.utf8))
        artifacts.assign(name: "report.json", data: Data(report.utf8))
        artifacts.assign(
            name: "panel-effects.csv", data: Data(panelEffects.utf8))
        artifacts.assign(name: "summaries.csv", data: Data(summaries.utf8))
        artifacts.assign(name: "experiment.json", data: snapshotData)
        let remote = RunResults.remoteModel(runID: "r1", artifacts: artifacts)

        // Same bytes ⇒ identical semantics on every derived layer,
        // INCLUDING the historically drifted effect-size lift and panel
        // effects (only the runDirectory and the live-workspace epoch
        // context may differ).
        #expect(remote.records == local.records)
        #expect(remote.report == local.report)
        #expect(remote.summaries == local.summaries)
        #expect(remote.effectSizes == local.effectSizes)
        #expect(remote.effectSizes?.count == 1)  // lifted inline, both sides
        #expect(remote.panelEffects == local.panelEffects)
        #expect(remote.panelEffects?.count == 1)
        #expect(remote.choiceMatrix == local.choiceMatrix)
        #expect(remote.armSummaries == local.armSummaries)
        #expect(remote.parseFailures == local.parseFailures)
        #expect(remote.agreements == local.agreements)
        #expect(remote.completion == local.completion)
        #expect(remote.categoricalConditions == local.categoricalConditions)
        #expect(remote.hasManifestSnapshot == local.hasManifestSnapshot)
        #expect(remote.conditionInterventions == local.conditionInterventions)
        // Unique manifest name ⇒ no live manifest on either side: both
        // classify the frozen snapshot as epoch-unverified.
        #expect(remote.classification == local.classification)
        #expect(remote.classification.runClass == .frozenUnverified)
    }

    // MARK: - Unreadable manifest snapshot (F8): caution, never silence

    /// A snapshot whose `status` carries an unknown enum value (engine
    /// version skew): the strict decode fails, but the banner must render a
    /// caution state — bytes-present drives the header, the tolerant read
    /// drives the class, and `snapshotUnreadable` drives the warning line.
    @Test func unknownStatusEnumStillRendersACautionState() throws {
        let snapshot = Data(
            #"{"name":"skewed","status":"superFrozen","freezeForced":false}"#
                .utf8)
        let model = RunResults.remoteModel(
            runID: "r1",
            generationsText: choiceLine,
            generationsTruncated: false,
            reportData: nil,
            snapshotData: snapshot)
        #expect(model.hasManifestSnapshot)  // bytes present ⇒ header renders
        #expect(model.classification.snapshotUnreadable)
        #expect(model.classification.snapshotWarning != nil)
        // Unknown status cannot claim evidence grade.
        #expect(model.classification.runClass != .frozenEvidenceGrade)
    }

    @Test func unreadableSnapshotHonorsTolerantStatusAndForce() throws {
        // status "frozen" is legible even though the strict decode fails
        // (missing required fields) — classify frozen-unverified, flagged.
        let frozen = RunResults.remoteModel(
            runID: "r1", generationsText: nil, generationsTruncated: false,
            reportData: nil,
            snapshotData: Data(#"{"name":"x","status":"frozen"}"#.utf8))
        #expect(frozen.classification.runClass == .frozenUnverified)
        #expect(frozen.classification.snapshotUnreadable)

        // freezeForced survives the tolerant read: still non-citable.
        let forced = RunResults.remoteModel(
            runID: "r2", generationsText: nil, generationsTruncated: false,
            reportData: nil,
            snapshotData: Data(
                #"{"name":"x","status":"weird","freezeForced":true}"#.utf8))
        #expect(forced.classification.runClass == .forcedFreeze)
        #expect(forced.classification.snapshotUnreadable)
    }

    @Test func localLoadFlagsUnreadableSnapshotForParity() throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "run-results-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try #"{"name":"skewed-local","status":"superFrozen"}"#.write(
            to: directory.appending(component: "experiment.json"),
            atomically: true, encoding: .utf8)

        let model = RunResults.load(runDirectory: directory)
        #expect(model.hasManifestSnapshot)
        #expect(model.classification.snapshotUnreadable)
        #expect(model.classification.snapshotWarning != nil)
    }

    @Test func readableSnapshotsAreNotFlaggedUnreadable() throws {
        var manifest = ExperimentManifest(
            name: "ok-\(UUID().uuidString.prefix(8))",
            description: "fixture", modelID: "m")
        manifest.status = .frozen
        let model = RunResults.remoteModel(
            runID: "r1", generationsText: nil, generationsTruncated: false,
            reportData: nil,
            snapshotData: try JSONEncoder().encode(manifest))
        #expect(!model.classification.snapshotUnreadable)
        #expect(model.classification.snapshotWarning == nil)
    }

    // MARK: - Epoch chips defer to the frozen labels

    @Test func epochChipsSuppressedWhereTheLabelEncodesEpoch() {
        func classification(_ runClass: RunResults.RunClass) -> RunResults.Classification {
            RunResults.Classification(
                runClass: runClass, epoch: .unknown,
                unpinnedResolvedRevision: nil)
        }
        // Frozen classes bake the epoch state into the label — chips off,
        // so header and chip can never contradict.
        #expect(!classification(.frozenEvidenceGrade).showsEpochChips)
        #expect(!classification(.frozenEpochMismatch).showsEpochChips)
        #expect(!classification(.frozenUnverified).showsEpochChips)
        // Non-frozen classes carry no epoch in the label — chip is the
        // only signal.
        #expect(classification(.forcedFreeze).showsEpochChips)
        #expect(classification(.validatedDraft).showsEpochChips)
        #expect(classification(.draftPilot).showsEpochChips)
    }
}
