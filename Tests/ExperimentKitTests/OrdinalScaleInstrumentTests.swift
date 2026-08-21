import Foundation
import Testing

@testable import ExperimentKit

/// The general ordinal-scale outcome instrument (`ordinalScale`): pure
/// aggregation math, the declared-aggregation contract, record/report
/// fields, and the activation surface. The math fixtures are cross-engine
/// twins of `Server/tests/test_ordinal_scale.py` — SAME numbers asserted on
/// both engines. No model, no GPU.
@Suite(.serialized) struct OrdinalScaleInstrumentTests {

    private func isClose(_ a: Double, _ b: Double, tolerance: Double = 1e-9) -> Bool {
        abs(a - b) <= tolerance * max(1, max(abs(a), abs(b)))
    }

    func withTempRoot<T>(_ body: (URL) throws -> T) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "ordinal", body)
    }

    // MARK: - Pure aggregation math (server test_ordinal_scale.py twins)

    /// The canonical cross-engine fixture: ladder of 3, probs
    /// [0.2, 0.3, 0.5] → expected value 2.3, argmax position 3.
    @Test func expectedValueAndArgmaxOnCanonicalFixture() {
        let distribution = [0.2, 0.3, 0.5]
        #expect(isClose(
            LogprobInstrument.ordinalPosition(
                distribution: distribution, aggregation: .expectedValue),
            2.3))
        #expect(
            LogprobInstrument.ordinalPosition(
                distribution: distribution, aggregation: .argmax) == 3.0)
    }

    @Test func ordinalDistributionRenormalizes() {
        // [0.2, 0.2] renormalizes to [0.5, 0.5].
        let renormalized = LogprobInstrument.ordinalDistribution([0.2, 0.2])
        #expect(renormalized.count == 2)
        #expect(isClose(renormalized[0], 0.5))
        #expect(isClose(renormalized[1], 0.5))
        // Already-normalized input passes through unchanged.
        let unchanged = LogprobInstrument.ordinalDistribution([0.2, 0.3, 0.5])
        #expect(isClose(unchanged[0], 0.2))
        #expect(isClose(unchanged[1], 0.3))
        #expect(isClose(unchanged[2], 0.5))
        // Negative inputs clamp to 0 before renormalizing.
        let clamped = LogprobInstrument.ordinalDistribution([-1.0, 1.0, 1.0])
        #expect(isClose(clamped[0], 0.0))
        #expect(isClose(clamped[1], 0.5))
        #expect(isClose(clamped[2], 0.5))
        // Degenerate all-zero total → uniform; empty in → empty out.
        let uniform = LogprobInstrument.ordinalDistribution([0, 0, 0, 0])
        #expect(uniform == [0.25, 0.25, 0.25, 0.25])
        #expect(LogprobInstrument.ordinalDistribution([]).isEmpty)
    }

    /// Tie rule pinned cross-engine: FIRST max wins (Python `max()`
    /// semantics on the server).
    @Test func argmaxTieBreaksToFirstMaximum() {
        #expect(
            LogprobInstrument.ordinalPosition(
                distribution: [0.5, 0.5], aggregation: .argmax) == 1.0)
        #expect(
            LogprobInstrument.ordinalPosition(
                distribution: [0.2, 0.4, 0.4], aggregation: .argmax) == 2.0)
    }

    @Test func emptyDistributionDegeneratesToZero() {
        #expect(
            LogprobInstrument.ordinalPosition(
                distribution: [], aggregation: .expectedValue) == 0)
        #expect(
            LogprobInstrument.ordinalPosition(
                distribution: [], aggregation: .argmax) == 0)
    }

    /// `orderedProbabilities` preserves DECLARED option order and matches
    /// the dictionary softmax value-for-value (ladder positions are
    /// positional, so order is load-bearing).
    @Test func orderedProbabilitiesFollowLadderOrder() throws {
        let result = ChoiceResult(options: [
            OptionScore(option: "1", tokenIDs: [1], tokenLogprobs: [Float(log(0.2))]),
            OptionScore(option: "2", tokenIDs: [2], tokenLogprobs: [Float(log(0.3))]),
            OptionScore(option: "3", tokenIDs: [3], tokenLogprobs: [Float(log(0.5))]),
        ])
        let ordered = result.orderedProbabilities
        #expect(ordered.count == 3)
        #expect(isClose(ordered.reduce(0, +), 1.0))
        #expect(isClose(ordered[0], 0.2, tolerance: 1e-6))
        #expect(isClose(ordered[1], 0.3, tolerance: 1e-6))
        #expect(isClose(ordered[2], 0.5, tolerance: 1e-6))
        for (index, option) in result.optionNames.enumerated() {
            let byName = try #require(result.probability[option])
            #expect(isClose(ordered[index], byName))
        }
    }

    // MARK: - Vocabulary (cross-engine list parity)

    @Test func vocabulariesMatchTheServerLiterals() {
        // Twin of the server's KNOWN_OUTCOME_INSTRUMENTS /
        // KNOWN_ORDINAL_AGGREGATIONS assertions — same literals, verbatim.
        #expect(
            ExperimentStore.knownOutcomeInstruments == [
                "sampledText", "answerTokenLogprob", "choiceProbability",
                "repeReaderScore", "ordinalScale",
            ])
        #expect(ExperimentStore.knownOrdinalAggregations == ["expectedValue", "argmax"])
        #expect(
            ExperimentTasks.choiceInstruments
                == ["answerTokenLogprob", "choiceProbability", "ordinalScale"])
        #expect(
            OrdinalAggregation.allCases.map(\.rawValue) == ["expectedValue", "argmax"])
    }

    // MARK: - Store setters

    @Test func outcomeInstrumentsVocabularyAcceptsOrdinalScale() throws {
        try withTempRoot { _ in
            try ExperimentStore.create(
                name: "ord-a", description: "d", modelID: "test/model")
            try ExperimentStore.setOutcomeInstruments(
                ["ordinalScale"], experimentName: "ord-a")
            #expect(
                try ExperimentStore.load(name: "ord-a").outcomeInstruments
                    == ["ordinalScale"])
        }
    }

    @Test func ordinalAggregationSetterRoundTripsClearsAndRefusesUnknown() throws {
        try withTempRoot { _ in
            try ExperimentStore.create(
                name: "ord-b", description: "d", modelID: "test/model")
            try ExperimentStore.setOrdinalAggregation(
                "expectedValue", experimentName: "ord-b")
            #expect(
                try ExperimentStore.load(name: "ord-b").ordinalAggregation
                    == "expectedValue")
            try ExperimentStore.setOrdinalAggregation("argmax", experimentName: "ord-b")
            #expect(
                try ExperimentStore.load(name: "ord-b").ordinalAggregation == "argmax")
            // nil / empty clears to ABSENT (content-hash hygiene).
            try ExperimentStore.setOrdinalAggregation(nil, experimentName: "ord-b")
            #expect(try ExperimentStore.load(name: "ord-b").ordinalAggregation == nil)
            try ExperimentStore.setOrdinalAggregation(
                "expectedValue", experimentName: "ord-b")
            try ExperimentStore.setOrdinalAggregation("", experimentName: "ord-b")
            #expect(try ExperimentStore.load(name: "ord-b").ordinalAggregation == nil)
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.setOrdinalAggregation(
                    "median", experimentName: "ord-b")
            }
        }
    }

    // MARK: - Verify gates (declared, never defaulted)

    private func manifest(
        instruments: [String]?, aggregation: String?, thinking: Bool = false
    ) -> ExperimentManifest {
        var manifest = ExperimentManifest(
            name: "ord-verify", description: "", modelID: "test/model")
        manifest.outcomeInstruments = instruments
        manifest.ordinalAggregation = aggregation
        manifest.qwenThinkingEnabled = thinking
        return manifest
    }

    @Test func verifyRefusesOrdinalScaleWithoutAggregation() {
        let violations = ExperimentStore.verify(
            manifest(instruments: ["ordinalScale"], aggregation: nil))
        #expect(violations.contains {
            $0.contains("ordinalScale but ordinalAggregation")
        })
    }

    @Test func verifyRefusesUnknownAggregation() {
        let violations = ExperimentStore.verify(
            manifest(instruments: ["ordinalScale"], aggregation: "median"))
        #expect(violations.contains {
            $0.contains("unknown ordinalAggregation 'median'")
        })
    }

    @Test func verifyAcceptsDeclaredAggregations() {
        for aggregation in ["expectedValue", "argmax"] {
            let violations = ExperimentStore.verify(
                manifest(instruments: ["ordinalScale"], aggregation: aggregation))
            #expect(!violations.contains { $0.contains("ordinalAggregation") })
        }
        // No ordinalScale, no aggregation — nothing to declare.
        let none = ExperimentStore.verify(
            manifest(instruments: ["answerTokenLogprob"], aggregation: nil))
        #expect(!none.contains { $0.contains("ordinalAggregation") })
    }

    @Test func thinkingModeGateCoversOrdinalScale() {
        let violations = ExperimentStore.verify(
            manifest(
                instruments: ["ordinalScale"], aggregation: "expectedValue",
                thinking: true))
        #expect(violations.contains {
            $0.contains("thinking-mode answers are marginals")
        })
        // Off is fine.
        let off = ExperimentStore.verify(
            manifest(instruments: ["ordinalScale"], aggregation: "expectedValue"))
        #expect(!off.contains { $0.contains("marginals") })
    }

    // MARK: - Record fields (cross-engine keys)

    private func canonicalChoice() -> ChoiceResult {
        ChoiceResult(options: [
            OptionScore(option: "1", tokenIDs: [1], tokenLogprobs: [Float(log(0.2))]),
            OptionScore(option: "2", tokenIDs: [2], tokenLogprobs: [Float(log(0.3))]),
            OptionScore(option: "3", tokenIDs: [3], tokenLogprobs: [Float(log(0.5))]),
        ])
    }

    private func record(
        instruments: [String]?, aggregation: String?
    ) -> ExperimentTasks.ChoiceRecord {
        var manifest = ExperimentManifest(
            name: "ord-rec", description: "", modelID: "test/model")
        manifest.outcomeInstruments = instruments
        manifest.ordinalAggregation = aggregation
        return ExperimentTasks.choiceRecord(
            manifest: manifest, experimentHash: "hh",
            taskPromptsFile: "f.jsonl", taskPromptsHash: "ph",
            promptMode: .chatAssistant, systemPrompt: nil,
            qwenThinkingEnabled: false, condition: "baseline",
            promptIndex: 1,
            prompt: ExperimentTasks.StudyPrompt(
                id: "p1", text: "rate it", options: ["1", "2", "3"],
                target: nil, anchorMonths: nil, severity: nil, arm: nil,
                caseID: nil),
            choice: canonicalChoice())
    }

    @Test func choiceRecordStampsOrdinalFieldsWhenDeclared() throws {
        let stamped = record(
            instruments: ["ordinalScale"], aggregation: "expectedValue")
        let position = try #require(stamped.ordinalPosition)
        #expect(isClose(position, 2.3, tolerance: 1e-6))
        let distribution = try #require(stamped.ordinalDistribution)
        #expect(distribution.count == 3)
        #expect(isClose(distribution[2], 0.5, tolerance: 1e-6))
        // Open-issues #6: a ladder item declares no target, and the record
        // says so. It used to read `target: "1"` — the scale minimum,
        // synthesized from `options[0]` — which analyze then reported as a
        // declared choiceLogOdds endpoint.
        #expect(stamped.target == nil)
        #expect(stamped.targetSource == nil)

        let argmax = record(instruments: ["ordinalScale"], aggregation: "argmax")
        #expect(try #require(argmax.ordinalPosition) == 3.0)

        // Encoded keys are the cross-engine contract, verbatim.
        let object = try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(stamped))
        let keys = Set(try #require(object as? [String: Any]).keys)
        #expect(keys.contains("ordinalPosition"))
        #expect(keys.contains("ordinalDistribution"))
    }

    @Test func choiceRecordOmitsOrdinalFieldsWhenNotDeclared() throws {
        let plain = record(instruments: ["answerTokenLogprob"], aggregation: nil)
        #expect(plain.ordinalPosition == nil)
        #expect(plain.ordinalDistribution == nil)
        let object = try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(plain))
        let keys = Set(try #require(object as? [String: Any]).keys)
        #expect(!keys.contains("ordinalPosition"))
        #expect(!keys.contains("ordinalDistribution"))
    }

    // MARK: - Report summary (ordinalMean/ordinalSD)

    @Test func reportStampsPerConditionOrdinalMeanAndSD() throws {
        let manifest = ExperimentManifest(
            name: "ord-report", description: "", modelID: "test/model")
        // Two instrument readouts for the steered condition: positions 2.3
        // and 1.7 → mean 2.0, population SD 0.3 (server twin numbers).
        let readouts = [
            ExperimentTasks.ReportChoiceReadout(
                condition: "steered", promptID: "p1", sampleIndex: nil,
                source: "instrument", selected: "3", target: "3",
                ordinalPosition: 2.3),
            ExperimentTasks.ReportChoiceReadout(
                condition: "steered", promptID: "p2", sampleIndex: nil,
                source: "instrument", selected: "1", target: "1",
                ordinalPosition: 1.7),
            ExperimentTasks.ReportChoiceReadout(
                condition: "baseline", promptID: "p1", sampleIndex: nil,
                source: "instrument", selected: "2", target: "2",
                ordinalPosition: 2.0),
        ]
        let report = ExperimentTasks.report(
            experiment: manifest, experimentHash: "hh",
            taskPrompts: ("f.jsonl", "ph", []), rows: [], conditionCount: 2,
            concepts: [], choiceReadouts: readouts)
        let steered = try #require(report.conditions["steered"])
        #expect(isClose(try #require(steered.ordinalMean), 2.0))
        #expect(isClose(try #require(steered.ordinalSD), 0.3))
        let baseline = try #require(report.conditions["baseline"])
        #expect(isClose(try #require(baseline.ordinalMean), 2.0))
        // A single readout has a defined population SD of 0.
        #expect(try #require(baseline.ordinalSD) == 0.0)

        // Contract keys in the encoded report.json, verbatim.
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let json = String(decoding: try encoder.encode(report), as: UTF8.self)
        #expect(json.contains("\"ordinalMean\""))
        #expect(json.contains("\"ordinalSD\""))
    }

    @Test func reportOmitsOrdinalSummaryWithoutOrdinalReadouts() throws {
        let manifest = ExperimentManifest(
            name: "ord-none", description: "", modelID: "test/model")
        let readouts = [
            ExperimentTasks.ReportChoiceReadout(
                condition: "baseline", promptID: "p1", sampleIndex: nil,
                source: "instrument", selected: "A", target: "A"),
        ]
        let report = ExperimentTasks.report(
            experiment: manifest, experimentHash: "hh",
            taskPrompts: ("f.jsonl", "ph", []), rows: [], conditionCount: 1,
            concepts: [], choiceReadouts: readouts)
        let baseline = try #require(report.conditions["baseline"])
        #expect(baseline.ordinalMean == nil)
        #expect(baseline.ordinalSD == nil)
    }

    // MARK: - Paired effect sizes (the analyze/report layer)

    private func ordinalReadout(
        _ condition: String, _ promptID: String, _ position: Double?,
        source: String = "instrument"
    ) -> ExperimentTasks.ReportChoiceReadout {
        ExperimentTasks.ReportChoiceReadout(
            condition: condition, promptID: promptID, sampleIndex: nil,
            source: source, selected: "", target: nil,
            ordinalPosition: position)
    }

    /// ordinalPosition is one more per-item numeric metric through the SAME
    /// paired-to-baseline machinery (bootstrap CI + Wilcoxon) — and the
    /// metric name is the pinned cross-engine contract (server twin pins
    /// "ordinalPosition" in test_ordinal_scale.py).
    @Test func ordinalPositionsJoinThePairedEffectMachinery() throws {
        let readouts = [
            ordinalReadout("baseline", "p1", 2.0),
            ordinalReadout("baseline", "p2", 1.0),
            ordinalReadout("steered", "p1", 2.3),
            ordinalReadout("steered", "p2", 1.7),
        ]
        let entries = ExperimentTasks.effectSizes(
            rows: [], concepts: [], choiceReadouts: readouts, replicates: 500)
        #expect(entries.count == 1)
        let entry = try #require(entries.first)
        #expect(entry.metric == "ordinalPosition")
        #expect(entry.condition == "steered")
        #expect(entry.n == 2)
        // Diffs are +0.3 and +0.7 → paired mean +0.5; the bootstrap CI stays
        // inside the diff range and Wilcoxon is defined (all non-zero).
        #expect(isClose(entry.meanDiff, 0.5, tolerance: 1e-9))
        #expect(entry.ciLower >= 0.3 - 1e-9)
        #expect(entry.ciUpper <= 0.7 + 1e-9)
        #expect(entry.wilcoxonW != nil)
        #expect(entry.wilcoxonP != nil)
    }

    @Test func ordinalEffectsNeedAnInstrumentBaselineAndPairing() {
        // No baseline readouts → no entries.
        #expect(
            ExperimentTasks.effectSizes(
                rows: [], concepts: [],
                choiceReadouts: [ordinalReadout("steered", "p1", 2.3)],
                replicates: 100
            ).isEmpty)
        // Unpaired items → no entries.
        #expect(
            ExperimentTasks.effectSizes(
                rows: [], concepts: [],
                choiceReadouts: [
                    ordinalReadout("baseline", "p1", 2.0),
                    ordinalReadout("steered", "p2", 2.3),
                ],
                replicates: 100
            ).isEmpty)
        // Parsed-source and position-less readouts never join (the ordinal
        // metric is the deterministic instrument's, not sampled parses).
        #expect(
            ExperimentTasks.effectSizes(
                rows: [], concepts: [],
                choiceReadouts: [
                    ordinalReadout("baseline", "p1", 2.0, source: "parsed"),
                    ordinalReadout("steered", "p1", 2.3, source: "parsed"),
                    ordinalReadout("baseline", "p2", nil),
                    ordinalReadout("steered", "p2", nil),
                ],
                replicates: 100
            ).isEmpty)
    }

    /// The run-path report stamps the ordinal effect entry inline (report.json
    /// `effectSizes`), so an ordinalScale run needs no separate analyze pass
    /// to show its paired effect.
    @Test func reportEffectSizesCarryTheOrdinalEntry() throws {
        let manifest = ExperimentManifest(
            name: "ord-effects", description: "", modelID: "test/model")
        let readouts = [
            ordinalReadout("baseline", "p1", 2.0),
            ordinalReadout("baseline", "p2", 1.0),
            ordinalReadout("steered", "p1", 2.3),
            ordinalReadout("steered", "p2", 1.7),
        ]
        let report = ExperimentTasks.report(
            experiment: manifest, experimentHash: "hh",
            taskPrompts: ("f.jsonl", "ph", []), rows: [], conditionCount: 2,
            concepts: [], choiceReadouts: readouts)
        let entry = try #require(
            report.effectSizes?.first {
                $0.condition == "steered" && $0.metric == "ordinalPosition"
            })
        #expect(entry.n == 2)
        #expect(isClose(entry.meanDiff, 0.5, tolerance: 1e-9))
    }

    // MARK: - RunResults decode (report.json summary + record fields)

    @Test func runResultsDecodesOrdinalConditionSummary() throws {
        let json = """
            {"experiment": "ord", "conditions": {
                "steered": {"generations": 0, "choiceReadouts": 2,
                            "ordinalMean": 2.0, "ordinalSD": 0.3},
                "baseline": {"generations": 0, "choiceReadouts": 2}},
             "effectSizes": [{"condition": "steered",
                              "metric": "ordinalPosition", "n": 2,
                              "meanDiff": 0.5, "ciLower": 0.3,
                              "ciUpper": 0.7}]}
            """
        let report = try #require(RunResults.report(fromJSON: Data(json.utf8)))
        let steered = try #require(report.conditions["steered"])
        #expect(steered.ordinalMean == 2.0)
        #expect(steered.ordinalSD == 0.3)
        // Absence decodes as absence, never invented.
        let baseline = try #require(report.conditions["baseline"])
        #expect(baseline.ordinalMean == nil)
        #expect(baseline.ordinalSD == nil)
        // The inline effect entry surfaces as a generic effect row — the
        // effect views render it with no ordinal-specific wiring.
        let row = try #require(report.effectSizes.first)
        #expect(row.metric == "ordinalPosition")
        #expect(row.meanDiff == 0.5)
        #expect(row.ciExcludesZero)
    }

    @Test func runResultsDecodesOrdinalRecordFields() {
        let line = """
            {"condition": "steered", "promptID": "p1", \
            "instrument": "answerTokenLogprob", "selected": "3", \
            "ordinalPosition": 2.3, "ordinalDistribution": [0.2, 0.3, 0.5]}
            """
        let decoded = RunResults.records(fromJSONL: line)
        #expect(decoded.skippedLines == 0)
        #expect(decoded.records.first?.ordinalPosition == 2.3)
        #expect(decoded.records.first?.ordinalDistribution == [0.2, 0.3, 0.5])
        // Non-ordinal records stay nil.
        let plain = RunResults.records(
            fromJSONL: #"{"condition": "baseline", "promptID": "p1"}"#)
        #expect(plain.records.first?.ordinalPosition == nil)
        #expect(plain.records.first?.ordinalDistribution == nil)
    }

    // MARK: - Manifest round-trip

    @Test func ordinalAggregationRoundTripsAndOmitsNil() throws {
        var manifest = ExperimentManifest(
            name: "ord-code", description: "", modelID: "test/model")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        // Absent stays absent (legacy manifests keep their content hash).
        let absent = String(decoding: try encoder.encode(manifest), as: UTF8.self)
        #expect(!absent.contains("ordinalAggregation"))
        manifest.ordinalAggregation = "argmax"
        let data = try encoder.encode(manifest)
        #expect(String(decoding: data, as: UTF8.self)
            .contains("\"ordinalAggregation\":\"argmax\""))
        let decoded = try JSONDecoder().decode(ExperimentManifest.self, from: data)
        #expect(decoded.ordinalAggregation == "argmax")
    }

    // MARK: - Activation surface (auxiliary-instrument pattern)

    @Test func ordinalScaleIsAnOptionConsumingAuxiliary() {
        // Fed by options (no "declare an instrument" warning once declared)…
        #expect(InstrumentActivation.categoricalDeclared(["ordinalScale"]))
        #expect(
            InstrumentActivation.activationWarning(
                optionsItemCount: 3, instruments: ["ordinalScale"]) == nil)
        // …but NOT owned by the Outcome Mode picker: it reads back as its
        // own auxiliary row and survives mode edits.
        #expect(!InstrumentActivation.pickerOwnedInstruments.contains("ordinalScale"))
        #expect(
            InstrumentActivation.OutcomeMode.from(["ordinalScale"]) == .notDeclared)
        #expect(
            InstrumentActivation.auxiliaryInstruments(
                of: ["sampledText", "ordinalScale"]) == ["ordinalScale"])
        let edited = InstrumentActivation.applying(
            .answerTokenProbability, to: ["ordinalScale", "sampledText"])
        #expect(edited?.contains("ordinalScale") == true)
        #expect(edited?.contains("sampledText") == false)
        // Deterministic logprob readout — no sampled generation implied.
        #expect(!InstrumentActivation.auxiliaryImpliesSampledGeneration("ordinalScale"))
        #expect(
            InstrumentActivation.auxiliaryDescription("ordinalScale")
                .contains("ordered options"))
        // Save & Pin's enabled line names it.
        #expect(
            InstrumentActivation.savePinSummary(
                optionsItemCount: 2, itemCount: 2,
                instruments: ["ordinalScale"]
            ).contains("ordinalScale"))
    }
}
