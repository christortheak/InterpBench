import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// Declared record-exclusion rules (manifest `exclusionRules` + per-item
/// `attentionCheck`): closed vocabulary, plain-language violations, battery
/// grading reuse, analysis-time application with pairwise deletion, and the
/// cross-engine report stamp. Message strings, descriptions, the stamp note,
/// and the stamp fixture are the cross-engine contract — pinned VALUE-FOR-
/// VALUE against `Server/tests/test_exclusions.py`.
@Suite(.serialized) struct ExclusionRulesTests {

    func withTempRoot<T>(_ body: (URL) throws -> T) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "exclusion") { root in
            try body(root)
        }
    }

    private func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - Rule vocabulary + violation messages (cross-engine strings)

    @Test func ruleVocabularyIsClosedAndOrdered() {
        #expect(
            ExclusionEngine.ruleVocabulary == [
                "failedAttentionCheck", "unparseableEndpoint", "outOfRange",
            ])
        #expect(ExclusionEngine.defaultEndpoint == "parsedMonths")
    }

    @Test func violationMessagesAreThePinnedCrossEngineStrings() {
        // Each fixture is (rules, expected message) — the message literals
        // are pinned identically in Server/tests/test_exclusions.py.
        #expect(
            ExclusionEngine.violations([ExclusionRule(rule: "outliers")]) == [
                "exclusion rule 'outliers' is not recognized — declared rules "
                    + "must be one of: failedAttentionCheck, unparseableEndpoint, "
                    + "outOfRange"
            ])
        #expect(
            ExclusionEngine.violations([
                ExclusionRule(rule: "unparseableEndpoint"),
                ExclusionRule(rule: "unparseableEndpoint"),
            ]) == [
                "exclusion rule 'unparseableEndpoint' is declared more than "
                    + "once — declare each rule at most once"
            ])
        #expect(
            ExclusionEngine.violations([ExclusionRule(rule: "outOfRange")]) == [
                "exclusion rule outOfRange declares no bounds — declare 'min', "
                    + "'max', or both"
            ])
        #expect(
            ExclusionEngine.violations([
                ExclusionRule(rule: "outOfRange", min: 10, max: 2)
            ]) == [
                "exclusion rule outOfRange has min (10) greater than max (2) — "
                    + "the rule keeps records with min <= value <= max"
            ])
        #expect(
            ExclusionEngine.violations([
                ExclusionRule(rule: "unparseableEndpoint", min: 1)
            ]) == [
                "exclusion rule 'unparseableEndpoint' does not take 'min'/'max' "
                    + "— a numeric range applies only to outOfRange"
            ])
        #expect(
            ExclusionEngine.violations([
                ExclusionRule(rule: "failedAttentionCheck", endpoint: "parsedMonths")
            ]) == [
                "exclusion rule failedAttentionCheck does not take 'endpoint' — "
                    + "it grades each record's output against its item's declared "
                    + "attentionCheck"
            ])
        #expect(
            ExclusionEngine.violations([
                ExclusionRule(rule: "outOfRange", endpoint: "  ", max: 600)
            ]) == [
                "exclusion rule 'outOfRange' declares an empty 'endpoint' — "
                    + "omit the key for the default (parsedMonths) or name the "
                    + "record's parsed-value key"
            ])
        // A well-formed declaration is silent; absent = no rules at all.
        #expect(
            ExclusionEngine.violations([
                ExclusionRule(rule: "failedAttentionCheck"),
                ExclusionRule(rule: "unparseableEndpoint"),
                ExclusionRule(rule: "outOfRange", min: 0, max: 600),
            ]).isEmpty)
        #expect(ExclusionEngine.violations(nil).isEmpty)
    }

    // MARK: - Ladder-window advisory (2026-08-06)

    @Test func ladderRangeReadsFullyNumericLaddersOnly() {
        #expect(ExclusionEngine.ladderRange(optionLadders: []) == nil)
        // A partially numeric ladder implies nothing.
        #expect(
            ExclusionEngine.ladderRange(optionLadders: [["1", "2", "high"]])
                == nil)
        // Non-numeric ladders contribute nothing; numeric ones span min..max.
        #expect(
            ExclusionEngine.ladderRange(
                optionLadders: [["1", "2", "7"], ["Strongly disagree", "Agree"]])
                == 1.0...7.0)
    }

    @Test func ladderWindowWarningIsThePinnedString() {
        // The field case: min 0 / max 100 declared on a 1–7 scale endpoint —
        // syntactically valid, semantically inert. Message string pinned
        // identically in Server/tests/test_exclusions.py
        // (test_out_of_range_window_outside_the_ladder_warns).
        let warnings = ExclusionEngine.ladderWarnings(
            rules: [ExclusionRule(rule: "outOfRange", min: 0, max: 100)],
            optionLadders: [(1...7).map(String.init)])
        #expect(
            warnings == [
                "exclusion rule outOfRange declares min 0 and max 100, but "
                    + "the task items' options ladder spans 1 to 7 — every "
                    + "declared bound lies outside the ladder, so if the "
                    + "endpoint takes ladder values the rule can never bind "
                    + "(it would exclude nothing or everything); align the "
                    + "bounds with the scale the items use, or drop the rule"
            ])
    }

    @Test func aBoundInsideTheLadderIsNotWarned() {
        let ladders = [(1...7).map(String.init)]
        // max 5 can bind (values 6, 7 excluded) — a real window, no warning.
        #expect(
            ExclusionEngine.ladderWarnings(
                rules: [ExclusionRule(rule: "outOfRange", min: 0, max: 5)],
                optionLadders: ladders).isEmpty)
        #expect(
            ExclusionEngine.ladderWarnings(
                rules: [ExclusionRule(rule: "outOfRange", min: 2)],
                optionLadders: ladders).isEmpty)
    }

    @Test func aSingleOutsideBoundWarns() {
        let warnings = ExclusionEngine.ladderWarnings(
            rules: [ExclusionRule(rule: "outOfRange", min: 0)],
            optionLadders: [["1", "7"]])
        #expect(warnings.count == 1)
        #expect(warnings.first?.contains("declares min 0, but") == true)
    }

    @Test func nonNumericLaddersAndOtherRulesNeverWarn() {
        #expect(
            ExclusionEngine.ladderWarnings(
                rules: [ExclusionRule(rule: "outOfRange", min: 0, max: 100)],
                optionLadders: [["A", "B"]]).isEmpty)
        #expect(
            ExclusionEngine.ladderWarnings(
                rules: [ExclusionRule(rule: "unparseableEndpoint")],
                optionLadders: [["1", "7"]]).isEmpty)
        #expect(
            ExclusionEngine.ladderWarnings(rules: nil, optionLadders: [["1", "7"]])
                .isEmpty)
    }

    @Test func boundFormattingMatchesTheServer() {
        // Cross-engine _fmt fixtures (Python: int-collapse, else repr).
        #expect(ExclusionEngine.formatBound(0) == "0")
        #expect(ExclusionEngine.formatBound(600) == "600")
        #expect(ExclusionEngine.formatBound(0.5) == "0.5")
        #expect(ExclusionEngine.formatBound(-1.25) == "-1.25")
    }

    @Test func descriptionsAreThePinnedCrossEngineStrings() {
        #expect(
            ExclusionEngine.description(of: ExclusionRule(rule: "failedAttentionCheck"))
                == "the record's output failed its item's declared attention check")
        #expect(
            ExclusionEngine.description(of: ExclusionRule(rule: "unparseableEndpoint"))
                == "no parseable parsedMonths value (the endpoint parser produced null)")
        #expect(
            ExclusionEngine.description(
                of: ExclusionRule(rule: "outOfRange", min: 0, max: 600))
                == "parsed parsedMonths outside the declared range [0, 600]")
        #expect(
            ExclusionEngine.description(of: ExclusionRule(rule: "outOfRange", min: 6))
                == "parsed parsedMonths below the declared minimum 6")
        #expect(
            ExclusionEngine.description(of: ExclusionRule(rule: "outOfRange", max: 600))
                == "parsed parsedMonths above the declared maximum 600")
    }

    // MARK: - Manifest round-trip + hash stability

    @Test func legacyManifestBytesAndHashAreUntouched() throws {
        let plain = ExperimentManifest(
            name: "legacy", description: "", modelID: "test/model")
        let encoded = String(
            decoding: try JSONEncoder().encode(plain), as: UTF8.self)
        // Omit-when-nil: a manifest that never declared rules never carries
        // the key, so pre-existing manifests keep their content hash.
        #expect(!encoded.contains("exclusionRules"))

        var declared = plain
        declared.exclusionRules = [ExclusionRule(rule: "unparseableEndpoint")]
        #expect(
            ExperimentStore.manifestHash(plain)
                != ExperimentStore.manifestHash(declared))
        // Same declaration = same hash (the rules are ordinary hashed data).
        var again = plain
        again.exclusionRules = [ExclusionRule(rule: "unparseableEndpoint")]
        #expect(
            ExperimentStore.manifestHash(again)
                == ExperimentStore.manifestHash(declared))
    }

    @Test func rulesRoundTripThroughManifestCoding() throws {
        var manifest = ExperimentManifest(
            name: "rt", description: "", modelID: "test/model")
        manifest.exclusionRules = [
            ExclusionRule(rule: "failedAttentionCheck"),
            ExclusionRule(rule: "outOfRange", endpoint: "parsedMonths", min: 0, max: 600),
        ]
        let decoded = try JSONDecoder().decode(
            ExperimentManifest.self, from: JSONEncoder().encode(manifest))
        #expect(decoded.exclusionRules == manifest.exclusionRules)

        // A legacy manifest JSON without the key decodes to nil.
        let legacy = try JSONDecoder().decode(
            ExperimentManifest.self,
            from: Data("""
                {"name": "old", "experimentDescription": "", "modelID": "m", \
                "createdAt": "2026-01-01T00:00:00Z", "status": "draft", \
                "concepts": [], "conditions": [], "seeds": [0], \
                "temperature": 0, "maxTokens": 128}
                """.utf8))
        #expect(legacy.exclusionRules == nil)
    }

    // MARK: - Per-item attention checks (battery grading vocabulary reuse)

    @Test func parserCarriesAttentionChecksAndValidatesThem() throws {
        let data = Data("""
            {"id": "p1", "prompt": "Decide the case."}
            {"id": "ac-1", "prompt": "End with the number 7.", "attentionCheck": {"expected": "7", "grading": "exact_number"}}
            {"id": "ac-2", "prompt": "Say yes.", "attentionCheck": {"expected": "yes"}}
            """.utf8)
        let prompts = try ExperimentTasks.parseTaskPrompts(data)
        #expect(prompts[0].attentionCheck == nil)
        #expect(prompts[1].attentionCheck?.expected == "7")
        #expect(prompts[1].attentionCheck?.grading == .exactNumber)
        // Absent grading = the battery's inferred-mode rule at grade time.
        #expect(prompts[2].attentionCheck?.expected == "yes")
        #expect(prompts[2].attentionCheck?.grading == nil)

        // Cross-engine messages: unknown grading mode, missing expected.
        #expect {
            _ = try ExperimentTasks.parseTaskPrompts(
                Data("""
                    {"id": "bad", "prompt": "x", "attentionCheck": {"expected": "7", "grading": "vibes"}}
                    """.utf8))
        } throws: { error in
            (error as? ExperimentError)?.reason
                == "task prompts: item 'bad' attentionCheck grading 'vibes' is "
                + "not a known grading mode — one of: exact_number, yes_no, "
                + "token_exact, exact_normalized, regex"
        }
        #expect {
            _ = try ExperimentTasks.parseTaskPrompts(
                Data("""
                    {"id": "bad2", "prompt": "x", "attentionCheck": {"expected": "  "}}
                    """.utf8))
        } throws: { error in
            (error as? ExperimentError)?.reason
                == "task prompts: item 'bad2' declares an attentionCheck "
                + "without a non-empty 'expected' string — declare the "
                + "expected answer"
        }
    }

    private func view(
        condition: String, promptID: String, output: String = "text",
        seed: UInt64 = 1, months: Double?? = nil
    ) -> ExclusionEngine.RecordView {
        var endpoints: [String: Double?] = [:]
        if case .some(let value) = months {
            endpoints.updateValue(value, forKey: "parsedMonths")
        }
        return ExclusionEngine.RecordView(
            condition: condition, seed: seed, promptID: promptID,
            output: output, endpoints: endpoints)
    }

    @Test func attentionCheckGradingReusesTheBatteryGrader() {
        let rules = [ExclusionRule(rule: "failedAttentionCheck")]
        let checks: [String: AttentionCheck] = [
            "ac-num": AttentionCheck(expected: "7", grading: .exactNumber),
            "ac-yn": AttentionCheck(expected: "yes", grading: nil),
        ]
        let outcome = ExclusionEngine.evaluate(
            rules: rules, checks: checks,
            views: [
                // Battery semantics: labeled final answer parses; the first
                // standalone yes/no token decides; unchecked items pass.
                view(condition: "steered", promptID: "ac-num", output: "The answer is 7."),
                view(condition: "steered", promptID: "ac-yn", output: "No, never."),
                view(condition: "steered", promptID: "p1", output: "anything"),
                view(condition: "baseline", promptID: "ac-num", output: "8"),
            ])
        #expect(outcome.stamp.excludedRecords == 2)
        #expect(outcome.stamp.excludedByRule["steered"] == ["failedAttentionCheck": 1])
        #expect(outcome.stamp.excludedByRule["baseline"] == ["failedAttentionCheck": 1])
        #expect(outcome.stamp.survivingN == ["steered": 2, "baseline": 0])
        #expect(outcome.stamp.rules.first?.checkedItems == 2)
    }

    // MARK: - The cross-engine stamp fixture

    @Test func evaluateProducesThePinnedCrossEngineStamp() throws {
        let rules = [
            ExclusionRule(rule: "failedAttentionCheck"),
            ExclusionRule(rule: "unparseableEndpoint"),
            ExclusionRule(rule: "outOfRange", min: 0, max: 600),
        ]
        let checks = [
            "ac-1": AttentionCheck(expected: "7", grading: .exactNumber)
        ]
        let outcome = ExclusionEngine.evaluate(
            rules: rules, checks: checks,
            views: [
                view(condition: "baseline", promptID: "p1", output: "fine", months: 24),
                view(
                    condition: "baseline", promptID: "ac-1",
                    output: "The answer is 7.", months: 12),
                view(condition: "baseline", promptID: "p2", output: "fine", months: 12),
                view(condition: "steer", promptID: "p1", output: "words", months: .some(nil)),
                view(
                    condition: "steer", promptID: "ac-1",
                    output: "The answer is 8.", months: 12),
                view(condition: "steer", promptID: "p2", output: "fine", months: 900),
            ])
        let stamp = outcome.stamp
        // Value-for-value fixture shared with test_exclusions.py.
        #expect(stamp.consideredN == ["baseline": 3, "steer": 3])
        #expect(stamp.survivingN == ["baseline": 3, "steer": 0])
        #expect(stamp.excludedRecords == 3)
        #expect(
            stamp.excludedByRule == [
                "baseline": [
                    "failedAttentionCheck": 0, "unparseableEndpoint": 0,
                    "outOfRange": 0,
                ],
                "steer": [
                    "failedAttentionCheck": 1, "unparseableEndpoint": 1,
                    "outOfRange": 1,
                ],
            ])
        #expect(stamp.pairwiseDeletion == true)
        #expect(stamp.scope == ExclusionEngine.scopeAllRecordTypes)
        #expect(
            stamp.note
                == "Exclusions are applied at analysis time only; excluded "
                + "records remain in generations.jsonl. Scope: all record "
                + "types — endpoint rules (unparseableEndpoint, outOfRange) "
                + "read any record, sampled or deterministic instrument "
                + "readout, that itself carries the named endpoint (never by "
                + "proxy), and a failed attention check drops the whole "
                + "(condition, item) cell, instrument readouts included, once "
                + "every sampled record of the cell fails its check. Paired "
                + "statistics use pairwise deletion: an excluded record's "
                + "item drops from that condition's paired comparison, and an "
                + "item whose baseline record is excluded drops from every "
                + "condition's pairs. A record failing several rules is "
                + "excluded once and counted under each rule it failed; with "
                + "multiple samples per item, the cell keeps its surviving "
                + "samples and drops only when every sample is excluded.")
        #expect(stamp.rules.map(\.rule) == [
            "failedAttentionCheck", "unparseableEndpoint", "outOfRange",
        ])
        #expect(stamp.rules[0].checkedItems == 1)
        #expect(stamp.rules[0].endpoint == nil)
        #expect(stamp.rules[1].endpoint == "parsedMonths")
        #expect(stamp.rules[2].min == 0 && stamp.rules[2].max == 600)

        // Stamp JSON shape: the exact cross-engine key sets ("scope" is the
        // declared-record-types field, 2026-07-20).
        let encoded = try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(stamp))
        let object = try #require(encoded as? [String: Any])
        #expect(
            object.keys.sorted() == [
                "consideredN", "excludedByRule", "excludedRecords", "note",
                "pairwiseDeletion", "rules", "scope", "survivingN",
            ])
        let ruleObjects = try #require(object["rules"] as? [[String: Any]])
        #expect(ruleObjects[0].keys.sorted() == ["checkedItems", "description", "rule"])
        #expect(ruleObjects[1].keys.sorted() == ["description", "endpoint", "rule"])
        #expect(
            ruleObjects[2].keys.sorted() == [
                "description", "endpoint", "max", "min", "rule",
            ])
    }

    @Test func recordFailingSeveralRulesIsExcludedOnceCountedUnderEach() {
        let rules = [
            ExclusionRule(rule: "failedAttentionCheck"),
            ExclusionRule(rule: "unparseableEndpoint"),
        ]
        let outcome = ExclusionEngine.evaluate(
            rules: rules,
            checks: ["ac-1": AttentionCheck(expected: "7", grading: .exactNumber)],
            views: [
                view(
                    condition: "steer", promptID: "ac-1", output: "not a number",
                    months: .some(nil))
            ])
        #expect(outcome.stamp.excludedRecords == 1)
        #expect(outcome.stamp.survivingN == ["steer": 0])
        #expect(
            outcome.stamp.excludedByRule["steer"] == [
                "failedAttentionCheck": 1, "unparseableEndpoint": 1,
            ])
        #expect(outcome.excludedKeys.count == 1)
    }

    // MARK: - Declared scope: instrument readouts (review finding 4)

    private func instrumentView(
        condition: String, promptID: String,
        endpoints: [String: Double?] = [:]
    ) -> ExclusionEngine.InstrumentRecordView {
        ExclusionEngine.InstrumentRecordView(
            condition: condition, promptID: promptID, endpoints: endpoints)
    }

    /// The declared allRecordTypes scope, value-for-value: a cell whose
    /// EVERY sampled record fails its attention check drops its instrument
    /// readout too; a cell with any passing sample keeps it; endpoint rules
    /// fire on instrument records ONLY through endpoints the record itself
    /// carries (never the sampled records' parsedMonths by proxy). Twin of
    /// test_exclusions.py's instrument-scope fixtures.
    @Test func instrumentReadoutsFollowTheDeclaredScope() {
        let rules = [
            ExclusionRule(rule: "failedAttentionCheck"),
            ExclusionRule(
                rule: "outOfRange", endpoint: "ordinalPosition", min: 1, max: 3),
        ]
        let checks = [
            "ac-fail": AttentionCheck(expected: "7", grading: .exactNumber),
            "ac-pass": AttentionCheck(expected: "7", grading: .exactNumber),
        ]
        let outcome = ExclusionEngine.evaluate(
            rules: rules, checks: checks,
            views: [
                // ac-fail: the cell's ONLY sampled record fails the check.
                view(condition: "steer", promptID: "ac-fail", output: "8"),
                // ac-pass: the sampled record passes.
                view(
                    condition: "steer", promptID: "ac-pass",
                    output: "The answer is 7."),
                view(condition: "steer", promptID: "plain", output: "words"),
            ],
            instrumentViews: [
                // Dropped WITH its attention-failed cell.
                instrumentView(
                    condition: "steer", promptID: "ac-fail",
                    endpoints: ["ordinalPosition": 2.0]),
                // Kept — its cell's sampled record passed the check.
                instrumentView(
                    condition: "steer", promptID: "ac-pass",
                    endpoints: ["ordinalPosition": 2.0]),
                // Dropped by its OWN out-of-range ordinalPosition.
                instrumentView(
                    condition: "steer", promptID: "plain",
                    endpoints: ["ordinalPosition": 9.0]),
                // Kept: no sampled partner at all (nothing to grade) and no
                // endpoint the rules read.
                instrumentView(condition: "steer", promptID: "orphan"),
            ])
        // 3 sampled + 4 instrument records considered; ac-fail's sampled +
        // instrument records and plain's instrument record are excluded.
        #expect(outcome.stamp.consideredN == ["steer": 7])
        #expect(outcome.stamp.survivingN == ["steer": 4])
        #expect(outcome.stamp.excludedRecords == 3)
        #expect(
            outcome.stamp.excludedByRule["steer"] == [
                "failedAttentionCheck": 2, "outOfRange": 1,
            ])
        #expect(outcome.excludedKeys.count == 1)
        #expect(
            outcome.excludedInstrumentKeys == [
                ExclusionEngine.instrumentKey(
                    condition: "steer", promptID: "ac-fail"),
                ExclusionEngine.instrumentKey(
                    condition: "steer", promptID: "plain"),
            ])
        #expect(outcome.stamp.scope == "allRecordTypes")
    }

    /// Never by proxy: a default-endpoint (parsedMonths) rule reads NOTHING
    /// on an instrument record — the readout carries no parsedMonths — even
    /// when the cell's sampled record fails that same rule.
    @Test func endpointRulesNeverFireOnInstrumentRecordsByProxy() {
        let outcome = ExclusionEngine.evaluate(
            rules: [ExclusionRule(rule: "unparseableEndpoint")],
            checks: [:],
            views: [
                view(condition: "steer", promptID: "p1", months: .some(nil))
            ],
            instrumentViews: [
                instrumentView(
                    condition: "steer", promptID: "p1",
                    endpoints: ["ordinalPosition": 2.0])
            ])
        // The sampled record is excluded; the instrument readout survives.
        #expect(outcome.stamp.consideredN == ["steer": 2])
        #expect(outcome.stamp.survivingN == ["steer": 1])
        #expect(outcome.excludedInstrumentKeys.isEmpty)
    }

    /// With several samples per cell, the instrument readout drops only
    /// when EVERY sampled record failed the check (the note's multi-sample
    /// rule, extended to instrument readouts).
    @Test func instrumentReadoutSurvivesWhileAnySampledRecordPasses() {
        let rules = [ExclusionRule(rule: "failedAttentionCheck")]
        let checks = ["ac": AttentionCheck(expected: "7", grading: .exactNumber)]
        let mixed = ExclusionEngine.evaluate(
            rules: rules, checks: checks,
            views: [
                view(condition: "steer", promptID: "ac", output: "8", seed: 1),
                view(condition: "steer", promptID: "ac", output: "7", seed: 2),
            ],
            instrumentViews: [instrumentView(condition: "steer", promptID: "ac")])
        #expect(mixed.excludedInstrumentKeys.isEmpty)
        #expect(mixed.stamp.survivingN == ["steer": 2])

        let allFailed = ExclusionEngine.evaluate(
            rules: rules, checks: checks,
            views: [
                view(condition: "steer", promptID: "ac", output: "8", seed: 1),
                view(condition: "steer", promptID: "ac", output: "9", seed: 2),
            ],
            instrumentViews: [instrumentView(condition: "steer", promptID: "ac")])
        #expect(allFailed.excludedInstrumentKeys.count == 1)
        #expect(allFailed.stamp.survivingN == ["steer": 0])
    }

    /// The paired-judge scope (`sampledRecords`) leaves instrument views
    /// unconsidered even when passed — only sampled records are judged.
    @Test func sampledRecordsScopeIgnoresInstrumentViews() {
        let outcome = ExclusionEngine.evaluate(
            rules: [ExclusionRule(rule: "unparseableEndpoint")],
            checks: [:],
            views: [view(condition: "steer", promptID: "p1", months: 12)],
            instrumentViews: [
                instrumentView(
                    condition: "steer", promptID: "p1",
                    endpoints: ["ordinalPosition": 9.0])
            ],
            note: ExclusionEngine.evaluateNote,
            scope: ExclusionEngine.scopeSampledRecords)
        #expect(outcome.stamp.consideredN == ["steer": 1])
        #expect(outcome.stamp.scope == "sampledRecords")
        #expect(outcome.excludedInstrumentKeys.isEmpty)
    }

    /// Run-inline report parity: an excluded baseline instrument readout
    /// drops its item from every condition's ordinalPosition pairs (the
    /// same pairwise deletion the sampled metrics get).
    @Test func excludedInstrumentReadoutDropsItsOrdinalPairFromEffectSizes() {
        let manifest = ExperimentManifest(
            name: "ord-excl", description: "", modelID: "test/model")
        let readouts = [
            ExperimentTasks.ReportChoiceReadout(
                condition: "baseline", promptID: "p1", sampleIndex: nil,
                source: "instrument", selected: "1", target: nil,
                ordinalPosition: 9.0),
            ExperimentTasks.ReportChoiceReadout(
                condition: "baseline", promptID: "p2", sampleIndex: nil,
                source: "instrument", selected: "1", target: nil,
                ordinalPosition: 2.0),
            ExperimentTasks.ReportChoiceReadout(
                condition: "steer", promptID: "p1", sampleIndex: nil,
                source: "instrument", selected: "1", target: nil,
                ordinalPosition: 2.5),
            ExperimentTasks.ReportChoiceReadout(
                condition: "steer", promptID: "p2", sampleIndex: nil,
                source: "instrument", selected: "1", target: nil,
                ordinalPosition: 2.5),
        ]
        // The BASELINE p1 readout is off-scale → excluded → the steer p1
        // ordinal pair drops with it.
        let outcome = ExclusionEngine.evaluate(
            rules: [
                ExclusionRule(
                    rule: "outOfRange", endpoint: "ordinalPosition",
                    min: 1, max: 3)
            ],
            checks: [:], views: [],
            instrumentViews: [
                instrumentView(
                    condition: "baseline", promptID: "p1",
                    endpoints: ["ordinalPosition": 9.0]),
                instrumentView(
                    condition: "baseline", promptID: "p2",
                    endpoints: ["ordinalPosition": 2.0]),
                instrumentView(
                    condition: "steer", promptID: "p1",
                    endpoints: ["ordinalPosition": 2.5]),
                instrumentView(
                    condition: "steer", promptID: "p2",
                    endpoints: ["ordinalPosition": 2.5]),
            ])
        let report = ExperimentTasks.report(
            experiment: manifest, experimentHash: "hh",
            taskPrompts: ("f.jsonl", "ph", []), rows: [], conditionCount: 2,
            concepts: [], choiceReadouts: readouts, exclusions: outcome)
        let ordinal = report.effectSizes?.first {
            $0.condition == "steer" && $0.metric == "ordinalPosition"
        }
        #expect(ordinal?.n == 1)
        #expect(ordinal?.meanDiff == 0.5)
        // Descriptive blocks keep the raw readout counts; the stamp says
        // what survived.
        #expect(report.conditions["baseline"]?.choiceReadouts == 2)
        #expect(report.exclusions?.survivingN == ["baseline": 1, "steer": 2])
    }

    // MARK: - Paired statistics: pairwise deletion through effect sizes

    private func metricRow(
        condition: String, promptID: String, wordCount: Int, seed: UInt64 = 1
    ) -> ExperimentTasks.MetricRow {
        ExperimentTasks.MetricRow(
            condition: condition, seed: seed, promptIndex: 0,
            promptID: promptID, wordCount: wordCount, distinct2: 0.9,
            markerDensity: [:])
    }

    @Test func excludedBaselinePartnerDropsThePairFromEffectSizes() {
        let manifest = ExperimentManifest(
            name: "pairwise", description: "", modelID: "test/model")
        let rows = [
            metricRow(condition: "baseline", promptID: "p1", wordCount: 100),
            metricRow(condition: "baseline", promptID: "p2", wordCount: 90),
            metricRow(condition: "steered", promptID: "p1", wordCount: 111),
            metricRow(condition: "steered", promptID: "p2", wordCount: 99),
        ]
        // The BASELINE p1 record is unparseable → excluded → the steered p1
        // pair drops with it (pairwise deletion), even though the steered
        // record itself survives.
        let outcome = ExclusionEngine.evaluate(
            rules: [ExclusionRule(rule: "unparseableEndpoint")],
            checks: [:],
            views: [
                view(condition: "baseline", promptID: "p1", months: .some(nil)),
                view(condition: "baseline", promptID: "p2", months: 12),
                view(condition: "steered", promptID: "p1", months: 12),
                view(condition: "steered", promptID: "p2", months: 12),
            ])
        let report = ExperimentTasks.report(
            experiment: manifest, experimentHash: "hh",
            taskPrompts: ("f.jsonl", "ph", []), rows: rows, conditionCount: 2,
            concepts: [], exclusions: outcome)
        let wordCount = report.effectSizes?.first {
            $0.condition == "steered" && $0.metric == "wordCount"
        }
        #expect(wordCount?.n == 1)
        #expect(wordCount?.meanDiff == 9)
        // Descriptive per-condition blocks keep the RAW record counts — the
        // stamp carries consideredN vs survivingN.
        #expect(report.conditions["baseline"]?.generations == 2)
        #expect(report.exclusions?.survivingN == ["baseline": 1, "steered": 2])

        // No rules → identical entries to today's behavior, no stamp key.
        let plain = ExperimentTasks.report(
            experiment: manifest, experimentHash: "hh",
            taskPrompts: ("f.jsonl", "ph", []), rows: rows, conditionCount: 2,
            concepts: [])
        #expect(
            plain.effectSizes?.first {
                $0.condition == "steered" && $0.metric == "wordCount"
            }?.n == 2)
        #expect(plain.exclusions == nil)
    }

    // MARK: - Preflight

    @Test func preflightRefusesMalformedRulesAndUnusedAttentionRule() throws {
        #expect {
            try ExclusionEngine.preflight(
                rules: [ExclusionRule(rule: "outliers")], checks: [:])
        } throws: { error in
            ((error as? ExperimentError)?.reason ?? "").contains("not recognized")
        }
        #expect {
            try ExclusionEngine.preflight(
                rules: [ExclusionRule(rule: "failedAttentionCheck")], checks: [:])
        } throws: { error in
            (error as? ExperimentError)?.reason
                == "exclusion rule failedAttentionCheck is declared but no "
                + "task-prompt item declares an attentionCheck — add checks to "
                + "items or drop the rule"
        }
        // Valid + checked, and the no-rules cases, pass silently.
        try ExclusionEngine.preflight(
            rules: [ExclusionRule(rule: "failedAttentionCheck")],
            checks: ["ac-1": AttentionCheck(expected: "7", grading: nil)])
        try ExclusionEngine.preflight(rules: nil, checks: [:])
    }

    // MARK: - Verify + confirmation draft + analyze verb (store-backed)

    @Test func verifySurfacesRuleViolations() throws {
        try withTempRoot { _ in
            var manifest = try ExperimentStore.create(
                name: "verify-rules", description: "", modelID: "test/model")
            manifest.exclusionRules = [ExclusionRule(rule: "outliers")]
            try ExperimentStore.save(manifest)
            let violations = ExperimentStore.verify(manifest)
            #expect(
                violations.contains {
                    $0.contains("exclusion rule 'outliers' is not recognized")
                })
        }
    }

    @Test func confirmationDraftCarriesExclusionRules() throws {
        try withTempRoot { _ in
            var screen = try ExperimentStore.create(
                name: "screen-excl", description: "", modelID: "test/model")
            screen.exclusionRules = [
                ExclusionRule(rule: "outOfRange", min: 0, max: 600)
            ]
            try ExperimentStore.save(screen)
            let draft = try ExperimentStore.createConfirmationDraft(
                fromScreen: "screen-excl", named: "confirm-excl")
            #expect(draft.exclusionRules == screen.exclusionRules)
        }
    }

    /// Fabricates a completed study run (the AnalyzeEffectSizesTests
    /// pattern) whose steered p1 record has a NULL parsedMonths — the
    /// unparseable-endpoint case.
    private func fabricateRun(for manifest: ExperimentManifest) throws -> URL {
        let dir = ExperimentStore.runsDirectory.appending(
            component: "20260720T000000000Z-exp-\(manifest.name)-run")
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        try JSONEncoder().encode(manifest).write(
            to: dir.appending(component: "experiment.json"))
        try ExperimentStore.manifestHash(manifest).write(
            to: dir.appending(component: "experiment-hash.txt"),
            atomically: true, encoding: .utf8)
        let lines = """
            {"condition": "baseline", "seed": 1, "promptID": "p1", "promptIndex": 1, "wordCount": 100, "distinct2": 0.9, "markerDensity": {}, "output": "a", "parsedMonths": 24}
            {"condition": "baseline", "seed": 1, "promptID": "p2", "promptIndex": 2, "wordCount": 90, "distinct2": 0.9, "markerDensity": {}, "output": "b", "parsedMonths": 12}
            {"condition": "steered", "seed": 1, "promptID": "p1", "promptIndex": 1, "wordCount": 111, "distinct2": 0.8, "markerDensity": {}, "output": "c", "parsedMonths": null}
            {"condition": "steered", "seed": 1, "promptID": "p2", "promptIndex": 2, "wordCount": 99, "distinct2": 0.8, "markerDensity": {}, "output": "d", "parsedMonths": 18}

            """
        try lines.write(
            to: dir.appending(component: "generations.jsonl"),
            atomically: true, encoding: .utf8)
        try "{}".write(
            to: dir.appending(component: "report.json"),
            atomically: true, encoding: .utf8)
        return dir
    }

    private func makeVerifiedManifest(name: String) throws -> ExperimentManifest {
        var manifest = try ExperimentStore.create(
            name: name, description: "", modelID: "test/model")
        let french = try StimulusSet(
            directory: VectorCatalog.conceptsDirectory.appending(component: "french"))
        manifest.concepts.append(
            ExperimentStore.makeConceptRef(
                name: "french", stimulusSetHash: french.hash, options: .init()))
        try ExperimentStore.save(manifest)
        return manifest
    }

    @Test func analyzeAppliesDeclaredExclusionsAndStampsThem() throws {
        try withTempRoot { _ in
            var manifest = try makeVerifiedManifest(name: "analyze-excl")
            manifest.exclusionRules = [ExclusionRule(rule: "unparseableEndpoint")]
            try ExperimentStore.save(manifest)
            _ = try fabricateRun(for: manifest)

            let out = try ExperimentTasks.analyze(experimentName: "analyze-excl")
            let object = try #require(
                try JSONSerialization.jsonObject(
                    with: Data(
                        contentsOf: out.appending(component: "analysis.json")))
                    as? [String: Any])
            // The steered p1 record is excluded → its pair drops → n == 1.
            let entries = try #require(object["effectSizes"] as? [[String: Any]])
            let wordCount = try #require(
                entries.first {
                    $0["metric"] as? String == "wordCount"
                        && $0["condition"] as? String == "steered"
                })
            #expect(wordCount["n"] as? Int == 1)
            #expect(wordCount["meanDiff"] as? Double == 9)
            let stampObject = try #require(object["exclusions"] as? [String: Any])
            #expect(stampObject["excludedRecords"] as? Int == 1)
            #expect(
                stampObject["survivingN"] as? [String: Int]
                    == ["baseline": 2, "steered": 1])
            // The stamp file the server also writes.
            let stampFile = try #require(
                try JSONSerialization.jsonObject(
                    with: Data(
                        contentsOf: out.appending(component: "exclusions.json")))
                    as? [String: Any])
            #expect(
                stampFile.keys.sorted() == [
                    "consideredN", "excludedByRule", "excludedRecords", "note",
                    "pairwiseDeletion", "rules", "scope", "survivingN",
                ])
            #expect(stampFile["scope"] as? String == "allRecordTypes")
        }
    }

    @Test func analyzeWithoutRulesIsUnchangedAndStampsNothing() throws {
        try withTempRoot { _ in
            let manifest = try makeVerifiedManifest(name: "analyze-plain")
            _ = try fabricateRun(for: manifest)
            let out = try ExperimentTasks.analyze(experimentName: "analyze-plain")
            let object = try #require(
                try JSONSerialization.jsonObject(
                    with: Data(
                        contentsOf: out.appending(component: "analysis.json")))
                    as? [String: Any])
            #expect(object["exclusions"] == nil)
            #expect(
                !FileManager.default.fileExists(
                    atPath: out.appending(component: "exclusions.json").path))
            let entries = try #require(object["effectSizes"] as? [[String: Any]])
            let wordCount = try #require(
                entries.first {
                    $0["metric"] as? String == "wordCount"
                        && $0["condition"] as? String == "steered"
                })
            #expect(wordCount["n"] as? Int == 2)
        }
    }

    @Test func analyzeAttentionCheckRuleNeedsPinnedPromptsAndRealChecks() throws {
        try withTempRoot { root in
            var manifest = try makeVerifiedManifest(name: "analyze-ac")
            manifest.exclusionRules = [ExclusionRule(rule: "failedAttentionCheck")]
            try ExperimentStore.save(manifest)
            _ = try fabricateRun(for: manifest)

            // Unpinned prompts: the join cannot be trusted — refuse.
            #expect {
                try ExperimentTasks.analyze(experimentName: "analyze-ac")
            } throws: { error in
                (error as? ExperimentError)?.reason
                    == "exclusion rule failedAttentionCheck needs the task "
                    + "prompts pinned (taskPromptsFile + taskPromptsHash) so "
                    + "analysis grades the same items the run saw — pin the "
                    + "prompt set first"
            }

            // Pinned prompts with NO checks: declared-but-inert rule — refuse.
            let noChecks = root.appending(component: "prompts-nochecks.jsonl")
            let noChecksData = Data("""
                {"id": "p1", "prompt": "a"}
                {"id": "p2", "prompt": "b"}

                """.utf8)
            try noChecksData.write(to: noChecks)
            manifest.taskPromptsFile = noChecks.path
            manifest.taskPromptsHash = sha256Hex(noChecksData)
            try ExperimentStore.save(manifest)
            _ = try fabricateRun(for: manifest)
            #expect {
                try ExperimentTasks.analyze(experimentName: "analyze-ac")
            } throws: { error in
                (error as? ExperimentError)?.reason
                    == "exclusion rule failedAttentionCheck is declared but no "
                    + "task-prompt item declares an attentionCheck — add checks "
                    + "to items or drop the rule"
            }

            // With a real check, the failing record is excluded end-to-end:
            // the steered p1 output "c" fails expected "a" (token_exact), so
            // the p1 pair drops and wordCount n == 1.
            let checked = root.appending(component: "prompts-checked.jsonl")
            let checkedData = Data("""
                {"id": "p1", "prompt": "a", "attentionCheck": {"expected": "a", "grading": "token_exact"}}
                {"id": "p2", "prompt": "b"}

                """.utf8)
            try checkedData.write(to: checked)
            manifest.taskPromptsFile = checked.path
            manifest.taskPromptsHash = sha256Hex(checkedData)
            try ExperimentStore.save(manifest)
            _ = try fabricateRun(for: manifest)
            let out = try ExperimentTasks.analyze(experimentName: "analyze-ac")
            let object = try #require(
                try JSONSerialization.jsonObject(
                    with: Data(
                        contentsOf: out.appending(component: "exclusions.json")))
                    as? [String: Any])
            #expect(object["excludedRecords"] as? Int == 1)
            #expect(
                object["survivingN"] as? [String: Int]
                    == ["baseline": 2, "steered": 1])
        }
    }

    /// End-to-end analyze under the declared allRecordTypes scope: a cell
    /// whose sampled record fails its attention check loses its ordinal
    /// instrument readout too, and the dropped BASELINE-side pairing falls
    /// out of the same promptID join as the sampled metrics.
    @Test func analyzeDropsExcludedInstrumentReadoutsWithTheirCell() throws {
        try withTempRoot { root in
            var manifest = try makeVerifiedManifest(name: "analyze-inst")
            manifest.exclusionRules = [ExclusionRule(rule: "failedAttentionCheck")]
            let checked = root.appending(component: "prompts-inst.jsonl")
            let checkedData = Data("""
                {"id": "p1", "prompt": "a", "attentionCheck": {"expected": "a", "grading": "token_exact"}}
                {"id": "p2", "prompt": "b"}

                """.utf8)
            try checkedData.write(to: checked)
            manifest.taskPromptsFile = checked.path
            manifest.taskPromptsHash = sha256Hex(checkedData)
            try ExperimentStore.save(manifest)

            let dir = ExperimentStore.runsDirectory.appending(
                component: "20260720T000000000Z-exp-analyze-inst-run")
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)
            try JSONEncoder().encode(manifest).write(
                to: dir.appending(component: "experiment.json"))
            try ExperimentStore.manifestHash(manifest).write(
                to: dir.appending(component: "experiment-hash.txt"),
                atomically: true, encoding: .utf8)
            // The steered p1 output "c" fails its check (expected "a"), so
            // its whole cell — the ordinal instrument readout included —
            // drops from the paired statistics.
            let lines = """
                {"condition": "baseline", "seed": 1, "promptID": "p1", "promptIndex": 1, "wordCount": 100, "distinct2": 0.9, "markerDensity": {}, "output": "a"}
                {"condition": "baseline", "seed": 1, "promptID": "p2", "promptIndex": 2, "wordCount": 90, "distinct2": 0.9, "markerDensity": {}, "output": "b"}
                {"condition": "steered", "seed": 1, "promptID": "p1", "promptIndex": 1, "wordCount": 111, "distinct2": 0.8, "markerDensity": {}, "output": "c"}
                {"condition": "steered", "seed": 1, "promptID": "p2", "promptIndex": 2, "wordCount": 99, "distinct2": 0.8, "markerDensity": {}, "output": "b"}
                {"instrument": "answerTokenLogprob", "condition": "baseline", "promptID": "p1", "ordinalPosition": 2.0}
                {"instrument": "answerTokenLogprob", "condition": "baseline", "promptID": "p2", "ordinalPosition": 2.0}
                {"instrument": "answerTokenLogprob", "condition": "steered", "promptID": "p1", "ordinalPosition": 3.0}
                {"instrument": "answerTokenLogprob", "condition": "steered", "promptID": "p2", "ordinalPosition": 2.5}

                """
            try lines.write(
                to: dir.appending(component: "generations.jsonl"),
                atomically: true, encoding: .utf8)
            try "{}".write(
                to: dir.appending(component: "report.json"),
                atomically: true, encoding: .utf8)

            let out = try ExperimentTasks.analyze(experimentName: "analyze-inst")
            let object = try #require(
                try JSONSerialization.jsonObject(
                    with: Data(
                        contentsOf: out.appending(component: "analysis.json")))
                    as? [String: Any])
            let entries = try #require(object["effectSizes"] as? [[String: Any]])
            // Sampled pairing: only the surviving p2 pair remains.
            let wordCount = try #require(
                entries.first {
                    $0["metric"] as? String == "wordCount"
                        && $0["condition"] as? String == "steered"
                })
            #expect(wordCount["n"] as? Int == 1)
            #expect(wordCount["meanDiff"] as? Double == 9)
            // Ordinal pairing: the excluded steered p1 readout drops its
            // pair — only p2's 0.5 shift remains.
            let ordinal = try #require(
                entries.first {
                    $0["metric"] as? String == "ordinalPosition"
                        && $0["condition"] as? String == "steered"
                })
            #expect(ordinal["n"] as? Int == 1)
            #expect(ordinal["meanDiff"] as? Double == 0.5)
            // The stamp counts BOTH record types under the declared scope:
            // steered lost its p1 sampled record and its p1 readout.
            let stamp = try #require(object["exclusions"] as? [String: Any])
            #expect(stamp["scope"] as? String == "allRecordTypes")
            #expect(stamp["excludedRecords"] as? Int == 2)
            #expect(
                stamp["consideredN"] as? [String: Int]
                    == ["baseline": 4, "steered": 4])
            #expect(
                stamp["survivingN"] as? [String: Int]
                    == ["baseline": 4, "steered": 2])
            #expect(
                (stamp["excludedByRule"] as? [String: [String: Int]])?["steered"]
                    == ["failedAttentionCheck": 2])
        }
    }

    // MARK: - Paired-judge evaluate (exclusions BEFORE judging)

    /// The evaluate-path note is its own pinned cross-engine literal (the
    /// wording differs from analyze on purpose: evaluate's exclusions save
    /// judge calls) — VALUE-pinned against `test_exclusions.py`.
    @Test func evaluateNoteIsThePinnedCrossEngineString() {
        #expect(
            ExclusionEngine.evaluateNote
                == "Exclusions are applied before judging: excluded records "
                + "are filtered from the pairs entering the judge panel, so "
                + "no judge call is spent on them; excluded records remain "
                + "in the source run's generations.jsonl. Scope: sampled "
                + "records only — instrument readouts are never judged, so "
                + "the rules read only the sampled generations entering the "
                + "panel. Paired judging uses pairwise deletion: an excluded "
                + "record's pair is not judged, and an item whose baseline "
                + "record is excluded drops from every condition's pairs. A "
                + "record failing several rules is excluded once and counted "
                + "under each rule it failed.")
    }

    /// A completed source run with three pairs: p1's STEERED record and
    /// p2's BASELINE record are unparseable; only p3 survives both sides.
    private func fabricateEvaluateSourceRun(
        for manifest: ExperimentManifest
    ) throws -> URL {
        let dir = ExperimentStore.runsDirectory.appending(
            component: "20260720T000000000Z-exp-\(manifest.name)-run")
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        try JSONEncoder().encode(manifest).write(
            to: dir.appending(component: "experiment.json"))
        try ExperimentStore.manifestHash(manifest).write(
            to: dir.appending(component: "experiment-hash.txt"),
            atomically: true, encoding: .utf8)
        let name = manifest.name
        let lines = """
            {"experiment": "\(name)", "condition": "baseline", "seed": 1, "promptID": "p1", "prompt": "q1", "output": "a", "parsedMonths": 24}
            {"experiment": "\(name)", "condition": "baseline", "seed": 1, "promptID": "p2", "prompt": "q2", "output": "b", "parsedMonths": null}
            {"experiment": "\(name)", "condition": "baseline", "seed": 1, "promptID": "p3", "prompt": "q3", "output": "c", "parsedMonths": 18}
            {"experiment": "\(name)", "condition": "steered", "seed": 1, "promptID": "p1", "prompt": "q1", "output": "d", "parsedMonths": null}
            {"experiment": "\(name)", "condition": "steered", "seed": 1, "promptID": "p2", "prompt": "q2", "output": "e", "parsedMonths": 12}
            {"experiment": "\(name)", "condition": "steered", "seed": 1, "promptID": "p3", "prompt": "q3", "output": "f", "parsedMonths": 30}

            """
        try lines.write(
            to: dir.appending(component: "generations.jsonl"),
            atomically: true, encoding: .utf8)
        try "{}".write(
            to: dir.appending(component: "report.json"),
            atomically: true, encoding: .utf8)
        return dir
    }

    /// Judged pairs recorded by the counting fake judge, keyed
    /// condition::promptID.
    private actor JudgedLog {
        var pairs: [String] = []
        func record(_ pair: String) { pairs.append(pair) }
    }

    @Test func evaluateAppliesExclusionsBeforeJudgingAndSavesJudgeCalls()
        async throws
    {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "eval-excl-\(UUID().uuidString)")
        ExperimentStore.rootOverride = temp
        let log = JudgedLog()
        ExperimentTasks.judgeOverrideForTesting = { _, prompt, _, _ in
            await log.record(prompt)
            return PairedJudgeResponse(
                aScores: nil, bScores: nil, structuredFields: nil,
                winner: "A", confidence: 0.9, briefReason: "fake")
        }
        defer {
            ExperimentTasks.judgeOverrideForTesting = nil
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }

        var manifest = try makeVerifiedManifest(name: "eval-excl")
        manifest.exclusionRules = [ExclusionRule(rule: "unparseableEndpoint")]
        try ExperimentStore.save(manifest)
        let source = try fabricateEvaluateSourceRun(for: manifest)

        let out = try await ExperimentTasks.evaluatePairedJudge(
            experimentName: "eval-excl",
            sourceRunDirectory: source,
            evaluation: .init(
                kind: .pairedJudge, judgeModel: "claude-test",
                judgePrompt: "prefer the better response"))

        // The judge saw ONLY the surviving pair: p1 lost its steered record,
        // p2 lost its baseline partner (pairwise deletion) — neither judge
        // call ever happened.
        let judged = await log.pairs
        #expect(judged == ["q3"])

        let report = try #require(
            try JSONSerialization.jsonObject(
                with: Data(
                    contentsOf: out.appending(component: "judge-report.json")))
                as? [String: Any])
        let stamp = try #require(report["exclusions"] as? [String: Any])
        #expect(stamp["excludedRecords"] as? Int == 2)
        #expect(
            stamp["consideredN"] as? [String: Int]
                == ["baseline": 3, "steered": 3])
        #expect(
            stamp["survivingN"] as? [String: Int]
                == ["baseline": 2, "steered": 2])
        #expect(stamp["note"] as? String == ExclusionEngine.evaluateNote)
        // Evaluate's declared scope: sampled records only — instrument
        // readouts are never judged.
        #expect(stamp["scope"] as? String == "sampledRecords")
        // Identical stamp shape to the analyze path's.
        #expect(
            stamp.keys.sorted() == [
                "consideredN", "excludedByRule", "excludedRecords", "note",
                "pairwiseDeletion", "rules", "scope", "survivingN",
            ])
        // The stamp file the server also writes.
        let stampFile = try #require(
            try JSONSerialization.jsonObject(
                with: Data(
                    contentsOf: out.appending(component: "exclusions.json")))
                as? [String: Any])
        #expect(stampFile["excludedRecords"] as? Int == 2)
        // Only the surviving pair was judged into the report.
        let conditions = try #require(report["conditions"] as? [String: Any])
        let steered = try #require(conditions["steered"] as? [String: Any])
        #expect(steered["pairs"] as? Int == 1)
    }

    @Test func evaluateWithoutRulesIsUnchangedAndStampsNothing() async throws {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "eval-plain-\(UUID().uuidString)")
        ExperimentStore.rootOverride = temp
        let log = JudgedLog()
        ExperimentTasks.judgeOverrideForTesting = { _, prompt, _, _ in
            await log.record(prompt)
            return PairedJudgeResponse(
                aScores: nil, bScores: nil, structuredFields: nil,
                winner: "B", confidence: 0.8, briefReason: "fake")
        }
        defer {
            ExperimentTasks.judgeOverrideForTesting = nil
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }

        let manifest = try makeVerifiedManifest(name: "eval-plain")
        let source = try fabricateEvaluateSourceRun(for: manifest)

        let out = try await ExperimentTasks.evaluatePairedJudge(
            experimentName: "eval-plain",
            sourceRunDirectory: source,
            evaluation: .init(
                kind: .pairedJudge, judgeModel: "claude-test",
                judgePrompt: "prefer the better response"))

        // No rules: every pair is judged, no stamp key, no stamp file —
        // today's behavior exactly.
        let judged = await log.pairs
        #expect(judged.sorted() == ["q1", "q2", "q3"])
        let report = try #require(
            try JSONSerialization.jsonObject(
                with: Data(
                    contentsOf: out.appending(component: "judge-report.json")))
                as? [String: Any])
        #expect(report["exclusions"] == nil)
        #expect(
            !FileManager.default.fileExists(
                atPath: out.appending(component: "exclusions.json").path))
    }
}
