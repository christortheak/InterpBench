import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit

/// Both engines pair the same outcomes, from the same records.
///
/// `analyze` reports one paired effect row per outcome a study measured. The
/// Python engine and this one must report the SAME outcomes, under the same
/// names and definitions, from the same records — otherwise a study's
/// results depend on where it ran.
///
/// The measurements come from the shared fixture
/// `Tests/Fixtures/cross-engine/effect-outcomes.json`: one record set that
/// reaches every outcome, with the rows and the outcome list the Python
/// engine's `analyze` writes for it. Server twin:
/// `Server/tests/test_effect_outcomes.py`. The bootstrap interval is left
/// out of the comparison on purpose: the two engines resample with different
/// generators, so its bounds agree only loosely.
///
/// Every expected value in the hand-computed tests is worked out from the
/// measurements, not read back from either engine.
///
/// Every test binds an explicit temporary workspace; none touches
/// `rootOverride`.
struct EffectOutcomeCoverageTests {

    // MARK: - The shared fixture

    private struct Fixture: Decodable {
        let numericParser: String
        let parserRegistry: PinnedFile
        let taxonomy: PinnedFile
        let outcomeFamilies: [StudyAnalysisOutcomes.Family]
        let cases: [Case]
    }

    /// A workspace file the study declares or pins, as exact text.
    private struct PinnedFile: Decodable {
        let path: String
        let text: String
    }

    private struct Case: Decodable {
        let label: String
        let phase: String?
        let exclusionRules: [ExclusionRule]
        let effectRows: [PythonRow]
        let outcomeCoverage: StudyAnalysisOutcomes.Coverage
    }

    /// One effect-sizes.csv row as the Python engine writes it (its column
    /// names; an empty cell is "" for text and null for a number).
    private struct PythonRow: Decodable {
        let condition: String
        let endpoint: String
        let n: Int
        let deltaMean: Double
        let wilcoxonW: Double?
        let wilcoxonP: Double?
        let adjustedP: Double?
        let correction: String
        let stratifyBy: String
        let stratum: String
        let unit: String
        let estimand: String
        let inference: String
    }

    private static let fixtureURL = CodeResources.compiledCheckoutPath.appending(
        components: "Tests", "Fixtures", "cross-engine", "effect-outcomes.json")

    private func fixture() throws -> Fixture {
        try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: Self.fixtureURL))
    }

    private func fixtureCase(_ label: String) throws -> Case {
        try #require(try fixture().cases.first { $0.label == label })
    }

    /// The fixture's records as the lines of a `generations.jsonl`, keys and
    /// nulls exactly as committed (a failed parse is `null`, and must stay
    /// one).
    private func recordLines() throws -> [String] {
        let object = try JSONSerialization.jsonObject(with: Data(contentsOf: Self.fixtureURL))
        let records = try #require((object as? [String: Any])?["records"] as? [[String: Any]])
        return try records.map {
            String(
                decoding: try JSONSerialization.data(withJSONObject: $0, options: [.sortedKeys]),
                as: UTF8.self)
        }
    }

    /// This engine's manifest for the fixture study: the same declarations
    /// the Python suite's manifest makes.
    private func manifest(
        _ fixture: Fixture, phase: String? = nil, exclusionRules: [ExclusionRule] = []
    ) -> ExperimentManifest {
        var manifest = ExperimentManifest(
            name: "outcomes", description: "", modelID: "test/model",
            createdAt: Date(timeIntervalSince1970: 0))
        manifest.numericParser = fixture.numericParser
        manifest.reasoningStyleTaxonomyPath = fixture.taxonomy.path
        manifest.reasoningStyleTaxonomyHash = Self.sha256(fixture.taxonomy.text)
        manifest.phase = phase
        manifest.exclusionRules = exclusionRules.isEmpty ? nil : exclusionRules
        return manifest
    }

    private static func sha256(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - The real analysis entry point, in a temporary workspace

    private struct Analysis {
        let entries: [ExperimentTasks.EffectSizeEntry]
        let exclusions: ExclusionStamp?
        let csv: String
        let coverage: StudyAnalysisOutcomes.Coverage
        let files: Set<String>
    }

    /// Writes the lines as a completed run in a fresh temporary workspace
    /// holding `files`, runs `experiment analyze`'s workflow over it, and
    /// reads back what it published.
    private func analyze(
        _ manifest: ExperimentManifest, lines: [String], files: [PinnedFile]
    ) throws -> Analysis {
        let root = FileManager.default.temporaryDirectory.appending(
            component: "effect-outcomes-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        for file in files {
            let url = root.appending(path: file.path)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(file.text.utf8).write(to: url)
        }
        let run = root.appending(path: "runs/20261004T000000000Z-exp-outcomes-run")
        try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
        try JSONEncoder().encode(manifest).write(
            to: run.appending(component: "experiment.json"))
        try Data(ExperimentStore.manifestHash(manifest).utf8).write(
            to: run.appending(component: "experiment-hash.txt"))
        try Data((lines.joined(separator: "\n") + "\n").utf8).write(
            to: run.appending(component: "generations.jsonl"))
        try Data("{}".utf8).write(to: run.appending(component: "report.json"))

        let out = try StudyAnalysisWorkflow.analyze(
            manifest: manifest,
            repository: StudyAnalysisRepository(workspaceRoot: root, promptRoot: root),
            allowUnverifiedEpoch: false)
        let report = try JSONDecoder().decode(
            ExperimentTasks.AnalyzeReport.self,
            from: Data(contentsOf: out.appending(component: "analysis.json")))
        return Analysis(
            entries: report.effectSizes,
            exclusions: report.exclusions,
            csv: try String(
                contentsOf: out.appending(component: "effect-sizes.csv"), encoding: .utf8),
            coverage: try JSONDecoder().decode(
                StudyAnalysisOutcomes.Coverage.self,
                from: Data(contentsOf: out.appending(component: "outcome-coverage.json"))),
            files: Set(try FileManager.default.contentsOfDirectory(atPath: out.path)))
    }

    /// The fixture study, analyzed by this engine.
    private func analyzeFixture(
        phase: String? = nil, exclusionRules: [ExclusionRule] = []
    ) throws -> Analysis {
        let fixture = try fixture()
        return try analyze(
            manifest(fixture, phase: phase, exclusionRules: exclusionRules),
            lines: try recordLines(), files: [fixture.parserRegistry, fixture.taxonomy])
    }

    private func entry(
        _ entries: [ExperimentTasks.EffectSizeEntry], _ metric: String,
        stratifyBy: String? = nil, stratum: String? = nil, condition: String = "steered"
    ) throws -> ExperimentTasks.EffectSizeEntry {
        let matches = entries.filter {
            $0.metric == metric && $0.stratifyBy == stratifyBy && $0.stratum == stratum
                && $0.condition == condition
        }
        #expect(
            matches.count == 1,
            "expected one \(stratifyBy ?? "pooled") \(stratum ?? "") \(condition) \(metric) row, found \(matches.count)")
        return try #require(matches.first)
    }

    private func absent(
        _ entries: [ExperimentTasks.EffectSizeEntry], _ metric: String,
        stratifyBy: String? = nil, stratum: String? = nil, condition: String = "steered"
    ) -> Bool {
        !entries.contains {
            $0.metric == metric && $0.stratifyBy == stratifyBy && $0.stratum == stratum
                && $0.condition == condition
        }
    }

    /// The normal-approximation p of a signed-rank W, written out from the
    /// textbook formula rather than through the engine: mean n(n+1)/4,
    /// variance n(n+1)(2n+1)/24 less the tie correction,
    /// continuity-corrected. `n` counts the NONZERO differences.
    private func twoSidedP(w: Double, n: Double, tieCorrection: Double = 0) -> Double {
        let mean = n * (n + 1) / 4
        let variance = n * (n + 1) * (2 * n + 1) / 24 - tieCorrection / 48
        let z = (w - mean + 0.5) / variance.squareRoot()
        return min(1, 1 + erf(z / 2.0.squareRoot()))
    }

    /// What a row says its differences are.
    private enum Stamps {
        /// A pooled row: its unit is the run's default, the item.
        case pooled
        /// One difference per item, a member of its correction family.
        case item
        /// One item's samples: a diagnostic, held out of every family.
        case sample
    }

    /// One row against its hand-computed values. `w`/`p` are nil when every
    /// difference is zero (the test is undefined). `adjusted` is nil for a
    /// correction family of one (the adjusted p is the raw p).
    private func check(
        _ row: ExperimentTasks.EffectSizeEntry, n: Int, mean: Double, w: Double?, p: Double?,
        _ stamps: Stamps, adjusted: Double? = nil, correction: String = "bh",
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let name = "\(row.stratifyBy ?? "pooled") \(row.stratum ?? "") \(row.condition) \(row.metric)"
        #expect(row.n == n, "n of \(name)", sourceLocation: sourceLocation)
        #expect(
            abs(row.meanDiff - mean) <= 1e-12, "mean of \(name): \(row.meanDiff)",
            sourceLocation: sourceLocation)
        #expect(row.wilcoxonW == w, "W of \(name)", sourceLocation: sourceLocation)
        switch (row.wilcoxonP, p) {
        case (nil, nil): break
        case (let mine?, let expected?):
            #expect(
                abs(mine - expected) <= 1e-12, "p of \(name): \(mine)",
                sourceLocation: sourceLocation)
        default:
            Issue.record("p of \(name): \(String(describing: row.wilcoxonP))", sourceLocation: sourceLocation)
        }
        switch stamps {
        case .pooled:
            #expect(
                row.unit == nil && row.estimand == nil && row.inference == nil,
                "stamps of \(name)", sourceLocation: sourceLocation)
        case .item:
            #expect(
                row.unit == "item" && row.estimand == "itemLevel" && row.inference == "corrected",
                "stamps of \(name)", sourceLocation: sourceLocation)
        case .sample:
            #expect(
                row.unit == "sample" && row.estimand == "withinItemSamples"
                    && row.inference == "diagnostic",
                "stamps of \(name)", sourceLocation: sourceLocation)
            // A within-item row is a diagnostic: no adjusted p, no family.
            #expect(
                row.adjustedP == nil && row.correction == nil, "correction of \(name)",
                sourceLocation: sourceLocation)
            return
        }
        #expect(row.correction == correction, "correction of \(name)", sourceLocation: sourceLocation)
        guard let raw = row.wilcoxonP else {
            #expect(row.adjustedP == nil, "adjusted p of \(name)", sourceLocation: sourceLocation)
            return
        }
        let mine = row.adjustedP ?? .nan
        #expect(
            abs(mine - (adjusted ?? raw)) <= 1e-12, "adjusted p of \(name): \(mine)",
            sourceLocation: sourceLocation)
    }

    // MARK: - Agreement with the Python engine on the same records

    private static let outcomes = [
        "choiceLogOdds", "choiceRate", "distinct2", "meanMonths", "monthsSpread",
        "ordinalPosition", "parsedValueMean", "parsedValueSpread", "readerScore:warm",
        "rs_hedge", "warmMarkerDensity", "wordCount",
    ]

    /// Three readings of the one record set: an unphased study (corrected
    /// by Benjamini–Hochberg), a confirm-phase one (Holm), and an unphased
    /// one that declares an exclusion.
    @Test(arguments: ["unphased", "confirm", "excluded"])
    func analyzeAgreesWithThePythonEngineOnEveryOutcome(label: String) throws {
        let value = try fixtureCase(label)
        let analysis = try analyzeFixture(
            phase: value.phase, exclusionRules: value.exclusionRules)
        let entries = analysis.entries

        // One record set, every outcome — on both engines.
        #expect(Set(entries.map(\.metric)).sorted() == Self.outcomes)
        #expect(Set(value.effectRows.map(\.endpoint)).sorted() == Self.outcomes)

        func key(_ stratifyBy: String, _ stratum: String, _ condition: String, _ metric: String)
            -> String
        { [stratifyBy, stratum, condition, metric].joined(separator: "|") }
        // Same rows on both engines: nothing missing, nothing extra.
        #expect(
            Set(entries.map { key($0.stratifyBy ?? "pooled", $0.stratum ?? "", $0.condition, $0.metric) })
                == Set(value.effectRows.map { key($0.stratifyBy, $0.stratum, $0.condition, $0.endpoint) }))
        #expect(entries.count == value.effectRows.count)
        #expect(value.effectRows.count == 97)

        // The fixture's numbers carry the CSV's six significant digits.
        func close(_ mine: Double?, _ theirs: Double?) -> Bool {
            switch (mine, theirs) {
            case (nil, nil): return true
            case (let mine?, let theirs?): return abs(mine - theirs) <= 2e-6 * max(1, abs(theirs))
            default: return false
            }
        }
        for expected in value.effectRows {
            let name = key(
                expected.stratifyBy, expected.stratum, expected.condition, expected.endpoint)
            let matches = entries.filter {
                key($0.stratifyBy ?? "pooled", $0.stratum ?? "", $0.condition, $0.metric) == name
            }
            guard let mine = matches.first, matches.count == 1 else { continue }
            #expect(mine.n == expected.n, "n drift on \(name)")
            #expect(close(mine.meanDiff, expected.deltaMean), "mean drift on \(name)")
            #expect(close(mine.wilcoxonW, expected.wilcoxonW), "W drift on \(name)")
            #expect(close(mine.wilcoxonP, expected.wilcoxonP), "p drift on \(name)")
            #expect(close(mine.adjustedP, expected.adjustedP), "adjusted p drift on \(name)")
            // An empty CSV cell on the Python side is an unset field here.
            #expect((mine.correction ?? "") == expected.correction, "correction drift on \(name)")
            #expect((mine.unit ?? "") == expected.unit, "unit drift on \(name)")
            #expect((mine.estimand ?? "") == expected.estimand, "estimand drift on \(name)")
            #expect((mine.inference ?? "") == expected.inference, "inference drift on \(name)")
        }

        // The outcome list the analysis publishes — names, families,
        // definitions, and that nothing is unavailable — is the Python
        // engine's, entry for entry.
        #expect(analysis.coverage == value.outcomeCoverage)
        #expect(analysis.coverage.outcomes.map(\.name) == Self.outcomes)
        #expect(analysis.coverage.notAvailable.isEmpty)
        #expect(analysis.files.contains("outcome-coverage.json"))

        // The file's column names are this engine's own, unchanged.
        #expect(
            analysis.csv.hasPrefix(
                "condition,metric,n,meanDiff,ciLower,ciUpper,wilcoxonW,wilcoxonP,"
                    + "adjustedP,correction,stratifyBy,stratum,unit,estimand,inference\n"))
    }

    /// The definitions in words are the fixture's: this engine's text and
    /// the Python engine's are both held to it.
    @Test func outcomeDefinitionsAreTheSharedOnes() throws {
        let fixture = try fixture()
        #expect(StudyAnalysisOutcomes.families == fixture.outcomeFamilies)
        #expect(Set(StudyAnalysisOutcomes.families.map(\.id)).count == 12)
        for value in fixture.cases {
            #expect(value.outcomeCoverage.pairing == StudyAnalysisOutcomes.pairingDefinition)
            for outcome in value.outcomeCoverage.outcomes {
                #expect(
                    StudyAnalysisOutcomes.family(of: outcome.name, markerConcepts: ["warm"])
                        == outcome.family)
            }
        }
    }

    @Test func outcomeNamesResolveToTheirFamily() {
        for name in [
            "wordCount", "distinct2", "choiceRate", "meanMonths", "monthsSpread",
            "parsedValueMean", "parsedValueSpread", "choiceLogOdds", "ordinalPosition",
        ] {
            #expect(StudyAnalysisOutcomes.family(of: name) == name)
        }
        #expect(StudyAnalysisOutcomes.family(of: "rs_hedge") == "reasoningStyle")
        #expect(StudyAnalysisOutcomes.family(of: "readerScore:warm") == "readerScore")
        #expect(
            StudyAnalysisOutcomes.family(of: "warmMarkerDensity", markerConcepts: ["warm"])
                == "markerDensity")
        // A marker-density name is known by its concept, not its suffix: a
        // reasoning-style feature that happens to end the same way keeps
        // its own family.
        #expect(
            StudyAnalysisOutcomes.family(of: "rs_toneMarkerDensity", markerConcepts: ["warm"])
                == "reasoningStyle")
        #expect(
            StudyAnalysisOutcomes.family(of: "rs_toneMarkerDensity", markerConcepts: ["rs_tone"])
                == "markerDensity")
        // A family id is not an outcome name.
        for name in ["markerDensity", "reasoningStyle", "readerScore", "rs_", "readerScore:", "unheardOf"] {
            #expect(StudyAnalysisOutcomes.family(of: name) == "")
        }
    }

    // MARK: - Hand-computed: the outcomes this engine gained

    /// `choiceLogOdds`: the log-odds of the declared target, one readout per
    /// item and condition.
    ///
    ///                 baseline   steered   Δ       steeredHigh   Δ
    ///     item-1       −1.0       0.5     +1.5      1.0         +2.0
    ///     item-2        0.25      0.75    +0.5      1.25        +1.0
    ///     item-3        2.0       1.0     −1.0      1.5         −0.5
    ///     item-4       −0.5       2.0     +2.5      2.5         +3.0
    ///     item-5       declares no target: no value, on either engine
    ///
    /// steered: mean (1.5 + 0.5 − 1 + 2.5) / 4 = 0.875; the one negative
    /// difference has rank 2, so W = 2. steeredHigh: mean (2 + 1 − 0.5 + 3)
    /// / 4 = 1.375; the negative difference has rank 1, so W = 1.
    @Test func choiceLogOddsMatchesTheHandComputation() throws {
        let pSteered = twoSidedP(w: 2, n: 4)  // 0.361310
        let pHigh = twoSidedP(w: 1, n: 4)  // 0.201243
        #expect(abs(pSteered - 0.361310) < 5e-7 && abs(pHigh - 0.201243) < 5e-7)

        let unphased = try analyzeFixture().entries
        // Benjamini–Hochberg over the two conditions: the smaller p doubles
        // to 0.402485, which exceeds the larger, so both rows take the
        // larger.
        check(
            try entry(unphased, "choiceLogOdds"), n: 4, mean: 0.875, w: 2, p: pSteered, .pooled,
            adjusted: pSteered)
        check(
            try entry(unphased, "choiceLogOdds", condition: "steeredHigh"), n: 4, mean: 1.375,
            w: 1, p: pHigh, .pooled, adjusted: pSteered)

        let confirm = try analyzeFixture(phase: "confirm").entries
        // Holm: the smaller p doubles, and the larger may not fall below it.
        check(
            try entry(confirm, "choiceLogOdds"), n: 4, mean: 0.875, w: 2, p: pSteered, .pooled,
            adjusted: 2 * pHigh, correction: "holm")
        check(
            try entry(confirm, "choiceLogOdds", condition: "steeredHigh"), n: 4, mean: 1.375,
            w: 1, p: pHigh, .pooled, adjusted: 2 * pHigh, correction: "holm")

        // Arm x = {item-1, item-3}: one difference of each sign, the
        // negative one smaller, so W = 1 and z = 0. Arm y = {item-2,
        // item-4}: both positive, W = 0. The arm family holds all four rows
        // (two conditions × two arms): p = 1, 0.371093, 1, 0.371093.
        let pSameSign = twoSidedP(w: 0, n: 2)
        let arm: [(stratum: String, condition: String, mean: Double, w: Double, p: Double)] = [
            ("x", "steered", 0.25, 1, 1),  // +1.5, −1.0
            ("y", "steered", 1.5, 0, pSameSign),  // +0.5, +2.5
            ("x", "steeredHigh", 0.75, 1, 1),  // +2.0, −0.5
            ("y", "steeredHigh", 2.0, 0, pSameSign),  // +1.0, +3.0
        ]
        for expected in arm {
            // Benjamini–Hochberg, m = 4: the two 0.371093s take rank 2 →
            // 0.371093 × 4 / 2. Holm: 0.371093 × 4 > 1, so every row is 1.
            check(
                try entry(
                    unphased, "choiceLogOdds", stratifyBy: "arm", stratum: expected.stratum,
                    condition: expected.condition),
                n: 2, mean: expected.mean, w: expected.w, p: expected.p, .item,
                adjusted: expected.stratum == "x" ? 1 : 2 * pSameSign)
            check(
                try entry(
                    confirm, "choiceLogOdds", stratifyBy: "arm", stratum: expected.stratum,
                    condition: expected.condition),
                n: 2, mean: expected.mean, w: expected.w, p: expected.p, .item,
                adjusted: 1, correction: "holm")
        }

        // One readout per item: an item's own stratum is ONE difference,
        // never a within-item row (there is nothing sampled to pair).
        let own: [(item: String, steered: Double, high: Double)] = [
            ("item-1", 1.5, 2.0), ("item-2", 0.5, 1.0), ("item-3", -1.0, -0.5),
            ("item-4", 2.5, 3.0),
        ]
        for expected in own {
            for (condition, mean) in [("steered", expected.steered), ("steeredHigh", expected.high)] {
                check(
                    try entry(
                        unphased, "choiceLogOdds", stratifyBy: "promptID", stratum: expected.item,
                        condition: condition),
                    n: 1, mean: mean, w: 0, p: 1, .item, adjusted: 1)
            }
        }
        // The rating item reaches the scale position and not the log-odds.
        for condition in ["steered", "steeredHigh"] {
            #expect(
                absent(
                    unphased, "choiceLogOdds", stratifyBy: "promptID", stratum: "item-5",
                    condition: condition))
            #expect(
                try entry(
                    unphased, "ordinalPosition", stratifyBy: "promptID", stratum: "item-5",
                    condition: condition
                ).n == 1)
        }
        #expect(try entry(unphased, "ordinalPosition").n == 5)
    }

    /// `choiceRate`: the share of an item's responses that chose the target
    /// "A", among those with a readable choice ("–" is unreadable).
    ///
    ///     item-1   A B –  → 1/2    |  A A A → 1      +0.5
    ///     item-2   B B B  → 0      |  A B – → 1/2    +0.5
    ///     item-3   A A A  → 1      |  B – A → 1/2    −0.5
    ///     item-4   – – –  → none   |  A A A → 1      the item does not pair
    ///
    /// Three paired items (not four), mean (0.5 + 0.5 − 0.5) / 3 = 1/6. All
    /// three differences tie at rank 2: W+ = 4, W− = 2, W = 2.
    @Test func choiceRateMatchesTheHandComputation() throws {
        let entries = try analyzeFixture().entries
        check(
            try entry(entries, "choiceRate"), n: 3, mean: 1.0 / 6, w: 2,
            p: twoSidedP(w: 2, n: 3, tieCorrection: 24), .pooled)  // p = 0.772830
        // Only "steered" has sampled responses.
        #expect(absent(entries, "choiceRate", condition: "steeredHigh"))

        // Arm x = {item-1, item-3}: +0.5 and −0.5 tie, W = 1.5, p clamps
        // to 1.
        check(
            try entry(entries, "choiceRate", stratifyBy: "arm", stratum: "x"), n: 2, mean: 0,
            w: 1.5, p: 1, .item)
        // Arm y = {item-2, item-4}: only item-2 pairs. ONE paired item drops
        // to that item's samples — sample 0 (A against B: 1 − 0) and sample
        // 1 (B against B: 0 − 0); sample 2 is unreadable — and the row is a
        // diagnostic, so arm x above is a correction family of one.
        check(
            try entry(entries, "choiceRate", stratifyBy: "arm", stratum: "y"), n: 2, mean: 0.5,
            w: 0, p: 1, .sample)

        let within: [(item: String, mean: Double)] = [
            ("item-1", 0.5),  // sample 0: 1 − 1; sample 1: 1 − 0
            ("item-2", 0.5),  // sample 0: 1 − 0; sample 1: 0 − 0
            ("item-3", -0.5),  // sample 0: 0 − 1; sample 2: 1 − 1
        ]
        for expected in within {
            check(
                try entry(entries, "choiceRate", stratifyBy: "promptID", stratum: expected.item),
                n: 2, mean: expected.mean, w: 0, p: 1, .sample)
        }
        #expect(absent(entries, "choiceRate", stratifyBy: "promptID", stratum: "item-4"))
    }

    /// `meanMonths` and `monthsSpread` — and, because the study's parser
    /// reads a percentage rather than a duration, the same values again as
    /// `parsedValueMean` and `parsedValueSpread`.
    ///
    /// The parsed numbers ("–" is a response the parser could not read),
    /// each cell's mean and sample standard deviation:
    ///
    ///     item-1   10 20 30 → 20, 10   |  30 40 50   → 40, 10      +20     0
    ///     item-2   50 60 70 → 60, 10   |  55 –  75   → 65, √200    +5      √200 − 10
    ///     item-3   20 40 60 → 40, 20   |  30 30 30   → 30, 0       −10     −20
    ///     item-4   80 –  –  → 80, none |  90 100 110 → 100, 10     +20     does not pair
    ///
    /// Mean: four items, (20 + 5 − 10 + 20) / 4 = 8.75; ranks 1 (5), 2
    /// (−10), and 3.5 twice (the two 20s), so W− = 2. Spread: three items,
    /// since a cell with one readable response has no spread; the zero
    /// drops, and the two that remain have ranks 1 (+) and 2 (−), so W = 1
    /// and z = 0.
    @Test func parsedNumbersMatchTheHandComputation() throws {
        let entries = try analyzeFixture().entries
        let spread2 = 200.0.squareRoot() - 10  // 4.142136
        let pSameSign = twoSidedP(w: 0, n: 2)
        for (meanName, spreadName) in [
            ("meanMonths", "monthsSpread"), ("parsedValueMean", "parsedValueSpread"),
        ] {
            check(
                try entry(entries, meanName), n: 4, mean: 8.75, w: 2,
                p: twoSidedP(w: 2, n: 4, tieCorrection: 6), .pooled)  // p = 0.357273
            check(
                try entry(entries, spreadName), n: 3, mean: (200.0.squareRoot() - 30) / 3, w: 1,
                p: 1, .pooled)  // mean −5.285955

            // Arm x = {item-1, item-3}: +20, −10 → W = 1, z = 0. Arm y =
            // {item-2, item-4}: +5, +20 → W = 0. Benjamini–Hochberg over
            // the pair doubles the smaller p.
            check(
                try entry(entries, meanName, stratifyBy: "arm", stratum: "x"), n: 2, mean: 5,
                w: 1, p: 1, .item, adjusted: 1)
            check(
                try entry(entries, meanName, stratifyBy: "arm", stratum: "y"), n: 2, mean: 12.5,
                w: 0, p: pSameSign, .item, adjusted: 2 * pSameSign)
            // Spread, arm x: 0 and −20 → one nonzero difference. Arm y:
            // only item-2 pairs, and a spread has no sample axis to drop
            // to — it stays ONE item-level difference.
            check(
                try entry(entries, spreadName, stratifyBy: "arm", stratum: "x"), n: 2, mean: -10,
                w: 0, p: 1, .item, adjusted: 1)
            check(
                try entry(entries, spreadName, stratifyBy: "arm", stratum: "y"), n: 1,
                mean: spread2, w: 0, p: 1, .item, adjusted: 1)

            // An item's own spread: one difference. item-1's is exactly
            // zero, so its test is undefined; item-4 has none.
            check(
                try entry(entries, spreadName, stratifyBy: "promptID", stratum: "item-1"), n: 1,
                mean: 0, w: nil, p: nil, .item)
            check(
                try entry(entries, spreadName, stratifyBy: "promptID", stratum: "item-2"), n: 1,
                mean: spread2, w: 0, p: 1, .item, adjusted: 1)
            check(
                try entry(entries, spreadName, stratifyBy: "promptID", stratum: "item-3"), n: 1,
                mean: -20, w: 0, p: 1, .item, adjusted: 1)
            #expect(absent(entries, spreadName, stratifyBy: "promptID", stratum: "item-4"))
        }

        // An item's own mean, under the months name, drops to its samples
        // when two of them were read on both sides.
        check(
            try entry(entries, "meanMonths", stratifyBy: "promptID", stratum: "item-1"), n: 3,
            mean: 20, w: 0, p: twoSidedP(w: 0, n: 3, tieCorrection: 24), .sample)  // +20 +20 +20
        check(
            try entry(entries, "meanMonths", stratifyBy: "promptID", stratum: "item-2"), n: 2,
            mean: 5, w: 0, p: twoSidedP(w: 0, n: 2, tieCorrection: 6), .sample)  // +5 · +5
        check(
            try entry(entries, "meanMonths", stratifyBy: "promptID", stratum: "item-3"), n: 3,
            mean: -10, w: 1.5, p: twoSidedP(w: 1.5, n: 3, tieCorrection: 6), .sample)  // +10 −10 −30
        // item-4: one sample read on both sides is not enough — it stays
        // the one item-level difference, 100 − 80, and the only corrected
        // row of its family.
        check(
            try entry(entries, "meanMonths", stratifyBy: "promptID", stratum: "item-4"), n: 1,
            mean: 20, w: 0, p: 1, .item)
        // Under the neutral name an item's own row is ALWAYS the one
        // item-level difference (the Python engine has never read that name
        // sample by sample, and the two engines must agree).
        for (item, mean) in [("item-1", 20.0), ("item-2", 5.0), ("item-3", -10.0), ("item-4", 20.0)] {
            check(
                try entry(entries, "parsedValueMean", stratifyBy: "promptID", stratum: item),
                n: 1, mean: mean, w: 0, p: 1, .item, adjusted: 1)
        }
    }

    /// `readerScore:warm`: the recorded reader score, averaged over the
    /// responses that carry one.
    ///
    ///     item-1    1    1.5  2   → 1.5   |   2    2.5   3    → 2.5     +1
    ///     item-2    0.25 0.75 ·   → 0.5   |   1    1     1    → 1       +0.5
    ///     item-3   −1    0    1   → 0     |  −1   −0.75 −0.5  → −0.75   −0.75
    ///     item-4    2    2    2   → 2     |   4    4.5   5    → 4.5     +2.5
    ///
    /// (item-2's third baseline record carries no score, so its cell
    /// averages two.) Mean (1 + 0.5 − 0.75 + 2.5) / 4 = 0.8125; the negative
    /// difference has rank 2, so W = 2.
    @Test func readerScoresMatchTheHandComputation() throws {
        let entries = try analyzeFixture().entries
        let pSameSign = twoSidedP(w: 0, n: 2)
        check(
            try entry(entries, "readerScore:warm"), n: 4, mean: 0.8125, w: 2,
            p: twoSidedP(w: 2, n: 4), .pooled)  // p = 0.361310
        // Arm x = {item-1, item-3}: +1, −0.75 → W = 1. Arm y = {item-2,
        // item-4}: +0.5, +2.5 → W = 0.
        check(
            try entry(entries, "readerScore:warm", stratifyBy: "arm", stratum: "x"), n: 2,
            mean: 0.125, w: 1, p: 1, .item, adjusted: 1)
        check(
            try entry(entries, "readerScore:warm", stratifyBy: "arm", stratum: "y"), n: 2,
            mean: 1.5, w: 0, p: pSameSign, .item, adjusted: 2 * pSameSign)
        let within: [(item: String, n: Int, mean: Double, p: Double)] = [
            ("item-1", 3, 1, twoSidedP(w: 0, n: 3, tieCorrection: 24)),  // +1 +1 +1
            ("item-2", 2, 0.5, pSameSign),  // +0.75 +0.25 ·
            ("item-3", 3, -0.75, pSameSign),  // 0 −0.75 −1.5
            ("item-4", 3, 2.5, twoSidedP(w: 0, n: 3)),  // +2 +2.5 +3
        ]
        for expected in within {
            check(
                try entry(
                    entries, "readerScore:warm", stratifyBy: "promptID", stratum: expected.item),
                n: expected.n, mean: expected.mean, w: 0, p: expected.p, .sample)
        }
    }

    // MARK: - A declared exclusion reaches every outcome

    /// The study declares `unparseableEndpoint`: a response the numeric
    /// parser could not read is left out — of EVERY outcome, not only the
    /// parsed number. Three responses go: item-2's second steered response,
    /// and item-4's second and third baseline responses.
    ///
    /// wordCount (the excluded responses in brackets):
    ///
    ///     item-2   20 20 23 → 21      |  22 [25] 25 → 23.5    +2.5  (was +3)
    ///     item-4   40 [41] [45] → 40  |  50 52 54   → 52      +12   (was +10)
    ///
    /// so the mean is (6 + 2.5 − 1 + 12) / 4 = 4.875.
    ///
    /// choiceRate: item-2's steered cell loses its "B" and keeps "A" and an
    /// unreadable response → 1 of 1; baseline 0 → +1 (was +0.5). item-4's
    /// baseline had no readable choice and still has none. Mean (0.5 + 1 −
    /// 0.5) / 3 = 1/3; the two 0.5s tie at rank 1.5, so W− = 1.5.
    ///
    /// The parsed number itself does not move: the responses that went had
    /// no number to contribute. The answer-token readouts carry no parsed
    /// number, so the rule does not touch them.
    @Test func aDeclaredExclusionReachesEveryOutcome() throws {
        let analysis = try analyzeFixture(
            exclusionRules: [ExclusionRule(rule: ExclusionEngine.ruleUnparseableEndpoint)])
        #expect(analysis.exclusions?.excludedRecords == 3)
        let entries = analysis.entries

        check(
            try entry(entries, "wordCount"), n: 4, mean: 4.875, w: 1, p: twoSidedP(w: 1, n: 4),
            .pooled)
        check(
            try entry(entries, "choiceRate"), n: 3, mean: 1.0 / 3, w: 1.5,
            p: twoSidedP(w: 1.5, n: 3, tieCorrection: 6), .pooled)  // p = 0.586214
        check(
            try entry(entries, "meanMonths"), n: 4, mean: 8.75, w: 2,
            p: twoSidedP(w: 2, n: 4, tieCorrection: 6), .pooled)
        check(
            try entry(entries, "readerScore:warm"), n: 4, mean: 0.8125, w: 2,
            p: twoSidedP(w: 2, n: 4), .pooled)
        // item-2's steered marker densities 0.125 [0.25] 0.375 still average
        // 0.25, so this mean happens not to move either.
        check(
            try entry(entries, "warmMarkerDensity"), n: 4, mean: 0.09375, w: 2,
            p: twoSidedP(w: 2, n: 3), .pooled)
        check(
            try entry(entries, "choiceLogOdds"), n: 4, mean: 0.875, w: 2,
            p: twoSidedP(w: 2, n: 4), .pooled, adjusted: twoSidedP(w: 2, n: 4))

        // item-2's own word-count row pairs the two samples both arms still
        // have: 22 − 20 and 25 − 23. item-4 has one such sample, which is
        // not enough for a within-item row: it stays the one item-level
        // difference.
        check(
            try entry(entries, "wordCount", stratifyBy: "promptID", stratum: "item-2"), n: 2,
            mean: 2, w: 0, p: twoSidedP(w: 0, n: 2, tieCorrection: 6), .sample)
        check(
            try entry(entries, "wordCount", stratifyBy: "promptID", stratum: "item-4"), n: 1,
            mean: 12, w: 0, p: 1, .item)
        // item-2's choice rate has ONE sample readable on both sides now (A
        // against B), so it too is the item-level difference, 1 − 0.
        check(
            try entry(entries, "choiceRate", stratifyBy: "promptID", stratum: "item-2"), n: 1,
            mean: 1, w: 0, p: 1, .item)
        check(
            try entry(entries, "choiceRate", stratifyBy: "arm", stratum: "y"), n: 1, mean: 1,
            w: 0, p: 1, .item, adjusted: 1)
    }

    // MARK: - What an analysis cannot produce, it says

    /// The study declares a numeric parser, and its registry entry cannot
    /// be read. The parsed numbers are still paired — under the months
    /// names — and the two neutral names are SAID to be missing, with the
    /// reason, rather than silently absent. Nothing else moves.
    @Test func anUnreadableNumericParserIsSaidNotSilentlyDropped() throws {
        let fixture = try fixture()
        let whole = try analyzeFixture()
        // The same study, in a workspace with no parser registry.
        let analysis = try analyze(
            manifest(fixture), lines: try recordLines(), files: [fixture.taxonomy])

        let names = Set(analysis.entries.map(\.metric))
        #expect(names.contains("meanMonths") && names.contains("monthsSpread"))
        #expect(!names.contains("parsedValueMean") && !names.contains("parsedValueSpread"))
        // Every other row is exactly what it was with the registry present.
        #expect(
            analysis.entries
                == whole.entries.filter {
                    $0.metric != "parsedValueMean" && $0.metric != "parsedValueSpread"
                })

        #expect(analysis.coverage.outcomes.map(\.name) == Self.outcomes)
        let missing = analysis.coverage.notAvailable
        #expect(missing.map(\.name) == ["parsedValueMean", "parsedValueSpread"])
        for outcome in missing {
            #expect(outcome.family == outcome.name)
            let reason = try #require(outcome.reason)
            #expect(reason.hasPrefix("Not available in this analysis: "))
            #expect(reason.contains("the numeric parser 'percent'"))
            #expect(reason.contains("no parser registry exists at prompts/parsers/parser-registry.json"))
            #expect(reason.contains("meanMonths and monthsSpread"))
            #expect(!outcome.definition.isEmpty)
        }

        // A registry that changed since the study pinned it is not trusted
        // to say what the run's numbers were, either.
        var pinned = manifest(fixture)
        pinned.parserRegistryHash = Self.sha256("some other registry")
        let drifted = try analyze(
            pinned, lines: try recordLines(), files: [fixture.parserRegistry, fixture.taxonomy])
        #expect(drifted.coverage.notAvailable.map(\.name) == ["parsedValueMean", "parsedValueSpread"])
        #expect(
            try #require(drifted.coverage.notAvailable.first?.reason)
                .contains("the parser registry changed since the study pinned it"))

        // …and the right pin reads normally.
        pinned.parserRegistryHash = Self.sha256(fixture.parserRegistry.text)
        let held = try analyze(
            pinned, lines: try recordLines(), files: [fixture.parserRegistry, fixture.taxonomy])
        #expect(held.coverage.notAvailable.isEmpty)
        #expect(held.entries == whole.entries)
    }

    /// The neutral names belong to parsers that do NOT read durations. A
    /// duration parser, or a study that declares no parser, reports the
    /// months names alone — and nothing is missing.
    @Test func neutralNamesAppearOnlyForAParserThatDoesNotReadDurations() throws {
        let fixture = try fixture()
        let durations = PinnedFile(
            path: fixture.parserRegistry.path,
            text: #"""
                {"schemaVersion": 1, "parsers": {"percent": {"kind": "durationMonths",
                 "units": {"months": 1, "years": 12}}}}
                """#)
        var undeclared = manifest(fixture)
        undeclared.numericParser = nil
        for analysis in [
            try analyze(
                manifest(fixture), lines: try recordLines(), files: [durations, fixture.taxonomy]),
            try analyze(undeclared, lines: try recordLines(), files: [fixture.taxonomy]),
        ] {
            let names = Set(analysis.entries.map(\.metric))
            #expect(names.contains("meanMonths") && names.contains("monthsSpread"))
            #expect(!names.contains("parsedValueMean") && !names.contains("parsedValueSpread"))
            #expect(analysis.coverage.notAvailable.isEmpty)
            #expect(
                analysis.coverage.outcomes.map(\.name)
                    == Self.outcomes.filter { !$0.hasPrefix("parsedValue") })
        }
    }

    // MARK: - What must NOT change

    /// A reading of the wrong type is no reading — and costs the record
    /// nothing else. These two responses analyzed to a word-count row
    /// before the outcome readings were read, and still do.
    @Test func aMalformedReadingCostsOnlyItself() throws {
        var manifest = ExperimentManifest(
            name: "outcomes", description: "", modelID: "test/model",
            createdAt: Date(timeIntervalSince1970: 0))
        manifest.numericParser = nil
        let lines = [
            #"{"condition":"baseline","seed":1,"promptID":"p","wordCount":10,"distinct2":0.5,"target":"A","parsedMonths":"twelve","parsedChoice":3,"readerScores":"none"}"#,
            #"{"condition":"steered","seed":1,"promptID":"p","wordCount":14,"distinct2":0.5,"target":"A","parsedMonths":12,"parsedChoice":"A","readerScores":{"warm":1}}"#,
        ]
        let analysis = try analyze(manifest, lines: lines, files: [])
        let wordCount = try entry(analysis.entries, "wordCount")
        #expect(wordCount.n == 1 && wordCount.meanDiff == 4)
        // The baseline has no readable number, choice, or score, so none of
        // the three pairs.
        #expect(Set(analysis.entries.map(\.metric)) == ["wordCount", "distinct2"])
        #expect(analysis.coverage.outcomes.map(\.name) == ["distinct2", "wordCount"])
    }

    /// Records that carry none of the new readings analyze to exactly the
    /// rows they always did: no row is added, and the outcome list names
    /// only what the rows hold.
    @Test func recordsWithoutTheNewReadingsGainNoRows() throws {
        var manifest = ExperimentManifest(
            name: "outcomes", description: "", modelID: "test/model",
            createdAt: Date(timeIntervalSince1970: 0))
        manifest.numericParser = nil
        var lines: [String] = []
        for (item, base, steered) in [("p1", 100, 111), ("p2", 90, 99)] {
            lines.append(
                #"{"condition":"baseline","seed":1,"promptID":"\#(item)","promptIndex":1,"wordCount":\#(base),"distinct2":0.5,"markerDensity":{"french":0.0}}"#)
            lines.append(
                #"{"condition":"steered","seed":1,"promptID":"\#(item)","promptIndex":1,"wordCount":\#(steered),"distinct2":0.25,"markerDensity":{"french":0.5}}"#)
        }
        let analysis = try analyze(manifest, lines: lines, files: [])
        #expect(
            analysis.entries.filter { $0.stratifyBy == nil }.map(\.metric)
                == ["wordCount", "distinct2", "frenchMarkerDensity"])
        // Pooled, plus each of the two items' own stratum: 3 + 6.
        #expect(analysis.entries.count == 9)
        #expect(
            analysis.coverage.outcomes.map(\.name)
                == ["distinct2", "frenchMarkerDensity", "wordCount"])
        #expect(analysis.coverage.outcomes.map(\.family) == ["distinct2", "markerDensity", "wordCount"])
    }
}
