import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit

/// The evaluate subsample — self-contained, and pinned by literals.
///
/// Cross-engine twin: `Server/tests/test_evaluate_subsample.py`. The LITERAL
/// blocks below appear byte-identically in both suites: the same stream-seed
/// derivation, the same seeded promptID order, the same allotments, and the
/// same chosen `(condition, promptID, sampleIndex)` triples. Two things
/// depend on that.
///
/// 1. **A runtime upgrade cannot move the draw undetected.** The membership
///    of a subsample IS the evidence — a report that says
///    `samplePerCondition: 2400` and `sampleSeed: 0x…` claims those 2,400
///    records are recomputable from those two numbers forever. Every step is
///    64-bit integer arithmetic and SHA-256 (the argument
///    `TokenBankDownsampling` makes at length, and the reason its SplitMix64
///    is the primitive this borrows).
/// 2. **The two engines draw the SAME records.** A study whose corpus is
///    coded on the cluster and whose numbers are checked on the Mac must be
///    looking at one subsample, not two that happen to be the same size.
@Suite("Evaluate subsample")
struct EvaluateSubsampleTests {

    // ---------------------------------------------------------- literals
    //
    // Twin values, asserted verbatim on both engines.
    let literalStreamSeed42Baseline: UInt64 = 11_560_963_923_506_615_204
    let literalStreamSeed42BaselineP01: UInt64 = 1_292_222_067_763_893_346
    let literalSeededOrder5_42 = [3, 4, 2, 0, 1]
    let literalAllotments444_7 = [3, 2, 2]
    let literalRuleSHA256 =
        "92cca1c80532c5d2fbf96734dd3b5feb2d767ec90f6ac8fdf65152aab55f049d"

    /// The shared cross-engine fixture: a synthetic run of 2 conditions × 3
    /// promptIDs × 4 sampleIndexes = 24 codeable records, sampled at
    /// `--sample-per-condition 7 --sample-seed 0x2a`. Exactly these 14
    /// triples, in exactly this order, on both engines.
    let literalFixtureSelection: [Triple] = [
        Triple("baseline", "p01", 0),
        Triple("baseline", "p01", 1),
        Triple("baseline", "p01", 3),
        Triple("baseline", "p02", 1),
        Triple("baseline", "p02", 3),
        Triple("baseline", "p03", 0),
        Triple("baseline", "p03", 1),
        Triple("injected", "p01", 0),
        Triple("injected", "p01", 1),
        Triple("injected", "p02", 0),
        Triple("injected", "p02", 2),
        Triple("injected", "p02", 3),
        Triple("injected", "p03", 0),
        Triple("injected", "p03", 2),
    ]

    struct Triple: Equatable, CustomStringConvertible {
        let condition: String
        let promptID: String
        let sampleIndex: UInt64
        init(_ condition: String, _ promptID: String, _ sampleIndex: UInt64) {
            self.condition = condition
            self.promptID = promptID
            self.sampleIndex = sampleIndex
        }
        init(_ coordinate: EvaluateSubsample.Coordinate) {
            self.condition = coordinate.condition
            self.promptID = coordinate.promptID
            self.sampleIndex = coordinate.sampleIndex
        }
        var description: String { "(\(condition), \(promptID), \(sampleIndex))" }
    }

    /// The 24-record synthetic run the literal selection is drawn from.
    func fixtureRecords() -> [EvaluateSubsample.Coordinate] {
        var records: [EvaluateSubsample.Coordinate] = []
        for condition in ["baseline", "injected"] {
            for prompt in ["p01", "p02", "p03"] {
                for index in UInt64(0)..<4 {
                    records.append(
                        .init(
                            condition: condition, promptID: prompt,
                            sampleIndex: index))
                }
            }
        }
        return records
    }

    func chosen(
        _ records: [EvaluateSubsample.Coordinate],
        _ request: EvaluateSubsample.Request
    ) throws -> [Triple] {
        try EvaluateSubsample.selectedPositions(
            records, request: request, program: "steerlab-cli"
        ).map { Triple(records[$0]) }
    }

    // ---------------------------------------------------------- the draw

    @Test("The stream seed is the pinned derivation")
    func streamSeedIsThePinnedDerivation() {
        #expect(
            EvaluateSubsample.streamSeed(42, "baseline")
                == literalStreamSeed42Baseline)
        #expect(
            EvaluateSubsample.streamSeed(42, "baseline", "p01")
                == literalStreamSeed42BaselineP01)
    }

    @Test("The stream seed length-prefixes its parts")
    func streamSeedLengthPrefixesItsParts() {
        // ("ab", "c") and ("a", "bc") must not collide — without the length
        // prefix two different cells of one condition would share a draw.
        #expect(
            EvaluateSubsample.streamSeed(1, "ab", "c")
                != EvaluateSubsample.streamSeed(1, "a", "bc"))
    }

    @Test("The seeded order is the pinned permutation")
    func seededOrderIsThePinnedPermutation() {
        #expect(
            EvaluateSubsample.seededOrder(count: 5, seed: 42)
                == literalSeededOrder5_42)
        #expect(
            EvaluateSubsample.seededOrder(count: 9, seed: 7).sorted()
                == Array(0..<9))
    }

    @Test("Allotments are floor plus a seeded remainder")
    func allotmentsAreFloorPlusASeededRemainder() {
        #expect(
            EvaluateSubsample.allotments(
                populations: [4, 4, 4], total: 7,
                seed: EvaluateSubsample.streamSeed(42, "baseline"))
                == literalAllotments444_7)
        #expect(literalAllotments444_7.reduce(0, +) == 7)
    }

    @Test("A full cell is skipped and exactly n is still placed")
    func allotmentsSkipAFullCellAndStillPlaceExactlyN() {
        // A ragged source run (a partial or resumed one) still yields exactly
        // n: a promptID at its population is skipped and its quota passes on.
        let allot = EvaluateSubsample.allotments(
            populations: [1, 10, 10], total: 9, seed: 99)
        #expect(allot.reduce(0, +) == 9)
        #expect(allot[0] <= 1)
    }

    @Test("The selection is the pinned cross-engine fixture")
    func selectionIsThePinnedCrossEngineFixture() throws {
        let records = fixtureRecords()
        let request = EvaluateSubsample.Request(
            samplePerCondition: 7, seed: 0x2A)
        #expect(try chosen(records, request) == literalFixtureSelection)
        let stamp = EvaluateSubsample.stamp(request, sampled: 14, source: 24)
        #expect(stamp.samplePerCondition == 7)
        #expect(stamp.sampleSeed == "0x000000000000002a")
        #expect(stamp.sampledRecords == 14)
        #expect(stamp.sourceRecords == 24)
    }

    @Test("The selection is deterministic and seed-sensitive")
    func selectionIsDeterministicAndSeedSensitive() throws {
        let records = fixtureRecords()
        let first = try chosen(
            records, .init(samplePerCondition: 7, seed: 0x2A))
        let again = try chosen(
            records, .init(samplePerCondition: 7, seed: 0x2A))
        let other = try chosen(
            records, .init(samplePerCondition: 7, seed: 0x2B))
        #expect(first == again)
        #expect(first != other)
    }

    @Test("Every condition contributes exactly n")
    func everyConditionContributesExactlyN() throws {
        let picked = try chosen(
            fixtureRecords(), .init(samplePerCondition: 5, seed: 12345))
        var counts: [String: Int] = [:]
        for triple in picked { counts[triple.condition, default: 0] += 1 }
        #expect(counts == ["baseline": 5, "injected": 5])
    }

    @Test("Kept positions stay in source order")
    func keptPositionsStayInSourceOrder() throws {
        let positions = try EvaluateSubsample.selectedPositions(
            fixtureRecords(), request: .init(samplePerCondition: 6, seed: 9),
            program: "steerlab-cli")
        #expect(positions == positions.sorted())
    }

    @Test("A full-size sample keeps everything")
    func aFullSizeSampleKeepsEverything() throws {
        let records = fixtureRecords()
        let picked = try chosen(
            records, .init(samplePerCondition: 12, seed: 3))
        #expect(picked == records.map(Triple.init))
    }

    // ---------------------------------------------------------- refusals

    @Test("A sample without a seed refuses")
    func aSampleWithoutASeedRefuses() {
        #expect(throws: ExperimentError.self) {
            _ = try EvaluateSubsample.resolveRequest(
                samplePerCondition: "2400", sampleSeed: nil,
                program: "steerlab-cli")
        }
        do {
            _ = try EvaluateSubsample.resolveRequest(
                samplePerCondition: "2400", sampleSeed: nil,
                program: "steerlab-cli")
            Issue.record("expected a refusal")
        } catch let error as ExperimentError {
            #expect(error.reason.contains("nobody can redraw"))
            #expect(
                error.malformedInvocation?.repairAction.contains("--sample-seed")
                    == true)
        } catch {
            Issue.record("wrong error: \(error)")
        }
    }

    @Test("A seed without a sample size refuses")
    func aSeedWithoutASampleSizeRefuses() {
        do {
            _ = try EvaluateSubsample.resolveRequest(
                samplePerCondition: nil, sampleSeed: "0x2a",
                program: "steerlab-cli")
            Issue.record("expected a refusal")
        } catch let error as ExperimentError {
            #expect(error.reason.contains("did not shape"))
            #expect(
                error.malformedInvocation?.repairAction
                    .contains("--sample-per-condition") == true)
        } catch {
            Issue.record("wrong error: \(error)")
        }
    }

    @Test("Neither flag is the full corpus")
    func neitherFlagIsTheFullCorpus() throws {
        #expect(
            try EvaluateSubsample.resolveRequest(
                samplePerCondition: nil, sampleSeed: nil, program: "p") == nil)
        #expect(
            try EvaluateSubsample.resolveRequest(
                samplePerCondition: "", sampleSeed: "  ", program: "p") == nil)
    }

    @Test("Both flags resolve")
    func bothFlagsResolve() throws {
        let request = try #require(
            try EvaluateSubsample.resolveRequest(
                samplePerCondition: "2400", sampleSeed: "0x2a", program: "p"))
        #expect(request.samplePerCondition == 2400)
        #expect(request.seed == 42)
        #expect(request.seedText == "0x000000000000002a")
    }

    @Test(
        "A malformed seed refuses",
        arguments: ["not-a-seed", "-1", "0xfffffffffffffffff"])
    func aMalformedSeedRefuses(raw: String) {
        #expect(throws: ExperimentError.self) {
            _ = try EvaluateSubsample.resolveRequest(
                samplePerCondition: "1", sampleSeed: raw, program: "p")
        }
    }

    @Test("A seed spelled as a decimal means the decimal")
    func aDecimalSeedMeansTheDecimal() throws {
        #expect(try EvaluateSubsample.parseSeed("42", program: "p") == 42)
        #expect(try EvaluateSubsample.parseSeed("0x2a", program: "p") == 42)
        #expect(
            try EvaluateSubsample.parseSeed("abc123", program: "p") == 11_256_099)
    }

    @Test("A malformed sample size refuses", arguments: ["0", "-3", "two"])
    func aMalformedSampleSizeRefuses(raw: String) {
        #expect(throws: ExperimentError.self) {
            _ = try EvaluateSubsample.resolveRequest(
                samplePerCondition: raw, sampleSeed: "0x2a", program: "p")
        }
    }

    @Test("An over-ask refuses rather than clamping")
    func anOverAskRefusesRatherThanClamping() {
        do {
            _ = try EvaluateSubsample.selectedPositions(
                fixtureRecords(), request: .init(samplePerCondition: 13, seed: 1),
                program: "steerlab-cli")
            Issue.record("expected a refusal")
        } catch let error as ExperimentError {
            #expect(error.reason.contains("'baseline' has 12"))
            #expect(error.reason.contains("clamping"))
            #expect(
                error.malformedInvocation?.repairAction
                    .contains("--sample-per-condition 12 or less") == true)
        } catch {
            Issue.record("wrong error: \(error)")
        }
    }

    @Test("An over-ask names the binding stratum")
    func anOverAskNamesTheBindingStratum() {
        // The repair names the SMALLEST condition, because that is the number
        // the caller has to re-derive their design around — not whichever
        // condition happened to be enumerated first.
        let records = fixtureRecords().filter {
            !($0.condition == "injected" && $0.sampleIndex > 0)
        }
        do {
            _ = try EvaluateSubsample.selectedPositions(
                records, request: .init(samplePerCondition: 12, seed: 1),
                program: "p")
            Issue.record("expected a refusal")
        } catch let error as ExperimentError {
            #expect(error.reason.contains("'injected' has 3"))
            #expect(
                error.malformedInvocation?.repairAction
                    .contains("--sample-per-condition 3 or less") == true)
        } catch {
            Issue.record("wrong error: \(error)")
        }
    }

    @Test("The paired refusal names the unit of analysis")
    func thePairedRefusalNamesTheUnitOfAnalysis() {
        let refusal = EvaluateSubsample.pairedRefusal(program: "steerlab-cli")
        #expect(refusal.reason.contains("PAIR"))
        #expect(
            refusal.malformedInvocation?.repairAction
                .contains("perResponseCoding") == true)
    }

    // ---------------------------------------------------------- stamping

    @Test("The human phrase cannot be read as a full corpus")
    func theHumanPhraseCannotBeReadAsAFullCorpus() {
        let stamp = EvaluateSubsample.stamp(
            .init(samplePerCondition: 2400, seed: 42), sampled: 2400,
            source: 7200)
        #expect(
            EvaluateSubsample.codedPhrase(stamp, total: 7200)
                == "2400 of 7200 record(s) (seeded subsample)")
    }

    @Test("An absent block is the legacy full-corpus line")
    func anAbsentBlockIsTheLegacyFullCorpusLine() {
        #expect(
            EvaluateSubsample.codedPhrase(nil, total: 7200) == "7200 record(s)")
    }

    @Test("The stamp encodes exactly the five declared keys")
    func theStampEncodesExactlyTheFiveDeclaredKeys() throws {
        let stamp = EvaluateSubsample.stamp(
            .init(samplePerCondition: 3, seed: 7), sampled: 6, source: 24)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let object =
            try JSONSerialization.jsonObject(with: encoder.encode(stamp))
            as? [String: Any]
        #expect(
            Set(object?.keys ?? [:].keys)
                == [
                    "rule", "samplePerCondition", "sampleSeed",
                    "sampledRecords", "sourceRecords",
                ])
        #expect(Set(stamp.jsonObject.keys) == Set(object?.keys ?? [:].keys))
    }

    /// **Twin sentences.** Every refusal a researcher can hit reads
    /// identically on both engines, verbatim — the same complaint and the
    /// same runnable repair whether they typed `steerlab-cli` or
    /// `steerlab-server`. Pinned here and in
    /// `test_evaluate_subsample.LITERAL_REFUSALS`, so a reword on one side
    /// fails on the other.
    @Test("Every refusal reads identically on both engines")
    func everyRefusalReadsIdenticallyOnBothEngines() throws {
        func caught(_ body: () throws -> Void) throws -> (String, String) {
            do {
                try body()
                Issue.record("expected a refusal")
                return ("", "")
            } catch let error as ExperimentError {
                return (
                    error.reason,
                    error.malformedInvocation?.repairAction ?? "")
            }
        }
        let seedRepair =
            "steerlab-cli experiment evaluate <name> --sample-per-condition "
            + "<n> --sample-seed 0x5eed0a5e5eed0a5e  (any 64-bit value; "
            + "record it in the preregistration — the same seed always draws "
            + "the same records)"

        var refusal = try caught {
            _ = try EvaluateSubsample.resolveRequest(
                samplePerCondition: "2400", sampleSeed: nil,
                program: "steerlab-cli")
        }
        #expect(
            refusal.0
                == "--sample-per-condition 2400 was given without "
                    + "--sample-seed: a subsample nobody can redraw is not "
                    + "evidence, so the draw refuses rather than choosing a "
                    + "seed for you")
        #expect(refusal.1 == seedRepair)

        refusal = try caught {
            _ = try EvaluateSubsample.resolveRequest(
                samplePerCondition: nil, sampleSeed: "0x2a",
                program: "steerlab-cli")
        }
        #expect(
            refusal.0
                == "--sample-seed 0x2a was given without "
                    + "--sample-per-condition: with no sample size the full "
                    + "corpus is coded, and the seed would be stamped on a "
                    + "coding it did not shape")
        #expect(
            refusal.1
                == "add --sample-per-condition <n> to draw a subsample, or "
                    + "drop --sample-seed and run steerlab-cli experiment "
                    + "evaluate <name> to code the full corpus")

        refusal = try caught {
            _ = try EvaluateSubsample.resolveRequest(
                samplePerCondition: "x", sampleSeed: "0x2a",
                program: "steerlab-cli")
        }
        #expect(
            refusal.0
                == "--sample-per-condition must be a whole number of records "
                    + "of at least 1, not 'x' — a subsample of zero records "
                    + "is a design nobody can report")
        #expect(
            refusal.1
                == "steerlab-cli experiment evaluate <name> "
                    + "--sample-per-condition 2400 --sample-seed <seed>, or "
                    + "drop both flags to code the full corpus")

        refusal = try caught {
            _ = try EvaluateSubsample.resolveRequest(
                samplePerCondition: "1", sampleSeed: "zz",
                program: "steerlab-cli")
        }
        #expect(
            refusal.0
                == "--sample-seed 'zz' is not a 64-bit unsigned number — a "
                    + "seed is a decimal integer, or hexadecimal with or "
                    + "without a '0x' prefix, of at most 16 hex digits (the "
                    + "leading 16 of a digest are a fine seed, written down "
                    + "as such)")
        #expect(refusal.1 == seedRepair)

        let two = (0..<2).map {
            EvaluateSubsample.Coordinate(
                condition: "baseline", promptID: "p", sampleIndex: UInt64($0))
        }
        refusal = try caught {
            _ = try EvaluateSubsample.selectedPositions(
                two, request: .init(samplePerCondition: 5, seed: 1),
                program: "steerlab-cli")
        }
        #expect(
            refusal.0
                == "--sample-per-condition 5 exceeds what the source run "
                    + "holds: 'baseline' has 2 codeable record(s). A "
                    + "subsample cannot be larger than the stratum it is "
                    + "drawn from, and clamping it would code a smaller "
                    + "design than the one that was preregistered while "
                    + "every stamp still said 5")
        #expect(
            refusal.1
                == "re-run with --sample-per-condition 2 or less (condition "
                    + "'baseline' is the binding stratum), or drop both "
                    + "sample flags and run steerlab-cli experiment evaluate "
                    + "<name> to code all 2 record(s)")

        let paired = EvaluateSubsample.pairedRefusal(program: "steerlab-cli")
        #expect(
            paired.reason
                == "--sample-per-condition/--sample-seed apply to the "
                    + "per-response coding instrument only: this study's "
                    + "pinned rubric is a paired comparison, whose unit is a "
                    + "(baseline, variant) PAIR rather than a record, so a "
                    + "per-condition record count does not name a set of "
                    + "pairs to judge")
        #expect(
            paired.malformedInvocation?.repairAction
                == "drop both sample flags and run steerlab-cli experiment "
                    + "evaluate <name> to judge every pair, or pin a "
                    + "perResponseCoding rubric if per-record coding is the "
                    + "design")
    }

    @Test("The rule string is the pinned cross-engine derivation")
    func theRuleStringIsThePinnedCrossEngineDerivation() {
        // The rule is stamped VERBATIM into every sampled report, and its
        // whole purpose is that a reader can recompute the subsample's
        // membership from it — so its wording is part of the contract, not
        // documentation that may drift, and the two engines must stamp the
        // same bytes. Pinned by digest rather than by a second copy of the
        // prose: a duplicated paragraph is a paragraph that silently rots on
        // one side. Twin literal: `test_evaluate_subsample.LITERAL_RULE_SHA256`.
        #expect(
            SHA256.hash(data: Data(EvaluateSubsample.rule.utf8))
                .map { String(format: "%02x", $0) }.joined()
                == literalRuleSHA256)
        #expect(
            EvaluateSubsample.rule.hasPrefix("stratifiedByPromptID/v1 — "))
        #expect(EvaluateSubsample.rule.contains("SplitMix64"))
        #expect(EvaluateSubsample.rule.contains("UTF-8 bytes"))
        #expect(!EvaluateSubsample.rule.contains("\n"))
    }
}
