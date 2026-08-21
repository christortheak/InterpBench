import Foundation
import Testing
@testable import ExperimentKit

/// Mirrors `Server/tests/test_judicial.py` fixture-for-fixture — cross-engine
/// parity of the judicial endpoints rests on both substrates agreeing on
/// these exact values.
@Suite struct JudicialTests {

    private let options = ["Apply Kansas law", "Apply Nebraska law"]

    private func isClose(_ a: Double, _ b: Double, tolerance: Double = 1e-9) -> Bool {
        abs(a - b) <= tolerance * max(1, max(abs(a), abs(b)))
    }

    @Test func parseChoiceFromFencedJSON() {
        let text = "Reasoning...\n```json\n{\"answer\": \"Apply Kansas law\"}\n```"
        #expect(Judicial.parseChoice(text, options: options) == "Apply Kansas law")
    }

    @Test func parseChoiceFromBareJSONAndNormalization() {
        let text = "Conclusion: {\"answer\": \"apply nebraska law.\"} That is my ruling."
        #expect(Judicial.parseChoice(text, options: options) == "Apply Nebraska law")
    }

    @Test func parseChoiceEarliestMentionFallback() {
        let text = "I would apply Nebraska law, though Apply Kansas law was argued."
        #expect(Judicial.parseChoice(text, options: options) == "Apply Nebraska law")
    }

    @Test func parseChoiceFailureReturnsNil() {
        #expect(Judicial.parseChoice("The court declines to answer.", options: options) == nil)
        #expect(Judicial.parseChoice("", options: options) == nil)
    }

    @Test func parseMonthsSinglesYearsAndDecimals() {
        #expect(Judicial.parseMonths("I sentence the defendant to 18 months.") == 18.0)
        #expect(Judicial.parseMonths("A term of 2 years is appropriate.") == 24.0)
        #expect(Judicial.parseMonths("1.5 years of imprisonment") == 18.0)
        #expect(Judicial.parseMonths("una pena de 1,5 years") == 18.0)
    }

    @Test func parseMonthsRangeTakesMidpoint() {
        #expect(Judicial.parseMonths("a sentence of 18 to 24 months") == 21.0)
        #expect(Judicial.parseMonths("between 2-4 years in prison") == 36.0)
    }

    @Test func parseMonthsCompoundNeverDropsATerm() {
        // Cross-engine contract fixtures (twin of `parse_months`):
        // "X years Y months" must yield X*12 + Y, never just the first quantity.
        #expect(Judicial.parseMonths("8 years 3 months") == 99.0)
        #expect(Judicial.parseMonths("8 years and 3 months") == 99.0)
        #expect(Judicial.parseMonths("sentenced to 8 years 3 months in prison") == 99.0)
    }

    @Test func parseMonthsGermanUnits() {
        // German units carry identical semantics.
        #expect(Judicial.parseMonths("18 Monate") == 18.0)
        #expect(Judicial.parseMonths("2 Jahre 6 Monate") == 30.0)
        #expect(Judicial.parseMonths("eine Freiheitsstrafe von 1 Jahr") == 12.0)
        #expect(Judicial.parseMonths("2 Jahre und 6 Monate") == 30.0)
    }

    @Test func parseMonthsSingleFixtures() {
        // Locked cross-engine fixtures: single quantities.
        #expect(Judicial.parseMonths("3 months") == 3.0)
        #expect(Judicial.parseMonths("2 years") == 24.0)
    }

    @Test func parseMonthsJudicialYearsRegisterNumberWords() {
        // EXACT phrasings from a sentencing-anchoring cluster run
        // (20260810T194538745, maxTokens 512): the judicial years register
        // spells the numbers out, with markdown bold and curly apostrophes —
        // 58/1320 records unparsed, dose-dependently (steering pushes the
        // formal register). FIXTURE LIST SHARED WITH THE SERVER TWIN.
        #expect(
            Judicial.parseMonths(
                "I sentence the defendant, A, to **ten years and six months’** imprisonment")
                == 126.0)
        #expect(
            Judicial.parseMonths(
                "I hereby sentence the defendant, A, to **seven years and six months’ imprisonment**")
                == 90.0)
        #expect(
            Judicial.parseMonths(
                "Therefore, I sentence A to **six years and six months** imprisonment")
                == 78.0)
        #expect(Judicial.parseMonths("to **Five years and six months’ imprisonment**") == 66.0)
        #expect(Judicial.parseMonths("a term of ten years") == 120.0)
        #expect(Judicial.parseMonths("Ten Years And Six Months") == 126.0)
        #expect(Judicial.parseMonths("twelve months") == 12.0)
        // Digit and word terms mix freely.
        #expect(Judicial.parseMonths("ten years and 6 months") == 126.0)
        #expect(Judicial.parseMonths("10 years and six months") == 126.0)
    }

    @Test func parseMonthsNumberWordVocabularyStopsAtTwelve() {
        // One through twelve only — anything larger stays unparsed, never
        // guessed (unparsed is a counted coherence signal).
        #expect(Judicial.parseMonths("thirteen years") == nil)
        #expect(Judicial.parseMonths("twenty years") == nil)
    }

    @Test func parseMonthsNumberWordsRespectWordBoundaries() {
        // "sentenced" contains "ten"; "brighten" ends in "ten" — neither is
        // a number.
        #expect(Judicial.parseMonths("the defendant was sentenced") == nil)
        #expect(Judicial.parseMonths("brighten years of effort") == nil)
    }

    @Test func parseMonthsWordCompoundStillBeatsEarlierSingle() {
        // Precedence unchanged: a compound anywhere outranks an earlier
        // single-term mention, exactly as for digits — so a spelled-out
        // statutory minimum does not hijack the actual sentence.
        #expect(
            Judicial.parseMonths(
                "Considering the statutory minimum of five years under §212, "
                    + "I sentence the defendant to **seven years and six months’** imprisonment.")
                == 90.0)
    }

    @Test func parseMonthsFailure() {
        #expect(Judicial.parseMonths("The defendant is guilty.") == nil)
        #expect(Judicial.parseMonths("") == nil)
        // Unparseable stays unparseable — never a partial or guessed number.
        #expect(Judicial.parseMonths("a lengthy custodial term") == nil)
    }

    @Test func parseFailureRate() {
        #expect(Judicial.parseFailureRate([12.0, nil, 24.0, nil]) == 0.5)
        #expect(Judicial.parseFailureRate([]) == 0.0)
    }

    @Test func outcomeRateIgnoresParseFailures() {
        let choices: [String?] = ["A", "B", nil, "A"]
        #expect(isClose(Judicial.outcomeRate(choices, target: "A"), 2.0 / 3.0))
        #expect(Judicial.outcomeRate([nil, nil], target: "A").isNaN)
    }

    @Test func sympathyGapAndInteraction() {
        #expect(Judicial.sympathyGap(standardArmDelta: 0.30, ruleArmDelta: 0.05) == 0.25)
        let diffs = Judicial.ruleVsStandardInteraction(
            ruleBaseline: [0.5, 0.5], ruleTreated: [0.55, 0.6],
            standardBaseline: [0.5, 0.5], standardTreated: [0.8, 0.9])
        #expect(diffs == [(0.8 - 0.5) - (0.55 - 0.5), (0.9 - 0.5) - (0.6 - 0.5)])
    }

    @Test func anchorSlopeExactLine() {
        let anchors: [Double?] = [3.0, 9.0, 12.0]
        let sentences: [Double?] = [10.0, 22.0, 28.0]  // slope 2, intercept 4
        #expect(isClose(Judicial.anchorSlope(anchors: anchors, sentences: sentences), 2.0))
        #expect(Judicial.anchorSlope(anchors: [5.0], sentences: [10.0]).isNaN)
        #expect(Judicial.anchorSlope(anchors: [5.0, 5.0], sentences: [10.0, 12.0]).isNaN)
    }

    @Test func proportionalityPerfectAndDegenerate() {
        #expect(
            isClose(
                Judicial.proportionality(severities: [1, 2, 3], sentences: [10, 20, 30]),
                1.0))
        #expect(
            isClose(
                Judicial.proportionality(severities: [1, 2, 3], sentences: [30, 20, 10]),
                -1.0))
        #expect(Judicial.proportionality(severities: [1, 1], sentences: [10, 20]).isNaN)
    }

    @Test func summarizeQuantilesAndSpread() throws {
        let summary = try #require(Judicial.summarize([12.0, 24.0, nil, 18.0, 30.0]))
        #expect(summary.count == 4)
        #expect(summary.mean == 21.0)
        #expect(summary.min == 12.0 && summary.max == 30.0)
        #expect(summary.median == 21.0)
        #expect(isClose(summary.stdev, 7.745966692414834))
        #expect(summary.q25 == 16.5)
        #expect(Judicial.summarize([nil, nil]) == nil)
    }
}
