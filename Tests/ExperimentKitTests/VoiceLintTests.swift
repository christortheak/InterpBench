import Foundation
import Testing

@testable import ExperimentKit

/// Voice lint for multi-agent turns (spec §5, 2026-08-17).
///
/// A panel turn is supposed to be ONE participant's document; the measured
/// runs show three ways that fails (colleague signature blocks, whole-panel
/// screenplay turns, third-person self-reference). The lint records them and
/// never blocks: the failure rate is condition-correlated, so regenerating a
/// noncompliant turn would select on the dependent variable.
///
/// The load-bearing test is `fixtureCasesReplayExactly`: the goldens in
/// `prompts/fixtures/voice-lint/cases.jsonl` are verbatim excerpts of the real
/// failing transcripts, read by BOTH engines, so a divergence from
/// `Server/steerlab_server/experiment/voice_lint.py` fails on both sides.
///
/// Server twin: `Server/tests/test_voice_lint.py`.
@Suite("VoiceLintTests")
struct VoiceLintTests {

    /// Repo root derived from this file's compile-time path (the same anchor
    /// `TurnEndpointTests` uses).
    private static var fixtureURL: URL {
        URL(filePath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appending(components: "prompts", "fixtures", "voice-lint", "cases.jsonl")
    }

    private struct GoldenCase: Decodable {
        struct Expectation: Decodable {
            let version: Int
            let speaksForOthers: Bool
            let otherSpeakerLines: [String: Int]?
            let thirdPersonSelf: Int
        }
        let name: String
        let speakerName: String
        let otherNames: [String]
        let output: String
        let expected: Expectation
    }

    private let panel = ["Judge Whitfield", "Judge Marsden", "Judge Calloway"]

    private func lint(
        _ text: String, speaker: String = "Judge Marsden", others: [String]? = nil
    ) -> VoiceLintStamp {
        VoiceLint.stamp(
            in: text, speaker: speaker,
            others: others ?? panel.filter { $0 != speaker })
    }

    // MARK: the committed goldens

    @Test func fixtureCasesReplayExactly() throws {
        let text = try String(contentsOf: Self.fixtureURL, encoding: .utf8)
        let decoder = JSONDecoder()
        let cases = try text.split(separator: "\n", omittingEmptySubsequences: true)
            .map { try decoder.decode(GoldenCase.self, from: Data($0.utf8)) }
        #expect(cases.count >= 6, "the fixture is the calibration evidence")
        #expect(Set(cases.map(\.name)).count == cases.count)
        for golden in cases {
            let stamp = VoiceLint.stamp(
                in: golden.output, speaker: golden.speakerName,
                others: golden.otherNames)
            #expect(stamp.version == golden.expected.version, "\(golden.name)")
            #expect(
                stamp.speaksForOthers == golden.expected.speaksForOthers,
                "\(golden.name)")
            // Key PRESENCE is part of the contract: an absent
            // otherSpeakerLines is how a clean turn is written.
            #expect(
                stamp.otherSpeakerLines == golden.expected.otherSpeakerLines,
                "\(golden.name)")
            #expect(
                stamp.thirdPersonSelf == golden.expected.thirdPersonSelf,
                "\(golden.name)")
        }
    }

    // MARK: colleague signature blocks

    @Test func bareSurnameWithACommaIsASignatureBlock() {
        let stamp = lint("**WHITFIELD, Judge,** concurring. I join in full.")
        #expect(stamp.speaksForOthers)
        #expect(stamp.otherSpeakerLines == ["Judge Whitfield": 1])
    }

    /// The role half is not vocabulary — "J.", "Circuit Judge" and nothing at
    /// all all sign, because the rule keys on the bare surname.
    @Test func signatureMatchesAnyCaseAndAnyRoleText() {
        for line in [
            "**MARSDEN, J.,** concurring.",
            "MARSDEN, Circuit Judge, concurring in the judgment.",
            "Marsden, J., concurring.",
            "marsden.",
        ] {
            let stamp = lint(line, speaker: "Judge Whitfield")
            #expect(stamp.otherSpeakerLines == ["Judge Marsden": 1], "\(line)")
        }
    }

    /// The discrimination the rule set turns on: across 600 records the
    /// transcripts address a colleague by FULL name and sign by bare surname.
    @Test func fullNameAndCommaIsAddressNotASignature() {
        let stamp = lint("Judge Whitfield, I wholeheartedly agree with your analysis.")
        #expect(!stamp.speaksForOthers)
        #expect(stamp.otherSpeakerLines == nil)
    }

    @Test func colleagueNamedMidLineIsDiscussion() {
        #expect(!lint("I agree with Judge Whitfield, and with Judge Calloway too.")
            .speaksForOthers)
    }

    @Test func proseBeginningWithAColleaguesNameIsNotASignature() {
        let stamp = lint(
            "Judge Calloway raises a compelling point about timing.\n"
                + "Whitfield’s argument runs the other way.")
        #expect(!stamp.speaksForOthers)
    }

    @Test func panelCaptionIsNotASignature() {
        #expect(
            !lint("Before: MARSDEN, Judge; WHITFIELD, Judge; and CALLOWAY, Judge.")
                .speaksForOthers)
    }

    @Test func theSpeakersOwnSignatureIsNeverSpeakingForOthers() {
        let stamp = lint(
            "**MARSDEN, Judge,** writing for the Court.\nMarsden, J., concurring.")
        #expect(!stamp.speaksForOthers)
    }

    @Test func linesAreCountedPerParticipant() {
        let stamp = lint(
            "**WHITFIELD, J.,** concurring.\n**WHITFIELD, J.,** dissenting.\n"
                + "**CALLOWAY, J.,** concurring.")
        #expect(
            stamp.otherSpeakerLines == ["Judge Whitfield": 2, "Judge Calloway": 1])
    }

    // MARK: screenplay labels and stage directions

    @Test func bracketedSpeakerLabelIsAStageDirection() {
        let stamp = VoiceLint.stamp(
            in: "**(Judge A, as presiding judge):** Good morning.\n"
                + "**(Judge B):** Certainly. My initial vote is to affirm.",
            speaker: "Judge C", others: ["Judge A", "Judge B"])
        #expect(stamp.otherSpeakerLines == ["Judge A": 1, "Judge B": 1])
    }

    @Test func colonAfterAColleaguesNameIsASpeakerLabel() {
        let stamp = VoiceLint.stamp(
            in: "**Judge B:** I would affirm.",
            speaker: "Judge C", others: ["Judge A", "Judge B"])
        #expect(stamp.otherSpeakerLines == ["Judge B": 1])
    }

    /// A disposition package reporting each seat's vote is a legitimate turn
    /// type; without this exemption ten of ten such turns flagged.
    @Test func aMarkdownListItemIsARollCallNotAVoice() {
        let stamp = VoiceLint.stamp(
            in: """
                **I. JUDICIAL VOTES:**

                *   **Judge A:** Affirm
                *   **Judge B:** Affirm
                1. Judge B: Affirm
                """,
            speaker: "Judge A", others: ["Judge B", "Judge C"])
        #expect(!stamp.speaksForOthers)
    }

    /// "Judge A" must not make every line starting with "A" a signature.
    @Test func aShortSurnameNeverBecomesABareForm() {
        let stamp = VoiceLint.stamp(
            in: "A, the appellant, argues otherwise.\nA. The statute is clear.",
            speaker: "Judge C", others: ["Judge A", "Judge B"])
        #expect(!stamp.speaksForOthers)
    }

    @Test func longestFormWinsWhenNamesOverlap() {
        let stamp = VoiceLint.stamp(
            in: "Judge Marsden: I would reverse.",
            speaker: "Judge Whitfield", others: ["Judge Marsden", "Marsden Clerk"])
        #expect(stamp.otherSpeakerLines == ["Judge Marsden": 1])
    }

    // MARK: third-person self

    @Test func thirdPersonSelfCountsProseMentions() {
        let stamp = lint(
            "Despite Judge Marsden’s compelling arguments, the statute controls. "
                + "Judge Marsden would reverse.")
        #expect(stamp.thirdPersonSelf == 2)
    }

    @Test func thirdPersonSelfSkipsHeadersLabelsAndFirstPerson() {
        let text = """
            From: Judge Marsden
            **Opinion by:** Judge Marsden
            ## Judge Marsden — Private Notes
            **MARSDEN, Judge,** writing for the Court.
            I, Judge Marsden, concur.
            Okay, I am Judge Marsden. I have reviewed the record.
            As Judge Marsden, I write separately.
            You are Judge Marsden of this court.
            """
        #expect(lint(text).thirdPersonSelf == 0)
    }

    /// A bare surname is how a seat SIGNS; counting it would turn every own
    /// signature into a self-reference.
    @Test func thirdPersonSelfUsesTheFullNameOnly() {
        #expect(lint("Marsden, J., concurring.").thirdPersonSelf == 0)
    }

    @Test func thirdPersonSelfNeedsAWholeWord() {
        #expect(lint("Judge Marsdenite wrote otherwise.").thirdPersonSelf == 0)
    }

    @Test func thirdPersonSelfIsCaseInsensitive() {
        #expect(lint("The concerns raised by JUDGE MARSDEN stand.").thirdPersonSelf == 1)
    }

    /// Only the mention that OPENS the line is the signature; the prose after
    /// it on the same line still counts. This is the real shape of the
    /// documented consciousness-run failure.
    @Test func thirdPersonSelfCountsAfterALineInitialOwnSignature() {
        let stamp = lint(
            "**(Judge Marsden, presiding):** Good morning. Judge Marsden has a "
                + "scale position of 2.")
        #expect(stamp.thirdPersonSelf == 1)
    }

    // MARK: the stamp shape

    @Test func stampOmitsOtherSpeakerLinesWhenClean() throws {
        let stamp = lint("I would affirm.")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let json = String(decoding: try encoder.encode(stamp), as: UTF8.self)
        #expect(json == #"{"speaksForOthers":false,"thirdPersonSelf":0,"version":1}"#)
    }

    @Test func stampEncodesLinesWhenPresent() throws {
        let stamp = lint("**WHITFIELD, J.,** concurring.")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let json = String(decoding: try encoder.encode(stamp), as: UTF8.self)
        #expect(
            json == #"{"otherSpeakerLines":{"Judge Whitfield":1},"#
                + #""speaksForOthers":true,"thirdPersonSelf":0,"version":1}"#)
        #expect(stamp.version == VoiceLint.version)
    }

    @Test func aSoloScenarioHasNoColleaguesToSpeakFor() {
        let stamp = VoiceLint.stamp(
            in: "**WHITFIELD, J.,** concurring.", speaker: "Judge Whitfield",
            others: [])
        #expect(!stamp.speaksForOthers)
        #expect(stamp.thirdPersonSelf == 0)
    }

    @Test func emptyOutputIsCleanNotAnError() {
        let stamp = VoiceLint.stamp(in: "", speaker: "Judge A", others: ["Judge B"])
        #expect(!stamp.speaksForOthers)
        #expect(stamp.thirdPersonSelf == 0)
    }

    // MARK: the turn record

    /// The stamp is optional on the record so pre-lint turns.jsonl still
    /// decodes — absent means "written before the lint existed", never
    /// "clean".
    @Test func oldTurnRecordsStillDecode() throws {
        let line = """
            {"turnID":"t1","turnIndex":1,"title":"Opinion","speakerAgentID":"a",
            "speakerName":"Judge Marsden","prompt":"p","output":"o",
            "outputLabel":"o1","routedAgentIDs":[]}
            """.replacingOccurrences(of: "\n", with: "")
        let turn = try JSONDecoder().decode(
            MultiAgentTurnResult.self, from: Data(line.utf8))
        #expect(turn.voiceLint == nil)
    }

    /// The seam the run loop uses, exercised without a model — the same
    /// contract `endpointStamp` gets.
    @Test func theRunLoopSeamStampsAgainstTheScenariosAgents() {
        let agents = [
            MultiAgentScenario.Agent(id: "a", name: "Judge Marsden", baseModelID: "m"),
            MultiAgentScenario.Agent(id: "b", name: "Judge Whitfield", baseModelID: "m"),
        ]
        let turn = MultiAgentScenario.Turn(
            id: "t1", title: "Opinion", speakerAgentID: "a",
            promptTemplate: "Write.", outputLabel: "o1")
        let stamp = MultiAgentRunner.voiceLintStamp(
            for: turn,
            output: "**WHITFIELD, Judge,** concurring. I join Judge Marsden’s opinion.",
            agents: agents)
        #expect(stamp.otherSpeakerLines == ["Judge Whitfield": 1])
        #expect(stamp.thirdPersonSelf == 1)
    }
}
