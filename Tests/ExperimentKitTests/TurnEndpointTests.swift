import Foundation
import Testing

@testable import ExperimentKit

/// Declared turn endpoints (Wave-2 contract, 2026-08-05).
///
/// A panel turn may declare the quantity it is supposed to produce; the runner
/// parses the generated text at write time and stamps the parse onto the turn
/// record. The parse itself is a literal, case-insensitive
/// marker-then-80-characters scan with NO regex, byte-twinned with
/// `Server/steerlab_server/experiment/turn_endpoint.py` — the goldens below
/// are the committed fixture BOTH engines read
/// (`prompts/fixtures/panel-endpoints/`), so a divergence fails on both sides.
///
/// The refusals matter as much as the parses: the scenario is pinned,
/// reviewed data, and a typo'd declaration that silently parsed nothing would
/// be indistinguishable from a panel that never answered.
///
/// Server twin: `Server/tests/test_turn_endpoint.py`.
@Suite struct TurnEndpointTests {

    /// Repo root derived from this file's compile-time path (the same anchor
    /// the coding-judge goldens use).
    private static var fixtureDirectory: URL {
        URL(filePath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appending(components: "prompts", "fixtures", "panel-endpoints")
    }

    private struct GoldenCase: Decodable {
        struct Expectation: Decodable {
            let unparsed: Bool
            let choice: String?
            let number: Double?
        }
        let name: String
        let endpoint: String
        let text: String
        let expected: Expectation
    }

    private struct GoldenCases: Decodable {
        let cases: [GoldenCase]
    }

    private func fixtureScenario() throws -> MultiAgentScenario {
        try JSONDecoder().decode(
            MultiAgentScenario.self,
            from: try Data(
                contentsOf: Self.fixtureDirectory.appending(component: "scenario.json")))
    }

    private func fixtureEndpoints() throws -> [String: TurnEndpoint] {
        Dictionary(
            uniqueKeysWithValues: try fixtureScenario().turns.compactMap { turn in
                turn.endpoint.map { ($0.name, $0) }
            })
    }

    // MARK: the committed goldens

    @Test func fixtureScenarioDeclaresBothKinds() throws {
        let endpoints = try fixtureEndpoints()
        let vote = try #require(endpoints["vote"])
        #expect(vote.kind == .choice)
        #expect(vote.vocabulary == ["affirm", "reverse", "vacate", "remand"])
        let months = try #require(endpoints["months"])
        #expect(months.kind == .number)
        #expect(months.min == 0)
        #expect(months.max == 600)
    }

    @Test func parserMatchesTheCommittedGoldens() throws {
        let endpoints = try fixtureEndpoints()
        let golden = try JSONDecoder().decode(
            GoldenCases.self,
            from: try Data(
                contentsOf: Self.fixtureDirectory.appending(component: "cases.json")))
        #expect(!golden.cases.isEmpty)
        for testCase in golden.cases {
            let endpoint = try #require(endpoints[testCase.endpoint])
            let parsed = TurnEndpointParser.parse(endpoint, in: testCase.text)
            if testCase.expected.unparsed {
                #expect(
                    parsed == nil,
                    """
                    '\(testCase.name)' must be recorded as unparsed, never \
                    guessed — got \(String(describing: parsed))
                    """)
            } else if let choice = testCase.expected.choice {
                #expect(
                    parsed == .choice(choice),
                    "'\(testCase.name)': got \(String(describing: parsed))")
            } else {
                let number = try #require(testCase.expected.number)
                #expect(
                    parsed == .number(number),
                    "'\(testCase.name)': got \(String(describing: parsed))")
            }
        }
    }

    /// The parses are byte-twinned with the server's; if this drifts, change
    /// the fixture deliberately on BOTH engines.
    @Test func choiceValueIsTheDeclaredMemberNotTheMatchedText() throws {
        let vote = try #require(try fixtureEndpoints()["vote"])
        #expect(TurnEndpointParser.parse(vote, in: "Vote: AFFIRM") == .choice("affirm"))
    }

    @Test func wholeWordBoundaryIsLetterAdjacencyOnly() throws {
        let vote = try #require(try fixtureEndpoints()["vote"])
        // Punctuation and digits are not letters, so these are whole words.
        #expect(TurnEndpointParser.parse(vote, in: "Vote: (affirm)") == .choice("affirm"))
        #expect(TurnEndpointParser.parse(vote, in: "Vote: affirm/reverse") == .choice("affirm"))
        // Letters on either side are not.
        #expect(TurnEndpointParser.parse(vote, in: "Vote: xaffirm") == nil)
        #expect(TurnEndpointParser.parse(vote, in: "Vote: affirmx") == nil)
    }

    @Test func theWindowIsExactlyEightyCharacters() throws {
        let vote = try #require(try fixtureEndpoints()["vote"])
        // "Vote:" ends at index 5; the window is [5, 85). "affirm" must END
        // by 85.
        let fits = "Vote:" + String(repeating: " ", count: 85 - 5 - 6) + "affirm"
        #expect(TurnEndpointParser.parse(vote, in: fits) == .choice("affirm"))
        #expect(TurnEndpointParser.parse(vote, in: "Vote: " + fits.dropFirst(5)) == nil)
    }

    @Test func boundsAreInclusive() throws {
        let months = try #require(try fixtureEndpoints()["months"])
        #expect(TurnEndpointParser.parse(months, in: "Sentence: 0") == .number(0))
        #expect(TurnEndpointParser.parse(months, in: "Sentence: 600") == .number(600))
        #expect(TurnEndpointParser.parse(months, in: "Sentence: 600.5") == nil)
    }

    /// The FIRST number is the answer; a scan that kept looking would report
    /// the model's most agreeable number rather than its actual one.
    @Test func anOutOfRangeFirstNumberDoesNotFallThroughToASecond() throws {
        let months = try #require(try fixtureEndpoints()["months"])
        #expect(
            TurnEndpointParser.parse(months, in: "Sentence: 900, or 24 at the least")
                == nil)
    }

    // MARK: declaration validation

    @Test(arguments: [
        (#"{"name": "v", "kind": "ordinal", "marker": "V:"}"#, "unknown kind"),
        (#"{"name": "v", "kind": "choice", "marker": " ", "vocabulary": ["a"]}"#,
            "non-empty marker"),
        (#"{"name": " ", "kind": "choice", "marker": "V:", "vocabulary": ["a"]}"#,
            "needs a name"),
        (#"{"name": "v", "kind": "choice", "marker": "V:"}"#, "non-empty vocabulary"),
        (#"{"name": "v", "kind": "choice", "marker": "V:", "vocabulary": []}"#,
            "non-empty vocabulary"),
        (#"{"name": "v", "kind": "choice", "marker": "V:", "vocabulary": ["a", " "]}"#,
            "empty vocabulary member"),
        (
            #"{"name": "v", "kind": "choice", "marker": "V:", "vocabulary": ["a"], "min": 0}"#,
            "declares min/max"
        ),
        (#"{"name": "n", "kind": "number", "marker": "N:", "vocabulary": ["a"]}"#,
            "declares a vocabulary"),
        (#"{"name": "n", "kind": "number", "marker": "N:", "min": 10, "max": 1}"#,
            "min > max"),
    ])
    func aMalformedDeclarationRefusesLoudly(json: String, fragment: String) throws {
        do {
            _ = try JSONDecoder().decode(TurnEndpoint.self, from: Data(json.utf8))
            Issue.record("expected a refusal for \(json)")
        } catch let error as ExperimentError {
            #expect(
                error.reason.contains(fragment),
                "expected '\(fragment)' in: \(error.reason)")
        }
    }

    /// Loading is where it must fail: the scenario is pinned data, and a
    /// declaration that parses nothing must never reach a measured run.
    @Test func aMalformedDeclarationRefusesTheWholeScenario() throws {
        let json = """
            {"name": "p", "baseModelID": "m",
             "agents": [{"id": "a", "name": "A", "baseModelID": "m", "systemPrompt": ""}],
             "turns": [{"id": "t", "title": "T", "speakerAgentID": "a",
                        "promptTemplate": "go", "outputLabel": "", "routing": "all",
                        "routedAgentIDs": [], "includeScenarioMaterials": true,
                        "includeSpeakerContext": true,
                        "endpoint": {"name": "vote", "kind": "chioce",
                                     "marker": "Vote:", "vocabulary": ["affirm"]}}]}
            """
        do {
            _ = try JSONDecoder().decode(MultiAgentScenario.self, from: Data(json.utf8))
            Issue.record("expected a refusal")
        } catch let error as ExperimentError {
            #expect(error.reason.contains("unknown kind"))
        }
    }

    /// A scenario built in code never passed through the validating decoder.
    @Test func validateReChecksAProgrammaticallyBuiltDeclaration() throws {
        let scenario = MultiAgentScenario(
            name: "p", baseModelID: "m",
            agents: [.init(id: "a", name: "A", baseModelID: "m")],
            turns: [
                .init(
                    id: "t", title: "T", speakerAgentID: "a", promptTemplate: "go",
                    endpoint: TurnEndpoint(
                        name: "v", kind: .choice, marker: "V:", vocabulary: []))
            ])
        do {
            try MultiAgentRunner.validate(scenario)
            Issue.record("expected a refusal")
        } catch let error as ExperimentError {
            #expect(error.reason.contains("non-empty vocabulary"))
            #expect(error.reason.contains("turn 'T'"))
        }
    }

    /// Unknown keys are tolerated (they always were) and the declaration
    /// survives a decode/encode round trip.
    @Test func unknownKeysAreToleratedAndTheDeclarationSurvives() throws {
        let json = """
            {"name": "p", "baseModelID": "m", "createdAt": "2026-01-01T00:00:00Z",
             "agents": [{"id": "a", "name": "A", "baseModelID": "m",
                         "systemPrompt": "", "unknownSeatKey": 1}],
             "turns": [{"id": "t", "title": "T", "speakerAgentID": "a",
                        "promptTemplate": "go", "outputLabel": "", "routing": "all",
                        "routedAgentIDs": [], "includeScenarioMaterials": true,
                        "includeSpeakerContext": true, "unknownTurnKey": true,
                        "endpoint": {"name": "vote", "kind": "choice",
                                     "marker": "Vote:",
                                     "vocabulary": ["affirm", "reverse"]}}]}
            """
        let scenario = try JSONDecoder().decode(
            MultiAgentScenario.self, from: Data(json.utf8))
        #expect(scenario.turns[0].endpoint?.marker == "Vote:")
        let reEncoded = try JSONSerialization.jsonObject(
            with: try JSONEncoder().encode(scenario)) as? [String: Any]
        let turns = try #require(reEncoded?["turns"] as? [[String: Any]])
        let endpoint = try #require(turns[0]["endpoint"] as? [String: Any])
        #expect(endpoint["kind"] as? String == "choice")
        #expect(endpoint["vocabulary"] as? [String] == ["affirm", "reverse"])
    }

    /// Additive and optional: panels authored before the contract encode byte
    /// for byte as before.
    @Test func aTurnWithoutADeclarationEmitsNoEndpointKey() throws {
        let scenario = MultiAgentScenario(
            name: "p", baseModelID: "m",
            agents: [.init(id: "a", name: "A", baseModelID: "m")],
            turns: [.init(id: "t", title: "T", speakerAgentID: "a", promptTemplate: "go")])
        let encoded = try JSONSerialization.jsonObject(
            with: try JSONEncoder().encode(scenario)) as? [String: Any]
        let turns = try #require(encoded?["turns"] as? [[String: Any]])
        #expect(turns[0]["endpoint"] == nil)
    }

    // MARK: the runner stamp

    /// The seam the run loop uses, exercised without a model.
    @Test func theRunnerStampsADeclaredTurnAndSkipsAnUndeclaredOne() throws {
        let vote = try #require(try fixtureEndpoints()["vote"])
        let declared = MultiAgentScenario.Turn(
            id: "t", title: "T", speakerAgentID: "a", promptTemplate: "go",
            endpoint: vote)
        let undeclared = MultiAgentScenario.Turn(
            id: "u", title: "U", speakerAgentID: "a", promptTemplate: "go")

        let parsed = try #require(
            MultiAgentRunner.endpointStamp(
                for: declared, output: "Vote: affirm — the judgment stands."))
        #expect(parsed.name == "vote")
        #expect(parsed.value == .choice("affirm"))
        #expect(parsed.unparsed == nil)

        let unreadable = try #require(
            MultiAgentRunner.endpointStamp(
                for: declared, output: "I never labelled my line this time."))
        #expect(unreadable.value == nil)
        #expect(unreadable.unparsed == true)

        // No declaration ⇒ no stamp at all ⇒ the record is what it always was.
        #expect(MultiAgentRunner.endpointStamp(for: undeclared, output: "anything") == nil)
    }

    /// The stamp's JSON is the cross-engine contract: a value, or an explicit
    /// `null` alongside `unparsed: true` — never an absent key standing in for
    /// a failed parse, and never a key at all when nothing was declared.
    @Test func stampJSONMatchesTheCrossEngineShape() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]

        let parsed = TurnEndpointStamp(name: "vote", value: .choice("affirm"))
        #expect(
            String(decoding: try encoder.encode(parsed), as: UTF8.self)
                == #"{"name":"vote","value":"affirm"}"#)

        let number = TurnEndpointStamp(name: "months", value: .number(24))
        #expect(
            String(decoding: try encoder.encode(number), as: UTF8.self)
                == #"{"name":"months","value":24}"#)

        let unparsed = TurnEndpointStamp(name: "vote", value: nil)
        #expect(
            String(decoding: try encoder.encode(unparsed), as: UTF8.self)
                == #"{"name":"vote","unparsed":true,"value":null}"#)

        // A turn record carrying no stamp emits no key.
        let bare = MultiAgentTurnResult(
            turnID: "t", turnIndex: 1, title: "T", speakerAgentID: "a",
            speakerName: "A", modelRevision: nil, prompt: "p", output: "o",
            outputLabel: "l", routedAgentIDs: [])
        #expect(
            !String(decoding: try encoder.encode(bare), as: UTF8.self)
                .contains("endpoint"))
        let stamped = MultiAgentTurnResult(
            turnID: "t", turnIndex: 1, title: "T", speakerAgentID: "a",
            speakerName: "A", modelRevision: nil, prompt: "p", output: "o",
            outputLabel: "l", routedAgentIDs: [], endpoint: parsed)
        #expect(
            String(decoding: try encoder.encode(stamped), as: UTF8.self)
                .contains(#""endpoint":{"name":"vote","value":"affirm"}"#))
    }
}
