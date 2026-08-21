import Foundation
import Testing

@testable import ExperimentKit

/// Row hashes must be byte-identical across engines for ORDINARY text.
///
/// `json.dumps` defaults to `ensure_ascii=True`, escaping every non-ASCII and
/// every control character; a hand-rolled Swift escaper that handled only
/// backslash, quote and newline diverged on all of them — "café" hashed one
/// way on the server and another on the Mac. The earlier fixture was
/// ASCII-only and never compared row hashes, so nothing caught it, and
/// LLM-generated prose contains smart quotes as a matter of course.
struct ScenarioRowHashParityTests {

    @Test func rowHashesMatchThePythonTwinForAwkwardText() throws {
        let url = CodeResources.compiledCheckoutPath.appending(
            components: "Tests", "Fixtures", "cross-engine",
            "scenario-row-hashes.json")
        let cases = try #require(
            try JSONSerialization.jsonObject(with: try Data(contentsOf: url))
                as? [[String: Any]])
        #expect(cases.count >= 20)
        for entry in cases {
            let label = entry["label"] as? String ?? "?"
            let text = try #require(entry["text"] as? String)
            let expresses = try #require(entry["expresses"] as? Bool)
            #expect(
                ScenarioDiagnostics.rowHash(text: text, label: expresses)
                    == entry["rowHash"] as? String,
                "row-hash drift on case '\(label)': the same scenario would carry two identities across substrates")
        }
    }

    /// Spot-checks of the individual rules the fixture exercises in bulk.
    @Test func theEscaperReproducesJSONDumps() {
        let accented = ScenarioDiagnostics.quoted("caf\u{e9}")
        #expect(accented == "\"caf\\u00e9\"")

        let tabbed = ScenarioDiagnostics.quoted("a\tb")
        #expect(tabbed == "\"a\\tb\"")

        let returned = ScenarioDiagnostics.quoted("a\rb")
        #expect(returned == "\"a\\rb\"")

        // A control character with no short escape becomes \uXXXX.
        let bell = ScenarioDiagnostics.quoted("\u{07}")
        #expect(bell == "\"\\u0007\"")

        // Backslash and quote keep their short escapes.
        #expect(ScenarioDiagnostics.quoted("a\\b") == "\"a\\\\b\"")
        #expect(ScenarioDiagnostics.quoted("a\"b") == "\"a\\\"b\"")

        // Astral characters become surrogate pairs, which UTF-16 gives us.
        let emoji = ScenarioDiagnostics.quoted("\u{1F600}")
        #expect(emoji == "\"\\ud83d\\ude00\"")

        // Plain ASCII is untouched.
        #expect(ScenarioDiagnostics.quoted("plain") == "\"plain\"")
    }
}
