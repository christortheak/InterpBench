import Foundation
import Testing

@testable import ExperimentKit

/// Where judging would happen, said BEFORE it happens (2026-07-24).
/// Cross-engine twin of `Server/tests/test_judging_custody_visibility.py`.
///
/// The case this exists for: the judge key holds ONE kind, so a panel of one
/// claude and one openrouter judge on a host with an openrouter key
/// credentials half the panel and defers the whole thing — despite a key
/// having been deliberately placed. Deferring is not a bug; being unable to
/// find out in advance was.
struct JudgingCustodyAdvisoryTests {

    private func manifest(
        _ judges: [ExperimentManifest.JudgeRef]
    ) -> ExperimentManifest {
        var m = ExperimentManifest(
            name: "s", description: "", modelID: "org/m")
        m.judges = judges
        return m
    }

    /// A host holding a key for exactly these credential kinds.
    private func host(_ kinds: String...) -> (String) -> String? {
        let held = Set(kinds)
        return { held.contains($0) ? "sk-fake" : nil }
    }

    @Test func silentOnTheUnsurprisingCases() {
        // An advisory that fires on every study is one nobody reads.
        #expect(
            ExperimentStore.judgingCustodyAdvisory(
                manifest([.init(name: "a", kind: "openrouter",
                                model: "x/y", provider: "deepinfra")]),
                keyForKind: host("openrouter")) == nil)
        #expect(
            ExperimentStore.judgingCustodyAdvisory(
                manifest([.init(name: "a", kind: "local", model: "org/m")]),
                keyForKind: host()) == nil)
        #expect(
            ExperimentStore.judgingCustodyAdvisory(
                manifest([]), keyForKind: host()) == nil)
    }

    @Test func mixedPanelWithOneKindOfKeyDefersTheWholePanel() {
        let note = ExperimentStore.judgingCustodyAdvisory(
            manifest([
                .init(name: "claude-j", kind: "claude"),
                .init(name: "or-j", kind: "openrouter",
                      model: "anthropic/claude-opus-4.8", provider: "anthropic"),
            ]),
            keyForKind: host("openrouter"))
        let text = try! #require(note)
        #expect(text.contains("would DEFER"))
        // Names the judge that caused it — "which one do I fix" is the
        // whole question.
        #expect(text.contains("'claude-j' (claude)"))
        #expect(text.contains("including any judge that IS credentialed here"))
        // Explains the mechanism, because the mechanism is the surprise.
        #expect(text.contains("holds ONE kind"))
    }

    @Test func keylessExternalPanelDefers() {
        let note = ExperimentStore.judgingCustodyAdvisory(
            manifest([.init(name: "or-j", kind: "openrouter",
                            model: "x/y", provider: "deepinfra")]),
            keyForKind: host())
        #expect(note?.contains("would DEFER") == true)
    }

    @Test func splitLocalAndUncredentialedExternalRefuses() {
        let note = ExperimentStore.judgingCustodyAdvisory(
            manifest([
                .init(name: "local-j", kind: "local", model: "org/m"),
                .init(name: "or-j", kind: "openrouter",
                      model: "x/y", provider: "deepinfra"),
            ]),
            keyForKind: host())
        let text = try! #require(note)
        #expect(text.contains("would REFUSE"))
        #expect(text.contains("cannot defer coherently"))
    }

    @Test func claudeJudgeNeedsAnAnthropicKeyNotAnOpenRouterOne() {
        // The kind mapping is the trap's mechanism: judge kind "claude"
        // needs credential kind "anthropic".
        #expect(
            ExperimentStore.judgingCustodyAdvisory(
                manifest([.init(name: "c", kind: "claude")]),
                keyForKind: host("anthropic")) == nil)
        #expect(
            ExperimentStore.judgingCustodyAdvisory(
                manifest([.init(name: "c", kind: "claude")]),
                keyForKind: host("openrouter")) != nil)
    }

    @Test @MainActor func newJudgesDefaultToOpenRouter() {
        // External judging standardises on OpenRouter: it reaches Anthropic
        // models via provider `anthropic` and, unlike the direct path,
        // reports which backend actually served the verdict.
        let panel = ExperimentPanel()
        panel.addJudge()
        #expect(panel.judges.first?.kind == "openrouter")
    }
}
