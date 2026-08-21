import Foundation

@testable import ExperimentKit

/// Checkout-shape predicates for tests whose fixture is a file that the
/// RELEASE tree deliberately does not carry.
///
/// The instrument ships as a curated export (`scripts/export-allowlist.txt`):
/// study material, internal engineering documents, and the export tooling
/// itself stay research-side. The Swift suite, however, ships whole — the
/// same test files run in both trees — so a test whose fixture is
/// research-only must SKIP in a released checkout rather than fail there.
///
/// Two rules keep that from becoming a hole:
///
/// - Skip, never soften. A gate is disabled by `.enabled(if:)` with a named
///   reason and reports as skipped; it is never rewritten to pass vacuously
///   against a missing file. In the research tree, where every fixture is
///   present, every gate still runs with full force.
/// - Predicate on the FIXTURE, not on a tree flag. There is no
///   "am I the release tree?" switch to get wrong: each property asks
///   whether its own file is on disk, so a fixture that silently disappears
///   from the research tree shows up as a skip in the run summary.
enum ResearchTreeFixtures {

    /// The code checkout this test binary was compiled from.
    static var checkoutRoot: URL { CodeResources.compiledCheckoutPath }

    private static func exists(_ relativePath: String) -> Bool {
        FileManager.default.fileExists(
            atPath: checkoutRoot.appending(path: relativePath).path)
    }

    // MARK: - Research-only fixtures

    /// `docs/AGENTS-WORKSPACE-DRAFT.md` — the human source of truth the
    /// shipped `AgentContract` constant is held byte-equal to. It SHIPS
    /// (the contract text is neutral and a released reader benefits from
    /// reviewing it), so this should be true in both trees; the predicate
    /// stays so the drift gate reports honestly if the allowlist entry is
    /// ever removed.
    static var hasAgentContractDraft: Bool { exists("docs/AGENTS-WORKSPACE-DRAFT.md") }

    /// `scripts/export-denylist.txt` — the private-name list. It must NEVER
    /// ship (it carries the very identifiers the release must not contain),
    /// so the help-text neutrality gate is research-tree-only by design.
    /// The release tree's neutrality is proved instead by the public-tier
    /// scan in CI and by `AgentContractTests`' literal denylist.
    static var hasExportDenylist: Bool { exists("scripts/export-denylist.txt") }

    /// `docs/examples/starter-study-pack.json` — the paste-ready worked
    /// example pack. `SampleWorkspace/` ships; the pack and its walkthrough
    /// document are research-side today (the shipped docs set is curated,
    /// `docs/STARTER-PACK.md` is not in it), so nothing in a released tree
    /// can import it and nothing there needs the gate.
    static var hasStarterStudyPack: Bool { exists("docs/examples/starter-study-pack.json") }

    /// `prompts/panels/templates/deliberative-appellate-panel-v1.json` —
    /// study material (an appellate deliberation protocol), excluded from
    /// the export. The protocol-template MACHINERY is covered by the rest
    /// of the suite against in-code fixtures; only the shipped-file gate
    /// needs the file.
    static var hasAppellateProtocolTemplate: Bool {
        exists("prompts/panels/templates/deliberative-appellate-panel-v1.json")
    }

    // MARK: - Build-output fixtures

    /// `web/results-explorer/` — the embedded Results Explorer bundle. It is
    /// a committed BUILD OUTPUT in the research tree and is built from
    /// source (`results-explorer/`, `npm run build:embed`) in the release
    /// tree, so a cold clone has `web/` only after that build runs. Tests
    /// that assert the developer-checkout resource layout skip until it is
    /// there.
    static var hasBuiltWebAssets: Bool { exists("web/results-explorer/index.html") }
}
