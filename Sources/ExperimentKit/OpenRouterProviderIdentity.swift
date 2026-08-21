import Foundation

/// Canonical identity for an OpenRouter serving provider.
///
/// OpenRouter ROUTES by lowercase slug but REPORTS the serving provider by
/// display name, so the two spellings of one endpoint have to resolve to a
/// single identity — otherwise a judgment served by exactly the pinned
/// provider is refused as off-pin, and a correct run dies on a spelling.
///
/// The mapping is the committed fixture
/// `prompts/fixtures/openrouter/providers.json`, fetched from OpenRouter's
/// public provider list. It replaced a hand-written alias list that was
/// missing 86 of ~96 providers and wrong about Vertex: OpenRouter's display
/// name for it is `Google`, not `Google Vertex`, so a judgment correctly
/// served by a pinned `google-vertex` reported `"Google"` and was refused.
///
/// Python twin: `paired_judge.canonical_openrouter_provider`.
public enum OpenRouterProviderIdentity {

    /// Fixture path relative to the repo root. Cross-engine constant.
    public static let fixtureRelativePath =
        "prompts/fixtures/openrouter/providers.json"

    private struct Fixture: Decodable {
        struct Provider: Decodable {
            let name: String
            let slug: String
        }
        let providers: [Provider]
    }

    /// `[lowercased display name or slug: slug]`, loaded once.
    ///
    /// An unreadable or absent fixture degrades to an EMPTY table rather
    /// than trapping. Canonicalization then falls back to plain lowercasing
    /// — exactly the pre-fixture behaviour, so pins spelled as slugs keep
    /// working and everything else still fails closed. A missing fixture
    /// must not take down judging that does not depend on it.
    private static let aliases: [String: String] = {
        guard let url = fixtureURL() else { return [:] }
        return loadAliases(from: url)
    }()

    /// The alias table as read from one fixture URL.
    ///
    /// Split out of the cached `aliases` so the RESOLUTION can be tested
    /// independently of the cache: `aliases` is a `static let`, resolved
    /// once per process, so a staged-bundle test could never observe a
    /// second resolution. Internal, not private, for exactly that reason.
    static func loadAliases(from url: URL) -> [String: String] {
        guard let data = try? Data(contentsOf: url),
            let fixture = try? JSONDecoder().decode(Fixture.self, from: data)
        else { return [:] }
        var table: [String: String] = [:]
        for provider in fixture.providers {
            let slug = provider.slug
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !slug.isEmpty else { continue }
            table[slug.lowercased()] = slug
            let name = provider.name
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !name.isEmpty, table[name.lowercased()] == nil {
                table[name.lowercased()] = slug
            }
        }
        return table
    }

    /// Where the fixture resolves right now, or nil when it cannot be
    /// found. Exposed so tests can assert the RELEASE path finds it inside
    /// a staged bundle — the case a source-checkout derivation silently
    /// failed at.
    static func resolvedFixtureURL() -> URL? { fixtureURL() }

    /// The fixture describes OpenRouter, not a study, so it is an immutable
    /// SHIPPED resource — and therefore resolves through `CodeResources`,
    /// the one authority for those (see that file's header).
    ///
    /// It previously derived a path from the compiled-in source location,
    /// which works in a checkout and silently fails in a distributed `.app`
    /// that has no source tree. That failure would have been quiet and bad:
    /// an unloadable fixture degrades canonicalization to plain lowercasing,
    /// which is exactly the state in which OpenRouter's `Google` stops
    /// matching a pinned `google-vertex` — the bug this fixture exists to
    /// fix. `ReleaseModeResourceTests` fences that pattern; this file was
    /// tripping it.
    ///
    /// `clusterPayload` is the right family: it is the ROOT-LAYOUT tree that
    /// carries `prompts/fixtures/` in both developer mode (the checkout
    /// root) and a bundle (`ClusterPayload/`), because the cluster push
    /// already ships the parity fixtures. It used to resolve through
    /// `workspaceSeed`, which worked only while that family's dev fallback
    /// was the checkout root; WP1 narrowed the seed family to the curated
    /// `WorkspaceSeed/` tree, which carries seed DATA and no fixtures.
    private static func fixtureURL() -> URL? {
        guard let payloadRoot = try? CodeResources.clusterPayload() else {
            return nil
        }
        let url = payloadRoot.appending(path: fixtureRelativePath)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// Number of known spellings. 0 means the fixture did not load — tests
    /// assert against this so a lost fixture cannot pass silently.
    public static var knownSpellingCount: Int { aliases.count }

    public static func canonical(_ raw: String) -> String {
        let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        guard !value.isEmpty else { return value }
        return aliases[value] ?? value
    }

    public static func equivalent(_ lhs: String, _ rhs: String) -> Bool {
        canonical(lhs) == canonical(rhs)
    }
}
