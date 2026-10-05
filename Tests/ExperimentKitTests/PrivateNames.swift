import Foundation
import Testing

/// The private-name list, read from OUTSIDE the repository — the one loader
/// every Swift neutrality guard uses. (Python twin:
/// `scripts/ci/private_names.py`, same file, same rules.)
///
/// The guards exist to keep one researcher's site, study, and account names
/// out of what ships. They used to carry those names as string literals,
/// which put the very thing they guard against into a public repository and,
/// through the bundled payloads, into the app. A digest would not have fixed
/// that: the terms are short words, and a short word's hash is recovered by
/// trying words. So the list lives in a file the repository never sees:
///
/// - `$STEERLAB_PRIVATE_NAMES_FILE` when set, otherwise
///   `~/.steerlab/private-names.txt`;
/// - one lowercase term per line, matched case-insensitively as a substring
///   (deliberately blunt — a false positive costs one rename, a false
///   negative ships a name);
/// - blank lines and `#` comments are ignored, and so are `allow:` lines,
///   which belong to the artifact scan (`scripts/ci/artifact_scan.py`) and
///   never soften a test guard.
///
/// What a guard does when the list is not there:
///
/// - normally it is SKIPPED, with the reason in the run summary — a
///   contributor's machine and a fork's CI have no list and nothing to leak;
/// - with `STEERLAB_REQUIRE_PRIVATE_NAMES=1` it RUNS AND FAILS, so the
///   maintainer's release gate and the main repository's CI cannot pass
///   vacuously because a file went missing.
enum PrivateNames {

    static let fileVariable = "STEERLAB_PRIVATE_NAMES_FILE"
    static let requireVariable = "STEERLAB_REQUIRE_PRIVATE_NAMES"
    /// Relative to the home folder.
    static let defaultRelativeLocation = ".steerlab/private-names.txt"

    /// A list that was found and holds at least one term.
    ///
    /// A term never goes into a message. A failing guard's output can land
    /// in a log anyone may read, so a hit is named by its POSITION in the
    /// list ("list entry 3 (6 letters)") — the list's owner can look that
    /// up and nobody else can. For the same reason the value describes and
    /// reflects itself without its terms: a test runner prints the values
    /// in a failed expectation.
    struct List: Sendable, Equatable, CustomStringConvertible,
        CustomDebugStringConvertible, CustomReflectable
    {
        let url: URL
        let terms: [String]

        var description: String { "PrivateNames.List(\(terms.count) term(s))" }
        var debugDescription: String { description }
        var customMirror: Mirror { Mirror(self, children: [], displayStyle: .struct) }

        func label(forTermAt index: Int) -> String {
            "list entry \(index + 1) (\(terms[index].count) letters)"
        }

        /// The LABELS of the terms `text` contains, case-insensitively, as
        /// substrings.
        func hits(in text: String) -> [String] {
            let lowered = text.lowercased()
            return terms.indices
                .filter { lowered.contains(terms[$0]) }
                .map(label(forTermAt:))
        }
    }

    static func listURL(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> URL {
        if let override = environment[fileVariable], !override.isEmpty {
            return URL(filePath: (override as NSString).expandingTildeInPath)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appending(path: defaultRelativeLocation)
    }

    static func isRequired(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        ["1", "true", "yes"].contains(
            (environment[requireVariable] ?? "").lowercased())
    }

    /// The terms in one list file's text. Lowercased, de-duplicated in order.
    static func terms(fromListText text: String) -> [String] {
        var seen = Set<String>()
        var terms: [String] = []
        for line in text.split(whereSeparator: \.isNewline) {
            let entry = line.trimmingCharacters(in: .whitespaces).lowercased()
            if entry.isEmpty || entry.hasPrefix("#") || entry.hasPrefix("allow:") {
                continue
            }
            if seen.insert(entry).inserted { terms.append(entry) }
        }
        return terms
    }

    /// The list, or nil when the file is missing, unreadable, or holds no
    /// term — an empty list would let every guard pass vacuously, so it
    /// counts as absent.
    static func load(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> List? {
        let url = listURL(environment: environment)
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        let terms = terms(fromListText: text)
        return terms.isEmpty ? nil : List(url: url, terms: terms)
    }

    /// This process's list, read once.
    static let shared: List? = load()

    /// The condition behind `.needsPrivateNames`: a guard runs when it has a
    /// list to check against, and also when the list is REQUIRED — so that a
    /// required-but-missing list reaches `required()` and fails there.
    static var guardsShouldRun: Bool { shared != nil || isRequired() }

    static let skipReason: Comment = """
        no private-name list on this machine (~/.steerlab/private-names.txt, \
        or the file $STEERLAB_PRIVATE_NAMES_FILE names), so there is nothing \
        to check for; set STEERLAB_REQUIRE_PRIVATE_NAMES=1 to make that a failure
        """

    /// The list, for a guard's body. Fails the test when it is absent, which
    /// by `guardsShouldRun` only happens when the list was required.
    static func required(
        sourceLocation: SourceLocation = #_sourceLocation
    ) throws -> List {
        try #require(
            shared,
            """
            \(requireVariable) is set, but no private-name list with at least \
            one term could be read at \(listURL().path). Put the list there, \
            or point \(fileVariable) at it.
            """,
            sourceLocation: sourceLocation)
    }
}

extension Trait where Self == ConditionTrait {
    /// Marks a neutrality guard that checks against the private-name list:
    /// skipped (with the reason) when there is no list, unless the list is
    /// required. Pair it with `try PrivateNames.required()` in the body.
    static var needsPrivateNames: Self {
        .enabled(if: PrivateNames.guardsShouldRun, PrivateNames.skipReason)
    }
}
