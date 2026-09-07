import CryptoKit
import Foundation

// =============================================================================
// AGENTS.md — the contract every workspace carries (WP0 step 10)
//
// EDITS GO TO `docs/AGENTS-WORKSPACE-DRAFT.md` FIRST, then get mirrored here.
// scripts/ci/check-workspace-bootstrap.py generates both clients from it.
// That document is the human source of truth; this binding is the shipping
// copy, and `AgentContractTests.agentContractMatchesTheDraftDocument` asserts
// the two are byte-identical (modulo the generated header line below and the
// draft's own `<!-- … -->` markers, which are stripped). Editing only one of
// them fails that gate rather than drifting silently.
//
// The maintained guide is packaged as a resource for Python and compiled into
// WorkspaceBootstrapText for the Mac. Both use the same generated-file hash.
// =============================================================================

/// The workspace-facing agent contract: what `AGENTS.md` says, and the file
/// name it is written under. `WorkspaceStore` writes it at creation, lazily on
/// open, and — since the file's header started carrying a hash of its own body
/// — refreshes it in place while that hash proves nobody has edited it. A file
/// the hash does not vouch for is never touched.
public enum AgentContract {

    /// The file, at the workspace root.
    public static let fileName = "AGENTS.md"

    // MARK: - The header line

    /// **The header format.** One line, first in the file, one HTML comment so
    /// it does not render as prose and so stripping comment lines recovers the
    /// draft exactly. Shape:
    ///
    /// ```
    /// <!-- …fixed prose… sha256:<64 lowercase hex> -->
    /// ```
    ///
    /// The hex is the SHA-256 of **the body bytes this writer emitted after
    /// the header**, in the same normalized form `classify` compares: the
    /// file's text after the header line's newline, with at most one leading
    /// blank line removed. So `contents()` — header, blank line, body — hashes
    /// exactly `body`, and a copy that lost the blank line still verifies.
    ///
    /// That hash is what turns the old header-intact HEURISTIC into a proof:
    /// a file whose header hash matches its body is one SteerLab wrote and
    /// nobody has edited, which is the only condition under which this build
    /// rewrites it.
    static let headerPrefix =
        "<!-- Written by SteerLab workspace seeding. SteerLab keeps this file "
        + "current for you while this line's hash still matches the text under "
        + "it, and never touches it once you edit that text. sha256:"

    /// Closes the HTML comment. Everything between prefix and suffix is hex.
    static let headerSuffix = " -->"

    /// The header builds before the hash existed: the same promise, no proof,
    /// and it named a manual repair (delete + reopen) because that was the
    /// only one there was. **Recognised forever, never written again.** A file
    /// carrying it is treated exactly as this build's predecessors treated it
    /// — heuristic classification, advisory only, never rewritten — and the
    /// one manual regeneration the advisory names is what graduates it into
    /// the hashed regime.
    static let legacyGeneratedHeader =
        "<!-- Written by SteerLab workspace seeding; safe to regenerate — "
        + "delete this file and reopen the workspace to get it back. SteerLab "
        + "never overwrites an existing AGENTS.md, so local edits survive. -->"

    /// The header line for a given body — the writer's half of the format
    /// above, and the seam a test uses to age a fixture backwards honestly
    /// (an older build's body under an older build's *correct* hash).
    public static func header(for body: String) -> String {
        headerPrefix + sha256Hex(body) + headerSuffix
    }

    /// The header line this build writes, over the body this build ships.
    public static var generatedHeader: String { header(for: body) }

    /// The hash a header line declares, or nil when the line is not one of
    /// ours in the hashed format (a legacy header, a tampered one, prose).
    static func declaredBodyHash(inHeaderLine line: String) -> String? {
        guard line.hasPrefix(headerPrefix), line.hasSuffix(headerSuffix),
            line.count >= headerPrefix.count + headerSuffix.count + 64
        else { return nil }
        let hex = line.dropFirst(headerPrefix.count).dropLast(headerSuffix.count)
        guard hex.count == 64, hex.allSatisfy(\.isHexDigit),
            hex.lowercased() == hex
        else { return nil }
        return String(hex)
    }

    /// Lowercase hex SHA-256 of a string's UTF-8 bytes.
    static func sha256Hex(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
    }

    /// The contract text, byte-identical to `docs/AGENTS-WORKSPACE-DRAFT.md`
    /// with its draft-only comment markers removed. Ends with exactly one
    /// newline.
    public static let body: String = literal + "\n"

    /// The bytes written into a workspace: header line, blank line, body.
    public static func contents() -> String {
        generatedHeader + "\n\n" + body
    }

    // MARK: - Staleness

    /// What a workspace's `AGENTS.md` is, relative to the contract THIS build
    /// ships.
    ///
    /// The contract is written at workspace creation or on the first open of
    /// an older workspace. A file that is **provably** still the machine's —
    /// its header hash matches its body — is refreshed in place when the
    /// shipped contract moves on; anything else is never overwritten. That
    /// asymmetry is the whole design: the contract is documentation, it alters
    /// no run, and a workspace made before a contract revision otherwise keeps
    /// the old text forever, silently, while its runner agent reads
    /// instructions that no longer describe the CLI.
    ///
    /// Nothing in *this* type writes: it is a cheap read, a classification and
    /// two sentences. `WorkspaceStore` owns the one write it enables.
    public enum Status: Sendable, Equatable {

        /// No `AGENTS.md` at the workspace root. Silent by design: the
        /// upkeep path regenerates an absent contract on its own, so there is
        /// nothing for a person to repair.
        case absent

        /// Byte-identical to what this build would write. (Also the answer for
        /// a legacy hashless header over the current body: there is nothing to
        /// do and nothing to say.)
        case current

        /// **Proven machine-owned and behind.** The header carries a hash, and
        /// that hash is the hash of this file's own body — so SteerLab wrote
        /// these exact bytes and nobody has changed them since — but the body
        /// is not the body this build ships. This is the one state that gets
        /// rewritten automatically.
        ///
        /// `linesBehind` counts lines of the SHIPPED body that this copy does
        /// not have (see `missingLineCount` for exactly what that means and
        /// what it deliberately is not).
        case staleProven(linesBehind: Int)

        /// **A legacy header, intact, over an older body.** Written by a build
        /// from before the hash: the header says SteerLab wrote the file and
        /// nobody has touched the line that says so, which is a heuristic and
        /// not a proof. Advisory only — this state is never rewritten, and the
        /// one manual regeneration the advisory names graduates the file into
        /// the hashed regime, after which it refreshes itself.
        case staleUnedited(linesBehind: Int)

        /// The header is gone, altered, or hashed-but-not-matching — or the
        /// file is there but unreadable as text. **The researcher owns this
        /// file now.** Silent on every surface and never written: they chose
        /// their text, and a tool that nags about a file it promised never to
        /// touch is a tool that will be worked around.
        case edited
    }

    /// **Proof where there used to be a heuristic.**
    ///
    /// The first line of the file decides everything, and there are three
    /// kinds of it:
    ///
    /// 1. **Hashed header** (`headerPrefix … sha256:<hex> -->`). Recompute the
    ///    hash over the body actually present. Match → SteerLab wrote these
    ///    exact bytes and nobody has edited them: `current` if the body is the
    ///    shipped body, else `staleProven` — the state that is safe to rewrite
    ///    without asking, because we can *show* no human text is at risk.
    ///    Mismatch → someone edited the body under our header: `edited`, hands
    ///    off, no notice.
    /// 2. **Legacy hashless header**, byte-for-byte. Exactly the pre-hash
    ///    behaviour, deliberately unchanged: `current` or `staleUnedited`,
    ///    advisory only, never written. The heuristic's one wrong direction —
    ///    a body edited under an intact header reads as `staleUnedited` — is
    ///    still wrong in the harmless direction *because nothing writes here*.
    /// 3. **Anything else**, including a tampered header and a file with no
    ///    newline at all: `edited`.
    ///
    /// A pure seam: text in, classification out, no filesystem.
    public static func classify(_ text: String) -> Status {
        guard let split = splitHeader(text) else { return .edited }
        let (headerLine, rest) = split

        if let declared = declaredBodyHash(inHeaderLine: headerLine) {
            guard declared == sha256Hex(rest) else { return .edited }
            if rest == body { return .current }
            return .staleProven(
                linesBehind: missingLineCount(shipped: body, workspace: rest))
        }

        guard headerLine == legacyGeneratedHeader else { return .edited }
        if rest == body { return .current }
        return .staleUnedited(
            linesBehind: missingLineCount(shipped: body, workspace: rest))
    }

    /// A contract file's first line and the body under it, in the ONE
    /// normalized form everything downstream uses: the text after the header
    /// line's newline, with at most one leading blank line removed
    /// (`contents()` writes that blank line; a copy that lost it is not an
    /// edit). Nil when there is no first line to speak of.
    ///
    /// The header's hash is computed over exactly this, which is what makes
    /// the proof survive the same normalization the comparison does.
    static func splitHeader(_ text: String) -> (header: String, body: String)? {
        guard let breakIndex = text.firstIndex(of: "\n") else { return nil }
        var rest = String(text[text.index(after: breakIndex)...])
        if rest.hasPrefix("\n") { rest.removeFirst() }
        return (String(text[text.startIndex..<breakIndex]), rest)
    }

    /// The body a contract file carries, normalized as above — the text a
    /// refresh is replacing, and therefore the text a "lines changed" count
    /// must be about. Empty for a file with no header line at all.
    static func bodyText(of fileText: String) -> String {
        splitHeader(fileText)?.body ?? ""
    }

    /// Classify the `AGENTS.md` at a workspace root. Never throws and never
    /// writes: an unreadable-but-present file is `edited` (hands off — we
    /// cannot see it, so we cannot claim it is ours), a missing one is
    /// `absent`.
    public static func status(at workspaceRoot: URL) -> Status {
        let url = workspaceRoot.appending(component: fileName)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return .absent
        }
        guard let text = try? String(contentsOf: url, encoding: .utf8) else {
            return .edited
        }
        return classify(text)
    }

    /// How many lines the shipped body has that the workspace's copy does
    /// not, counting duplicates — a multiset difference, one pass, no
    /// allocation per line beyond the tally.
    ///
    /// **Honest about what it is not:** this is not an LCS diff and does not
    /// claim to be. It cannot tell a moved line from a deleted one, and it
    /// reports 0 for a copy that only ADDED lines. It answers exactly one
    /// question — "how much of the current contract is missing here" — which
    /// is the question the advisory asks, and it answers it in linear time
    /// over a file read once. `stalenessAdvisory` words the 0 case as
    /// "out of date with" rather than "0 lines behind", so the number is
    /// never asked to mean more than it does.
    static func missingLineCount(shipped: String, workspace: String) -> Int {
        var have: [Substring: Int] = [:]
        for line in workspace.split(
            separator: "\n", omittingEmptySubsequences: false)
        {
            have[line, default: 0] += 1
        }
        var missing = 0
        for line in shipped.split(separator: "\n", omittingEmptySubsequences: false) {
            if let count = have[line], count > 0 {
                have[line] = count - 1
            } else {
                missing += 1
            }
        }
        return missing
    }

    /// How many lines differ between two bodies, in both directions —
    /// `missingLineCount` symmetrized, so a refresh that only ADDS lines still
    /// reports a number rather than 0.
    ///
    /// Same honesty caveat as its half: a multiset difference, not an LCS
    /// diff. Two bodies with the same lines in a different order report 0, and
    /// `refreshNotice` words that case rather than printing "0 lines changed".
    static func changedLineCount(from old: String, to new: String) -> Int {
        missingLineCount(shipped: new, workspace: old)
            + missingLineCount(shipped: old, workspace: new)
    }

    /// The one advisory sentence, or nil when there is nothing to say.
    ///
    /// Fires for `staleUnedited` ONLY — the LEGACY hashless header, the one
    /// state we can neither vouch for nor rewrite. `current` has nothing to
    /// report, `absent` regenerates itself, `staleProven` is refreshed in
    /// place and speaks through `refreshNotice`, and `edited` is the
    /// researcher's file. Non-blocking everywhere it is used: it changes no
    /// exit code and gates nothing.
    ///
    /// Prefix-free on purpose — the CLI stamps its own `advisory: ` in front,
    /// the app's notice feed carries a severity instead, and neither surface
    /// has to own the other's punctuation.
    ///
    /// The repair it names is real on both surfaces — the app rewrites an
    /// absent contract on open (`WorkspaceStore.open`), and the CLI does the
    /// same once per invocation on its resolution path
    /// (`ExperimentCLIRunner.agentContractUpkeepLine`). Delete the file, run
    /// anything, get the current contract back — and the file that comes back
    /// carries a hashed header, so this is the last time the repair is
    /// manual.
    public static func stalenessAdvisory(at workspaceRoot: URL) -> String? {
        stalenessAdvisory(for: status(at: workspaceRoot), at: workspaceRoot)
    }

    /// The wording, from an already-computed status — so a caller that has to
    /// classify before acting (both surfaces do: they classify, then write)
    /// does not read the file twice.
    public static func stalenessAdvisory(
        for status: Status, at workspaceRoot: URL
    ) -> String? {
        guard case .staleUnedited(let linesBehind) = status else { return nil }
        let extent =
            linesBehind > 0
            ? "\(linesBehind) line\(linesBehind == 1 ? "" : "s") behind"
            : "out of date with"
        let path = workspaceRoot.appending(component: fileName).path
        return "this workspace's \(fileName) is \(extent) the agent contract "
            + "this build ships, and its machine header shows it unedited — "
            + "delete \(path) and reopen the workspace (or run any workspace "
            + "verb) to regenerate it. That one manual refresh is the last: "
            + "the regenerated file carries a hashed header, and SteerLab "
            + "refreshes a hashed, unedited contract for you from then on"
    }

    /// The one notice sentence for a contract this build just rewrote.
    ///
    /// Fires for `staleProven` and nothing else, once per open on each
    /// surface — the same discipline and the same channels as the advisory
    /// above (CLI stderr, the app's `"Workspace"` notice feed), and the same
    /// prefix-free wording for the same reason.
    ///
    /// It is a report, not a request: the work is already done and there is
    /// nothing for the reader to repair. It says so, and it says why the
    /// rewrite was safe — the header hashed the text it wrote, and that hash
    /// still matched.
    public static func refreshNotice(
        linesChanged: Int, at workspaceRoot: URL
    ) -> String {
        let extent =
            linesChanged > 0
            ? "\(linesChanged) line\(linesChanged == 1 ? "" : "s") changed"
            : "same lines, reordered"
        let path = workspaceRoot.appending(component: fileName).path
        return "refreshed \(path) to the agent contract this build ships "
            + "(\(extent)) — its machine header hashed the text it wrote and "
            + "that hash still matched, so nobody had edited it; nothing else "
            + "in the workspace was touched"
    }

    // Raw literal (`#"""`) on purpose: the contract is full of shell
    // continuations (`\` at end of line) and one escaped table pipe, none of
    // which may be interpreted as Swift escapes.
    private static let literal = String(WorkspaceBootstrapText.agentBody.dropLast())
}
