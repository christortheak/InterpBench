import Foundation

// =============================================================================
// The generated regions of `docs/CLI-REFERENCE.md` (WP0-AGENT-SURFACE-AUDIT
// §5, §7 step 11).
//
// The audit's disposition, followed literally:
//
//   * §5.2 — well over half of the reference document is ESSAY (rationale,
//     incident history, the security clause), and that material is the
//     document's real value. Only the SYNOPSIS — verb, arity, flags, one-line
//     purpose — is generated. Every hand-written paragraph, every semantic
//     flag table with its defaults, and every trap survives untouched outside
//     the markers.
//   * §5.3 — generation is NOT a build step. Each engine prints its own
//     regions, the result is COMMITTED, and a test compares. The document must
//     be readable on GitHub with no toolchain, and a committed artifact diffs
//     reviewably.
//
// Both engines write into the same file and each owns disjoint region ids:
// `swift-*` here, `server-*` in `Server/steerlab_server/cli_reference.py`. A
// generator never touches a region it does not own, so the two can run in
// either order.
// =============================================================================

/// Generates this engine's marked regions, and splices them into the committed
/// document.
public enum CLIReferenceDocument {

    /// The marker idiom. `BEGIN`/`END` on their own lines, with the region id
    /// in both, so a truncated region is a parse error rather than a silent
    /// swallow of the prose after it.
    public static func beginMarker(_ id: String) -> String {
        "<!-- GENERATED:\(id) BEGIN -->"
    }

    public static func endMarker(_ id: String) -> String {
        "<!-- GENERATED:\(id) END -->"
    }

    /// One generated region: which verbs it covers, in reading order.
    public struct Region: Sendable {
        public let id: String
        public let verbLabels: [String]

        public init(id: String, verbLabels: [String]) {
            self.id = id
            self.verbLabels = verbLabels
        }
    }

    /// The Swift engine's regions, in the document's reading order. A verb
    /// missing from this list is a generation failure, not a silent omission
    /// (`CLIReferenceGenerationTests.everyDeclaredVerbIsInExactlyOneRegion`).
    public static let regions: [Region] = [
        // `init` (the home layout) shares the workspace region: it is the verb
        // a first-run reader meets immediately before `workspace init`, and
        // splitting it into a region of its own would renumber every §3
        // heading below for one three-line synopsis.
        .init(id: "swift-workspace", verbLabels: ["init", "workspace init"]),
        .init(
            id: "swift-experiment-authoring",
            verbLabels: [
                "experiment list", "experiment create", "experiment attach",
                "experiment detach",
                "experiment pin-prompts", "experiment pin-rubric",
                "experiment declare-condition", "experiment set-sweep-selection",
                "experiment set-sweep-grid",
                "experiment set-instruments", "experiment set-sampling",
                "experiment set-exclusions", "experiment set-system-prompt",
                "experiment set-parser",
                "experiment set-instrument-scope",
                "experiment set-evaluation-sampling",
                "experiment set-style-taxonomy",
                "experiment verify", "experiment freeze", "experiment duplicate",
            ]),
        .init(
            id: "swift-experiment-running",
            verbLabels: [
                "experiment extract", "experiment validate", "experiment sweep",
                "experiment run",
            ]),
        .init(
            id: "swift-experiment-analysis",
            verbLabels: [
                "experiment analyze", "experiment rescore-style",
                "experiment evaluate",
            ]),
        .init(
            id: "swift-experiment-promotion",
            verbLabels: ["experiment promote", "experiment confirm"]),
        .init(
            id: "swift-panel",
            verbLabels: ["panel list", "panel check", "panel compile"]),
        .init(
            id: "swift-model",
            verbLabels: ["model capabilities", "model set-capability"]),
        .init(
            id: "swift-diagnostics",
            verbLabels: [
                "data check", "vectors compare", "vectors backfill-norms",
                "vectors mirror-poles",
            ]),
        .init(
            id: "swift-remote",
            verbLabels: [
                "remote capabilities", "remote package", "remote upload",
                "remote submit-bundle", "remote jobs", "remote logs",
                "remote cancel", "remote resubmit", "remote fetch",
                "remote import", "remote import-chain", "remote variants",
                "remote chat",
            ]),
        .init(id: "swift-authoring", verbLabels: ["authoring prompt"]),
        .init(id: "swift-docs", verbLabels: ["docs cli-reference"]),
        .init(
            id: "swift-install",
            verbLabels: [
                "install version", "install stamp", "install verify",
            ]),
    ]

    /// The cluster family has its own region, generated from `ClusterCLIVerb`
    /// rather than from the experiment table — the two declarative surfaces the
    /// audit names, one document.
    public static let clusterRegionID = "swift-cluster"

    /// The top-level page: which families the binary dispatches, and the flags
    /// that are read before any verb sees the argument list. Generated from
    /// `ExperimentCLIHelp.topLevelEntries`, which is also what
    /// `steerlab-cli --help` prints — so drift finding D8 (a hand-written
    /// top-level list that omitted two families) cannot recur silently.
    public static let globalRegionID = "swift-global"

    /// Every region id this engine owns.
    public static var ownedRegionIDs: [String] {
        regions.map(\.id) + [clusterRegionID, globalRegionID]
    }

    // MARK: Rendering

    /// The note every region carries, so a reader who lands mid-document knows
    /// the text is generated and knows the one command that regenerates it.
    static let regenerationNote =
        "<!-- Generated from the declarative verb table — `steerlab-cli docs "
        + "cli-reference --write`. Edit the table, not this block. -->"

    static let sharedFlagNote =
        "Every verb above also accepts `--help` (print its arguments and run "
        + "nothing), `--json` (one envelope on stdout), and `--out <file>`."

    /// One region's body, markers excluded.
    public static func body(of region: Region) throws -> String {
        let specs = try region.verbLabels.map { label -> ExperimentCLIVerbSpec in
            guard
                let spec = ExperimentCLIParser.specs.first(where: { $0.label == label })
            else {
                throw ExperimentError(
                    reason: "CLI-REFERENCE region '\(region.id)' names an "
                        + "undeclared verb '\(label)'")
            }
            return spec
        }
        var lines: [String] = [regenerationNote, "", "```"]
        for spec in specs {
            lines.append(
                ExperimentCLIHelp.synopsis(spec, includeSharedFlags: false))
        }
        lines.append("```")
        lines.append("")
        lines.append("| Verb | Purpose |")
        lines.append("|---|---|")
        for spec in specs {
            lines.append("| `\(spec.label)` | \(spec.purpose) |")
        }
        lines.append("")
        lines.append(sharedFlagNote)
        return lines.joined(separator: "\n")
    }

    /// The cluster region's body.
    public static func clusterBody() -> String {
        var lines: [String] = [regenerationNote, "", "```"]
        for verb in ClusterCLIVerb.allCases {
            lines.append(verb.synopsis)
        }
        lines.append("```")
        lines.append("")
        lines.append("| Verb | Purpose |")
        lines.append("|---|---|")
        for verb in ClusterCLIVerb.allCases {
            lines.append("| `cluster \(verb.rawValue)` | \(verb.purpose) |")
        }
        lines.append("")
        lines.append(
            "Every verb above also accepts `--json` and `--help`. "
                + "`--site <id>` is required wherever it is listed.")
        return lines.joined(separator: "\n")
    }

    /// The top-level region's body: the `--help` page itself, fenced, plus the
    /// global-flag table. The page is short enough to reproduce verbatim, and
    /// reproducing it is the point — a reader of the document and a caller
    /// running `--help` must see the same list of families.
    public static func globalBody() -> String {
        var lines: [String] = [regenerationNote, "", "```"]
        lines.append(
            ExperimentCLIHelp.topLevelText.trimmingCharacters(in: .newlines))
        lines.append("```")
        lines.append("")
        lines.append("| Global flag | Effect |")
        lines.append("|---|---|")
        for flag in ExperimentCLIHelp.globalFlags {
            lines.append("| `\(flag.synopsis)` | \(flag.purpose) |")
        }
        return lines.joined(separator: "\n")
    }

    /// Every region this engine owns, by id.
    public static func generatedBodies() throws -> [String: String] {
        var bodies: [String: String] = [:]
        for region in regions { bodies[region.id] = try body(of: region) }
        bodies[clusterRegionID] = clusterBody()
        bodies[globalRegionID] = globalBody()
        return bodies
    }

    // MARK: Splicing

    /// Replace this engine's regions in `document`, leaving every other byte —
    /// including the other engine's regions — exactly as it was.
    ///
    /// A region id that is generated but absent from the document is a
    /// FAILURE, not a no-op: a marker deleted by hand would otherwise silently
    /// drop a verb family out of the manual.
    public static func rewrite(_ document: String) throws -> String {
        var text = document
        for (id, body) in try generatedBodies().sorted(by: { $0.key < $1.key }) {
            text = try replace(regionID: id, in: text, with: body)
        }
        return text
    }

    /// The ids whose committed text differs from what the tables generate.
    /// Empty means the document is in sync — the drift gate's whole answer.
    public static func drift(in document: String) throws -> [String] {
        var drifted: [String] = []
        for (id, body) in try generatedBodies().sorted(by: { $0.key < $1.key }) {
            let committed = try extract(regionID: id, from: document)
            if committed != body { drifted.append(id) }
        }
        return drifted
    }

    /// The committed body of one region, markers excluded, trailing newline
    /// trimmed.
    public static func extract(regionID: String, from document: String) throws
        -> String
    {
        let begin = beginMarker(regionID)
        let end = endMarker(regionID)
        guard let beginRange = document.range(of: begin) else {
            throw ExperimentError(
                reason: "docs/CLI-REFERENCE.md has no region marker \(begin)")
        }
        guard let endRange = document.range(of: end),
            endRange.lowerBound > beginRange.upperBound
        else {
            throw ExperimentError(
                reason: "docs/CLI-REFERENCE.md has no closing marker \(end) after "
                    + "\(begin)")
        }
        let inner = document[beginRange.upperBound ..< endRange.lowerBound]
        return String(inner).trimmingCharacters(in: .newlines)
    }

    private static func replace(regionID: String, in document: String, with body: String)
        throws -> String
    {
        let begin = beginMarker(regionID)
        let end = endMarker(regionID)
        guard let beginRange = document.range(of: begin),
            let endRange = document.range(of: end),
            endRange.lowerBound > beginRange.upperBound
        else {
            throw ExperimentError(
                reason: "docs/CLI-REFERENCE.md has no \(begin) … \(end) region")
        }
        var text = document
        text.replaceSubrange(
            beginRange.upperBound ..< endRange.lowerBound, with: "\n" + body + "\n")
        return text
    }

    // MARK: Location

    /// Where the committed document lives in a code checkout. Nil when the
    /// binary is running outside one — an installed CLI ships no `docs/`, and
    /// saying so is better than writing a document nobody will read.
    public static var committedURL: URL? {
        let url = CodeResources.compiledCheckoutPath
            .appending(path: "docs/CLI-REFERENCE.md")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }
}
