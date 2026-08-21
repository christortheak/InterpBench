import Foundation
import Testing

@testable import ExperimentKit

/// The drift gate for the generated regions of `docs/CLI-REFERENCE.md`, and
/// the `--help` surface they share a source with (WP0-AGENT-SURFACE-AUDIT §5,
/// §7 step 11).
///
/// The audit's finding this suite closes: the reference document claims to be
/// "read out of the dispatch code", nine drift instances said otherwise, and
/// the one section with ZERO verified drift (§3.9, `cluster`) was the one
/// backed by a declarative table. Generating the synopsis from that table and
/// comparing the committed bytes makes the class of drift unreproducible —
/// a flag added to a verb and not to the document now fails this suite.
///
/// Deliberately NOT a build step: generation runs here and in the CLI verb,
/// the result is COMMITTED, and the repair for a red test is
/// `steerlab-cli docs cli-reference --write` followed by a commit. The
/// document has to read on GitHub with no toolchain.
struct CLIReferenceGenerationTests {

    /// The committed document, resolved from the code checkout — never from a
    /// workspace root, which may be anywhere.
    static func committedDocument() throws -> String {
        let url = try #require(
            CLIReferenceDocument.committedURL,
            "docs/CLI-REFERENCE.md must exist in the checkout")
        return try String(contentsOf: url, encoding: .utf8)
    }

    // MARK: The gate

    @Test func generatedRegionsMatchCommittedDocument() throws {
        let document = try Self.committedDocument()
        let drifted = try CLIReferenceDocument.drift(in: document)
        #expect(
            drifted.isEmpty,
            """
            docs/CLI-REFERENCE.md is out of date in \(drifted.count) region(s): \
            \(drifted.joined(separator: ", ")). \
            Repair: steerlab-cli docs cli-reference --write, then commit.
            """)
    }

    /// Rewriting an in-sync document must be a no-op — otherwise `--check` and
    /// `--write` could disagree and the gate would be unrepairable.
    @Test func rewritingAnInSyncDocumentChangesNothing() throws {
        let document = try Self.committedDocument()
        #expect(try CLIReferenceDocument.rewrite(document) == document)
    }

    /// Every verb the parser declares appears in exactly one region. A verb
    /// added to the table and to no region would otherwise be documented
    /// nowhere while every test still passed.
    @Test func everyDeclaredVerbIsInExactlyOneRegion() {
        var counts: [String: Int] = [:]
        for region in CLIReferenceDocument.regions {
            for label in region.verbLabels { counts[label, default: 0] += 1 }
        }
        for spec in ExperimentCLIParser.specs {
            #expect(
                counts[spec.label] == 1,
                "\(spec.label) appears in \(counts[spec.label] ?? 0) generated region(s); it must appear in exactly one")
        }
        #expect(counts.count == ExperimentCLIParser.specs.count)
    }

    /// The two engines own disjoint region ids, so each can regenerate without
    /// touching the other's text.
    @Test func swiftOwnsOnlySwiftRegions() throws {
        for id in CLIReferenceDocument.ownedRegionIDs {
            #expect(id.hasPrefix("swift-"), "\(id) is not a Swift-owned region")
        }
        // The server's regions are present in the same document and are left
        // alone by a Swift rewrite.
        let document = try Self.committedDocument()
        for id in ["server-experiment", "server-study", "server-jobs",
                   "server-vectors", "server-data"] {
            #expect(document.contains(CLIReferenceDocument.beginMarker(id)))
        }
        let rewritten = try CLIReferenceDocument.rewrite(document)
        for id in ["server-experiment", "server-data"] {
            let after = try CLIReferenceDocument.extract(
                regionID: id, from: rewritten)
            let before = try CLIReferenceDocument.extract(
                regionID: id, from: document)
            #expect(after == before, "a Swift rewrite touched \(id)")
        }
    }

    /// A deleted marker is a failure, never a silent skip.
    @Test func aMissingRegionMarkerRefuses() throws {
        let document = try Self.committedDocument()
            .replacingOccurrences(
                of: CLIReferenceDocument.beginMarker("swift-workspace"), with: "")
        #expect(throws: (any Error).self) {
            try CLIReferenceDocument.drift(in: document)
        }
    }

    // MARK: `ClusterUsageTextIsGenerated`

    /// The audit's exemplar drift: `ClusterCLIVerb.usageText` was a
    /// hand-written literal with nothing asserting it covered `allCases` or
    /// agreed with the flag sets. It is generated now, and this is the test
    /// that says so.
    @Test func clusterUsageTextIsGenerated() {
        let text = ClusterCLIVerb.usageText
        for verb in ClusterCLIVerb.allCases {
            #expect(text.contains(verb.rawValue), "usage omits \(verb.rawValue)")
            #expect(
                !verb.purpose.isEmpty, "\(verb.rawValue) declares no purpose")
        }
        // Both target and job-class vocabularies are rendered from their own
        // enums rather than retyped.
        for target in ClusterLifecycleTarget.allCases {
            #expect(text.contains(target.cliName))
        }
    }

    /// Every cluster verb's synopsis names every flag its parser accepts —
    /// the property no test asserted before this step.
    @Test func clusterSynopsisNamesEveryAcceptedFlag() {
        for verb in ClusterCLIVerb.allCases {
            let synopsis = verb.synopsis
            for flag in verb.booleanFlags.union(verb.valueFlags) {
                #expect(
                    synopsis.contains(flag),
                    "\(verb.displayName) accepts \(flag) but its synopsis omits it")
            }
            #expect(synopsis.contains("--site") == verb.requiresSite)
        }
    }

    // MARK: `--help` renders from the same table

    @Test func everyVerbHasAPurposeAndASynopsis() {
        for spec in ExperimentCLIParser.specs {
            #expect(!spec.purpose.isEmpty, "\(spec.label) declares no purpose")
            let text = ExperimentCLIHelp.text(for: spec)
            #expect(text.hasPrefix("usage: steerlab-cli \(spec.label)"))
            #expect(text.contains(spec.purpose))
            for flag in spec.declaredFlags {
                #expect(
                    text.contains(flag),
                    "\(spec.label) --help omits its declared flag \(flag)")
            }
            #expect(text.hasSuffix("\n"))
        }
    }

    /// Contract text, not prose: the pages must not name an institution, a
    /// study, or an incident. The denylist is the same one the release scanner
    /// applies to the shipped tree.
    @Test(
        .enabled(
            if: ResearchTreeFixtures.hasExportDenylist,
            """
            scripts/export-denylist.txt must never ship (it carries the very \
            names the release must not contain), so this gate is \
            research-tree-only — the released tree's neutrality is proved by \
            CI's public-tier scan and AgentContractTests' literal denylist
            """))
    func helpTextIsNeutral() throws {
        let denylistURL = CodeResources.compiledCheckoutPath
            .appending(path: "scripts/export-denylist.txt")
        // Same rule the release scanner applies: one case-insensitive regex
        // per line, word-boundary anchored unless the line opens with `raw:`.
        let patterns = try String(contentsOf: denylistURL, encoding: .utf8)
            .split(separator: "\n")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }
            .map { line -> String in
                line.hasPrefix("raw:")
                    ? String(line.dropFirst(4)) : "\\b\(line)\\b"
            }
        var pages = ExperimentCLIParser.specs.map(ExperimentCLIHelp.text(for:))
        pages += ExperimentCLIHelp.families.map(ExperimentCLIHelp.familyText(for:))
        pages += ClusterCLIVerb.allCases.map(\.helpText)
        pages.append(ClusterCLIVerb.usageText)
        pages.append(ExperimentCLIHelp.topLevelText)
        for page in pages {
            for pattern in patterns {
                let regex = try NSRegularExpression(
                    pattern: pattern, options: [.caseInsensitive])
                let range = NSRange(page.startIndex ..< page.endIndex, in: page)
                #expect(
                    regex.firstMatch(in: page, range: range) == nil,
                    "a help page matches the denylist pattern '\(pattern)'")
            }
        }
    }

    // MARK: The top-level page (step 12)

    /// The audit's drift finding D8 in test form: the top-level page was a
    /// hand-written literal, and it omitted `panel` and `vectors`. An unlisted
    /// family is indistinguishable from an absent one to a caller with no
    /// checkout to read, so every family the binary dispatches must appear.
    @Test func topLevelPageNamesEveryDispatchedFamily() throws {
        let page = ExperimentCLIHelp.topLevelText
        for family in ExperimentCLIHelp.families {
            #expect(page.contains(family), "the top-level page omits '\(family)'")
        }
        // The families that live outside the declarative table — `cluster` has
        // its own, and these four are not on the agent path at all — are
        // exactly the ones a table-driven check would silently miss, so they
        // are named here by hand and read out of `main.swift`'s dispatch below.
        for family in ["cluster", "panel", "artifacts", "serve", "--config"] {
            #expect(page.contains(family), "the top-level page omits '\(family)'")
        }
        #expect(page.contains("--version"))
        #expect(page.contains(ExperimentCLIHelp.exitCodeLine))
    }

    /// …and the list is checked against what the binary actually dispatches,
    /// not against itself. `main.swift` compares `arguments[1]` against a
    /// literal for each family that is not in the declarative table.
    @Test func everyDispatchedFamilyInMainIsOnTheTopLevelPage() throws {
        let main = try String(
            contentsOf: CodeResources.compiledCheckoutPath
                .appending(path: "Sources/steerlab-cli/main.swift"),
            encoding: .utf8)
        let page = ExperimentCLIHelp.topLevelText
        // `arguments[1] == "serve"` and friends — the dispatch's own spelling.
        let pattern = #"arguments\[1\] == "([a-z-]+)""#
        let regex = try NSRegularExpression(pattern: pattern)
        let range = NSRange(main.startIndex ..< main.endIndex, in: main)
        var found: [String] = []
        for match in regex.matches(in: main, range: range) {
            guard let captured = Range(match.range(at: 1), in: main) else { continue }
            found.append(String(main[captured]))
        }
        #expect(!found.isEmpty, "no dispatched families were found in main.swift")
        for family in found {
            #expect(
                page.contains(family),
                "main.swift dispatches '\(family)' and the top-level page omits it")
        }
    }

    /// A family page names every verb of that family.
    @Test func familyPagesNameEveryVerbOfTheirFamily() {
        for family in ExperimentCLIHelp.families {
            let text = ExperimentCLIHelp.familyText(for: family)
            for spec in ExperimentCLIHelp.specs(inFamily: family) {
                // A bare-verb family (`init`) has no sub-verb to name, and its
                // family page IS its verb page — so the name to look for is
                // the label. `contains("")` is false in Swift, so the empty
                // sub-verb cannot be checked directly.
                #expect(text.contains(spec.verb.isEmpty ? spec.label : spec.verb))
            }
        }
    }
}
