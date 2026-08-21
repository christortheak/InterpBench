import Foundation
import Testing

@testable import ExperimentKit

/// WP0 step 10's gates. Three of them are drift gates, and each holds a
/// different pair of things together:
///
/// - `agentContractMatchesTheDraftDocument` — the shipped constant against
///   `docs/AGENTS-WORKSPACE-DRAFT.md`, the human source of truth. Edit one
///   without the other and this fails instead of the two silently diverging.
/// - `agentContractNamesEveryAgentPathVerb` — the contract against
///   `ExperimentCLIParser.specs`, the declarative surface itself. A verb added
///   to the CLI that the contract does not name is a lie by omission to every
///   agent that reads only the contract.
/// - `agentContractIsNeutral` / `shippedSeedTreesAreNeutral` /
///   `seededWorkspaceContentIsNeutral` — the contract, the two shipped data
///   trees (`WorkspaceSeed/`, `SampleWorkspace/`), and a freshly created
///   workspace against the private name denylist. The shipped instrument
///   must not carry this researcher's study.
/// - `seedManifestAndSeedTreeAreTheSameSet` — WP1's explicit-allowlist
///   promise: no file seeds that `WorkspaceStore.seedManifest` does not
///   name, and no manifest line names a file that is not there.
@Suite(.serialized) struct AgentContractTests {

    /// The private denylist as literals, not read from
    /// `scripts/export-denylist.txt`: that file must never ship (it carries
    /// the identifiers the release must not contain), and a test that reads
    /// it would pass vacuously if it ever went missing. Terms are matched
    /// case-insensitively as substrings — deliberately blunter than the
    /// scanner's word-boundary regexes, because a false positive here costs
    /// one rename and a false negative ships a name.
    static let denylist = [
        "sapelo", "gacrc", "uga.edu", "cmtlab",
        "katz", "zamir", "imhoff", "posner",
        "turner", "cturner",
    ]

    static func denylistHits(in text: String) -> [String] {
        let lowered = text.lowercased()
        return denylist.filter { lowered.contains($0) }
    }

    private static var repoRoot: URL {
        URL(filePath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func tempDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appending(component: "agents-\(UUID().uuidString)")
    }

    // MARK: - Neutrality

    @Test func agentContractIsNeutral() {
        let hits = Self.denylistHits(in: AgentContract.contents())
        #expect(hits.isEmpty, "AGENTS.md names: \(hits.joined(separator: ", "))")
    }

    /// Every regular file under a tree, workspace-relative, with `.git`
    /// pruned (a workspace's own history is not content).
    static func regularFiles(under root: URL) throws -> [(relative: String, url: URL)] {
        let fm = FileManager.default
        // Resolve both sides: the enumerator hands back `/private/var/…`
        // while `root` may be the `/var/…` symlink, and a prefix strip that
        // misses leaves an absolute path in `relative`.
        let base = root.resolvingSymlinksInPath().standardizedFileURL.path + "/"
        var out: [(String, URL)] = []
        guard
            let enumerator = fm.enumerator(
                at: root, includingPropertiesForKeys: [.isRegularFileKey])
        else { return [] }
        for case let url as URL in enumerator {
            let full = url.resolvingSymlinksInPath().standardizedFileURL.path
            let relative =
                full.hasPrefix(base)
                ? String(full.dropFirst(base.count)) : url.lastPathComponent
            if relative == ".git" || relative.hasPrefix(".git/") {
                enumerator.skipDescendants()
                continue
            }
            guard
                (try? url.resourceValues(forKeys: [.isRegularFileKey]))?
                    .isRegularFile == true
            else { continue }
            out.append((relative, url))
        }
        return out.sorted { $0.0 < $1.0 }
    }

    /// Denylist offenders in a tree: the RELATIVE path (the absolute one is
    /// the test machine's, not the tree's content) and the file's bytes.
    static func denylistOffenders(under root: URL) throws -> [String] {
        var offenders: [String] = []
        for (relative, url) in try regularFiles(under: root) {
            for hit in denylistHits(in: relative) {
                offenders.append("\(relative) (path names '\(hit)')")
            }
            guard let text = try? String(contentsOf: url, encoding: .utf8) else {
                continue  // not text; nothing shipped here is binary today
            }
            for hit in denylistHits(in: text) {
                offenders.append("\(relative) (contains '\(hit)')")
            }
        }
        return offenders.sorted()
    }

    /// The two SHIPPED data trees, walked file by file against the private
    /// name denylist: `WorkspaceSeed/` (what every new workspace is born
    /// with) and `SampleWorkspace/` (the recipe-only worked example). Before
    /// WP1 the seed tree was the research checkout's own `prompts/`, so
    /// seeding was a live path by which this study's names reached a
    /// workspace; the allowlist plus this gate is what replaced the old
    /// sweep-minus-exclusions boundary.
    @Test func shippedSeedTreesAreNeutral() throws {
        for tree in ["WorkspaceSeed", "SampleWorkspace"] {
            let root = Self.repoRoot.appending(path: tree)
            let offenders = try Self.denylistOffenders(under: root)
            #expect(offenders.isEmpty, "\(tree) carries private names: \(offenders)")
            #expect(
                !(try Self.regularFiles(under: root)).isEmpty,
                "\(tree) is empty — the gate would pass vacuously")
        }
    }

    /// The explicit-allowlist promise, enforced in both directions: every
    /// manifest entry exists in `WorkspaceSeed/`, and every file in
    /// `WorkspaceSeed/` is in the manifest. A stowaway — a file dropped into
    /// the seed tree without a manifest line — is exactly what the old
    /// directory sweep made possible.
    @Test func seedManifestAndSeedTreeAreTheSameSet() throws {
        let seedRoot = Self.repoRoot.appending(path: "WorkspaceSeed")
        let onDisk = Set(try Self.regularFiles(under: seedRoot).map(\.relative))
            .subtracting([".DS_Store"])
        let declared = Set(WorkspaceStore.seedManifest)

        let missing = declared.subtracting(onDisk).sorted()
        let stowaways = onDisk.subtracting(declared).sorted()
        #expect(
            missing.isEmpty,
            "manifest names files WorkspaceSeed/ does not have: \(missing)")
        #expect(
            stowaways.isEmpty,
            "WorkspaceSeed/ carries files the manifest does not name: \(stowaways)")
        #expect(
            WorkspaceStore.seedManifest.count == declared.count,
            "the manifest lists a path twice")
    }

    /// The seeded workspace itself: neutral bytes, exactly the manifest
    /// files, and CONCEPT-EMPTY (the demo concepts and the starter pack are
    /// no longer seeded — `SampleWorkspace/` is where a worked example
    /// lives, opened on purpose).
    @Test func seededWorkspaceContentIsNeutral() throws {
        let root = tempDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let created = try WorkspaceStore.create(at: root)
        let fm = FileManager.default

        let offenders = try Self.denylistOffenders(under: created)
        #expect(
            offenders.isEmpty,
            "seeded workspace carries private names: \(offenders)")

        // Exactly the manifest, plus the three files creation GENERATES.
        let generated: Set<String> = [
            WorkspaceStore.markerFileName, AgentContract.fileName, ".gitignore",
        ]
        let present = Set(try Self.regularFiles(under: created).map(\.relative))
        let expected = Set(WorkspaceStore.seedManifest).union(generated)
        let extra = present.subtracting(expected).sorted()
        let absent = expected.subtracting(present).sorted()
        #expect(
            present == expected,
            "seeded workspace differs from the manifest: extra \(extra), missing \(absent)")

        // Spot-checks with teeth: the instruments a workspace needs on day
        // one are there…
        for expected in [
            "prompts/rubrics/default-paired-v1.md",
            "prompts/batteries/basic.jsonl",
            "prompts/dev/dev-prompts.jsonl",
            "prompts/neutral/corpus.jsonl",
            "prompts/parsers/parser-registry.json",
        ] {
            #expect(
                fm.fileExists(atPath: created.appending(path: expected).path),
                "missing seeded instrument: \(expected)")
        }
        // …and the study material that used to ride along is not.
        var conceptsIsDirectory: ObjCBool = false
        let concepts = created.appending(path: "prompts/concepts")
        #expect(
            fm.fileExists(atPath: concepts.path, isDirectory: &conceptsIsDirectory)
                && conceptsIsDirectory.boolValue,
            "the concepts directory must still be created, just empty")
        #expect(
            (try? fm.contentsOfDirectory(atPath: concepts.path))?
                .filter { $0 != ".DS_Store" }.isEmpty == true,
            "a fresh workspace must be concept-empty")
        for absent in [
            "prompts/concepts/french", "prompts/concepts/golden-gate-bridge",
            "prompts/concepts/formality", "prompts/tasks/starter-prompts.jsonl",
            "prompts/batteries/starter-battery.jsonl",
            "prompts/generation/COWORK-JOB-optvec-datasets.md",
            "prompts/panels/templates/deliberative-appellate-panel-v1.json",
            "prompts/panels/templates/deliberative-appellate-panel-v2.json",
        ] {
            #expect(
                !fm.fileExists(atPath: created.appending(path: absent).path),
                "no longer seeded, but present: \(absent)")
        }
        // The verb's own usage promises this directory; seeding still makes it.
        var isDirectory: ObjCBool = false
        #expect(
            fm.fileExists(
                atPath: created.appending(
                    path: ExperimentStore.taxonomiesRelativeDirectory()).path,
                isDirectory: &isDirectory) && isDirectory.boolValue)
    }

    // MARK: - Drift: the contract against the CLI surface

    /// Every verb the parser declares on the authoring/lifecycle surface must
    /// be named in the contract, inside a code span or a fenced block — the
    /// contract's own convention for naming a command. `remote` and `vectors`
    /// are out of scope here: they are the connection and parity families,
    /// documented in `CLI-REFERENCE`, and audit §4.2's table is scoped to the
    /// lifecycle an agent drives.
    @Test func agentContractNamesEveryAgentPathVerb() {
        // `panel` joined the list when `panel compile` landed (open-issues
        // §18): casting is authoring, it is the only headless way to make a
        // multi-agent study runnable, and a contract that did not name it
        // would send an agent back to hand-editing the manifest.
        let namespaces: Set<String> = ["workspace", "data", "experiment", "panel"]
        let code = Self.codeText(in: AgentContract.body)
        var missing: [String] = []
        for spec in ExperimentCLIParser.specs where namespaces.contains(spec.namespace) {
            if Self.mentions(verb: spec.verb, in: code) { continue }
            missing.append("\(spec.namespace) \(spec.verb)")
        }
        #expect(
            missing.isEmpty,
            "AGENTS.md does not name: \(missing.joined(separator: ", "))")

        // The gate has teeth: a verb that is not there is reported.
        #expect(!Self.mentions(verb: "teleport", in: code))
        // …and prose alone does not satisfy it — only code spans count.
        #expect(!Self.mentions(verb: "manipulation", in: code))
    }

    /// Text inside ``` fences and `inline spans`, which is where the contract
    /// writes commands. Prose mentions do not count: "run" and "list" are
    /// ordinary English, and a gate they satisfy is not a gate.
    static func codeText(in markdown: String) -> String {
        var out = ""
        var inFence = false
        for line in markdown.split(separator: "\n", omittingEmptySubsequences: false) {
            if line.hasPrefix("```") {
                inFence.toggle()
                continue
            }
            if inFence {
                out += line + "\n"
                continue
            }
            let parts = line.split(separator: "`", omittingEmptySubsequences: false)
            for (index, part) in parts.enumerated() where index % 2 == 1 {
                out += part + "\n"
            }
        }
        return out
    }

    static func mentions(verb: String, in code: String) -> Bool {
        code.range(
            of: "\\b\(NSRegularExpression.escapedPattern(for: verb))\\b",
            options: [.regularExpression]) != nil
    }

    // MARK: - Drift: the contract against the human document

    /// `docs/AGENTS-WORKSPACE-DRAFT.md` stays the document a person reads and
    /// reviews; `AgentContract.body` is the copy that ships. This holds them
    /// byte-identical, modulo the generated header (which is not in the
    /// draft) and the draft's own `<!-- … -->` planning markers (which do not
    /// ship).
    @Test(
        .enabled(
            if: ResearchTreeFixtures.hasAgentContractDraft,
            """
            docs/AGENTS-WORKSPACE-DRAFT.md is not in this checkout — it is \
            allowlisted to ship, so a skip here means the export dropped it
            """))
    func agentContractMatchesTheDraftDocument() throws {
        let draftURL = Self.repoRoot.appending(path: "docs/AGENTS-WORKSPACE-DRAFT.md")
        let draft = try String(contentsOf: draftURL, encoding: .utf8)

        let stripped =
            draft
            .split(separator: "\n", omittingEmptySubsequences: false)
            .filter {
                let trimmed = $0.trimmingCharacters(in: .whitespaces)
                return !(trimmed.hasPrefix("<!--") && trimmed.hasSuffix("-->"))
            }
            .joined(separator: "\n")

        #expect(
            AgentContract.body == stripped,
            """
            AgentContract.body has drifted from docs/AGENTS-WORKSPACE-DRAFT.md \
            — edit the draft first, then mirror it into AgentContract.swift
            """)

        // The draft's markers are the only difference, and they are real:
        // the emitted file must carry none of them.
        #expect(draft.contains("<!--"))
        #expect(!AgentContract.body.contains("<!--"))
        // The header is the one line the emitted file adds.
        #expect(AgentContract.contents() == AgentContract.generatedHeader + "\n\n" + stripped)
        #expect(AgentContract.generatedHeader.hasPrefix("<!--"))
    }
}
