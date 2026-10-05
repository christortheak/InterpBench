import Foundation
import Testing

@testable import ExperimentKit

/// WP0 step 10's gates. Three of them are drift gates, and each holds a
/// different pair of things together:
///
/// - `agentContractMatchesTheDraftDocument` — the shipped constant against
///   `WorkspaceGuide/core.md`, the human source of truth. Edit one
///   without the other and this fails instead of the two silently diverging.
/// - `agentContractNamesEveryAgentPathVerb` — the guide (the core contract
///   plus the topics `workspace guide` serves) against
///   `ExperimentCLIParser.specs`, the declarative surface itself. A verb added
///   to the CLI that the guide does not name is a lie by omission to every
///   agent that reads only the guide.
/// - `agentContractIsNeutral` / `shippedSeedTreesAreNeutral` /
///   `seededWorkspaceCarriesNoPrivateNames` — the contract, the two shipped
///   data trees (`WorkspaceSeed/`, `SampleWorkspace/`), and a freshly created
///   workspace against the private-name list (`PrivateNames`, read from
///   outside the repository). The shipped instrument must not carry this
///   researcher's study.
/// - `seedManifestAndSeedTreeAreTheSameSet` — WP1's explicit-allowlist
///   promise: no file seeds that `WorkspaceStore.seedManifest` does not
///   name, and no manifest line names a file that is not there.
@Suite(.serialized) struct AgentContractTests {

    // The private names themselves are NOT in this file, or anywhere in the
    // repository: `PrivateNames` reads them from a file outside it. The
    // guards below are skipped (and say so) on a machine with no list, and
    // fail when the list is required but missing — see `PrivateNames`.
    // Terms are matched case-insensitively as substrings, deliberately
    // blunt: a false positive costs one rename, a false negative ships a
    // name.

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

    @Test(.needsPrivateNames) func agentContractIsNeutral() throws {
        let names = try PrivateNames.required()
        let hits = names.hits(in: AgentContract.contents())
        #expect(hits.isEmpty, "AGENTS.md carries: \(hits.joined(separator: ", "))")
        // The topics are the same guide, served on demand: the same rule.
        for topic in WorkspaceGuide.topics {
            let hits = names.hits(in: topic.name + topic.summary + topic.text)
            #expect(
                hits.isEmpty,
                "guide topic \(topic.name) carries: \(hits.joined(separator: ", "))")
        }
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

    /// Private-name offenders in a tree: the RELATIVE path (the absolute one
    /// is the test machine's, not the tree's content) and the file's bytes.
    static func privateNameOffenders(
        under root: URL, names: PrivateNames.List
    ) throws -> [String] {
        var offenders: [String] = []
        for (relative, url) in try regularFiles(under: root) {
            for hit in names.hits(in: relative) {
                offenders.append("\(relative) (its path carries \(hit))")
            }
            guard let text = try? String(contentsOf: url, encoding: .utf8) else {
                continue  // not text; nothing shipped here is binary today
            }
            for hit in names.hits(in: text) {
                offenders.append("\(relative) (contains \(hit))")
            }
        }
        return offenders.sorted()
    }

    /// The two SHIPPED data trees, walked file by file against the
    /// private-name list: `WorkspaceSeed/` (what every new workspace is born
    /// with) and `SampleWorkspace/` (the recipe-only worked example). Before
    /// WP1 the seed tree was the research checkout's own `prompts/`, so
    /// seeding was a live path by which this study's names reached a
    /// workspace; the allowlist plus this gate is what replaced the old
    /// sweep-minus-exclusions boundary.
    @Test(.needsPrivateNames) func shippedSeedTreesAreNeutral() throws {
        let names = try PrivateNames.required()
        for tree in ["WorkspaceSeed", "SampleWorkspace"] {
            let root = Self.repoRoot.appending(path: tree)
            let offenders = try Self.privateNameOffenders(under: root, names: names)
            #expect(offenders.isEmpty, "\(tree) carries private names: \(offenders)")
            #expect(
                !(try Self.regularFiles(under: root)).isEmpty,
                "\(tree) is empty — the gate would pass vacuously")
        }
    }

    /// The guard has teeth: a planted term is caught in a file's bytes and
    /// in a file's name, case-insensitively — and reported by its position
    /// in the list, never by the term itself. Runs everywhere: it uses a
    /// made-up list, never the private one.
    @Test func aPlantedPrivateNameIsCaught() throws {
        let root = tempDirectory()
        let fm = FileManager.default
        defer { try? fm.removeItem(at: root) }
        try fm.createDirectory(
            at: root.appending(path: "prompts"), withIntermediateDirectories: true)
        try "A clean file.\n".write(
            to: root.appending(path: "prompts/clean.md"), atomically: true, encoding: .utf8)
        try "Run it on the Quuxcluster login node.\n".write(
            to: root.appending(path: "prompts/notes.md"), atomically: true, encoding: .utf8)
        try "nothing here\n".write(
            to: root.appending(path: "prompts/zorblab-items.md"), atomically: true,
            encoding: .utf8)

        let listURL = root.appending(path: "list.txt")
        try "# a made-up list\nquuxcluster\n\nallow:zorblabish\nZorbLab\n".write(
            to: listURL, atomically: true, encoding: .utf8)
        let names = try #require(
            PrivateNames.load(environment: [PrivateNames.fileVariable: listURL.path]))
        #expect(names.terms == ["quuxcluster", "zorblab"])

        let offenders = try Self.privateNameOffenders(
            under: root.appending(path: "prompts"), names: names)
        #expect(
            offenders == [
                "notes.md (contains list entry 1 (11 letters))",
                "zorblab-items.md (its path carries list entry 2 (7 letters))",
            ])
        #expect(!String(describing: names).contains("quuxcluster"))
        #expect(!String(reflecting: names).contains("quuxcluster"))
    }

    /// The loader's own contract: a missing or term-less list is ABSENT (so a
    /// guard can never pass vacuously against it), and the requirement switch
    /// is read from the environment.
    @Test func aMissingOrEmptyPrivateNameListIsAbsent() throws {
        let root = tempDirectory()
        let fm = FileManager.default
        defer { try? fm.removeItem(at: root) }
        try fm.createDirectory(at: root, withIntermediateDirectories: true)

        let missing = root.appending(path: "no-such-list.txt")
        #expect(PrivateNames.load(environment: [PrivateNames.fileVariable: missing.path]) == nil)

        let empty = root.appending(path: "empty.txt")
        try "# only a comment\n\nallow:something\n".write(
            to: empty, atomically: true, encoding: .utf8)
        #expect(PrivateNames.load(environment: [PrivateNames.fileVariable: empty.path]) == nil)

        #expect(
            PrivateNames.listURL(environment: [PrivateNames.fileVariable: missing.path]).path
                == missing.path)
        #expect(
            PrivateNames.listURL(environment: [:]).path.hasSuffix(
                "/" + PrivateNames.defaultRelativeLocation))
        #expect(!PrivateNames.isRequired(environment: [:]))
        #expect(!PrivateNames.isRequired(environment: [PrivateNames.requireVariable: "0"]))
        #expect(PrivateNames.isRequired(environment: [PrivateNames.requireVariable: "1"]))
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

    /// The seeded workspace's bytes and file names against the private-name
    /// list — the same walk as the shipped trees above, over what creation
    /// actually writes (the generated contract and marker included).
    @Test(.needsPrivateNames) func seededWorkspaceCarriesNoPrivateNames() throws {
        let names = try PrivateNames.required()
        let root = tempDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let created = try WorkspaceStore.create(at: root)
        let offenders = try Self.privateNameOffenders(under: created, names: names)
        #expect(
            offenders.isEmpty,
            "seeded workspace carries private names: \(offenders)")
    }

    /// The seeded workspace itself: exactly the manifest files, and
    /// CONCEPT-EMPTY (the demo concepts and the starter pack are no longer
    /// seeded — `SampleWorkspace/` is where a worked example lives, opened
    /// on purpose). Its private-name check is the test above.
    @Test func seededWorkspaceContentIsNeutral() throws {
        let root = tempDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let created = try WorkspaceStore.create(at: root)
        let fm = FileManager.default

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
    /// be named in the guide, inside a code span or a fenced block — the
    /// guide's own convention for naming a command. Since the guide was split
    /// into a short core and on-demand topics, "the guide" is the core plus
    /// every topic this client serves: the reference text moved, and this
    /// assertion moved with it. `remote` and `vectors`
    /// are out of scope here: they are the connection and parity families,
    /// documented in `CLI-REFERENCE`, and audit §4.2's table is scoped to the
    /// lifecycle an agent drives.
    @Test func agentContractNamesEveryAgentPathVerb() {
        // `panel` joined the list when `panel compile` landed (open-issues
        // §18): casting is authoring, it is the only headless way to make a
        // multi-agent study runnable, and a contract that did not name it
        // would send an agent back to hand-editing the manifest.
        // `authoring` joined it when the generation-prompt emitter landed:
        // §4.15's missing-data rule is unfollowable without naming the verb
        // that emits the prompt, and a contract that described the rule
        // without the command would send an agent back to improvising one.
        let namespaces: Set<String> = [
            "workspace", "data", "experiment", "panel", "authoring", "design", "agent",
        ]
        let code = Self.codeText(
            in: ([AgentContract.body] + WorkspaceGuide.topics.map(\.text))
                .joined(separator: "\n"))
        var missing: [String] = []
        for spec in ExperimentCLIParser.specs where namespaces.contains(spec.namespace) {
            if Self.mentions(verb: spec.verb, in: code) { continue }
            missing.append("\(spec.namespace) \(spec.verb)")
        }
        #expect(
            missing.isEmpty,
            "the agent guide does not name: \(missing.joined(separator: ", "))")

        // The gate has teeth: a verb that is not there is reported.
        #expect(!Self.mentions(verb: "teleport", in: code))
        // …and prose alone does not satisfy it — only code spans count.
        #expect(!Self.mentions(verb: "manipulation", in: code))
    }

    /// Text inside ``` fences and `inline spans`, which is where the guide
    /// writes commands. Prose mentions do not count: "run" and "list" are
    /// ordinary English, and a gate they satisfy is not a gate. Inline spans
    /// are read per paragraph (`WorkspaceGuideTests.codeSpans`), so a command
    /// wrapped across two lines is still one command.
    static func codeText(in markdown: String) -> String {
        WorkspaceGuideTests.codeSpans(in: markdown).map { $0 + "\n" }.joined()
    }

    static func mentions(verb: String, in code: String) -> Bool {
        code.range(
            of: "\\b\(NSRegularExpression.escapedPattern(for: verb))\\b",
            options: [.regularExpression]) != nil
    }

    // MARK: - Drift: the contract against the human document

    /// `WorkspaceGuide/core.md` stays the document a person reads and
    /// reviews; `AgentContract.body` is the copy that ships. This holds them
    /// byte-identical, modulo the generated header (which is not in the
    /// source) and the source's own `<!-- … -->` markers (which do not
    /// ship).
    @Test(
        .enabled(
            if: ResearchTreeFixtures.hasAgentContractDraft,
            """
            WorkspaceGuide/core.md is not in this checkout — it is the guide's \
            source, so a skip here means the checkout is incomplete
            """))
    func agentContractMatchesTheDraftDocument() throws {
        let draftURL = Self.repoRoot.appending(path: "WorkspaceGuide/core.md")
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
            AgentContract.body has drifted from WorkspaceGuide/core.md — edit \
            the source, then run scripts/ci/check-workspace-bootstrap.py --write
            """)

        // The draft's markers are the only difference, and they are real:
        // the emitted file must carry none of them.
        #expect(draft.contains("<!--"))
        #expect(!AgentContract.body.contains("<!--"))
        // The header is the one line the emitted file adds, and it carries the
        // hash of exactly the body under it — the proof that lets a later
        // build refresh this file without asking.
        #expect(AgentContract.contents() == AgentContract.generatedHeader + "\n\n" + stripped)
        #expect(AgentContract.generatedHeader.hasPrefix("<!--"))
        #expect(AgentContract.generatedHeader.hasSuffix("-->"))
        #expect(!AgentContract.generatedHeader.contains("\n"))
        #expect(
            AgentContract.declaredBodyHash(inHeaderLine: AgentContract.generatedHeader)
                == AgentContract.sha256Hex(stripped))
    }
}
