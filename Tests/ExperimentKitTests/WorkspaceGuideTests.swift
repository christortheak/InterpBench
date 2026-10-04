import Foundation
import Testing

@testable import ExperimentKit

/// The agent guide after its split: a short, client-neutral core written into
/// every workspace as `AGENTS.md`, and topics served on demand by
/// `workspace guide [<topic>]` with this client's own commands.
///
/// What these hold together:
///
/// - the core stays within its size budget, so it cannot quietly grow back
///   into a verb reference;
/// - both clients offer the same topic names, and the core's topic list is
///   exactly that set;
/// - every topic resolves through the verb, in both output modes;
/// - no topic names a command this client does not have, and none shows the
///   other client's commands.
@Suite(.serialized) struct WorkspaceGuideTests {

    private static var repoRoot: URL {
        URL(filePath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    // MARK: - The core's budget

    /// At most about 300 lines. The budget is the point of the split: a core
    /// that grows past it should lose text to a topic, not gain a waiver.
    @Test func theCoreGuideStaysWithinItsBudget() {
        let lines = AgentContract.body.split(
            separator: "\n", omittingEmptySubsequences: false
        ).count - 1
        #expect(lines <= 300, "the core guide is \(lines) lines; move depth into a topic")
        #expect(
            AgentContract.body.utf8.count <= 18_000,
            "the core guide is \(AgentContract.body.utf8.count) bytes; move depth into a topic")
        // …and it is not a stub either: the sections the brief names are there.
        for heading in [
            "## Collaborate at the researcher's level", "## Ask before you spend",
            "## Choose the installed client", "## Discover what the client can do",
            "## The study lifecycle", "## The machine contract, in brief",
            "## Immutability", "## What not to do", "## Topics on demand",
        ] {
            #expect(AgentContract.body.contains(heading), "the core guide lost: \(heading)")
        }
    }

    /// The core is byte-identical from both clients: the Python client's
    /// packaged copy is this build's body.
    @Test func theCoreGuideIsTheSameBytesOnBothClients() throws {
        let packaged = try String(
            contentsOf: Self.repoRoot.appending(
                path: "Server/steerlab_server/client/resources/agent-guide.md"),
            encoding: .utf8)
        #expect(packaged == AgentContract.body)
    }

    // MARK: - Topics

    @Test func everyTopicResolvesAndIsListedInTheCore() {
        let topics = WorkspaceGuide.topics
        #expect(topics.count >= 10)
        #expect(Set(topics.map(\.name)).count == topics.count, "a topic name repeats")
        for topic in topics {
            #expect(WorkspaceGuide.topic(named: topic.name) == topic)
            #expect(topic.text.hasPrefix("# "), "\(topic.name) has no title line")
            #expect(topic.text.hasSuffix("\n") && !topic.text.hasSuffix("\n\n"))
            #expect(!topic.text.contains("{{"), "\(topic.name) kept an unrendered token")
            #expect(!topic.text.contains("<!--"), "\(topic.name) kept a source marker")
            // The core's own list is how a reader learns the name exists.
            #expect(
                AgentContract.body.contains("- `\(topic.name)` — \(topic.summary)\n"),
                "the core guide does not list the topic \(topic.name)")
        }
        #expect(WorkspaceGuide.topic(named: "teleport") == nil)
    }

    /// Both clients offer the same topic names, in the same order, with the
    /// same summaries. The texts differ on purpose; the names must not.
    @Test func bothClientsOfferTheSameTopicNames() throws {
        struct Index: Decodable {
            struct Entry: Decodable { let name: String; let summary: String }
            let topics: [Entry]
        }
        let resources = Self.repoRoot.appending(
            path: "Server/steerlab_server/client/resources")
        let python = try JSONDecoder().decode(
            Index.self,
            from: Data(contentsOf: resources.appending(component: "agent-guide-topics.json")))
        #expect(python.topics.map(\.name) == WorkspaceGuide.topics.map(\.name))
        #expect(python.topics.map(\.summary) == WorkspaceGuide.topics.map(\.summary))
        for entry in python.topics {
            let file = resources.appending(component: "agent-guide-topic-\(entry.name).md")
            #expect(
                FileManager.default.fileExists(atPath: file.path),
                "the Python client lists \(entry.name) but packages no text for it")
        }
        let source = try JSONDecoder().decode(
            Index.self,
            from: Data(
                contentsOf: Self.repoRoot.appending(path: "WorkspaceGuide/topics.json")))
        #expect(source.topics.map(\.name) == WorkspaceGuide.topics.map(\.name))
    }

    // MARK: - The verb

    private func withTemporaryRoot<T>(_ body: (URL) async throws -> T) async rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        let root = FileManager.default.temporaryDirectory
            .appending(component: "guide-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        ExperimentStore.rootOverride = root
        defer {
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: root)
            ExperimentRootOverrideLock.release()
        }
        return try await body(root)
    }

    @Test func workspaceGuideListsTheTopics() async throws {
        try await withTemporaryRoot { root in
            let recorder = ExperimentCLIRecorder()
            let outcome = await ExperimentCLIRunner(sink: recorder.sink)
                .run(namespace: "workspace", ["guide", "--json"])
            #expect(outcome.exitCode == 0)
            #expect(outcome.envelope.verb == "workspace guide")
            #expect(outcome.envelope.changed == false)
            let result = try #require(outcome.envelope.result)
            #expect(result["topics"] == WorkspaceGuide.index)
            #expect(result["client"] == .string("steerlab-cli"))
            #expect(result["text"] == nil)

            let human = ExperimentCLIRecorder()
            _ = await ExperimentCLIRunner(sink: human.sink)
                .run(namespace: "workspace", ["guide"])
            for topic in WorkspaceGuide.topics {
                #expect(human.standardOutput.contains("\(topic.name) — \(topic.summary)\n"))
            }
            // Reading guidance never writes: no contract is seeded into a
            // folder that is not a workspace, and nothing else appears.
            #expect(try FileManager.default.contentsOfDirectory(atPath: root.path).isEmpty)
        }
    }

    @Test func workspaceGuidePrintsOneTopicInBothModes() async throws {
        try await withTemporaryRoot { _ in
            for topic in WorkspaceGuide.topics {
                let recorder = ExperimentCLIRecorder()
                let outcome = await ExperimentCLIRunner(sink: recorder.sink)
                    .run(namespace: "workspace", ["guide", topic.name, "--json"])
                #expect(outcome.exitCode == 0, "\(topic.name) did not resolve")
                let result = try #require(outcome.envelope.result)
                #expect(result["topic"] == .string(topic.name))
                #expect(result["text"] == .string(topic.text))
                #expect(result["topics"] == WorkspaceGuide.index)
            }
            let human = ExperimentCLIRecorder()
            let outcome = await ExperimentCLIRunner(sink: human.sink)
                .run(namespace: "workspace", ["guide", "freeze"])
            #expect(outcome.exitCode == 0)
            #expect(human.standardOutput == WorkspaceGuide.topic(named: "freeze")?.text)
        }
    }

    /// An unknown topic is a malformed request with a repair the reader can
    /// carry out: the names to choose from, and the command that lists them.
    @Test func anUnknownTopicNamesTheOnesThatExist() async throws {
        try await withTemporaryRoot { _ in
            let recorder = ExperimentCLIRecorder()
            let outcome = await ExperimentCLIRunner(sink: recorder.sink)
                .run(namespace: "workspace", ["guide", "teleport", "--json"])
            #expect(outcome.exitCode != 0)
            #expect(outcome.envelope.state == .blocked)
            let error = try #require(outcome.envelope.error)
            #expect(error.code == "usage")
            #expect(error.reason.contains("teleport"))
            let repair = try #require(error.repairAction)
            for topic in WorkspaceGuide.topics { #expect(repair.contains(topic.name)) }
            #expect(repair.contains("steerlab-cli workspace guide"))

            let extra = await ExperimentCLIRunner(sink: ExperimentCLIRecorder().sink)
                .run(namespace: "workspace", ["guide", "freeze", "sweep", "--json"])
            #expect(extra.envelope.state == .blocked)
        }
    }

    // MARK: - No topic names a verb this client does not have

    /// Fenced lines, plus inline code spans read per PARAGRAPH so that a span
    /// wrapped across two lines is still one span.
    static func codeSpans(in markdown: String) -> [String] {
        var spans: [String] = []
        var paragraph: [Substring] = []
        var inFence = false
        func flush() {
            let parts = paragraph.joined(separator: "\n")
                .split(separator: "`", omittingEmptySubsequences: false)
            for (index, part) in parts.enumerated() where index % 2 == 1 {
                spans.append(
                    part.split(whereSeparator: \.isWhitespace).joined(separator: " "))
            }
            paragraph.removeAll()
        }
        for line in markdown.split(separator: "\n", omittingEmptySubsequences: false) {
            if line.hasPrefix("```") {
                flush()
                inFence.toggle()
            } else if inFence {
                spans.append(line.trimmingCharacters(in: .whitespaces))
            } else if line.allSatisfy(\.isWhitespace) {
                flush()
            } else {
                paragraph.append(line)
            }
        }
        flush()
        return spans
    }

    /// Every verb family either client has. A code span that opens with one of
    /// these words followed by a verb is a command, and must be this client's.
    static let families: Set<String> = [
        "workspace", "setup", "science", "experiment", "concept", "bundle", "pack",
        "design", "agent", "model", "authoring", "runner", "run", "panel", "data",
        "vectors", "remote", "cluster", "docs", "install", "init",
    ]

    static func isWord(_ token: Substring) -> Bool {
        guard let first = token.first, first.isLowercase, first.isASCII else { return false }
        return token.allSatisfy { ($0.isLowercase && $0.isASCII) || $0 == "-" }
    }

    /// A bare hyphenated word shaped like one of the two clients' verbs
    /// (`set-protocol`, `pin-rubric`, `submit-bundle`): it has to be a verb
    /// THIS client declares.
    static func looksLikeAVerb(_ token: Substring) -> Bool {
        guard isWord(token), let dash = token.firstIndex(of: "-") else { return false }
        return [
            "set", "pin", "declare", "remove", "attach", "detach", "import", "inspect",
            "submit", "model", "science", "cleanup", "extract", "rescore", "mirror",
            "backfill",
        ].contains(String(token[..<dash]))
    }

    /// What a text says that `table` (family → verbs) does not back.
    static func unknownCommands(
        in text: String, executable: String, otherExecutable: String,
        table: [String: Set<String>]
    ) -> [String] {
        let verbNames = Set(table.values.flatMap { $0 })
        var found: [String] = []
        for span in codeSpans(in: text) {
            var tokens = span.split(separator: " ")[...]
            guard let first = tokens.first, !first.hasPrefix("#"),
                first != "steerlab-server"
            else { continue }
            if first == otherExecutable, tokens.count > 1, isWord(tokens[tokens.startIndex + 1]) {
                found.append("the other client's command: \(span)")
                continue
            }
            let explicit = first == executable
            if explicit { tokens = tokens.dropFirst() }
            guard let head = tokens.first else { continue }
            if families.contains(String(head)) {
                let next = tokens.dropFirst().first.map {
                    $0.split(separator: "/", omittingEmptySubsequences: false).first ?? $0
                }
                guard let verb = next, isWord(verb) else {
                    if explicit, table[String(head)] == nil {
                        found.append("no such family: \(span)")
                    }
                    continue
                }
                if table[String(head)]?.contains(String(verb)) != true {
                    found.append("no such verb: \(span)")
                }
            } else if explicit, isWord(head) {
                found.append("no such family: \(span)")
            } else if looksLikeAVerb(head), !verbNames.contains(String(head)) {
                found.append("a verb this client does not have: \(span)")
            }
        }
        return found
    }

    /// The Mac command line's verbs, from the parsers themselves.
    static var macVerbTable: [String: Set<String>] {
        var table: [String: Set<String>] = [:]
        for spec in ExperimentCLIParser.specs {
            table[spec.namespace, default: []].insert(spec.verb)
        }
        table["cluster"] = Set(
            ClusterCLIVerb.allCases.compactMap {
                $0.rawValue.split(separator: " ").first.map(String.init)
            })
        return table
    }

    @Test func noTopicNamesAVerbTheMacCommandLineLacks() {
        let table = Self.macVerbTable
        for topic in WorkspaceGuide.topics {
            let unknown = Self.unknownCommands(
                in: topic.text, executable: "steerlab-cli", otherExecutable: "steerlab",
                table: table)
            #expect(unknown.isEmpty, "topic \(topic.name): \(unknown)")
        }
        // The core is client-neutral, so every command it spells has to exist
        // on this client too.
        let core = Self.unknownCommands(
            in: AgentContract.body, executable: "steerlab-cli",
            otherExecutable: "\u{0}", table: table)
        #expect(core.isEmpty, "core guide: \(core)")

        // The gate has teeth, in each direction it guards.
        let planted = """
            Use `steerlab-cli experiment teleport <name>` and then `runner serve`.
            Also `set-protocol`, and the Python client's `steerlab run <name>`.
            A wrapped span still counts: `steerlab-cli experiment
            teleport`.
            """
        let caught = Self.unknownCommands(
            in: planted, executable: "steerlab-cli", otherExecutable: "steerlab",
            table: table)
        #expect(caught.count == 5, "\(caught)")
        // …and it does not cry wolf at real commands or at ordinary spans.
        let fine = """
            `steerlab-cli experiment attach <name>`, `data check <name>`, `cluster auth
            open`, `experiment.json`, `runs/<timestamp>-<slug>/`, `validate`, `--json`,
            `steerlab-server battery run <file>`, `set-sampling`, `remote <verb> --site <id>`.
            """
        #expect(
            Self.unknownCommands(
                in: fine, executable: "steerlab-cli", otherExecutable: "steerlab",
                table: table
            ).isEmpty)
    }
}
