import Foundation
import ExperimentKit
import SteeringKit

// Headless runner. The GUI calls into this layer, not the reverse:
// anything needed for the paper must work from here (CLAUDE.md › Data &
// reproducibility). Configs carry a "task" field; smoke-test is the default
// for configs that predate it.

// The SAME claim `SteerLabApp.init` makes, for the SAME reason: this binary
// also ships inside SteerLab.app (Contents/Helpers/steerlab-cli, staged by
// scripts/build-app.sh), and a copy running out of a bundle is a distributed
// build whatever happens to exist at the compiled-in source path of the
// machine that assembled it. Without this, the bundled helper would silently
// stay in DEVELOPER mode on the maintainer's own Mac — resolving families out
// of that checkout instead of failing closed on one packaging forgot.
// A CLI installed by scripts/install-cli.sh is not in a bundle and is
// untouched.
if CodeResources.enclosingAppBundle != nil {
    CodeResources.releaseModeAsserted = true
}

var arguments = CommandLine.arguments

// Global --workspace <path>: applied before ANY verb runs (workspace-root
// resolution precedence #2 — see WorkspaceRoot; STEERLAB_WORKSPACE is #1 and
// needs no flag). Stripped from the argument list so verbs never see it.
if let flagIndex = arguments.firstIndex(of: "--workspace") {
    guard arguments.count > flagIndex + 1 else {
        FileHandle.standardError.write(
            Data("steerlab-cli: --workspace requires a directory path\n".utf8))
        exit(64)  // EX_USAGE
    }
    WorkspaceRoot.programmaticOverride =
        URL(filePath: arguments[flagIndex + 1]).standardizedFileURL
    arguments.removeSubrange(flagIndex ... flagIndex + 1)
}

// `--version` is the spelling every command-line tool answers to, and it is a
// rewrite rather than a second implementation: the report, its envelope, its
// `--json` mode, and its `--help` page all come from `install version`
// (WP0-AGENT-SURFACE-AUDIT §6.2, step 12). Anything after it — `--json`,
// `--out <file>` — is passed through untouched.
if arguments.count >= 2, arguments[1] == ExperimentCLIRunner.versionFlag {
    arguments.replaceSubrange(1 ... 1, with: ["install", "version"])
}

// The agent-path verb families — `init`, `workspace`, `data`, `vectors`,
// `remote`, `experiment`, `panel` — live in `ExperimentCLIRunner`
// (WP0-AGENT-SURFACE-AUDIT §7
// steps 4 and 5), the way `cluster` lives in `ClusterCLIRunner`. This rung is
// deliberately thin: parse → run → serialize, so the verbs are driveable
// in-process by tests and the binary is not a second implementation of them.
//
// `--json` produces exactly ONE `SteerLabCLIEnvelope` document on stdout, and
// the exit code then comes from the envelope's state vocabulary; without it,
// human prose and today's exit codes, unchanged.
if arguments.count >= 2, ExperimentCLIRunner.namespaces.contains(arguments[1]) {
    await runAgentPathCommand(arguments[1], Array(arguments.dropFirst(2)))
}

func runAgentPathCommand(_ namespace: String, _ args: [String]) async -> Never {
    // Honour --json even when parsing itself fails: an agent that asked for
    // machine output must not get prose back on the one path it cannot parse
    // (audit §2.2, the rule `cluster` already follows below).
    let invocation: ExperimentCLIInvocation
    do {
        invocation = try ExperimentCLIParser.parse(namespace: namespace, args)
    } catch {
        // The only throw is an undeclared flag. `run(namespace:)` re-parses
        // and produces the same refusal, so the fallback stays one code path.
        let outcome = await ExperimentCLIRunner(
            sink: args.contains("--json") ? .diagnostics : .standard
        ).run(namespace: namespace, args)
        emitAgentPathOutcome(outcome, json: args.contains("--json"), outPath: nil)
    }

    // In JSON mode stdout must carry the document and NOTHING else. Two
    // mechanisms, because the dispatch is not the only writer: the sink sends
    // the runner's own output to stderr, and file descriptor 1 is pointed at
    // stderr for the duration so that library code deeper in the call graph —
    // `ExperimentTasks` alone has 69 bare `print` sites on the model-loading
    // verbs — cannot land a progress line inside the document. The real
    // stdout is kept in `documentHandle` and is where the envelope goes.
    var documentHandle = FileHandle.standardOutput
    var savedStandardOutput: Int32 = -1
    if invocation.json {
        fflush(stdout)
        savedStandardOutput = dup(STDOUT_FILENO)
        if savedStandardOutput >= 0 {
            documentHandle = FileHandle(fileDescriptor: savedStandardOutput)
            dup2(STDERR_FILENO, STDOUT_FILENO)
        }
    }

    let outcome = await ExperimentCLIRunner(
        sink: invocation.json ? .diagnostics : .standard
    ).run(invocation)

    if invocation.json, savedStandardOutput >= 0 {
        fflush(stdout)
        dup2(savedStandardOutput, STDOUT_FILENO)
    }
    emitAgentPathOutcome(
        outcome, json: invocation.json, outPath: invocation.outPath,
        handle: documentHandle)
}

/// Serialize one outcome: the envelope in JSON mode, today's stderr prose
/// otherwise, plus the optional `--out` copy — which is written in BOTH modes,
/// because "give me the document in a file" is a separate request from "put
/// the document on stdout".
func emitAgentPathOutcome(
    _ outcome: ExperimentCLIOutcome, json: Bool, outPath: String?,
    handle: FileHandle = .standardOutput
) -> Never {
    let document =
        (try? outcome.envelope.jsonText())
        ?? "{\"schemaVersion\":1,\"state\":\"failed\"}\n"
    if json {
        handle.write(Data(document.utf8))
    } else if let text = ExperimentCLIRenderer.standardErrorText(outcome) {
        FileHandle.standardError.write(Data(text.utf8))
    }
    if let outPath {
        do {
            try document.write(
                to: URL(filePath: outPath), atomically: true, encoding: .utf8)
        } catch {
            FileHandle.standardError.write(
                Data("steerlab-cli: could not write --out \(outPath): \(error)\n".utf8))
        }
    }
    exit(outcome.exitCode(json: json))
}

// Web front end for remote/cluster use: same engine, browser client.
if arguments.count >= 2, arguments[1] == "serve" {
    let port: UInt16 =
        if let flagIndex = arguments.firstIndex(of: "--port"),
            arguments.count > flagIndex + 1,
            let value = UInt16(arguments[flagIndex + 1])
        { value } else { 8080 }
    do {
        try await SteerLabWebServer.run(port: port)
    } catch {
        FileHandle.standardError.write(Data("steerlab-cli serve: \(error)\n".utf8))
        exit(1)
    }
    exit(0)
}

// Artifact audit: reports legacy/ambiguous vector sidecars without mutating
// immutable run directories.
if arguments.count >= 2, arguments[1] == "artifacts" {
    do {
        try runArtifactsCommand(Array(arguments.dropFirst(2)))
    } catch {
        FileHandle.standardError.write(Data("steerlab-cli artifacts: \(error)\n".utf8))
        exit(1)
    }
    exit(0)
}

// Cluster lifecycle (docs/CLUSTER-CLI-LIFECYCLE-PLAN.md Phase C). This block is
// deliberately thin: parse → ExperimentKit → serialize. All orchestration lives
// in `ClusterCLIRunner`, which the wizard's own operations back, so the CLI can
// never become a second cluster implementation.
if arguments.count >= 2, arguments[1] == "cluster" {
    await runClusterCommand(Array(arguments.dropFirst(2)))
}

func runClusterCommand(_ args: [String]) async -> Never {
    // Honour --json even when parsing itself fails: an agent that asked for
    // machine output must not get prose back on the one path it cannot parse.
    let wantsJSON = args.contains("--json")

    // `cluster --help` with no verb: the generated family page, exit 0.
    // Asking what the verbs are is not a usage error, which is what it cost
    // before step 11.
    if args.contains("--help"), ClusterCLIVerb.match(args) == nil {
        FileHandle.standardOutput.write(Data((ClusterCLIVerb.usageText + "\n").utf8))
        exit(0)
    }

    func fail(_ error: ClusterCLIError) -> Never {
        if wantsJSON {
            let envelope = ClusterCLIEnvelope.failure(
                verb: "cluster", code: error.code, reason: error.reason,
                repairAction: error.repairAction, state: .blocked)
            let text = (try? envelope.jsonText()) ?? "{\"schemaVersion\":1}\n"
            FileHandle.standardOutput.write(Data(text.utf8))
        } else {
            FileHandle.standardError.write(
                Data("steerlab-cli cluster: \(error.reason)\n  \(error.repairAction)\n\n".utf8))
            FileHandle.standardError.write(Data((ClusterCLIVerb.usageText + "\n").utf8))
        }
        exit(error.exitCode)
    }

    guard !args.isEmpty else { fail(.unknownVerb("")) }
    let invocation: ClusterCLIInvocation
    do {
        invocation = try ClusterCLIParser.parse(args)
    } catch let error as ClusterCLIError {
        fail(error)
    } catch {
        fail(.unknownVerb(args.first ?? ""))
    }

    // `cluster <verb> --help`: the generated page on stdout, nothing run. In
    // JSON mode it travels as the envelope's `message`, so the
    // one-document rule holds in both modes.
    if invocation.help, !invocation.json {
        FileHandle.standardOutput.write(Data(invocation.verb.helpText.utf8))
        exit(0)
    }

    // Streamed log lines (`controller logs --follow`) go to stderr in JSON mode
    // so stdout carries EXACTLY ONE JSON document (plan §6.5).
    let sink: FileHandle = invocation.json ? .standardError : .standardOutput
    let outcome = await ClusterCLIRunner().run(invocation) { line in
        sink.write(Data((line + "\n").utf8))
    }
    if invocation.json {
        let text = (try? outcome.envelope.jsonText())
            ?? "{\"schemaVersion\":1,\"state\":\"failed\"}\n"
        FileHandle.standardOutput.write(Data(text.utf8))
    } else {
        FileHandle.standardOutput.write(
            Data(ClusterCLIRenderer.humanText(outcome.envelope).utf8))
    }
    exit(outcome.exitCode)
}

func runArtifactsCommand(_ args: [String]) throws {
    switch args.first {
    case "audit":
        let findings = VectorCatalog.auditArtifacts()
        if findings.isEmpty {
            print("artifact audit OK — no legacy or ambiguous vector sidecars found")
            return
        }
        for finding in findings {
            print("[\(finding.severity.rawValue)] \(finding.artifactPath)")
            print("  issue: \(finding.issue)")
            print("  recommendation: \(finding.recommendation)")
        }
        if args.contains("--json") {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(findings)
            let url = VectorCatalog.runsDirectory.appending(component: "artifact-audit.json")
            try data.write(to: url)
            print("wrote \(url.path)")
        }
    default:
        throw ExperimentError(reason: "usage: artifacts audit [--json]")
    }
}

// The top-level page. Every family the binary dispatches is named on it — the
// audit's drift finding D8 was this page's hand-written literal omitting
// `panel` and `vectors`, which is exactly the class of omission a caller
// cannot recover from: an unlisted verb is indistinguishable from an absent
// one. It is generated from `ExperimentCLIHelp.topLevelEntries` now, the
// document's `swift-global` region renders from the same table, and
// `topLevelPageNamesEveryDispatchedFamily` is the gate.
func topLevelUsageText() -> String { ExperimentCLIHelp.topLevelText }

// `steerlab-cli --help` with no family: the top-level page, exit 0.
if arguments.count >= 2, arguments[1] == "--help" {
    FileHandle.standardOutput.write(Data(topLevelUsageText().utf8))
    exit(0)
}

guard arguments.count >= 3, arguments[1] == "--config" else {
    FileHandle.standardError.write(Data(topLevelUsageText().utf8))
    exit(64)  // EX_USAGE
}

let configURL = URL(filePath: arguments[2])

struct TaskPeek: Decodable {
    let task: String?
}

do {
    let data = try Data(contentsOf: configURL)
    let task = (try JSONDecoder().decode(TaskPeek.self, from: data)).task ?? "smoke-test"

    switch task {
    case "smoke-test":
        let config = try JSONDecoder().decode(SmokeTestConfig.self, from: data)
        try await SmokeTest.run(config: config)
        print("SMOKE TEST PASSED (\(config.models.map(\.id).joined(separator: ", ")))")
    case "toy-concept":
        let config = try JSONDecoder().decode(ToyConceptConfig.self, from: data)
        try await ToyConceptRun.run(config: config)
        print("TOY CONCEPT PASSED (\(config.models.map(\.id).joined(separator: ", ")))")
    default:
        FileHandle.standardError.write(Data("steerlab-cli: unknown task '\(task)'\n".utf8))
        exit(64)
    }
} catch {
    FileHandle.standardError.write(Data("steerlab-cli: \(error)\n".utf8))
    exit(1)
}

