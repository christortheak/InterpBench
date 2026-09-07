import Foundation

/// The Python archive owner is the portable format authority. Mac adapters call
/// its local-only process entry point; no HTTP, model load or shell is involved.
public enum DiagnosticWorkspace {
    public static let actions = ["sae-check", "sae-show", "sae-pin-plan", "sae-pin", "interview", "draft", "publish", "input-plan", "package", "import", "custody", "verify-custody"]

    public static func perform(_ action: String, payload: [String: JSONValue],
                               python: URL? = nil, checkout: URL? = nil) async throws -> JSONValue {
        guard actions.contains(action) else { throw ExperimentError(reason: "Unknown diagnostic workspace action.") }
        guard let interpreter = python ?? LocalPythonRuntime.venvPython,
              let source = checkout ?? LocalPythonRuntime.repoRoot,
              FileManager.default.isExecutableFile(atPath: interpreter.path) else {
            throw ExperimentError.malformed("Diagnostic transport requires the local Python client environment.", repair: LocalPythonRuntime.setupHint)
        }
        let input = try JSONEncoder().encode(JSONValue.object(["action": .string(action), "payload": .object(payload)]))
        return try await Task.detached {
            let temporary = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString)
            try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
            defer { try? FileManager.default.removeItem(at: temporary) }
            let inputURL = temporary.appending(component: "input.json")
            try input.write(to: inputURL)
            let errorsURL = temporary.appending(component: "errors.txt")
            FileManager.default.createFile(atPath: errorsURL.path, contents: Data())
            let stdin = try FileHandle(forReadingFrom: inputURL)
            let stderr = try FileHandle(forWritingTo: errorsURL)
            defer { try? stdin.close(); try? stderr.close() }
            let process = Process(); let output = Pipe()
            process.executableURL = interpreter
            process.arguments = ["-m", "steerlab_server.client.diagnostic_workspace"]
            process.currentDirectoryURL = source
            var environment = ProcessInfo.processInfo.environment
            environment["PYTHONPATH"] = source.appending(component: "Server").path
            process.environment = environment
            process.standardInput = stdin; process.standardOutput = output; process.standardError = stderr
            try process.run()
            let bytes = output.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            guard let value = try? JSONDecoder().decode(JSONValue.self, from: bytes), case .object(let object) = value else {
                throw ExperimentError(reason: "The local diagnostic client returned no result. Check the Python client installation.")
            }
            guard process.terminationStatus == 0, object["ok"] == .bool(true), let result = object["result"] else {
                let reason: String = if case .string(let text) = object["reason"] { text } else { "Diagnostic workspace operation refused." }
                throw ExperimentError.malformed(reason, repair: "Inspect the archive, receipt and originating workspace; retain remote originals until custody verifies.")
            }
            return result
        }.value
    }

    static func http(_ action: String, body: Data, root: URL) async -> StudyAuthoringHTTP.Response {
        do {
            let payload = try JSONDecoder().decode([String: JSONValue].self, from: body)
            guard case .string(let path) = payload["workspaceRoot"],
                  try ManifestFileTransaction.canonicalPath(URL(filePath: path)) == ManifestFileTransaction.canonicalPath(root) else {
                throw ExperimentError(reason: "The workbench is serving another workspace.")
            }
            return .json(try await perform(action, payload: payload))
        } catch { return StudyAuthoringHTTP.failure(error) }
    }
}

struct DiagnosticArguments {
    let verb: String
    let positional: String?
    let flags: [String: String]
    init(_ args: [String], namespace: String, takesValue: Bool) throws {
        guard let verb = args.first, let spec = ExperimentCLIParser.spec(namespace: namespace, verb: verb) else { throw ExperimentError(reason: "Unknown diagnostic command.") }
        self.verb = verb
        var flags: [String: String] = [:]; var positionals: [String] = []; var index = 1
        while index < args.count {
            let word = args[index]
            if spec.valueFlags.contains(word) || spec.booleanFlags.contains(word) {
                guard flags[word] == nil else { throw ExperimentError(reason: "Supply each flag once.") }
                if spec.valueFlags.contains(word) {
                    guard args.indices.contains(index + 1), !args[index + 1].hasPrefix("--") else { throw ExperimentError(reason: "Missing value for " + word) }
                    flags[word] = args[index + 1]; index += 2
                } else { flags[word] = ""; index += 1 }
            } else { positionals.append(word); index += 1 }
        }
        guard positionals.count == (takesValue ? 1 : 0), spec.requiredFlags.allSatisfy({ flags[$0] != nil }) else {
            throw ExperimentError.malformed("Supply the declared positional and required flags, including explicit removal confirmation.", repair: namespace + " " + verb + " --help")
        }
        self.flags = flags; self.positional = positionals.first
    }
}

enum DiagnosticWorkspaceCLI {
    static func run(_ invocation: ExperimentCLIInvocation, sink: ExperimentCLISink) async throws -> ExperimentCLIResult {
        let arguments = try DiagnosticArguments(invocation.args, namespace: "science", takesValue: invocation.verb != "custody")
        var payload: [String: JSONValue] = ["workspaceRoot": .string(ExperimentStore.workspaceRoot.path)]
        if let value = arguments.positional {
            let key = arguments.verb.hasPrefix("sae-") ? "path" : ["interview", "draft", "publish"].contains(arguments.verb) ? "operation" : (["input-plan", "package"].contains(arguments.verb) ? "requestFile" : (arguments.verb == "import" ? "archivePath" : "receiptSHA256"))
            payload[key] = .string(value)
        }
        if let path = arguments.flags["--answers"] { payload["answersText"] = .string(try String(contentsOfFile: path, encoding: .utf8)) }
        for (flag, key) in [("--experiment", "experiment"), ("--destination", "destination"), ("--archive", "archivePath"), ("--sha256", "archiveSHA256"), ("--plan-sha256", "planSHA256")] {
            if let value = arguments.flags[flag] { payload[key] = .string(value) }
        }
        let result = try await DiagnosticWorkspace.perform(arguments.verb, payload: payload)
        sink.out(String(decoding: try JSONEncoder().encode(result), as: UTF8.self))
        var changed = ["package", "import"].contains(arguments.verb)
        if case .object(let object) = result, case .bool(let value) = object["changed"] { changed = value }
        return .init(message: "Diagnostic workspace operation completed.", changed: changed, payload: ["response": result])
    }
}
