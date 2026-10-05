import Foundation

/// The Python archive owner is the portable format authority. Mac adapters call
/// its local-only process entry point; no HTTP, model load or shell is involved.
/// Release builds read the bundled ServerPayload; only the interpreter and
/// client dependencies live outside the signed bundle.
public enum DiagnosticWorkspace {
    public static let actions = ["evidence-analyze", "policy-list", "policy-inspect", "policy-review", "policy-publish", "policy-attach-review", "policy-attach", "measurements-review", "measurements-save", "probe-list", "probe-inspect", "staged-request", "corpus-preview", "corpus-publish", "artifact-plan", "artifact-import", "setup-start", "setup-inspect", "sae-check", "sae-show", "sae-pin-plan", "sae-pin", "interview", "draft", "publish", "input-plan", "package", "import", "custody", "verify-custody", "results-export", "report", "results-report"]

    /// Asks the Python client only to confirm that its files are the source
    /// this build was compiled against: nothing is imported, read, or
    /// written. Internal to the bridge, so not one of the `actions` the
    /// command line routes.
    static let identityCheckAction = "client-identity"

    /// A well-formed action the Python client declined — a run folder with no
    /// report, an archive that fails custody, a plan that changed. The command
    /// line answers it `refused` (65) with this code and repair, as the Python
    /// client's own `science` verbs do. A request in the wrong shape is not
    /// one of these: it stays the `blocked` (64) malformed invocation it has
    /// always been. The Python client decides which is which, in one place
    /// (`diagnostic_commands.refusal_fields`), and says so in its answer.
    public struct Refusal: Error, Sendable, Equatable, CustomStringConvertible {
        public let code: String
        public let reason: String
        public let repairAction: String

        public var description: String { reason }
    }

    /// The error a declined answer from the Python client amounts to.
    static func failure(from object: [String: JSONValue]) -> any Error {
        let reason: String = if case .string(let text) = object["reason"] { text } else { "Diagnostic workspace operation refused." }
        let repair: String = if case .string(let text) = object["repairAction"] { text } else { ScientificPythonRuntime.setupHint }
        guard case .string("refused") = object["state"] else {
            return ExperimentError.malformed(reason, repair: repair)
        }
        let code: String = if case .string(let text) = object["code"], !text.isEmpty { text } else { "refused" }
        return Refusal(code: code, reason: reason, repairAction: repair)
    }

    /// Confirms, locally, that the Python client files match this build, and
    /// throws the same typed failure `perform` would. A remote step that
    /// ends in a local one calls this first, so a mismatch is found before
    /// the server has done any work.
    public static func confirmClientIdentity(python: URL? = nil, source: URL? = nil) async throws {
        _ = try await perform(identityCheckAction, payload: [:], python: python, source: source)
    }

    public static func perform(_ action: String, payload: [String: JSONValue],
                               python: URL? = nil, source: URL? = nil) async throws -> JSONValue {
        guard actions.contains(action) || action == identityCheckAction else {
            throw ExperimentError(reason: "Unknown diagnostic workspace action.")
        }
        let source = try source ?? CodeResources.serverPayload()
        guard FileManager.default.fileExists(atPath: source.appending(path: "steerlab_server/__init__.py").path),
              FileManager.default.fileExists(atPath: source.appending(path: "steerlab_server/client/diagnostic_workspace.py").path) else {
            throw ExperimentError.malformed("The scientific Python payload is incomplete.", repair: ScientificPythonRuntime.setupHint)
        }
        guard let interpreter = python ?? ScientificPythonRuntime.interpreter,
              FileManager.default.isExecutableFile(atPath: interpreter.path) else {
            throw ExperimentError.malformed("Scientific workspace actions require a local Python client environment.", repair: ScientificPythonRuntime.setupHint)
        }
        let expected = PythonClientIdentity.sourceSHA256
        let input = try JSONEncoder().encode(JSONValue.object(["action": .string(action), "payload": .object(payload), "clientSHA256": .string(expected)]))
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
            process.arguments = ["-B", "-s", "-m", "steerlab_server.client.diagnostic_workspace"]
            process.currentDirectoryURL = temporary
            var environment = ProcessInfo.processInfo.environment
            environment["PYTHONPATH"] = source.path
            environment.removeValue(forKey: "PYTHONHOME")
            process.environment = environment
            process.standardInput = stdin; process.standardOutput = output; process.standardError = stderr
            try process.run()
            let bytes = output.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            let answer = try? JSONDecoder().decode(JSONValue.self, from: bytes)
            if let failure = identityFailure(
                answer: answer, expected: expected, source: source,
                interpreter: interpreter, exitStatus: process.terminationStatus,
                errorOutput: { errorTail(errorsURL) })
            {
                throw ExperimentError.clientIdentity(failure)
            }
            guard case .object(let object) = answer else {
                throw ExperimentError(reason: "The local Python client returned no result.")
            }
            guard process.terminationStatus == 0, object["ok"] == .bool(true), let result = object["result"] else {
                throw failure(from: object)
            }
            return result
        }.value
    }

    /// The identity failure an answer from the Python client amounts to, or
    /// nil when it confirmed this build's identity. One cause per shape of
    /// answer (`ClientIdentityFailure.Cause`): no answer at all, an answer
    /// that stopped before it named an identity, or an identity that is not
    /// this build's.
    static func identityFailure(
        answer: JSONValue?, expected: String, source: URL, interpreter: URL,
        exitStatus: Int32, errorOutput: () -> String?
    ) -> ClientIdentityFailure? {
        let (layout, installation) = ClientIdentityFailure.classify(payload: source)
        func failure(_ cause: ClientIdentityFailure.Cause, actual: String? = nil,
                     reported: String? = nil, reason: String? = nil,
                     withOutput: Bool) -> ClientIdentityFailure {
            ClientIdentityFailure(
                cause: cause, expected: expected, actual: actual,
                payloadPath: reported ?? source.path, layout: layout,
                replacedWhileRunning: cause == .sourcesDiffer
                    && ClientIdentityFailure.replacedSinceLaunch(
                        payload: source, layout: layout, installation: installation),
                pythonReason: reason, errorOutput: withOutput ? errorOutput() : nil,
                exitStatus: cause == .sourcesDiffer ? nil : exitStatus,
                interpreterPath: interpreter.path)
        }
        guard case .object(let object) = answer else {
            return failure(.noAnswer, withOutput: true)
        }
        let reported: String? = if case .string(let path) = object["clientRoot"] { path } else { nil }
        let reason: String? = if case .string(let text) = object["reason"] { text } else { nil }
        guard case .string(let actual) = object["clientSHA256"] else {
            return failure(.identityNotReported, reported: reported, reason: reason,
                           withOutput: true)
        }
        guard actual == expected else {
            return failure(.sourcesDiffer, actual: actual, reported: reported,
                           reason: reason, withOutput: false)
        }
        return nil
    }

    /// The last few lines the Python client wrote to its error stream, for a
    /// failure that has nothing better to show.
    static func errorTail(_ url: URL) -> String? {
        guard let data = try? Data(contentsOf: url), !data.isEmpty else { return nil }
        let text = String(decoding: data.suffix(600), as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let lines = text.split(separator: "\n", omittingEmptySubsequences: true).suffix(3)
        let tail = lines.joined(separator: " | ")
        return tail.isEmpty ? nil : tail
    }

    static func http(_ action: String, body: Data, root: URL) async -> StudyAuthoringHTTP.Response {
        do {
            let payload = try JSONDecoder().decode([String: JSONValue].self, from: body)
            guard case .string(let path) = payload["workspaceRoot"],
                  try ManifestFileTransaction.canonicalPath(URL(filePath: path)) == ManifestFileTransaction.canonicalPath(root) else {
                throw ExperimentError(reason: "The workbench is serving another workspace.")
            }
            if action.hasPrefix("artifact-"), case .string(let source) = payload["descriptionFile"] {
                let file = URL(filePath: source, relativeTo: root).standardizedFileURL.resolvingSymlinksInPath()
                let base = root.standardizedFileURL.resolvingSymlinksInPath()
                guard file.path.hasPrefix(base.path + "/") else {
                    throw ExperimentError(reason: "Stage the description and source files in this workbench workspace first.")
                }
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
        let arguments = try DiagnosticArguments(invocation.args, namespace: "science", takesValue: !["custody", "probe-list", "policy-list"].contains(invocation.verb ?? ""))
        // `science report` owns its `--out` (the page) and its own result sentence.
        if arguments.verb == ScienceReport.action { return try await ScienceReport.run(arguments, root: ExperimentStore.workspaceRoot, sink: sink) }
        var payload: [String: JSONValue] = ["workspaceRoot": .string(ExperimentStore.workspaceRoot.path)]
        if let value = arguments.positional {
            let key = ["policy-inspect", "evidence-analyze"].contains(arguments.verb) ? "path" : arguments.verb.hasPrefix("policy-") ? "settingsText" : arguments.verb.hasPrefix("measurements-") ? "experiment" : arguments.verb == "corpus-preview" ? "specText" : arguments.verb == "corpus-publish" ? "previewID" : arguments.verb.hasPrefix("artifact-") ? "descriptionFile" : (arguments.verb.hasPrefix("sae-") || arguments.verb == "probe-inspect") ? "path" : ["interview", "draft", "publish"].contains(arguments.verb) ? "operation" : (["input-plan", "package"].contains(arguments.verb) ? "requestFile" : (arguments.verb == "import" ? "archivePath" : "receiptSHA256"))
            payload[key] = .string((arguments.verb == "corpus-preview" || key == "settingsText") ? try String(contentsOfFile: value, encoding: .utf8) : value)
        }
        if let path = arguments.flags["--settings"] { payload["settingsText"] = .string(try String(contentsOfFile: path, encoding: .utf8)) }
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
