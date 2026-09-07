import Foundation

/// Local setup is available before Python exists. The release installer owns
/// plans, approvals, locks, verification and atomic activation for every client.
public enum ClientSetup {
    public static func inspect(workspace: URL? = nil, python: URL? = nil, source: URL? = nil) async -> [String: JSONValue] {
        do {
            let payload: [String: JSONValue] = workspace.map { ["workspaceRoot": .string($0.path)] } ?? [:]
            guard case .object(let value) = try await DiagnosticWorkspace.perform("setup-inspect", payload: payload, python: python, source: source) else {
                throw ExperimentError(reason: "The readiness owner returned no report.")
            }
            return value
        } catch {
            return ["changed": .bool(false), "clientReady": .bool(false), "authoringReady": .bool(false),
                    "reason": .string(String(describing: error)), "repairAction": .string(ScientificPythonRuntime.setupHint),
                    "workspace": workspace.flatMap { try? WorkspaceBootstrap.inspect($0) }.map(JSONValue.object) ?? .null,
                    "execution": .object(["state": .string("notAssessed"), "requiredForAuthoring": .bool(false)])]
        }
    }

    public static func releaseDirectory(explicit: URL? = nil) throws -> URL {
        let override = ProcessInfo.processInfo.environment["STEERLAB_CLIENT_RELEASE"].map { URL(filePath: $0) }
        let release = try explicit ?? override ?? CodeResources.serverPayload().appending(component: "client-release")
        guard let stamp = try? String(contentsOf: release.appending(component: "source.sha256"), encoding: .utf8),
              stamp.trimmingCharacters(in: .whitespacesAndNewlines) == PythonClientIdentity.sourceSHA256 else {
            throw ExperimentError.malformed("The client installer is missing or belongs to another Mac build.", repair: "Reinstall the complete matching app. Developers: build scripts/build-client-release.py from the matching checkout and set STEERLAB_CLIENT_RELEASE to its output directory before launching.")
        }
        return release
    }

    public static func provision(_ operation: String, release: URL? = nil,
                                 runtime: URL = ScientificPythonRuntime.defaultEnvironment,
                                 expected: String? = nil, approved: Bool = false,
                                 logDirectory: URL? = nil) async throws -> [String: JSONValue] {
        guard ["plan", "apply", "repair"].contains(operation) else { throw ExperimentError(reason: "Unknown client setup operation.") }
        let directory = try releaseDirectory(explicit: release)
        var arguments = [directory.appending(component: "install-client.sh").path, operation == "apply" ? "install" : operation, "--runtime", runtime.path]
        if let expected { arguments += ["--expect", expected] }
        if approved { arguments.append("--yes") }
        let commandArguments = arguments
        return try await Task.detached {
            let logs = logDirectory ?? FileManager.default.homeDirectoryForCurrentUser.appending(path: "Library/Application Support/SteerLab/client-setup-logs")
            let temporary = FileManager.default.temporaryDirectory.appending(component: "client-setup-" + UUID().uuidString)
            let log = operation == "plan" ? temporary : logs.appending(component: UUID().uuidString + ".log")
            if operation != "plan" { try FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700]) }
            FileManager.default.createFile(atPath: log.path, contents: Data(), attributes: [.posixPermissions: 0o600])
            defer { if operation == "plan" { try? FileManager.default.removeItem(at: log) } }
            let stderr = try FileHandle(forWritingTo: log)
            defer { try? stderr.close() }
            let process = Process(), output = Pipe()
            process.executableURL = URL(filePath: "/bin/sh")
            process.arguments = commandArguments
            process.standardOutput = output; process.standardError = stderr
            process.currentDirectoryURL = directory
            try process.run()
            let bytes = output.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            guard case .object(var result) = try JSONDecoder().decode(JSONValue.self, from: bytes) else { throw ExperimentError(reason: "Client setup returned no structured result. Read " + log.path) }
            guard process.terminationStatus == 0, result["ok"] == .bool(true) else {
                let reason: String = if case .string(let value) = result["reason"] { value } else { "Client setup did not finish." }
                let repair: String = if case .string(let value) = result["repairAction"] { value } else { "Review a fresh setup plan." }
                throw ExperimentError.malformed(reason, repair: repair + (operation == "plan" ? "" : " Setup log: " + log.path))
            }
            if operation != "plan" { result["logPath"] = .string(log.path) }
            return result
        }.value
    }
}

enum ClientSetupCLI {
    static func run(_ invocation: ExperimentCLIInvocation) async throws -> ExperimentCLIResult {
        let args = try DiagnosticArguments(invocation.args, namespace: "setup", takesValue: invocation.verb == "start")
        let result: [String: JSONValue]
        if args.verb == "start" {
            guard let directory = args.positional,
                  case .object(let value) = try await DiagnosticWorkspace.perform("setup-start", payload: ["workspaceRoot": .string(URL(filePath: directory).path), "create": .bool(args.flags["--create"] != nil)]) else {
                throw ExperimentError(reason: "First run returned no result.")
            }
            var handoff = try WorkspaceBootstrap.handoff(URL(filePath: directory))
            handoff["changed"] = .bool(false)
            var response = value
            response["handoff"] = .object(handoff)
            result = response
        } else if args.verb == "inspect" {
            let root = ExperimentStore.workspaceRoot
            result = await ClientSetup.inspect(workspace: WorkspaceStore.isWorkspace(url: root) ? root : nil)
        } else {
            result = try await ClientSetup.provision(args.verb,
                release: args.flags["--release"].map { URL(filePath: $0) },
                runtime: args.flags["--runtime"].map { URL(filePath: $0) } ?? ScientificPythonRuntime.defaultEnvironment,
                expected: args.flags["--expect"], approved: args.flags["--yes"] != nil)
        }
        return .init(message: "Client setup operation completed.", changed: result["changed"] == .bool(true), payload: result)
    }
}
