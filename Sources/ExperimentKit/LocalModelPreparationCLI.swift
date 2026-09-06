import Foundation

/// Foreground local installation delegates to the app's installer. HTTP offers
/// observation/cancellation of the workbench's separate, in-process installer.
enum LocalModelPreparationCLI {
    @MainActor static func run(_ invocation: ExperimentCLIInvocation, sink: ExperimentCLISink,
                               installer supplied: LocalModelInstaller? = nil) async throws -> ExperimentCLIResult {
        let args = invocation.args
        guard args.count >= 2 else { throw ExperimentError.malformed("Name the model to plan or install.", repair: "steerlab-cli model --help") }
        let revision = args.firstIndex(of: "--revision").map { args[$0 + 1] }
        do {
            if args[0] == "plan" {
                return try result(LocalModelPreparation.plan(modelID: args[1], revision: revision), changed: false, sink: sink)
            }
            let installer = supplied ?? LocalModelInstaller()
            let started = try LocalModelPreparation.start(modelID: args[1], revision: revision, installer: installer)
            sink.err("Installing \(args[1]) on this Mac. This foreground command waits for completion; partial cache files are retained if interrupted.\n")
            _ = try await installer.completion(requestID: started.request!.id)
            let status = LocalModelPreparation.Status(installer)
            let encoded = try JSONEncoder().encode(status)
            let payload = try JSONDecoder().decode([String: JSONValue].self, from: encoded)
            guard status.state == "finished" else {
                throw ExperimentCLIStop(exitCode: 70, state: .failed,
                    code: status.state == "cancelled" ? "modelInstallationCancelled" : "modelInstallationFailed",
                    reason: status.reason ?? "Model installation did not complete.",
                    repairAction: "Inspect the reason, cache access and repository credentials, then explicitly rerun model install if appropriate.",
                    payload: payload, changed: true)
            }
            return try result(status, changed: true, sink: sink)
        } catch let error as LocalModelPreparationError {
            let malformed = error.code == "invalidModelInstallRequest"
            throw ExperimentCLIStop(exitCode: malformed ? 64 : 65, state: malformed ? .blocked : .refused,
                code: error.code, reason: error.reason, repairAction: error.repairAction)
        }
    }

    private static func result(_ value: some Encodable, changed: Bool, sink: ExperimentCLISink) throws -> ExperimentCLIResult {
        let data = try JSONEncoder().encode(value)
        sink.out(String(decoding: data, as: UTF8.self))
        return .init(message: changed ? "Local model installation completed." : "Local model cache plan inspected.", changed: changed,
            payload: try JSONDecoder().decode([String: JSONValue].self, from: data))
    }
}
