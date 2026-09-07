import Foundation

/// Lightweight client environment for shared authoring and evidence transport.
/// Source selection remains CodeResources' job; a release uses its own payload.
public enum ScientificPythonRuntime {
    public static let defaultEnvironment = FileManager.default.homeDirectoryForCurrentUser
        .appending(path: "Library/Application Support/SteerLab/client-runtime")

    public static var interpreter: URL? {
        resolve(environment: ProcessInfo.processInfo.environment,
                clientEnvironment: defaultEnvironment,
                checkoutPython: LocalPythonRuntime.venvPython)
    }

    static func resolve(environment: [String: String], clientEnvironment: URL,
                        checkoutPython: URL?) -> URL? {
        if let path = environment["STEERLAB_CLIENT_PYTHON"] {
            // An explicit invalid choice must refuse, never select a different
            // environment whose behavior the researcher did not request.
            guard path.hasPrefix("/") else { return nil }
            return URL(filePath: path)
        }
        let installed = clientEnvironment.appending(path: "bin/python")
        if FileManager.default.isExecutableFile(atPath: installed.path) { return installed }
        return checkoutPython
    }

    public static let setupHint =
        "Open Workspace → Research Setup in the app, or run steerlab-cli setup plan, then setup apply with its --expect hash and --yes. "
        + "This installs the lightweight client without a model server. An explicit STEERLAB_CLIENT_PYTHON override must be repaired or removed separately. "
        + "For development source mismatches, rebuild the Mac client and payload together. See docs/PYTHON-CLIENT-RUNTIME.md."
}
