import Foundation

/// Read-only presentation of credential state. Observation never opens Keychain,
/// migrates legacy settings, validates a provider, or publishes credential bytes.
/// An absent environment value says nothing about a stored credential.
public enum CredentialObservation {
    public enum State: String, Codable, Sendable {
        case available
        case absent
        case notChecked

        public init(present: Bool) { self = present ? .available : .absent }

        public var knownPresence: Bool? {
            switch self {
            case .available: true
            case .absent: false
            case .notChecked: nil
            }
        }
    }

    public static let deferredCheckMessage =
        "Stored credentials have not been checked. Judging checks the required provider's key before execution."

    public static func anthropic(environment: [String: String] = ProcessInfo.processInfo.environment) -> State {
        environmentState("ANTHROPIC_API_KEY", environment: environment)
    }

    public static func openRouter(environment: [String: String] = ProcessInfo.processInfo.environment) -> State {
        environmentState("OPENROUTER_API_KEY", environment: environment)
    }

    private static func environmentState(_ name: String, environment: [String: String]) -> State {
        (environment[name] ?? "").isEmpty ? .notChecked : .available
    }
}
