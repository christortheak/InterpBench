import Foundation
import Observation

/// First-run presentation state. Scientific owners and the installer retain
/// all validation and mutation responsibilities.
@MainActor @Observable public final class ResearchSetupModel {
    public private(set) var readiness: [String: JSONValue] = [:]
    public private(set) var plan: [String: JSONValue] = [:]
    public private(set) var busy = false
    public private(set) var message: String?
    public private(set) var error: String?
    public private(set) var handoff: String?
    public init() {}

    public var clientReady: Bool { readiness["clientReady"] == .bool(true) }
    public var basicClientReady: Bool { readiness["basicClientReady"] == .bool(true) }
    public var authoringReady: Bool { readiness["authoringReady"] == .bool(true) }
    public var planHash: String? { if case .string(let value) = plan["planSHA256"] { value } else { nil } }
    public var planDestination: String { if case .string(let value) = plan["runtime"] { value } else { "" } }
    public var planActions: [String] {
        guard case .array(let values) = plan["actions"] else { return [] }
        return values.compactMap { if case .string(let text) = $0 { text } else { nil } }
    }

    public func refresh(workspace: URL?) async {
        guard !busy else { return }
        busy = true
        readiness = await ClientSetup.inspect(workspace: workspace)
        handoff = workspace.flatMap { root in
            guard let value = try? WorkspaceBootstrap.handoff(root),
                  let bytes = try? JSONEncoder().encode(JSONValue.object(value)) else { return nil }
            return String(decoding: bytes, as: UTF8.self)
        }
        busy = false
    }

    public func preview() async {
        guard !busy else { return }
        busy = true; error = nil; message = nil; plan = [:]
        do { plan = try await ClientSetup.provision("plan") }
        catch { self.error = String(describing: error) }
        busy = false
    }

    public func install(workspace: URL?) async {
        guard !busy, let expected = planHash else { return }
        busy = true; error = nil; message = "Installing Python and the lightweight client. Downloads may take a few minutes."
        defer { busy = false; plan = [:] }
        do {
            let result = try await ClientSetup.provision(clientReady ? "repair" : "apply", expected: expected, approved: true)
            readiness = await ClientSetup.inspect(workspace: workspace)
            if clientReady { message = "Client setup complete. You can author studies." }
            else { message = "Installation finished; the selected interpreter still needs attention. Check for an explicit interpreter override or a development source mismatch." }
            if case .string(let log) = result["logPath"] { message = (message ?? "") + " Setup log: " + log }
        } catch { self.error = String(describing: error); message = nil }
    }
}
