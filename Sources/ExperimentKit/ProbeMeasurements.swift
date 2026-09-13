import Foundation
import CryptoKit

/// Presentation and byte-pin verification. Reviewed authorship uses the Python
/// declaration owner on every surface; no numeric algorithm lives here.
public enum ProbeMeasurements {
    public struct Reference: Sendable {
        public let path: String
        public let sha256: String
    }
    public static func references(_ value: JSONValue?) -> [Reference] {
        guard case .object(let config) = value, case .array(let probes) = config["probes"] else { return [] }
        return probes.compactMap { value in
            guard case .object(let item) = value, case .object(let ref) = item["probe"],
                  case .string(let path) = ref["path"], case .string(let sha) = ref["sha256"] else { return nil }
            return Reference(path: path, sha256: sha)
        }
    }
    public static func validShape(_ value: JSONValue) -> Bool {
        guard case .object(let config) = value,
              Set(config.keys) == ["schemaVersion", "probes", "onError", "maxReadings", "retainActivations", "maxActivationBytes"],
              config["schemaVersion"] == .number(1), case .array(let probes) = config["probes"], probes.count <= 32,
              case .string(let policy) = config["onError"], ["recordMissing", "stop"].contains(policy),
              case .bool = config["retainActivations"] else { return false }
        for (key, limit) in [("maxReadings", 65536.0), ("maxActivationBytes", 16777216.0)] {
            guard case .number(let n) = config[key], n.isFinite, n >= 1, n <= limit, n.rounded() == n else { return false }
        }
        var identifiers = Set<String>()
        for probe in probes {
            guard case .object(let item) = probe,
                  Set(item.keys) == ["id", "probe", "conditions", "agents", "stages", "recordingStage"],
                  case .string(let id) = item["id"], !id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                  identifiers.insert(id).inserted, case .object(let ref) = item["probe"], Set(ref.keys) == ["path", "sha256"],
                  case .string(let path) = ref["path"], !path.isEmpty, !path.hasPrefix("/"), !path.contains("\\"),
                  !path.split(separator: "/", omittingEmptySubsequences: false).contains(where: { $0.isEmpty || $0 == "." || $0 == ".." }),
                  case .string(let sha) = ref["sha256"], sha.count == 64, sha.allSatisfy({ "0123456789abcdef".contains($0) }),
                  case .string(let stage) = item["recordingStage"], ["preAction", "postAction"].contains(stage) else { return false }
            for key in ["conditions", "agents", "stages"] {
                guard case .array(let values) = item[key] else { return false }
                let names = values.compactMap { if case .string(let s) = $0 { return s }; return nil }
                guard names.count == values.count, !names.contains(""), Set(names).count == names.count else { return false }
                if key == "stages", names.isEmpty || !Set(names).isSubset(of: ["prefill", "decode"]) { return false }
            }
        }
        return true
    }
    public static func violations(_ value: JSONValue?, root: URL) -> [String] {
        guard let value else { return [] }
        guard validShape(value) else { return ["Review and save valid probe measurement settings before freezing."] }
        var issues: [String] = []
        for ref in references(value) {
            do {
                let url = ExperimentStore.resolveProjectPath(ref.path, root: root)
                let base = root.resolvingSymlinksInPath().standardizedFileURL.path + "/"
                guard url.resolvingSymlinksInPath().path.hasPrefix(base) else {
                    issues.append("The measurement probe must be inside this workspace."); continue
                }
                var component = root
                for part in ref.path.split(separator: "/") {
                    component.append(component: String(part))
                    let attrs = try FileManager.default.attributesOfItem(atPath: component.path)
                    if attrs[.type] as? FileAttributeType == .typeSymbolicLink { throw ExperimentError(reason: "Measurement paths cannot contain symbolic links.") }
                }
                let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
                guard attrs[.type] as? FileAttributeType == .typeRegular, (attrs[.size] as? NSNumber)?.intValue ?? Int.max <= 67108864 else {
                    issues.append("Choose an ordinary probe file no larger than 64 MiB."); continue
                }
                let data = try Data(contentsOf: url, options: .mappedIfSafe)
                let sha = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
                if sha != ref.sha256 { issues.append("Measurement probe bytes changed: \(ref.path). Review the selection again.") }
            } catch { issues.append("Cannot read measurement probe \(ref.path): \(error.localizedDescription)") }
        }
        return issues
    }
    public static func request(_ action: String, experiment: String, settings: JSONValue,
                               root: URL, planSHA256: String? = nil) async throws -> JSONValue {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        var payload: [String: JSONValue] = ["workspaceRoot": .string(root.path), "experiment": .string(experiment),
            "settingsText": .string(String(decoding: try encoder.encode(settings), as: UTF8.self))]
        if let planSHA256 { payload["planSHA256"] = .string(planSHA256) }
        return try await DiagnosticWorkspace.perform(action, payload: payload)
    }
}
