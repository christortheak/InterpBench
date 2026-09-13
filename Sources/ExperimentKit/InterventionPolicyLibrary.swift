import CryptoKit
import Foundation

/// The portable Python owner validates and publishes policies. Swift preserves
/// their exact text, and supplies guided authoring without duplicating rule math.
public enum InterventionPolicyLibrary {
    public static let executionHint = "This agent has intervention policies. Run it in a study using Python Compute, or through the Python agent chat API. Playground’s editable steering controls cannot represent policies yet."

    public static func requireNativeExecution(_ agent: ModelVariantArtifact) throws {
        guard (agent.interventionPolicies ?? []).isEmpty else { throw ExperimentError(reason: executionHint) }
    }

    public static func validateAttachments(_ values: [JSONValue]?) throws {
        guard let values else { return }
        guard values.count <= 16 else { throw ExperimentError(reason: "Attach at most 16 intervention policies.") }
        var seen = Set<String>()
        for value in values {
            guard case .object(let item) = value, Set(item.keys) == ["json", "sha256"],
                  case .string(let text) = item["json"], case .string(let sha) = item["sha256"],
                  text.utf8.count <= 16 * 1024 * 1024,
                  SHA256.hash(data: Data(text.utf8)).map({ String(format: "%02x", $0) }).joined() == sha,
                  seen.insert(sha).inserted else {
                throw ExperimentError(reason: "An attached policy’s exact bytes do not match its digest, or the attachment is duplicated.")
            }
            guard case .object(let policy) = try JSONDecoder().decode(JSONValue.self, from: Data(text.utf8)),
                  policy["schemaVersion"] == .number(1) else {
                throw ExperimentError(reason: "Use a version-1 intervention policy published by science policy-publish.")
            }
        }
    }

    public static func call(_ action: String, root: URL, settings: JSONValue? = nil, hash: String? = nil, path: String? = nil) async throws -> JSONValue {
        var payload: [String: JSONValue] = ["workspaceRoot": .string(root.path)]
        if let settings { payload["settingsText"] = .string(try formatted(settings)) }
        if let hash { payload["planSHA256"] = .string(hash) }
        if let path { payload["path"] = .string(path) }
        return try await DiagnosticWorkspace.perform(action, payload: payload)
    }

    public static func formatted(_ value: JSONValue) throws -> String {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return String(decoding: try encoder.encode(value), as: UTF8.self)
    }

    /// One-probe starter. Multi-probe and expert definitions use the same owner
    /// through a selected settings file, not a second UI-only schema.
    public static func starter(probe: ProbeLibrary.Record, name: String, rule: String, action: String,
                               threshold: Double, strength: Double, lower: Double, upper: Double,
                               slope: Double, intercept: Double, vectorPath: String, tokens: [Int]) throws -> JSONValue {
        guard case .object(let document) = probe.document, case .object(let input) = document["input"],
              case .object(let reading) = input["reading"], let site = input["site"] else {
            throw ExperimentError(reason: "Select a portable fitted probe before creating a policy.")
        }
        var binding: [String: JSONValue] = [:]
        for key in ["modelID", "revision", "tokenizerSHA256", "coordinateConvention"] { binding[key] = input[key] }
        binding["rendering"] = reading["rendering"]
        let logit = ["logitBias", "allowTokens", "forceToken"].contains(action)
        var target: [String: JSONValue] = ["id": .string("action"), "kind": .string(action), "bounds": .array([.number(lower), .number(upper)])]
        if logit { target["tokens"] = .array(tokens.map { .number(Double($0)) }) }
        else { target["vectorArtifactID"] = .string(vectorPath) }
        var decision: [String: JSONValue] = ["action": .string("action"), "kind": .string(rule)]
        if rule == "fixed" { decision["value"] = .number(strength) }
        else {
            decision["weights"] = .object(["reader": .number(1)])
            if rule == "threshold" {
                decision["threshold"] = .number(threshold); decision["below"] = .number(0); decision["above"] = .number(strength)
            } else { decision["slope"] = .number(slope); decision["intercept"] = .number(intercept) }
        }
        return .object(["schemaVersion": .number(1), "name": .string(name), "binding": .object(binding),
                        "site": logit ? .object(["kind": .string("logitsPreSelection")]) : site,
                        "stages": .array([.string("prefill"), .string("decode")]), "positions": .string("lastPosition"),
                        "probes": .array([.object(["id": .string("reader"), "path": .string(probe.path)])]),
                        "actions": .array([.object(target)]), "rules": .array([.object(decision)]),
                        "onError": .string("stop"), "maxEvents": .number(2048)])
    }
}
