import Foundation

/// Versioned runtime support is separate from model qualification and study identity.
public enum InstrumentationSupport {
    public static func requirements(_ value: JSONValue) -> Set<String> {
        switch value {
        case .object(let object):
            var result = Set<String>()
            if case .array(let policies) = object["interventionPolicies"], !policies.isEmpty {
                result.formUnion(["policy-v1", "policy-evidence-v2"])
            }
            if case .object(let measurements) = object["probeMeasurements"], case .array(let probes) = measurements["probes"], !probes.isEmpty {
                result.insert("probe-readings-v1")
            }
            if case .array(let declared) = object["runtimeRequirements"] {
                for value in declared {
                    if case .string(let name) = value { result.insert(name) }
                    else { result.insert("invalid-runtime-requirements") }
                }
            } else if object["runtimeRequirements"] != nil { result.insert("invalid-runtime-requirements") }
            for child in object.values { result.formUnion(requirements(child)) }
            return result
        case .array(let values): return values.reduce(into: Set<String>()) { $0.formUnion(requirements($1)) }
        default: return []
        }
    }

    public static func require(_ required: Set<String>, offered: [String]?) throws {
        let missing = required.subtracting(offered ?? [])
        guard missing.isEmpty else {
            throw ExperimentError(reason: "This engine cannot execute the study’s declared probes or policies: \(missing.sorted().joined(separator: ", ")). Update and restart the engine, then reconnect and review the submission. Existing artifacts do not need editing.")
        }
    }

    static func references(_ value: JSONValue) -> Set<String> {
        switch value {
        case .object(let object):
            var result = Set<String>()
            for key in ["artifactPath", "variantArtifactPath", "multiAgentScenarioPath"] {
                if case .string(let path) = object[key], !path.isEmpty { result.insert(path) }
            }
            for child in object.values { result.formUnion(references(child)) }
            return result
        case .array(let values): return values.reduce(into: Set<String>()) { $0.formUnion(references($1)) }
        default: return []
        }
    }
}

extension ClusterClient {
    func requireInstrumentation(_ value: JSONValue) async throws {
        let required = InstrumentationSupport.requirements(value)
        if !required.isEmpty { try InstrumentationSupport.require(required, offered: await capabilities().instrumentation) }
    }

    func requireBundleInstrumentation(path: String) async throws {
        let inspected: JSONValue = try await post("/api/bundles/inspect", body: ["bundlePath": path])
        if case .object(let metadata) = inspected, metadata["runtimeRequirements"] != nil {
            try await requireInstrumentation(inspected)
        } else {
            // An unmarked archive can contain an additive field an older engine ignores.
            try InstrumentationSupport.require(["policy-v1", "policy-evidence-v2", "probe-readings-v1"], offered: await capabilities().instrumentation)
        }
    }

    func requireStudyInstrumentation(experiment: String) async throws {
        let document = try JSONDecoder().decode(JSONValue.self, from: await experimentManifestBody(name: experiment))
        var required = InstrumentationSupport.requirements(document)
        var pending = Array(InstrumentationSupport.references(document)); var visited = Set<String>()
        while let path = pending.popLast() {
            guard visited.insert(path).inserted else { continue }
            guard visited.count <= 1024 else { throw ExperimentError(reason: "The study has too many linked agent documents to inspect. Package it locally before submission.") }
            let raw: JSONValue
            if path.hasPrefix("runs/") || path.contains("/runs/") {
                raw = try await get("/api/variant/detail", queryItems: [.init(name: "path", value: path)])
            } else {
                raw = try await get("/api/scenario", queryItems: [.init(name: "path", value: path)])
            }
            required.formUnion(InstrumentationSupport.requirements(raw))
            pending.append(contentsOf: InstrumentationSupport.references(raw))
        }
        if !required.isEmpty { try InstrumentationSupport.require(required, offered: await capabilities().instrumentation) }
    }

    func requireChatInstrumentation(_ selection: VariantChatSelection, strip: Bool, battery: Bool = false) async throws {
        guard !strip else { return }
        let document: JSONValue
        switch selection {
        case .inline(let agent): document = try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(agent))
        case .stored(let path, _):
            // Read exact saved bytes; an older server's typed decoder can discard new fields.
            document = try await get("/api/variant/detail", queryItems: [.init(name: "path", value: path)])
        }
        let required = InstrumentationSupport.requirements(document)
        if battery && !required.isEmpty { throw ExperimentError(reason: "This battery path does not execute policies. Compare sampled responses in a study, or explicitly select the unmodified baseline.") }
        if !required.isEmpty { try InstrumentationSupport.require(required, offered: await capabilities().instrumentation) }
    }
}
