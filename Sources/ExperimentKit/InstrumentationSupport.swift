import Foundation

/// Versioned runtime support is separate from model qualification and study identity.
public enum InstrumentationSupport {
    public static func requirements(_ value: JSONValue) -> Set<String> {
        switch value {
        case .object(let object):
            var result = Set<String>()
            if case .array(let policies) = object["interventionPolicies"], !policies.isEmpty {
                result.formUnion(["policy-v1", "policy-evidence-v2"])
                // An engine that knows only version 1 would refuse these at execution; say so before submission.
                if policies.contains(where: { policySchemaVersion($0) == .number(2) }) { result.insert("policy-v2") }
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

    /// An attachment's schema version, read from its exact bytes; unreadable bytes are refused elsewhere.
    /// Twin: `instrumentation_contract.policy_versions`.
    static func policySchemaVersion(_ attachment: JSONValue) -> JSONValue? {
        guard case .object(let item) = attachment, case .string(let text) = item["json"],
              case .object(let policy)? = try? JSONDecoder().decode(JSONValue.self, from: Data(text.utf8)) else { return nil }
        return policy["schemaVersion"]
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

    /// Returns the server's inspection of the staged bundle, which names the
    /// study it carries (`requireDeclaredSeedPolicy` reads it).
    @discardableResult
    func requireBundleInstrumentation(path: String) async throws -> JSONValue {
        let inspected: JSONValue = try await post("/api/bundles/inspect", body: ["bundlePath": path])
        if case .object(let metadata) = inspected, metadata["runtimeRequirements"] != nil {
            try await requireInstrumentation(inspected)
        } else {
            // An unmarked archive can contain an additive field an older engine ignores.
            try InstrumentationSupport.require(["policy-v1", "policy-evidence-v2", "probe-readings-v1"], offered: await capabilities().instrumentation)
        }
        return inspected
    }

    /// Refuse to submit a staged bundle whose study runs at temperature 0
    /// with several seeds (public issue #9), on every server: one from
    /// before the rule would refuse it only after the queue wait and the
    /// model load. Same evidence rule as `requireDeclaredSeedPolicy`. Python
    /// twin: `runner.require_greedy_seed_list`.
    func requireGreedySeedList(inspectedBundle inspected: JSONValue, verb: String) throws {
        guard SubstrateRouting.matrixExecutingVerbs.contains(verb),
            case .object(let metadata) = inspected,
            case .string(let name)? = metadata["experiment"], !name.isEmpty,
            case .string(let hash)? = metadata["experimentContentHash"],
            let manifest = try? ExperimentStore.load(name: name),
            ExperimentStore.manifestHash(manifest) == hash
        else { return }
        // The agent-arm rules (FR6-B round 2) ride the same evidence rule.
        if let armRefusal = SubstrateRouting.agentArmSubmissionRefusal(
            manifest: manifest, verb: verb)
        {
            throw armRefusal
        }
        guard let refusal = SubstrateRouting.greedySeedListSubmissionRefusal(
            manifest: manifest, verb: verb)
        else { return }
        throw ExperimentError.refusing(
            .studyDeclaration, refusal,
            repair: StudySampling.greedySeedListRepair(
                name: name, frozen: manifest.status != .draft))
    }

    /// Refuse to submit a staged bundle whose study runs at temperature 0
    /// with several seeds or samples per item, to an engine without
    /// `remoteStudy.greedyRulesSampledTextOnly`, which would refuse it after
    /// the queue wait and the model load. Same evidence rule as
    /// `requireDeclaredSeedPolicy`. Python twin: `runner.require_greedy_rules`.
    func requireGreedyRulesEngine(inspectedBundle inspected: JSONValue, verb: String) async throws {
        guard SubstrateRouting.matrixExecutingVerbs.contains(verb),
            case .object(let metadata) = inspected,
            case .string(let name)? = metadata["experiment"], !name.isEmpty,
            case .string(let hash)? = metadata["experimentContentHash"],
            let manifest = try? ExperimentStore.load(name: name),
            ExperimentStore.manifestHash(manifest) == hash,
            SubstrateRouting.greedyRulesNeeded(manifest: manifest, verb: verb)
        else { return }
        if let refusal = SubstrateRouting.greedyRulesSubmissionRefusal(
            manifest: manifest, verb: verb, capabilities: try await capabilities(),
            server: profile.baseURL.absoluteString)
        {
            throw ExperimentError.refusing(
                refusal.gate, refusal.reason, repair: refusal.repair)
        }
    }

    /// Refuse to submit a staged bundle whose study an engine without the
    /// declared seed policy would seed with the fixed list (issue #3). The
    /// bundle names its study and that study's content hash; the study is
    /// read from this workspace only when both match, because a bundle this
    /// client did not build says nothing a local manifest can vouch for.
    /// Python twin: `runner.require_declared_seed_policy`.
    func requireDeclaredSeedPolicy(inspectedBundle inspected: JSONValue, verb: String) async throws {
        guard SubstrateRouting.matrixExecutingVerbs.contains(verb),
            case .object(let metadata) = inspected,
            case .string(let name)? = metadata["experiment"], !name.isEmpty,
            case .string(let hash)? = metadata["experimentContentHash"],
            let manifest = try? ExperimentStore.load(name: name),
            ExperimentStore.manifestHash(manifest) == hash,
            StudySampling.declaredPolicyChangesSeeds(manifest)
        else { return }
        if let refusal = SubstrateRouting.declaredSeedPolicySubmissionRefusal(
            manifest: manifest, verb: verb, capabilities: try await capabilities(),
            server: profile.baseURL.absoluteString)
        {
            throw ExperimentError.refusing(
                .missingPrerequisite, refusal, repair: SubstrateRouting.declaredSeedPolicyRepair)
        }
    }

    /// Refuse a staged bundle whose study declares a prompt reading, before
    /// submitting it to an engine without `remoteStudy.promptReading` (issue
    /// #6). Same evidence rule as `requireDeclaredSeedPolicy`: the study is
    /// read from the local workspace only when the bundle's name and content
    /// hash match it. Python twin: `runner.require_prompt_reading`.
    func requirePromptReading(inspectedBundle inspected: JSONValue, verb: String) async throws {
        guard case .object(let metadata) = inspected,
            case .string(let name)? = metadata["experiment"], !name.isEmpty,
            case .string(let hash)? = metadata["experimentContentHash"],
            let manifest = try? ExperimentStore.load(name: name),
            ExperimentStore.manifestHash(manifest) == hash,
            manifest.promptReading != nil
        else { return }
        if let refusal = SubstrateRouting.promptReadingSubmissionRefusal(
            manifest: manifest, verb: verb, capabilities: try await capabilities(),
            server: profile.baseURL.absoluteString)
        {
            throw ExperimentError.refusing(
                .missingPrerequisite, refusal, repair: PromptReading.outdatedEngineRepair)
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
