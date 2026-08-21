import CryptoKit
import Foundation

/// Confirmation-study perturbation machinery: test a promoted agent on
/// held-out material under a DECLARED perturbation policy — never
/// hand-picked post-hoc points.
///
/// A perturbation policy is manifest DATA that expands MECHANICALLY into
/// ordinary hashed conditions at AUTHORING time (draft manifests only), so
/// the frozen manifest shows exactly what will run and the existing
/// pin/freeze/verify firewall applies unchanged. No new manifest type, no
/// run-time expansion.
///
/// Cross-engine contract with the server's `confirmation.py`: identical JSON
/// (`perturbationPolicy` manifest block), identical generated condition
/// names, layers, alphas, and controlType values.
public enum ConfirmationStudy {

    // MARK: - Pure expansion (unit-testable without a store)

    /// Deterministic expansion of a policy into ordinary conditions, given
    /// the agent's anchor (concept c, layer L, alpha a):
    ///
    /// - `<agent>-anchor` at α = a
    /// - per δ in `alphaDeltas` (ascending): `<agent>-minus-<δ>` at a−δ and
    ///   `<agent>-plus-<δ>` at a+δ (δ formatted `%g`, minimal)
    /// - `<agent>-control` (controlType "randomMatchedNorm") when
    ///   `includeMatchedNormControl` — both engines already implement
    ///   randomMatchedNorm in their condition-injection paths.
    ///
    /// The no-injection baseline is deliberately NOT a generated condition:
    /// both study runners already treat baseline as implicit/paired —
    /// verified: Swift `ExperimentTasks.run` prepends a `baseline` condition
    /// whenever the manifest lacks one, and the server's `tasks.run` does
    /// the same (`conditions = [Condition(name="baseline", slots=[])] + …`).
    ///
    /// Throws when any a−δ ≤ 0 — a nonpositive alpha is a refusal, never a
    /// silent clamp.
    public static func expandedConditions(
        policy: ExperimentManifest.PerturbationPolicy,
        bandWidth: Int
    ) throws -> [ExperimentManifest.Condition] {
        let agent = policy.sourceAgent.name
        let concept = policy.concept
        let layer = policy.cell.layer
        let alpha = policy.cell.alpha

        func condition(
            _ name: String, alpha: Double, controlType: String? = nil
        ) -> ExperimentManifest.Condition {
            .init(
                name: name,
                slots: [.init(concept: concept, layer: layer, alpha: alpha)],
                bandWidth: bandWidth,
                alphaInNormUnits: true,
                controlType: controlType)
        }

        var conditions = [condition("\(agent)-anchor", alpha: alpha)]
        for delta in policy.alphaDeltas.sorted() {
            let low = alpha - delta
            guard low > 0 else {
                throw ExperimentError(
                    reason: "alpha delta \(format(delta)) drives alpha nonpositive "
                        + "(\(format(alpha)) − \(format(delta)) = \(format(low)) ≤ 0) "
                        + "— choose a smaller delta")
            }
            conditions.append(
                condition("\(agent)-minus-\(format(delta))", alpha: low))
            conditions.append(
                condition("\(agent)-plus-\(format(delta))", alpha: alpha + delta))
        }
        if policy.includeMatchedNormControl {
            conditions.append(
                condition(
                    "\(agent)-control", alpha: alpha,
                    controlType: "randomMatchedNorm"))
        }
        return conditions
    }

    /// `%g`-style minimal formatting shared by names and refusal messages —
    /// must agree with the server's `f"{value:g}"` (e.g. 0.2 → "0.2",
    /// 1.0 → "1").
    static func format(_ value: Double) -> String {
        String(format: "%g", value)
    }

    /// True when `name` is one this policy's expansion generates for
    /// `agent` — the collide-safe replace rule: re-running the authoring op
    /// on the same draft REPLACES previously generated conditions for that
    /// agent (matched by the generated-name prefix) and re-stamps the policy
    /// block.
    static func isGeneratedName(_ name: String, agent: String) -> Bool {
        name == "\(agent)-anchor" || name == "\(agent)-control"
            || name.hasPrefix("\(agent)-minus-") || name.hasPrefix("\(agent)-plus-")
    }

    // MARK: - Authoring operation

    /// Attach a perturbation policy to a DRAFT experiment: locate the agent
    /// (variant-library name via `ModelVariantStore.scan()`, or an explicit
    /// path), pin it by content hash, refuse everything the design refuses
    /// (frozen manifests, adapter-bearing / zero- / multi-injection agents,
    /// unattached concepts, nonpositive perturbed alphas), expand the policy
    /// into ordinary hashed conditions, and save.
    @discardableResult
    public static func attach(
        experimentName: String,
        agent agentNameOrPath: String,
        deltas: [Double] = [0.2],
        includeControl: Bool = true,
        log: (String) -> Void = { print($0) }
    ) throws -> ExperimentManifest {
        var manifest = try ExperimentStore.load(name: experimentName)
        guard manifest.status == .draft else {
            throw ExperimentError.refusing(
                .statusImmutable,
                "cannot attach perturbations: '\(experimentName)' is "
                    + "\(manifest.status.rawValue) — duplicate first",
                repair: ExperimentStore.duplicateToIterateRepair(experimentName))
        }

        let normalizedDeltas = Array(Set(deltas)).sorted()
        guard !normalizedDeltas.isEmpty else {
            throw ExperimentError(reason: "at least one alpha delta is required")
        }
        if let bad = normalizedDeltas.first(where: { $0 <= 0 }) {
            throw ExperimentError(
                reason: "alpha deltas must be positive (got \(format(bad)))")
        }

        let (url, artifact) = try resolveAgent(agentNameOrPath)
        // Pin the artifact by content hash — the same file-bytes SHA-256
        // convention VariantCondition.artifactHash uses (ModelVariantStore.hash).
        let artifactHash = try ModelVariantStore.hash(url)

        guard artifact.adapters.isEmpty else {
            throw ExperimentError.refusing(
                .confirmationAgentShape,
                "perturbation of adapter-bearing agents is not supported "
                    + "yet — vector-only agents only",
                repair: Self.promoteAVectorAgentRepair(experimentName))
        }
        guard let injection = artifact.injections.first else {
            throw ExperimentError.refusing(
                .confirmationAgentShape,
                "agent '\(artifact.name)' has no injections — nothing "
                    + "to perturb",
                repair: Self.promoteAVectorAgentRepair(experimentName))
        }
        guard artifact.injections.count == 1 else {
            throw ExperimentError.refusing(
                .confirmationAgentShape,
                "agent '\(artifact.name)' has \(artifact.injections.count) "
                    + "injections — multi-injection agents are not supported yet; "
                    + "the anchor must be a single injection",
                repair: Self.promoteAVectorAgentRepair(experimentName))
        }
        guard injection.effectiveMode == .add else {
            throw ExperimentError.refusing(
                .confirmationAgentShape,
                ConfirmationStudyCopy.ablationRefusal(
                    agent: artifact.name),
                repair: Self.promoteAVectorAgentRepair(experimentName))
        }
        let concept = injection.concept
        guard manifest.concepts.contains(where: { $0.name == concept }) else {
            throw ExperimentError.refusing(
                .missingPrerequisite,
                "attach concept '\(concept)' to '\(experimentName)' first",
                repair: "steerlab-cli experiment attach \(experimentName) \(concept)")
        }

        let policy = ExperimentManifest.PerturbationPolicy(
            sourceAgent: .init(
                name: artifact.name,
                artifactPath: FineTuneStore.relativePath(for: url),
                artifactHash: artifactHash,
                promoted: artifact.promotion != nil),
            concept: concept,
            cell: .init(layer: injection.layer, alpha: injection.alpha),
            alphaDeltas: normalizedDeltas,
            includeMatchedNormControl: includeControl,
            declaredAt: ISO8601DateFormatter().string(from: Date()))

        let generated = try expandedConditions(
            policy: policy, bandWidth: artifact.bandWidth)

        // Collide-safe replace: drop everything a previous run of this op
        // generated for this agent, then append the fresh expansion.
        manifest.conditions.removeAll {
            isGeneratedName($0.name, agent: artifact.name)
        }
        manifest.conditions.append(contentsOf: generated)
        manifest.perturbationPolicy = policy
        try ExperimentStore.save(manifest)
        log(
            "attached perturbation policy for agent '\(artifact.name)' "
                + "(\(concept), L\(policy.cell.layer), α\(format(policy.cell.alpha)), "
                + "δ \(normalizedDeltas.map(format).joined(separator: ","))) → "
                + "\(generated.count) condition(s)")
        return manifest
    }

    /// Locate the agent artifact: an explicit path (absolute, or relative to
    /// the workspace root) wins when a file exists there; otherwise the
    /// variant library is scanned by name (newest first, matching
    /// `ModelVariantStore.scan()` ordering).
    /// The repair for every agent-SHAPE refusal (WP0 step 7): a perturbation
    /// policy anchors on ONE vector injection, added, not ablated. Every
    /// wrong shape is repaired the same way — promote a vector agent and
    /// anchor on that.
    static func promoteAVectorAgentRepair(_ experiment: String) -> String {
        "steerlab-cli experiment promote \(experiment) <concept> && "
            + "steerlab-cli experiment confirm \(experiment) --agent <agent>  "
            + "(a perturbation policy anchors on a single-injection, additive, "
            + "vector-only agent)"
    }

    static func resolveAgent(
        _ nameOrPath: String
    ) throws -> (url: URL, artifact: ModelVariantArtifact) {
        let fm = FileManager.default
        let candidate = ModelVariantStore.absoluteURL(nameOrPath)
        if fm.fileExists(atPath: candidate.path) {
            let data = try Data(contentsOf: candidate)
            guard
                let artifact = try? JSONDecoder().decode(
                    ModelVariantArtifact.self, from: data)
            else {
                throw ExperimentError(
                    reason: "'\(nameOrPath)' is not a readable model-variant artifact")
            }
            return (candidate, artifact)
        }
        // scan() is newest-first, so the first name match is the newest save.
        guard
            let record = ModelVariantStore.scan().first(
                where: { $0.artifact.name == nameOrPath })
        else {
            throw ExperimentError(
                reason: "no variant named '\(nameOrPath)' in the library and no "
                    + "artifact file at that path")
        }
        return (record.url, record.artifact)
    }
}

/// Researcher-facing wording for the confirmation path, factored out so the
/// remedies are testable rather than only readable.
public enum ConfirmationStudyCopy {

    /// Why a confirmation study cannot perturb an ablation agent, and what to
    /// do instead. A refusal without an alternative is a dead end.
    public static func ablationRefusal(agent: String) -> String {
        "agent '\(agent)' ABLATES its concept, and a confirmation study "
            + "perturbs a steering dose: it would generate α−δ / α+δ arms "
            + "around a λ that is not a dose, and a matched-norm random "
            + "control that has no norm to match. What to do instead: attach "
            + "the ablation agent as an ordinary variant condition, and for a "
            + "dose-response add further conditions at other λ (0.5 partial, "
            + "1 full, 2 reflection). A confirmation study remains the right "
            + "tool for a STEERING agent."
    }
}
