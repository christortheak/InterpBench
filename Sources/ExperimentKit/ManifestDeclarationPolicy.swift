import Foundation
import SteeringKit

/// Declaration rules over manifest values. No selected workspace or file access.
enum ManifestDeclarationPolicy {
    static let judgeDtypeVocabulary = ["bfloat16", "float16", "float32"]

    static let judgeDtypeAliases = [
        "bfloat16": "bfloat16", "bf16": "bfloat16",
        "float16": "float16", "fp16": "float16",
        "float32": "float32", "fp32": "float32",
    ]

    static let pipelineValidStages = [
        "extract", "validate", "sweep", "promote", "run", "evaluate",
        "analyze",
    ]

    struct ResolvedJudgeIdentity: Hashable {
        let kind: String
        let model: String
        let provider: String
    }

    static func modelOutputSurfacesOperative(
        _ manifest: ExperimentManifest
    ) -> Bool {
        manifest.studyKind == .modelOutput
    }

    static func optvecPinnedConcepts(
        _ manifest: ExperimentManifest
    ) -> [(name: String, pin: ExperimentManifest.ConceptRef.VectorArtifactPin)] {
        manifest.concepts.compactMap { ref in
            guard ref.options.method == .pinnedArtifact,
                let pin = ref.vectorArtifact,
                pin.sourceMethod == ExtractionMethod.optvec.rawValue
            else { return nil }
            return (ref.name, pin)
        }
    }

    static func optvecExemptFromValidateGate(
        _ manifest: ExperimentManifest
    ) -> Bool {
        guard !manifest.concepts.isEmpty, manifest.variantConditions.isEmpty
        else { return false }
        return optvecPinnedConcepts(manifest).count == manifest.concepts.count
    }

    static func symbolicRevisionProblem(
        _ manifest: ExperimentManifest
    ) -> String? {
        var offenders: [String] = []
        let study = (manifest.modelRevision ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if !study.isEmpty, !isCommitLike(study) {
            offenders.append("the study model pins '\(study)'")
        }
        for judge in manifest.judges ?? [] where judge.kind == "local" {
            let revision = (judge.revision ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !revision.isEmpty, !isCommitLike(revision) {
                offenders.append("judge '\(judge.name)' pins '\(revision)'")
            }
        }
        guard !offenders.isEmpty else { return nil }
        return "revision pin(s) name a moving reference rather than a commit: "
            + offenders.joined(separator: "; ")
            + ". A branch or tag is re-pointed by definition, so it cannot "
            + "identify the weights a run used — two runs a week apart would "
            + "record the same pin having loaded different bytes. Use the "
            + "commit hash (the Resolve button reads it from whichever "
            + "substrate will run the model)"
    }

    static func studyModelJudgePinConflict(
        _ manifest: ExperimentManifest
    ) -> String? {
        guard manifest.sweep?.selection?.objective?.metric == "judgeScore" else {
            return nil
        }
        let studyRevision = (manifest.modelRevision ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let studyDtype = (manifest.dtype ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        var offenders: [String] = []
        for judge in manifest.judges ?? [] where judge.kind == "local" {
            let declared = (judge.model ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            // Blank model AND explicit study model both resolve to the
            // study model.
            guard declared.isEmpty || declared == manifest.modelID else {
                continue
            }
            let revision = (judge.revision ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !revision.isEmpty, revision != studyRevision {
                offenders.append(
                    "'\(judge.name)' pins revision '\(revision)' but the study "
                        + (studyRevision.isEmpty
                            ? "has no revision pinned"
                            : "is pinned at '\(studyRevision)'"))
            }
            let dtype = (judge.dtype ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !dtype.isEmpty,
                normalizeJudgeDtype(dtype) != normalizeJudgeDtype(studyDtype)
            {
                offenders.append(
                    "'\(judge.name)' pins dtype '\(dtype)' but the study "
                        + (studyDtype.isEmpty
                            ? "pins none (the device decides)"
                            : "is pinned at '\(studyDtype)'"))
            }
        }
        guard !offenders.isEmpty else { return nil }
        return "this study selects on judgeScore, and local judge(s) "
            + "resolving to the STUDY model cannot pin a different identity: "
            + offenders.joined(separator: "; ")
            + ". Such a judge IS the study model — a sweep judges with the "
            + "already-held weights and never loads anything else, so the "
            + "divergent pin would be silently ignored. Drop the pin to "
            + "inherit the study's, or name a different model to make it a "
            + "genuinely separate judge"
    }

    static func unloadableStudyDtypeProblem(
        _ manifest: ExperimentManifest
    ) -> String? {
        let spelled = (manifest.dtype ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !spelled.isEmpty, normalizeJudgeDtype(spelled) == nil else {
            return nil
        }
        return "study dtype '\(spelled)' is not one this engine can load — "
            + "the loader accepts only "
            + judgeDtypeVocabulary.joined(separator: ", ")
            + " (aliases bf16/fp16/fp32). Leave it unset to let the device "
            + "decide, which is what every study did before this pin existed"
    }

    static func unpinnedForeignLocalJudgeProblem(
        _ manifest: ExperimentManifest
    ) -> String? {
        var offenders: [String] = []
        var unknown: [String] = []
        for judge in manifest.judges ?? [] where judge.kind == "local" {
            // A dtype OUTSIDE the closed vocabulary is checked for every
            // local judge, pinned or not: the server's loader refuses it at
            // run time, and discovering that on a compute node after a queue
            // wait is exactly the failure this firewall exists to move
            // forward in time (external review round 4, finding 2).
            let spelled = (judge.dtype ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !spelled.isEmpty, normalizeJudgeDtype(spelled) == nil {
                unknown.append("'\(judge.name)' declares dtype '\(spelled)'")
            }
            let declared = (judge.model ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !declared.isEmpty, declared != manifest.modelID else { continue }
            var missing: [String] = []
            if (judge.revision ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            {
                missing.append("revision")
            }
            if (judge.dtype ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            {
                missing.append("dtype")
            }
            if !missing.isEmpty {
                offenders.append(
                    "'\(judge.name)' (model '\(declared)') is missing "
                        + missing.joined(separator: " and "))
            }
        }
        if !unknown.isEmpty {
            return "local judge(s) declare a dtype this engine cannot load: "
                + unknown.joined(separator: "; ")
                + ". The loader accepts only "
                + judgeDtypeVocabulary.joined(separator: ", ")
                + " (aliases bf16/fp16/fp32). An unrecognized value used to "
                + "load float32 silently, so the pin would be a false claim"
        }
        guard !offenders.isEmpty else { return nil }
        return "local judge(s) naming a model other than the study model "
            + "must pin the exact bytes that will judge: "
            + offenders.joined(separator: "; ")
            + ". Without a revision pin two judging sessions can load "
            + "different defaults while both records say 'none', so a "
            + "resumed evaluation cannot prove its reused verdicts came "
            + "from the same judge. Pin judges[].revision and "
            + "judges[].dtype, or use the study model as judge"
    }

    static func foreignLocalJudges(_ manifest: ExperimentManifest) -> [String] {
        (manifest.judges ?? []).compactMap { judge -> String? in
            guard !judge.name.isEmpty else { return nil }
            let kind = judge.kind.trimmingCharacters(in: .whitespacesAndNewlines)
            guard kind == "local" else { return nil }
            let declared = (judge.model ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !declared.isEmpty, declared != manifest.modelID else {
                return nil
            }
            return "'\(judge.name)' (model '\(declared)')"
        }
    }

    static func localJudgePipelineProblem(_ manifest: ExperimentManifest) -> String? {
        guard pipelineBlockViolations(manifest.pipeline).isEmpty,
            let draft = PipelineDraft.parse(manifest.pipeline),
            draft.stages.contains("sweep"),
            manifest.sweep?.selection?.objective?.metric == "judgeScore"
        else { return nil }
        let offenders = foreignLocalJudges(manifest)
        guard !offenders.isEmpty else { return nil }
        return "the declared pipeline's sweep stage holds ONE model — "
            + "the study model '\(manifest.modelID)' — but local judge(s) "
            + offenders.joined(separator: ", ")
            + " resolve to a different model, which cannot load inside the "
            + "chain (the judge fan-out covers the evaluate stage only). "
            + "Leave a local judge's model empty to judge with the study "
            + "model, pin claude/openrouter judges, or select on logprobShift"
    }

    static func localJudgeFanoutNote(_ manifest: ExperimentManifest) -> String? {
        guard pipelineBlockViolations(manifest.pipeline).isEmpty,
            let draft = PipelineDraft.parse(manifest.pipeline),
            draft.stages.contains("evaluate")
        else { return nil }
        let offenders = foreignLocalJudges(manifest)
        guard !offenders.isEmpty else { return nil }
        return "the pipeline's evaluate stage will judge local judge(s) "
            + offenders.joined(separator: ", ")
            + " as a post-generation judge fan-out (one worker job per "
            + "distinct judge model; available on Slurm run-first pipeline "
            + "submissions — elsewhere the emitted packets await deferred "
            + "judging)"
    }

    static func judgePanelIndistinctProblem(_ manifest: ExperimentManifest) -> String? {
        let judges = (manifest.judges ?? []).filter { !$0.name.isEmpty }
        guard judges.count >= 2 else { return nil }
        var identities: [ResolvedJudgeIdentity: [String]] = [:]
        var order: [ResolvedJudgeIdentity] = []
        for judge in judges {
            let identity = resolvedJudgeIdentity(
                judge, studyModelID: manifest.modelID)
            if identities[identity] == nil { order.append(identity) }
            identities[identity, default: []].append(judge.name)
        }
        guard identities.count < 2, let identity = order.first else { return nil }
        let names = identities[identity] ?? []
        let quoted = names.map { "'\($0)'" }
        let joined =
            quoted.count == 2
            ? quoted.joined(separator: " and ")
            : quoted.dropLast().joined(separator: ", ") + " and "
                + (quoted.last ?? "")
        let quantifier = quoted.count == 2 ? "both" : "all"
        let what: String
        if identity.kind == "local", identity.model == manifest.modelID {
            what = "the study model at temperature 0"
        } else if !identity.provider.isEmpty {
            what =
                "the \(identity.kind) judge '\(identity.model)' via "
                + "'\(identity.provider)'"
        } else {
            what = "the \(identity.kind) judge '\(identity.model)'"
        }
        return "judges \(joined) \(quantifier) resolve to the same "
            + "deterministic judge (\(what)) — they would agree perfectly by "
            + "construction; use judges with different models, kinds, or "
            + "providers"
    }

    static func singleJudgePanelAdvisory(
        _ manifest: ExperimentManifest
    ) -> String? {
        let judges = (manifest.judges ?? []).filter { !$0.name.isEmpty }
        let judgeEvaluated =
            manifest.evaluation?.kind == .pairedJudge || !judges.isEmpty
        guard judgeEvaluated, judges.count == 1 else { return nil }
        return singleJudgePanelAdvisoryText
    }

    static func checkJudgeEvaluationValidity(_ manifest: ExperimentManifest) throws {
        let judgeEvaluated =
            manifest.evaluation?.kind == .pairedJudge || !(manifest.judges ?? []).isEmpty
        guard judgeEvaluated else { return }
        let name = manifest.name
        guard manifest.judgeRubricFile != nil, manifest.judgeRubricHash != nil else {
            throw ExperimentError(
                reason: "cannot freeze '\(name)': judge-evaluated study has no pinned "
                    + "judge rubric file — pin one: 'steerlab-cli experiment "
                    + "pin-rubric \(name) \(JudgeRubricStore.defaultRubricFile)'; "
                    + "inline rubric text is draft-only. Or freeze --force")
        }
        // ONE judge is a legal design (maintainer ruling, 2026-08-28): a
        // single-coder study is a real methodology, and the gate's job is to
        // refuse the INVALID state — a judged instrument with no judge —
        // not to legislate the panel size. What the ≥2 rule was protecting
        // (inter-rater agreement) survives as the non-blocking
        // `singleJudgePanelAdvisory`, said at freeze and at declaration.
        let judgeCount = (manifest.judges ?? []).count
        guard judgeCount >= 1 else {
            throw ExperimentError(
                reason: "cannot freeze '\(name)': "
                    + Self.noJudgeDeclaredReason(experimentName: name))
        }
        if let indistinct = judgePanelIndistinctProblem(manifest) {
            throw ExperimentError(reason: "cannot freeze '\(name)': \(indistinct)")
        }
        if let pipelineProblem = localJudgePipelineProblem(manifest) {
            throw ExperimentError(reason: "cannot freeze '\(name)': \(pipelineProblem)")
        }
        if let unpinned = unpinnedForeignLocalJudgeProblem(manifest) {
            throw ExperimentError(reason: "cannot freeze '\(name)': \(unpinned)")
        }
        if let conflict = studyModelJudgePinConflict(manifest) {
            throw ExperimentError(reason: "cannot freeze '\(name)': \(conflict)")
        }
    }

    static func isCommitLike(_ revision: String) -> Bool {
        let stripped = revision.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !stripped.isEmpty else { return false }
        return stripped.allSatisfy(\.isHexDigit)
    }

    static func normalizeJudgeDtype(_ value: String?) -> String? {
        judgeDtypeAliases[
            (value ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()]
    }

    static func resolvedJudgeIdentity(
        _ judge: ExperimentManifest.JudgeRef, studyModelID: String
    ) -> ResolvedJudgeIdentity {
        let rawKind = judge.kind.trimmingCharacters(in: .whitespacesAndNewlines)
        let kind = rawKind.isEmpty ? "claude" : rawKind
        var model = (judge.model ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let rawProvider = (judge.provider ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let provider =
            kind == "openrouter"
            ? OpenRouterProviderIdentity.canonical(rawProvider)
            : rawProvider
        if kind == "local", model.isEmpty {
            model = studyModelID
        } else if kind == "claude", model.isEmpty {
            model = ClaudePairedJudge.defaultModel
        }
        return ResolvedJudgeIdentity(kind: kind, model: model, provider: provider)
    }

    static func noJudgeDeclaredReason(experimentName name: String) -> String {
        "judge-evaluated study pins no judge — a judged instrument with no "
            + "judge codes nothing; pin a panel: 'steerlab-cli experiment "
            + "pin-rubric \(name) <rubric> --judges <name>:<kind>[,…]'. Or "
            + "freeze --force"
    }

    static func pipelineBlockViolations(_ block: JSONValue?) -> [String] {
        guard let block else { return [] }
        guard case .object(let pipeline) = block else {
            return ["pipeline block invalid: must be an object"]
        }
        var violations: [String] = []
        for key in pipeline.keys where key != "stages" && key != "gates" {
            violations.append(
                "pipeline block invalid: unknown pipeline key '\(key)' — a "
                    + "typo'd key silently ignored would un-declare a gate")
        }
        // The effective stage list (absent/empty = the default chain) — the
        // gate-membership check below needs it.
        var stages = ["extract", "validate", "sweep", "promote", "run"]
        if let rawStages = pipeline["stages"], !isNull(rawStages) {
            guard case .array(let items) = rawStages else {
                return violations
                    + ["pipeline block invalid: 'stages' must be a list of stage names"]
            }
            var names: [String] = []
            for item in items {
                guard case .string(let name) = item else {
                    return violations
                        + ["pipeline block invalid: 'stages' must be a list of stage names"]
                }
                names.append(name)
            }
            if !names.isEmpty {
                let order = Dictionary(
                    uniqueKeysWithValues: pipelineValidStages.enumerated()
                        .map { ($1, $0) })
                let unknown = names.filter { order[$0] == nil }
                if !unknown.isEmpty {
                    violations.append(
                        "pipeline block invalid: unknown stage(s) "
                            + unknown.joined(separator: ", "))
                }
                if Set(names).count != names.count {
                    violations.append(
                        "pipeline block invalid: 'stages' contains duplicates")
                }
                let indices = names.compactMap { order[$0] }
                if indices != indices.sorted() {
                    violations.append(
                        "pipeline block invalid: 'stages' must follow the "
                            + "canonical order "
                            + pipelineValidStages.joined(separator: ", "))
                }
                if names.contains("evaluate") || names.contains("analyze"),
                    !names.contains("run")
                {
                    violations.append(
                        "pipeline block invalid: evaluate/analyze require "
                            + "'run' in the same chain")
                }
                stages = names
            }
        }
        if let rawGates = pipeline["gates"], !isNull(rawGates) {
            guard case .object(let gates) = rawGates else {
                return violations + ["pipeline block invalid: 'gates' must be an object"]
            }
            let knownKeys: [String: Set<String>] = [
                "validate": [
                    "minScenarioAccuracy", "maxCrossConceptCosine",
                    "accuracyFloor",
                ],
                "sweep": ["requireSelectionForEveryConcept"],
            ]
            for (gateName, gate) in gates {
                guard let keys = knownKeys[gateName] else {
                    violations.append(
                        "pipeline block invalid: no gate is defined for "
                            + "stage '\(gateName)'")
                    continue
                }
                if !stages.contains(gateName) {
                    violations.append(
                        "pipeline block invalid: gate '\(gateName)' names a "
                            + "stage that is not in the stage list")
                }
                guard case .object(let fields) = gate else {
                    if !isNull(gate) {
                        violations.append(
                            "pipeline block invalid: gate '\(gateName)' must "
                                + "be an object")
                    }
                    continue
                }
                for (key, value) in fields {
                    guard keys.contains(key) else {
                        violations.append(
                            "pipeline block invalid: unknown \(gateName)-gate "
                                + "key '\(key)'")
                        continue
                    }
                    if gateName == "validate", key == "accuracyFloor",
                        !isNull(value)
                    {
                        violations += accuracyFloorViolations(value)
                        continue
                    }
                    if gateName == "validate", !isNull(value) {
                        guard case .number(let threshold) = value,
                            threshold >= 0, threshold <= 1
                        else {
                            violations.append(
                                "pipeline block invalid: gate "
                                    + "'\(gateName).\(key)' must be a number "
                                    + "in [0, 1]")
                            continue
                        }
                    }
                }
                // The legacy key IS the transferAccuracy floor — declaring
                // it beside an accuracyFloor is one ambiguity, refused on
                // both engines (server resolver twin).
                if gateName == "validate",
                    let legacy = fields["minScenarioAccuracy"], !isNull(legacy),
                    let declared = fields["accuracyFloor"], !isNull(declared)
                {
                    violations.append(
                        "pipeline block invalid: both minScenarioAccuracy "
                            + "and accuracyFloor are declared — declare "
                            + "exactly one")
                }
            }
        }
        return violations
    }

    static func isNull(_ value: JSONValue) -> Bool {
        if case .null = value { return true }
        return false
    }

    static func accuracyFloorViolations(_ value: JSONValue) -> [String] {
        guard case .object(let floor) = value,
            Set(floor.keys) == ["metric", "minimum"]
        else {
            return [
                "pipeline block invalid: 'validate.accuracyFloor' must be "
                    + "an object {\"metric\": …, \"minimum\": …}"
            ]
        }
        var violations: [String] = []
        if case .string(let metric)? = floor["metric"] {
            if !pipelineAccuracyFloorMetrics.contains(metric) {
                violations.append(
                    "pipeline block invalid: unknown accuracyFloor metric "
                        + "'\(metric)' — declare one of "
                        + pipelineAccuracyFloorMetrics.joined(separator: ", "))
            }
        } else {
            violations.append(
                "pipeline block invalid: accuracyFloor metric must be one of "
                    + pipelineAccuracyFloorMetrics.joined(separator: ", "))
        }
        if case .number(let minimum)? = floor["minimum"],
            minimum >= 0, minimum <= 1
        {
            // In range — fine.
        } else {
            violations.append(
                "pipeline block invalid: gate 'validate.accuracyFloor.minimum' "
                    + "must be a number in [0, 1]")
        }
        return violations
    }

    static func plainGateReason(_ reason: String, experimentName: String) -> String {
        let prefix = "cannot freeze '\(experimentName)': "
        guard reason.hasPrefix(prefix) else { return reason }
        return String(reason.dropFirst(prefix.count))
    }

    static func validateCLI(forRunSubstrate substrate: String) -> String {
        substrate == WorkspaceScoping.serverSubstrate
            ? "steerlab-server" : "steerlab-cli"
    }
    static let singleJudgePanelAdvisoryText =
        "single-coder design: this study pins 1 judge, so no inter-rater "
        + "agreement statistics (percent agreement, Cohen's kappa) will exist "
        + "for its codings — the coding report records fieldAgreement as "
        + "absent with that reason rather than empty"
    static let pipelineAccuracyFloorMetrics = [
        "transferAccuracy", "calibratedAccuracy",
        "calibratedBalancedAccuracy", "auc",
    ]
}
