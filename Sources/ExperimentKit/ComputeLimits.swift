import Foundation

/// Say, while a study is still being designed, what the workspace's compute
/// cannot run. Advice only: nothing here refuses, and verify and freeze stay
/// open, because a study authored on this Mac and run on the Python engine
/// is legitimate. The run paths keep their own refusals.
///
/// What each compute choice can run, and the sentence that says so, are the
/// shipped science catalog's `whereItRuns`, which the generator writes from
/// `docs/substrate-capabilities.json`. This type decides only which of those
/// declarations a manifest makes and which choice the workspace is on.
/// Python twin: `compute_limits`.
public enum ComputeLimits {

    /// The catalog's account, read once. Nil only if the compiled resource
    /// fails to decode, which the catalog tests catch.
    public static let shipped: ScienceCatalog.WhereItRuns? =
        try? ScienceCatalog.catalog().whereItRuns

    /// The study features a manifest declares, by catalog id, in catalog
    /// order. A multi-agent study runs a scenario and never arms the
    /// model-output features it may carry from before a kind switch, so
    /// only probe measurements count for it, as on the run path.
    public static func declaredFeatures(_ manifest: ExperimentManifest) -> [String] {
        var found: Set<String> = []
        if manifest.probeMeasurements != nil { found.insert("probeMeasurements") }
        if manifest.studyKind == .modelOutput {
            if manifest.variantConditions.contains(where: {
                !($0.artifact.interventionPolicies ?? []).isEmpty
            }) {
                found.insert("interventionPolicies")
            }
            // Counted the way `latentArmsNotExecutableProblem` counts them.
            if case .array(let entries)? = manifest.saeLatentConditions, !entries.isEmpty {
                found.insert("saeLatentArms")
            }
            if manifest.jlensReadout != nil { found.insert("jlensReadout") }
        }
        return (shipped?.studyFeatures ?? []).map(\.id).filter(found.contains)
    }

    /// The declared features `choice` cannot run, in catalog order.
    public static func unrunnable(
        _ manifest: ExperimentManifest, on choice: ComputeChoice
    ) -> [ScienceCatalog.WhereItRuns.StudyFeature] {
        let features = Dictionary(
            (shipped?.studyFeatures ?? []).map { ($0.id, $0) },
            uniquingKeysWith: { first, _ in first })
        return declaredFeatures(manifest).compactMap { id in
            guard let feature = features[id], feature.runsOn[choice.rawValue] == false
            else { return nil }
            return feature
        }
    }

    /// One sentence per declared feature `choice` cannot run, exactly as the
    /// catalog words it.
    public static func advisories(
        for manifest: ExperimentManifest, choice: ComputeChoice
    ) -> [String] {
        unrunnable(manifest, on: choice).compactMap { $0.advisories[choice.rawValue] }
    }

    /// How a researcher on this Mac reaches a choice that can run the study.
    public static let switchClause =
        " Choose one in the app\u{2019}s Workspace menu as where this workspace "
        + "runs studies."

    /// The choice this workspace is on, as the readiness checklist and
    /// verify read it: declared, else read from its own runs, else the quick
    /// start (`WorkspaceCompute.resolvedChoice`).
    public static func workspaceChoice(root: URL) -> ComputeChoice {
        WorkspaceCompute.resolvedChoice(root: root)
    }

    /// What `experiment verify` says on this workspace: each catalog
    /// sentence, then the way to switch.
    public static func workspaceAdvisories(
        for manifest: ExperimentManifest, root: URL
    ) -> [String] {
        advisories(for: manifest, choice: workspaceChoice(root: root))
            .map { $0 + switchClause }
    }

    /// The readiness checklist's row for the same fact. Absent when the
    /// study declares none of the features; present when the choice runs all
    /// of them; partial when it cannot run one. Never missing or invalid, so
    /// never a blocker.
    public static func requirement(
        for manifest: ExperimentManifest, choice: ComputeChoice
    ) -> DataRequirement? {
        let declared = declaredFeatures(manifest)
        guard !declared.isEmpty, let shipped else { return nil }
        let phrases = shipped.studyFeatures.filter { declared.contains($0.id) }
            .map(\.phrase)
        let notes = advisories(for: manifest, choice: choice)
        let path = "experiments/\(manifest.name)/experiment.json"
        guard !notes.isEmpty else {
            return DataRequirement(
                id: requirementID, title: requirementTitle, kind: .computeChoice,
                status: .present, path: path,
                detail: "\u{201C}\(choice.title)\u{201D} can run everything this "
                    + "study declares that needs a particular engine: "
                    + "\(listed(phrases)).")
        }
        return DataRequirement(
            id: requirementID, title: requirementTitle, kind: .computeChoice,
            status: .partial, path: path,
            detail: notes.joined(separator: " ") + switchClause)
    }

    static let requirementID = "computeChoice"
    static let requirementTitle = "where this study can run"

    /// "a", "a and b", "a, b, and c": the same in every locale.
    static func listed(_ items: [String]) -> String {
        switch items.count {
        case 0: ""
        case 1: items[0]
        case 2: "\(items[0]) and \(items[1])"
        default: items.dropLast().joined(separator: ", ") + ", and " + items[items.count - 1]
        }
    }
}
