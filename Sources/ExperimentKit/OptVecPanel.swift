import Foundation
import Observation

/// Observable state for the app's OptVec surface (v1: read-only + the one
/// attach action). Thin by design: every fact comes from the two stateless
/// stores (`OptVecBundleStore` — dataset bundles with data-check verdicts;
/// `OptVecRunStore` — campaign/train/eval/interpretation artifacts), read
/// from the WORKSPACE once per `refresh()`/selection change, never per
/// frame. Server-side state is never canonical here: campaign "progress" is
/// the offline derivation from workspace files, honestly labeled where the
/// scheduler's live word is missing.
///
/// v2/v3 (authoring, duplicate-and-adjust) add actions on this panel that
/// write through the stores — the read model stays as it is. See the
/// version contract in `OptVecBundleStore`'s header.
@Observable @MainActor
public final class OptVecPanel {
    public internal(set) weak var host: ChatService?

    public init() {}

    // MARK: - Read model (rebuilt from disk, once per refresh)

    public private(set) var bundles: [OptVecBundleStore.Entry] = []
    /// Data-check report per bundle name — the Swift mirror of
    /// `steerlab-server data check optvec`, computed at refresh (it hashes
    /// every bundle file, which a SwiftUI body must not do).
    public private(set) var reports: [String: OptVecBundleStore.Report] = [:]
    public private(set) var runs: [OptVecRunStore.RunItem] = []

    public var selectedBundleName: String?
    public var selectedRunName: String?

    // Detail for the SELECTED run only (one decode per selection, not one
    // per listed run).
    public private(set) var campaignStatus: OptVecRunStore.CampaignStatus?
    public private(set) var trainProgress: OptVecRunStore.TrainProgress?
    public private(set) var evalReport: OptVecRunStore.EvalReport?
    public private(set) var interpretReport: OptVecRunStore.InterpretReport?
    public private(set) var familyReport: OptVecRunStore.FamilyReport?
    public private(set) var jspaceReport: OptVecRunStore.JSpaceReport?
    public private(set) var geometryReport: OptVecRunStore.GeometryReport?

    // MARK: - Attach form (the ONE v1 action)

    public var attachStudyName = ""
    public var attachConceptName = ""
    public var attachArtifactReference = ""
    public var attachEvalRun = ""
    public var status: String?
    /// The last attach refusal, verbatim — rendered beside the form.
    public var attachError: String?

    public func refresh() {
        bundles = OptVecBundleStore.list()
        var checked: [String: OptVecBundleStore.Report] = [:]
        for entry in bundles {
            checked[entry.name] = OptVecBundleStore.check(
                directory: entry.directory)
        }
        reports = checked
        runs = OptVecRunStore.list()
        // Disk-reading derivations happen HERE, once — a SwiftUI body must
        // never trigger a runs/ scan or a manifest decode per frame.
        attachableArtifacts = runs.filter { $0.kind == .train }
            .compactMap { run in
                OptVecRunStore.trainProgress(runURL: run.url)
                    .artifactReference.map { (run.name, $0) }
            }
        draftStudyNames = ExperimentStore.list()
            .filter { $0.status == .draft }
            .map(\.name)
        if let selectedBundleName,
            !bundles.contains(where: { $0.name == selectedBundleName })
        {
            self.selectedBundleName = nil
        }
        if let selectedRunName,
            !runs.contains(where: { $0.name == selectedRunName })
        {
            self.selectedRunName = nil
        }
        loadSelectedRunDetail()
    }

    public var selectedBundle: OptVecBundleStore.Entry? {
        bundles.first { $0.name == selectedBundleName }
    }

    public var selectedRun: OptVecRunStore.RunItem? {
        runs.first { $0.name == selectedRunName }
    }

    /// Re-decode the selected run's artifacts. Cheap (one run's JSON), so
    /// selection changes call it directly.
    public func loadSelectedRunDetail() {
        campaignStatus = nil
        trainProgress = nil
        evalReport = nil
        interpretReport = nil
        familyReport = nil
        jspaceReport = nil
        geometryReport = nil
        guard let run = selectedRun else { return }
        switch run.kind {
        case .campaign:
            campaignStatus = OptVecRunStore.campaignStatus(runURL: run.url)
        case .train:
            trainProgress = OptVecRunStore.trainProgress(runURL: run.url)
        case .eval:
            evalReport = OptVecRunStore.evalReport(runURL: run.url)
        case .interpret:
            interpretReport = OptVecRunStore.interpretReport(runURL: run.url)
        case .family:
            familyReport = OptVecRunStore.familyReport(runURL: run.url)
        case .jspace:
            jspaceReport = OptVecRunStore.jspaceReport(runURL: run.url)
        case .geometry:
            geometryReport = OptVecRunStore.geometryReport(runURL: run.url)
        }
    }

    // MARK: - Attach

    /// Trained artifacts this workspace holds: every `optvec-train` run
    /// whose saved artifact (sidecar + tensor at the run root) is present.
    /// Rebuilt by `refresh()`, never computed per frame.
    public private(set) var attachableArtifacts:
        [(runName: String, reference: String)] = []

    /// Draft studies an artifact can attach to (attach is draft-only; the
    /// store's immutability refusal backs this filter up). Rebuilt by
    /// `refresh()`.
    public private(set) var draftStudyNames: [String] = []

    /// The one v1 action: pin the chosen trained artifact into a draft
    /// study as a `pinnedArtifact` concept, through
    /// `ExperimentStore.attachArtifact` (the Swift mirror of the server's
    /// attach-artifact contract — same pins, same refusals).
    public func attachSelectedArtifact() {
        attachError = nil
        let study = attachStudyName.trimmingCharacters(in: .whitespaces)
        let concept = attachConceptName.trimmingCharacters(in: .whitespaces)
        let reference = attachArtifactReference
            .trimmingCharacters(in: .whitespaces)
        guard !study.isEmpty, !concept.isEmpty, !reference.isEmpty else {
            attachError =
                "pick a draft study, a concept name, and a trained artifact"
            return
        }
        let evalRun = attachEvalRun.trimmingCharacters(in: .whitespaces)
        do {
            let manifest = try ExperimentStore.attachArtifact(
                concept, artifact: reference,
                evalRun: evalRun.isEmpty ? nil : evalRun,
                experimentName: study)
            let pin = manifest.concepts.first { $0.name == concept }?
                .vectorArtifact
            note(
                "attached '\(reference)' to '\(study)' as concept "
                    + "'\(concept)' (tensor "
                    + "\(pin?.sha256TensorHash.prefix(12) ?? "")…)",
                severity: .success)
            host?.experiments.refresh()
        } catch {
            let reason = (error as? ExperimentError)?.reason ?? "\(error)"
            attachError = reason
            note("attach failed — \(reason)", severity: .error)
        }
    }

    func note(_ message: String, severity: PanelNotice.Severity = .info) {
        status = message
        PanelNotices.shared.record(
            source: "OptVec", severity: severity, message: message)
    }
}
