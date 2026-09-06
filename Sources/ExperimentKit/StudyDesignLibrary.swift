import Foundation
import Observation

/// Design inventory and cached lineage. Refresh reads the stores; display helpers read only this cache.
@Observable @MainActor
public final class StudyDesignLibrary {
    public init() {}
    // MARK: Study templates (the invariant half of a replication)

    /// The workspace's design library, refreshed with the study list.
    /// ("Template" is the artifact's stored name; the interface calls one a
    /// DESIGN, which is what it is.)
    public private(set) var templates: [StudyTemplate] = []
    @ObservationIgnored private var reviews: [String: StudyDesignSnapshot] = [:]
    public var selectedTemplateName: String?
    /// The template a just-completed "Load as Template" wants opened — the
    /// view consumes and clears it, exactly as `renameInvitation` works.
    /// Also the Templates tab's cross-section handoff: Instantiate sets it and
    /// navigates to Studies, which opens the flow on it (consumed on appear as
    /// well as on change — the Studies view is not on screen when it is set).
    /// Also carries the occupant pool "Create permuted siblings…" hands over,
    /// so the table opens already holding one row per distinct re-seating.
    public var templateInstantiationInvitation: TemplateInstantiationInvitation?
    /// The Templates tab's "New from Study" source. A study NAME, chosen from
    /// the whole list at any status.
    public var templateSourceStudyName: String?
    /// What the next new study starts from (the Studies tab's first control).
    public var newStudyDesign: StudyDesignChoice = .fromScratch

    public var selectedTemplate: StudyTemplate? {
        templates.first { $0.name == selectedTemplateName }
    }

    /// Live design lineage per minted study, keyed by study name — both facts
    /// (`agreement` + `designRevised`, see `StudyTemplateStore.DesignLineage`).
    ///
    /// Computed once per `refresh()` and never per frame: the check re-reads
    /// each minted study's compiled panel from disk, which a SwiftUI body must
    /// not do. Studies with no lineage are absent from the map.
    public private(set) var designLineage: [String: StudyTemplateStore.DesignLineage] = [:]

    public func refresh(experiments: [ExperimentManifest]) {
        let root = ExperimentStore.workspaceRoot
        let names = (try? FileManager.default.contentsOfDirectory(atPath: root.appending(component: "templates").path)) ?? []
        let snapshots = names.compactMap { try? StudyDesignSnapshot(workspaceRoot: root, name: $0) }
        reviews = Dictionary(uniqueKeysWithValues: snapshots.map { ($0.template.name, $0) })
        templates = snapshots.map(\.template).sorted { $0.createdAt > $1.createdAt }
        if let selectedTemplateName,
            !templates.contains(where: { $0.name == selectedTemplateName })
        {
            self.selectedTemplateName = nil
        }
        newStudyDesign = StudyDesignChoice.resolve(newStudyDesign, designs: templates)
        if let templateSourceStudyName,
            !experiments.contains(where: { $0.name == templateSourceStudyName })
        {
            self.templateSourceStudyName = nil
        }
        designLineage = [:]
        for manifest in experiments where manifest.templateProvenance != nil {
            designLineage[manifest.name] = StudyTemplateStore.lineage(of: manifest)
        }
    }

    /// A confirmation retains this value; later catalog refreshes cannot
    /// authorize its old source settings against a newly read design version.
    public func reviewedDesign(named name: String) throws -> StudyDesignSnapshot {
        guard let review = reviews[name],
            try ManifestFileTransaction.canonicalPath(review.workspaceRoot) == ManifestFileTransaction.canonicalPath(ExperimentStore.workspaceRoot) else {
            throw StudyDesignAuthoringError(code: "designChanged", reason: "The reviewed design is unavailable in this workspace.",
                repairAction: "Reload the design library and review the intended destination before saving.")
        }
        return review
    }

    /// The design-summary rows for a template — the Templates tab's read-only
    /// body. On the panel so the view calls one function and formats nothing.
    public func designSummary(_ template: StudyTemplate) -> [StudyDesignSummary.Row] {
        StudyDesignSummary.rows(for: template)
    }

    /// The design a study can be saved back ONTO, or nil.
    ///
    /// Nil is not an error state — most studies have no design behind them.
    /// Paired with `saveBackToDesignRefusal` so the control can be present and
    /// explain itself rather than vanish.
    public func saveBackToDesignTarget(
        for manifest: ExperimentManifest
    ) -> String? {
        guard let provenance = manifest.templateProvenance else { return nil }
        guard templates.contains(where: { $0.name == provenance.template })
        else { return nil }
        return provenance.template
    }

    /// Why the selected study cannot be saved back onto a design, or nil.
    ///
    /// TWO refusals, both about the design rather than the study: it has to
    /// exist, and this study has to name it.
    ///
    /// Status is deliberately NOT one of them (2026-08-06). A design carries no
    /// lifecycle stamps — `strippedBody` removes every one of them — so writing
    /// a frozen study's settings onto a design cannot make the design claim to
    /// be frozen, and it cannot touch the frozen study either: the write goes to
    /// `templates/<name>/`, the run record is untouched, and studies minted
    /// earlier keep the hash stamped at their own mint time. "New from Study"
    /// already accepts any status for exactly this reason; refusing here only
    /// forced the researcher to duplicate a frozen study into a draft to write
    /// the same bytes.
    public func saveBackToDesignRefusal(
        for manifest: ExperimentManifest
    ) -> String? {
        guard let provenance = manifest.templateProvenance else {
            return "'\(manifest.name)' was not minted from a design — use Save "
                + "as new design"
        }
        guard templates.contains(where: { $0.name == provenance.template })
        else {
            return "design '\(provenance.template)' is no longer in the library "
                + "(renamed or deleted) — use Save as new design"
        }
        return nil
    }

    /// The lineage line for a minted study: which recipe it came from, and
    /// which batch of siblings it belongs to.
    ///
    /// Sibling counting is by `batchGroup` across the workspace, because
    /// panel castings are sibling STUDIES by necessity (one scenario per
    /// manifest on both engines) and the batch id is the only thing tying
    /// them together.
    /// Divergence is READ FROM THE CACHE built at refresh (`designLineage`),
    /// never recomputed here: this is called from a view body, and the check
    /// re-reads files.
    public func templateLineage(_ manifest: ExperimentManifest, experiments: [ExperimentManifest])
        -> String?
    {
        guard let provenance = manifest.templateProvenance else { return nil }
        // Editable-but-visibly-diverged is the policy: nothing here blocks an
        // edit, and the word changes so the lineage line stops claiming a
        // replication the study no longer is.
        let lineage =
            designLineage[manifest.name]
            ?? .init(agreement: .matches, designRevised: false)
        let agreement = lineage.agreement
        var line =
            agreement == .diverged
            ? "diverged from design '\(provenance.template)' "
            : "from design '\(provenance.template)' "
        line += "@ \(provenance.templateHash.prefix(12))…"
        if agreement == .designMissing {
            line += " (no longer in the library)"
        }
        // The SECOND, independent fact. Without it a study minted before a
        // design was revised reads as a plain "from design 'X'" forever, and
        // the researcher takes that as a replication of the recipe now
        // filed under X — which it is not. The study's own stamp comparison
        // is unchanged and still leads the line.
        if lineage.designRevised {
            line +=
                agreement == .diverged
                ? " · the design has since been revised too"
                : " · matches its design as minted · the design has since "
                    + "been revised"
        }
        if let batch = provenance.batchGroup {
            let siblings = experiments.filter {
                $0.templateProvenance?.batchGroup == batch
            }
            line += " · batch \(batch)"
            if siblings.count > 1 {
                line += " (\(siblings.count) studies minted together)"
            }
        }
        if agreement == .diverged {
            line +=
                " — edited since minting; edits are allowed and the stamp "
                + "records where it started"
        }
        return line
    }
}
