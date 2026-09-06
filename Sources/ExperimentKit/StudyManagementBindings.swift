import Foundation

/// Compatibility entry points for consumers migrating to management and design owners.
extension ExperimentPanel {
    private var studyCreationContext: StudyCreationContext? {
        guard let host else { return nil }
        return StudyCreationContext(
            workspaceDefaultModelID: host.workspaceSelectedModelID ?? host.selectedModelID,
            modelOptions: modelOptions)
    }

    public var experiments: [ExperimentManifest] {
        management.experiments
    }

    public var selectedName: String? {
        get { management.selectedName }
        set { management.selectedName = newValue }
    }

    public var selected: ExperimentManifest? {
        management.selected
    }

    public var displayLabels: [String: String] {
        management.displayLabels
    }

    public var renameInvitation: String? {
        get { management.renameInvitation }
        set { management.renameInvitation = newValue }
    }

    public var deleteSelectedStudyRefusal: String? {
        management.deleteSelectedStudyRefusal
    }

    public func displayName(_ manifest: ExperimentManifest) -> String {
        management.displayName(manifest)
    }

    public func newStudy() {
        management.newStudy(context: studyCreationContext)
    }

    public func create() {
        management.create(context: studyCreationContext)
    }

    public func renameSelected(canonicalName: String?, label: String?) {
        management.renameSelected(canonicalName: canonicalName, label: label)
    }

    public func refreshTemplates() {
        management.refreshTemplates()
    }

    public func designSummary(_ template: StudyTemplate) -> [StudyDesignSummary.Row] {
        management.designs.designSummary(template)
    }

    public func newDesignFromStudy(named name: String) {
        management.newDesignFromStudy(named: name)
    }

    public func updateTemplateDescription(_ name: String, to description: String) {
        management.updateTemplateDescription(name, to: description)
    }

    @discardableResult
    public func editDesign(_ name: String) -> String? {
        management.editDesign(name)
    }

    @discardableResult
    public func newDesignDraft() -> String? {
        management.newDesignDraft(context: studyCreationContext)
    }

    public func saveBackToDesignTarget(
        for manifest: ExperimentManifest
    ) -> String? {
        management.designs.saveBackToDesignTarget(for: manifest)
    }

    public func saveBackToDesignRefusal(
        for manifest: ExperimentManifest
    ) -> String? {
        management.designs.saveBackToDesignRefusal(for: manifest)
    }

    public func saveSelectedStudyBackToDesign() {
        management.saveSelectedStudyBackToDesign()
    }

    public func renameTemplate(_ oldName: String, to newName: String) {
        management.renameTemplate(oldName, to: newName)
    }

    public func deleteTemplate(_ name: String) {
        management.deleteTemplate(name)
    }

    public func templateLineage(_ manifest: ExperimentManifest) -> String? {
        management.designs.templateLineage(manifest, experiments: management.experiments)
    }

    public func batchSiblings(_ manifest: ExperimentManifest) -> [String] {
        management.batchSiblings(manifest)
    }

    public func deleteSelectedDraft() {
        management.deleteSelectedDraft()
    }

    public func duplicateSelected() {
        management.duplicateSelected()
    }

}
