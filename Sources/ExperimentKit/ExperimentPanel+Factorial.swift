import Foundation

/// The Generate & Pin action behind the factorial-design sheet (Usability
/// Plan Phase 4, items 20–21). Own file on purpose: `ExperimentPanel.swift`
/// stays untouched; this mirrors `importTaskPromptsTable`'s shape exactly —
/// draft-only guard, the transactional `FactorialImport` (write emitted
/// JSONL + provenance design JSON → pin → persist inside the transaction),
/// and the plain problem string returned for the sheet to display.
extension ExperimentPanel {

    /// Generates the design's full crossing, lands it as the study's
    /// task-prompts file (never overwriting differing bytes unless the
    /// sheet's explicit replace affordance is set), pins its hash, saves the
    /// manifest INSIDE the import's transaction, and reloads the editor.
    /// Returns the plain problem to show in the sheet, or nil when the
    /// generation landed (dismiss).
    public func generateFactorialTaskPrompts(
        design: FactorialDesign, replacingExisting: Bool
    ) -> String? {
        guard var manifest = selected, manifest.status == .draft else {
            return "select a draft study first — the generated prompts file "
                + "pins into the draft manifest"
        }
        do {
            let result = try FactorialImport.generateIntoStudy(
                design: design, manifest: &manifest,
                replacingExisting: replacingExisting,
                persist: { try ExperimentStore.save($0) })
            taskPromptsFile = result.file
            refresh()
            loadTaskPrompts()
            note(
                "generated \(result.itemCount) factorial "
                    + "item\(result.itemCount == 1 ? "" : "s") → \(result.file), "
                    + "pinned @ \(result.hash.prefix(12))… (design spec saved to "
                    + "\(result.designFile) for provenance)",
                severity: .success)
            return nil
        } catch {
            return "\(error)"
        }
    }
}
