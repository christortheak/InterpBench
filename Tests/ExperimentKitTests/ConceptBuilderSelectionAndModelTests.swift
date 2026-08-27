import Foundation
import Testing

@testable import ExperimentKit

/// Two hands-on-testing bugs in the Concept Vector Builder (2026-08-19):
///
/// 1. The panel carried TWO concept selections — the dataset selection
///    (`selectedExisting`, the Concept Index / Dataset Builder) and the build
///    target (`vectorBuilderSelectedExisting`, the Concept Vector Builder's own
///    picker) — that never synced, so the panel could show one concept while a
///    build extracted another.
/// 2. Local extraction read whatever model the Playground had resident instead
///    of the builder's own model selection.
///
/// Both fixes live at the model layer so every route in — the two dataset
/// pickers, the Data-section inventory's "Open in Concept Builder", the web
/// client — gets them.
@Suite(.serialized) @MainActor
struct ConceptBuilderSelectionAndModelTests {

    private func withTempWorkspace<T>(_ body: (URL) throws -> T) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "builder-selection") { root in
            let previous = WorkspaceRoot.programmaticOverride
            WorkspaceRoot.programmaticOverride = root
            defer { WorkspaceRoot.programmaticOverride = previous }
            return try body(root)
        }
    }

    private func write(_ contents: String, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try contents.write(to: url, atomically: true, encoding: .utf8)
    }

    private func textRows(_ texts: [String]) -> String {
        texts.map { #"{"text": "\#($0)"}"# }.joined(separator: "\n") + "\n"
    }

    /// A concept with a usable paired dataset (>= 4 rows per side, the build
    /// gate's floor).
    private func makeConcept(_ name: String, in root: URL, rows: Int = 5) throws {
        let directory = root.appending(components: "prompts", "concepts", name)
        try write(
            textRows((0 ..< rows).map { "\(name) positive \($0)" }),
            to: directory.appending(component: "positive.jsonl"))
        try write(
            textRows((0 ..< rows).map { "\(name) negative \($0)" }),
            to: directory.appending(component: "negative.jsonl"))
    }

    // MARK: Bug 1 — the two selections

    /// The dataset selection DRIVES the build target. Before the fix these
    /// were independent, so picking a concept in the Dataset Builder left the
    /// vector build pointed at the previous one.
    @Test func settingTheDatasetSelectionSyncsTheBuildTarget() throws {
        try withTempWorkspace { root in
            try makeConcept("french", in: root)
            try makeConcept("tired", in: root)
            let builder = ConceptBuilder()
            builder.refreshConceptList()

            builder.selectedExisting = "tired"
            #expect(builder.vectorBuilderSelectedExisting == "tired")

            builder.selectedExisting = "french"
            #expect(builder.vectorBuilderSelectedExisting == "french")
            #expect(builder.vectorBuilderConceptName == "french")
        }
    }

    /// The Data-section inventory's "Open in Concept Builder" routing seam
    /// (`SectionContainers.openInConceptBuilder`): refresh the index, then set
    /// the dataset selection. It set only ONE of the two selections when it
    /// landed, which is how a build silently targeted a different concept.
    @Test func inventoryRoutingEndsWithBothSelectionsOnTheSameConcept() throws {
        try withTempWorkspace { root in
            try makeConcept("french", in: root)
            try makeConcept("tired", in: root)
            let builder = ConceptBuilder()
            builder.selectedExisting = "tired"

            // Exactly what the inventory action does.
            builder.refreshConceptList()
            builder.selectedExisting = "french"

            #expect(builder.selectedExisting == "french")
            #expect(builder.vectorBuilderSelectedExisting == "french")
        }
    }

    /// The sync is ONE-WAY: after it lands, the build picker is still freely
    /// changeable, and that deliberate divergence survives until the next
    /// dataset-selection change.
    @Test func buildTargetDivergencePersistsUntilTheNextDatasetSelection() throws {
        try withTempWorkspace { root in
            try makeConcept("french", in: root)
            try makeConcept("tired", in: root)
            let builder = ConceptBuilder()

            builder.selectedExisting = "french"
            builder.vectorBuilderSelectedExisting = "tired"

            // Divergence stands: changing the build target does not drag the
            // dataset selection with it.
            #expect(builder.selectedExisting == "french")
            #expect(builder.vectorBuilderSelectedExisting == "tired")

            // ...and survives unrelated edits.
            builder.recipeFamily = .pairedDifferencePCA
            #expect(builder.vectorBuilderSelectedExisting == "tired")

            // The next dataset-selection change re-converges them.
            builder.selectedExisting = "french"  // unchanged: no sync
            #expect(builder.vectorBuilderSelectedExisting == "tired")
            builder.selectedExisting = "tired"
            builder.selectedExisting = "french"
            #expect(builder.vectorBuilderSelectedExisting == "french")
        }
    }

    /// The build's enablement gate and the build itself resolve the SAME name.
    /// A build target whose dataset is unusable must not light the button just
    /// because the dataset selection happens to have rows.
    @Test func theBuildGateFollowsTheBuildTargetNotTheDatasetSelection() throws {
        try withTempWorkspace { root in
            try makeConcept("french", in: root)
            // Named, but with too few rows to build.
            try makeConcept("tired", in: root, rows: 1)
            let builder = ConceptBuilder()
            builder.recipeFamily = .caaMeanDifference

            builder.selectedExisting = "french"
            #expect(builder.canSaveAndExtract)

            builder.vectorBuilderSelectedExisting = "tired"
            #expect(!builder.canSaveAndExtract)

            builder.vectorBuilderSelectedExisting = "french"
            #expect(builder.canSaveAndExtract)
        }
    }

    // MARK: Bug 2 — the selected model drives a local build

    /// `ChatService.ensureSelectedLocalModelLoaded` calls `loadModel()`, which
    /// needs real weights; the DECISION is factored out engine-pure so the
    /// rule is testable without one.
    @Test func aMismatchedSelectionDecidesToLoadTheSelectedModel() {
        let decision = ChatService.localModelLoadDecision(
            selectedModelID: "mlx-community/gemma-3-4b-it-4bit",
            loadedModelID: "mlx-community/gemma-3-12b-it-8bit",
            isLoading: false,
            isGenerating: false)
        #expect(decision == .load("mlx-community/gemma-3-4b-it-4bit"))
    }

    @Test func aMatchingSelectionLoadsNothing() {
        let decision = ChatService.localModelLoadDecision(
            selectedModelID: "mlx-community/gemma-3-4b-it-4bit",
            loadedModelID: "mlx-community/gemma-3-4b-it-4bit",
            isLoading: false,
            isGenerating: false)
        #expect(decision == .proceed)
    }

    /// Nothing loaded yet is still a load, not a proceed — a build must never
    /// run forward passes through a container that is not the selection's.
    @Test func noLoadedModelDecidesToLoad() {
        let decision = ChatService.localModelLoadDecision(
            selectedModelID: "Qwen/Qwen3-4B-MLX-4bit",
            loadedModelID: nil,
            isLoading: false,
            isGenerating: false)
        #expect(decision == .load("Qwen/Qwen3-4B-MLX-4bit"))
    }

    /// A model swap tears down the session and transcript, so an in-flight
    /// generation or load is refused rather than raced.
    @Test func anInFlightGenerationOrLoadRefusesInsteadOfRacing() {
        let generating = ChatService.localModelLoadDecision(
            selectedModelID: "a", loadedModelID: "b",
            isLoading: false, isGenerating: true)
        guard case .refuse(let generatingReason) = generating else {
            Issue.record("expected a refusal while generating, got \(generating)")
            return
        }
        #expect(generatingReason.contains("generation"))

        let loading = ChatService.localModelLoadDecision(
            selectedModelID: "a", loadedModelID: "b",
            isLoading: true, isGenerating: false)
        guard case .refuse(let loadingReason) = loading else {
            Issue.record("expected a refusal while loading, got \(loading)")
            return
        }
        #expect(loadingReason.contains("load"))

        // The already-correct model never refuses: no swap is needed, so a
        // running chat is irrelevant.
        #expect(
            ChatService.localModelLoadDecision(
                selectedModelID: "a", loadedModelID: "a",
                isLoading: false, isGenerating: true) == .proceed)
    }
}
