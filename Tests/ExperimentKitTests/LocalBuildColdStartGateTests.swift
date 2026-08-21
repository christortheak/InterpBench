import Foundation
import Testing

@testable import ExperimentKit

/// Cold-start local builds (2026-08-19).
///
/// The Concept panel's local build actions used to be gated on
/// `service.state != .ready` — "a model is loaded and idle". Once
/// `saveConceptAndExtract` / `buildReader` / `trainReadingProbe` /
/// `buildNeutralPCBasis` gained `ensureSelectedLocalModelLoaded()`, that gate
/// was obsolete AND wrong: a build loads the SELECTED model itself, so with
/// nothing resident the researcher was sent to the Playground to load a model
/// the build was about to replace anyway.
///
/// `ChatService.localBuildBlockedReason` is the replacement, and it lives at
/// the model layer for the same reason the load decision does: the rule is the
/// science-relevant part (never race the in-process model), the button is not.
/// It must agree with `localModelLoadDecision` — a state the gate admits but
/// the build then refuses would be a button that lies.
@Suite struct LocalBuildColdStartGateTests {

    private let selected = "mlx-community/gemma-3-4b-it-4bit"

    // MARK: The gate

    /// The headline: NOTHING loaded is enabled. The build's ensure step turns
    /// this into a load.
    @Test func aColdStartIsEnabled() {
        #expect(
            ChatService.localBuildBlockedReason(
                isBuilding: false, isLoading: false, isGenerating: false) == nil)
    }

    /// The ordinary warm case: the selection is resident and idle.
    @Test func readyAndMatchingIsEnabled() {
        #expect(
            ChatService.localBuildBlockedReason(
                isBuilding: false, isLoading: false, isGenerating: false) == nil)
        // ...and the load decision agrees there is nothing to do.
        #expect(
            ChatService.localModelLoadDecision(
                selectedModelID: selected, loadedModelID: selected,
                isLoading: false, isGenerating: false) == .proceed)
    }

    /// A streaming chat generation shares the in-process model, and a build
    /// would tear its session down mid-stream.
    @Test func aRunningGenerationDisables() {
        let reason = ChatService.localBuildBlockedReason(
            isBuilding: false, isLoading: false, isGenerating: true)
        #expect(reason != nil)
        #expect(reason?.contains("generation") == true)
    }

    /// A load already in flight: `loadModel()` would be re-entered against a
    /// half-built container.
    @Test func anInFlightLoadDisables() {
        let reason = ChatService.localBuildBlockedReason(
            isBuilding: false, isLoading: true, isGenerating: false)
        #expect(reason != nil)
        #expect(reason?.contains("load") == true)
    }

    /// The builder's own work is still a conflict — this is the term the old
    /// gate carried alongside `state != .ready` and the one worth keeping.
    @Test func aRunningBuildDisables() {
        let reason = ChatService.localBuildBlockedReason(
            isBuilding: true, isLoading: false, isGenerating: false)
        #expect(reason != nil)
        #expect(reason?.contains("build") == true)
    }

    /// Every blocker at once still reports exactly one reason (the caption has
    /// room for one), and reports the nearest one.
    @Test func aBuildInProgressWinsTheReason() {
        #expect(
            ChatService.localBuildBlockedReason(
                isBuilding: true, isLoading: true, isGenerating: true)
                == ChatService.localBuildBlockedReason(
                    isBuilding: true, isLoading: false, isGenerating: false))
    }

    // MARK: Agreement with the load decision

    /// The contract that makes the relaxation safe: every state the gate
    /// ENABLES is a state `localModelLoadDecision` acts on rather than
    /// refuses. Otherwise the button lights and the build immediately throws.
    @Test func everyEnabledStateIsOneTheBuildCanActOn() {
        for loaded in [nil, selected, "Qwen/Qwen3-4B-MLX-4bit"] as [String?] {
            for isLoading in [false, true] {
                for isGenerating in [false, true] {
                    let blocked = ChatService.localBuildBlockedReason(
                        isBuilding: false, isLoading: isLoading, isGenerating: isGenerating)
                    guard blocked == nil else { continue }
                    let decision = ChatService.localModelLoadDecision(
                        selectedModelID: selected, loadedModelID: loaded,
                        isLoading: isLoading, isGenerating: isGenerating)
                    if case .refuse(let reason) = decision {
                        let message: String =
                            "gate enabled (loaded=\(String(describing: loaded)), "
                            + "loading=\(isLoading), generating=\(isGenerating)) "
                            + "but the build refuses: \(reason)"
                        Issue.record(Comment(rawValue: message))
                    }
                }
            }
        }
    }

    // MARK: The caption

    /// The cold start gets its own honest wording rather than falling silent:
    /// the researcher should know a build is about to load 3+ GB of weights.
    @Test func noModelLoadedCaptionNamesTheSelection() {
        let caption = ChatService.localModelSelectionCaption(
            selectedModelID: selected, loadedModelID: nil)
        #expect(caption == "no model loaded — building will load \(selected)")
    }

    /// The mismatch caption is unchanged: it also warns about the chat reset,
    /// which a cold start cannot cause (there is no session to lose).
    @Test func aMismatchCaptionWarnsAboutTheChatReset() {
        let caption = ChatService.localModelSelectionCaption(
            selectedModelID: selected, loadedModelID: "mlx-community/gemma-3-12b-it-8bit")
        #expect(caption?.contains("mlx-community/gemma-3-12b-it-8bit is currently loaded") == true)
        #expect(caption?.contains(selected) == true)
        #expect(caption?.contains("reset the chat") == true)
    }

    /// Weights that are not on the Mac get the honest next step: the build
    /// would refuse rather than "load" them, so the caption must not promise
    /// a load it cannot perform.
    @Test func anUninstalledSelectionPointsAtTheDownload() {
        let caption = ChatService.localModelSelectionCaption(
            selectedModelID: selected, loadedModelID: nil, isInstalled: false)
        #expect(caption?.contains("not downloaded") == true)
        #expect(caption?.contains(selected) == true)
        #expect(caption?.contains("Add Model") == true)
    }

    /// Nothing to say when the selection is already resident.
    @Test func aMatchingSelectionHasNoCaption() {
        #expect(
            ChatService.localModelSelectionCaption(
                selectedModelID: selected, loadedModelID: selected) == nil)
    }
}
