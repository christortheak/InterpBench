import Testing

@testable import ExperimentKit

/// The agent editor's no-resurrection contract (pure rules, no UI): loading
/// an artifact preserves exactly what the artifact says (nil adapter stays
/// nil), a user-initiated base-model change clears dependent selections, and
/// an explicit pick sticks. Regression tests for the live-testing incident
/// where `adapterID = compatibleAdapters.first?.id` on every model change
/// silently re-selected an adapter the user had removed.
struct AgentEditorSelectionTests {

    // MARK: dependentSelection

    @Test func artifactLoadPreservesNil() {
        // Artifact records no adapter → the editor shows None, and the
        // change event fired by the programmatic model write must not
        // auto-pick one back.
        #expect(
            AgentEditorSelection.dependentSelection(
                after: .artifactLoad, current: nil) == nil)
    }

    @Test func artifactLoadPreservesRecordedSelection() {
        #expect(
            AgentEditorSelection.dependentSelection(
                after: .artifactLoad, current: "adapter-7") == "adapter-7")
    }

    @Test func userModelChangeClearsSelection() {
        // Never auto-pick a "compatible" replacement on a model change.
        #expect(
            AgentEditorSelection.dependentSelection(
                after: .userPick, current: "adapter-7") == nil)
        #expect(
            AgentEditorSelection.dependentSelection(
                after: .userPick, current: nil) == nil)
    }

    // MARK: ChangeClassifier

    @Test func programmaticLoadClassifiesAsArtifactLoad() {
        var classifier = AgentEditorSelection.ChangeClassifier()
        classifier.expectProgrammaticChange(to: "org/model-b", from: "org/model-a")
        #expect(classifier.classify(observed: "org/model-b") == .artifactLoad)
    }

    @Test func unexpectedChangeClassifiesAsUserPick() {
        var classifier = AgentEditorSelection.ChangeClassifier()
        #expect(classifier.classify(observed: "org/model-b") == .userPick)
    }

    @Test func tokenIsConsumedAfterOneEvent() {
        // The load fires exactly one change event; the NEXT change (even to
        // the same value via a different model and back) is the user's.
        var classifier = AgentEditorSelection.ChangeClassifier()
        classifier.expectProgrammaticChange(to: "org/model-b", from: "org/model-a")
        #expect(classifier.classify(observed: "org/model-b") == .artifactLoad)
        #expect(classifier.classify(observed: "org/model-b") == .userPick)
    }

    @Test func loadWithSameModelArmsNothing() {
        // Same-value programmatic write fires no change event — no stale
        // token may survive to misclassify a later user pick of that model.
        var classifier = AgentEditorSelection.ChangeClassifier()
        classifier.expectProgrammaticChange(to: "org/model-a", from: "org/model-a")
        #expect(classifier.classify(observed: "org/model-b") == .userPick)
    }

    @Test func userChangeToDifferentModelThanExpectedIsUserPick() {
        // Race-shaped: load armed for model-b, but the observed change is the
        // user picking model-c — that is a user pick and must clear.
        var classifier = AgentEditorSelection.ChangeClassifier()
        classifier.expectProgrammaticChange(to: "org/model-b", from: "org/model-a")
        #expect(classifier.classify(observed: "org/model-c") == .userPick)
    }

    // MARK: end-to-end shaped ("explicit pick sticks")

    @Test func explicitPickSurvivesUntilModelChanges() {
        // The user picks an adapter (no model change event fires for that),
        // then changes the model — only then does the selection clear.
        var classifier = AgentEditorSelection.ChangeClassifier()
        var adapterID: String? = "adapter-7"  // explicit user pick
        // No change event between the pick and the model change: sticks.
        let origin = classifier.classify(observed: "org/model-b")
        adapterID = AgentEditorSelection.dependentSelection(
            after: origin, current: adapterID)
        #expect(adapterID == nil)
    }
}
