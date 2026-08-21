import Foundation

/// Pure selection rules for the agent (variant) editor's model-dependent
/// pickers — adapter and neutral basis — extracted from the view so the
/// no-resurrection contract is unit-testable.
///
/// The contract (live-testing incident, 2026-07-07): the editor used to run
/// `adapterID = compatibleAdapters.first?.id` on EVERY base-model change,
/// including the change fired programmatically while loading a saved artifact
/// into the editor. That silently re-selected an adapter the user had
/// removed; saving wrote it back; multi-agent runs then failed on the missing
/// server adapter. The rules:
///
/// - **Loading an artifact preserves exactly what the artifact says.** No
///   adapter recorded means the selection is nil — never auto-picked back.
/// - **A user-initiated base-model change clears the dependent selection to
///   nil.** The old adapter/basis belongs to the old model; the editor must
///   never guess a "compatible" replacement the user did not pick.
/// - **An explicit user pick sticks** until the user changes it or changes
///   the base model.
public enum AgentEditorSelection {

    /// Why the base-model value changed.
    public enum ModelChangeOrigin: Sendable, Equatable {
        /// The editor was populated programmatically from a saved artifact
        /// (or a draft captured from the live steering controls).
        case artifactLoad
        /// The user changed the base model in the picker.
        case userPick
    }

    /// Distinguishes a programmatic base-model write (artifact load) from a
    /// user pick when the change is observed asynchronously (SwiftUI
    /// `onChange` fires after the state write, with no origin attached).
    ///
    /// Usage: call `expectProgrammaticChange(to:from:)` immediately before
    /// the programmatic write; in the change observer call
    /// `classify(observed:)`. A load that does not actually change the value
    /// fires no change event, so `expectProgrammaticChange` records nothing
    /// for it (no stale token to misclassify a later user pick).
    public struct ChangeClassifier: Sendable, Equatable {
        private var expectedProgrammaticValue: String?

        public init() {}

        /// Arm the classifier for a programmatic write of `newValue`.
        /// No-op when `newValue == currentValue` — no change event will fire.
        public mutating func expectProgrammaticChange(
            to newValue: String, from currentValue: String
        ) {
            expectedProgrammaticValue = newValue == currentValue ? nil : newValue
        }

        /// Classify one observed change event; always consumes the token.
        public mutating func classify(observed newValue: String) -> ModelChangeOrigin {
            defer { expectedProgrammaticValue = nil }
            return newValue == expectedProgrammaticValue ? .artifactLoad : .userPick
        }
    }

    /// The dependent selection (adapter id / neutral-basis id) after a
    /// base-model change: an artifact load preserves the value the load just
    /// set (including nil — a removed adapter must never resurrect); a user
    /// pick clears it (never auto-pick a compatible replacement).
    public static func dependentSelection(
        after origin: ModelChangeOrigin, current: String?
    ) -> String? {
        switch origin {
        case .artifactLoad: return current
        case .userPick: return nil
        }
    }
}
