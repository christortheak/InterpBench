import Foundation
import SteeringKit

/// What a single-string judge picker OFFERS — composed once, rendered by both
/// panes that have one (the Robustness Check's judge, and the ad-hoc judge
/// selector in the study pane). Pure: every input is a value, so the strings a
/// researcher reads are asserted in tests rather than inferred from a
/// screenshot.
///
/// Two rules the composition exists to hold:
///
/// - **A capability filter, not a narrower scan.** Candidates coming from the
///   local cache scan are checked with `LocalJudgeCapability`; curated entries
///   (the app's own model tiers, `claude-…`/`anthropic:…`, an OpenRouter
///   spelling) pass by construction — they are not cache artifacts and there
///   is nothing on disk to inspect.
/// - **A stored choice is never silently dropped.** The value the panel
///   already holds is ALWAYS listed. If it fails the filter it is listed
///   FLAGGED, carrying the engine's own reason, because a picker that quietly
///   deletes a saved selection leaves the researcher with a blank field and no
///   account of what happened to it.
public enum JudgeModelOffers {

    public struct Option: Sendable, Equatable, Identifiable {
        /// The value the binding stores — the tag.
        public let id: String
        /// What the row reads as.
        public let label: String
        /// nil when the option is usable; otherwise the engine's reason.
        public let caption: String?

        public var isFlagged: Bool { caption != nil }

        public init(id: String, label: String, caption: String? = nil) {
            self.id = id
            self.label = label
            self.caption = caption
        }
    }

    public struct Offers: Sendable, Equatable {
        /// Local and Claude judges, in offer order.
        public var models: [Option]
        /// OpenRouter judges. Empty unless a key is held, EXCEPT for a stored
        /// OpenRouter selection, which is listed either way.
        public var openRouter: [Option]
        /// Shown in place of the OpenRouter section when no key is held.
        public var openRouterHint: String?
        /// The caption for the current selection when it is flagged — the one
        /// line the pane puts under the picker.
        public var selectionCaption: String?

        public init(
            models: [Option] = [], openRouter: [Option] = [],
            openRouterHint: String? = nil, selectionCaption: String? = nil
        ) {
            self.models = models
            self.openRouter = openRouter
            self.openRouterHint = openRouterHint
            self.selectionCaption = selectionCaption
        }
    }

    // MARK: - Strings

    /// The bare `openrouter:` sentinel: choosing it reveals the model and
    /// provider fields, which write the full spelling back through
    /// `JudgeModelSpelling.spellOpenRouter`.
    public static let openRouterSentinel = JudgeModelSpelling.openRouterPrefix
    public static let openRouterSentinelLabel = "OpenRouter judge…"

    public static let openRouterKeyHint =
        "No OpenRouter key on this Mac — set the external judge key in "
        + "Compute (stored in the macOS Keychain), or put OPENROUTER_API_KEY "
        + "in the environment, and OpenRouter judges appear here."

    /// The flagged current selection's caption. Says what is wrong, that the
    /// row survives because it is stored, and what clears it.
    public static func flaggedSelectionCaption(
        model: String, reason: String
    ) -> String {
        "'\(model)' cannot judge: \(reason). It stays listed because it is "
            + "your stored choice — pick another judge to clear this."
    }

    /// The flagged row's own label suffix, so the list itself says it.
    public static func flaggedLabel(_ model: String) -> String {
        "\(model) (cannot judge)"
    }

    // MARK: - Composition

    /// The capability seam. Injectable so tests never touch the real cache.
    public typealias CapabilityCheck = @Sendable (String) -> LocalJudgeCapability.Verdict

    public static let liveCapability: CapabilityCheck = {
        LocalJudgeCapability.verdict(forModelID: $0)
    }

    /// One candidate id plus how it must be treated. Order is the CALLER's:
    /// cached and curated entries interleave in the list the pane already
    /// offered, so the fix changes what is listed, never the order.
    public struct Candidate: Sendable, Equatable {
        public let id: String
        /// True for ids that came from the local cache scan — there is a
        /// snapshot on disk to inspect, so inspect it. False for entries with
        /// nothing on disk to look at (the app's own model tiers, the default
        /// Claude judge, an OpenRouter spelling): those pass by construction.
        public let capabilityChecked: Bool

        public init(id: String, capabilityChecked: Bool) {
            self.id = id
            self.capabilityChecked = capabilityChecked
        }

        public static func cached(_ id: String) -> Candidate {
            Candidate(id: id, capabilityChecked: true)
        }

        public static func curated(_ id: String) -> Candidate {
            Candidate(id: id, capabilityChecked: false)
        }
    }

    /// - Parameters:
    ///   - selected: the value the panel currently holds (may be blank).
    ///   - candidates: everything the pane would list, in its own order.
    ///   - openRouterKeyPresent: PRESENCE only. The key itself is never read
    ///     here and never rendered.
    public static func compose(
        selected: String,
        candidates: [Candidate],
        openRouterKeyPresent: Bool,
        capability: CapabilityCheck = liveCapability
    ) -> Offers {
        var offers = Offers()
        var seen = Set<String>()

        func place(_ option: Option) {
            guard !option.id.isEmpty, seen.insert(option.id).inserted else { return }
            if option.id.lowercased().hasPrefix(JudgeModelSpelling.openRouterPrefix) {
                offers.openRouter.append(option)
            } else {
                offers.models.append(option)
            }
        }

        // 1. The stored selection, first and unconditionally.
        let selection = selected.trimmingCharacters(in: .whitespacesAndNewlines)
        if !selection.isEmpty {
            switch JudgeModelSpelling.parse(selection) {
            case .local(let model):
                let verdict = capability(model)
                if verdict.isCapable {
                    place(Option(id: selection, label: selection))
                } else {
                    let reason = verdict.reason ?? "it is not a text model"
                    offers.selectionCaption = flaggedSelectionCaption(
                        model: model, reason: reason)
                    place(
                        Option(
                            id: selection, label: flaggedLabel(model),
                            caption: reason))
                }
            default:
                // Claude and OpenRouter selections pass by construction —
                // there is no snapshot on this Mac to inspect.
                place(Option(id: selection, label: selection))
            }
        }

        // 2. The rest, in the caller's order: cache-scan ids filtered,
        //    curated ids listed as-is.
        for candidate in candidates {
            let trimmed = candidate.id.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            if candidate.capabilityChecked, !capability(trimmed).isCapable { continue }
            place(Option(id: trimmed, label: trimmed))
        }

        // 3. The OpenRouter section, gated on key PRESENCE alone.
        if openRouterKeyPresent {
            place(
                Option(
                    id: openRouterSentinel, label: openRouterSentinelLabel))
        } else {
            offers.openRouterHint = openRouterKeyHint
        }
        return offers
    }
}
