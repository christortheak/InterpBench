import Foundation
import SteeringKit

/// What a single-string judge picker OFFERS — composed once, rendered by both
/// panes that have one (the Robustness Check's judge, and the ad-hoc judge
/// selector in the study pane). Pure: every input is a value, so the strings a
/// researcher reads are asserted in tests rather than inferred from a
/// screenshot.
///
/// Three rules the composition exists to hold:
///
/// - **A capability filter, not a narrower scan.** Candidates coming from the
///   local cache scan are checked with `LocalJudgeCapability`; curated entries
///   (the app's own model tiers, `claude-…`/`anthropic:…`, an OpenRouter
///   spelling) are not cache artifacts and are never capability-inspected —
///   there may be nothing on disk to inspect.
/// - **Curated is not the same as present.** A curated model tier is a
///   CANDIDATE, not an inventory (`ChatService.availableModels`): a fresh Mac
///   has none of them downloaded. Passing every tier "by construction" made
///   them selectable, and selecting one sent `SteeredContainerLoader.load`
///   to the hub for up to 35 GB — outside the visible, cancellable Install
///   flow the model contract says is the only way weights arrive (review
///   round 7, finding 1). So a curated entry that names a LOCAL repo is
///   checked against the one installed test and, when absent, listed
///   FLAGGED — visible, because it is exactly the thing you would install,
///   and unrunnable, because it is not here yet.
/// - **A stored choice is never silently dropped.** The value the panel
///   already holds is ALWAYS listed. If it fails the filter it is listed
///   FLAGGED, carrying the engine's own reason, because a picker that quietly
///   deletes a saved selection leaves the researcher with a blank field and no
///   account of what happened to it.
///
/// Flagging, not hiding, is also how a server workspace treats local judges:
/// the server route runs generations remotely and judges from this Mac, so a
/// local pick is skipped with a warning. The picker says so in the row rather
/// than removing entries that are perfectly good on the Local workspace
/// (review round 7, finding 2).
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

    /// The two clauses that are not capability verdicts, restated from the
    /// shared precondition list so the picker and the Run gate quote the same
    /// words.
    public static let notInstalledReason = JudgeReadiness.notInstalledReason
    public static let localOnServerReason = JudgeReadiness.localOnServerReason

    // MARK: - Composition

    /// The capability seam. Injectable so tests never touch the real cache.
    public typealias CapabilityCheck = @Sendable (String) -> LocalJudgeCapability.Verdict
    /// The is-installed seam, same shape and same default as the Run gate's.
    public typealias InstalledCheck = JudgeReadiness.InstalledCheck

    public static let liveCapability: CapabilityCheck = {
        LocalJudgeCapability.verdict(forModelID: $0)
    }
    public static let liveInstalled: InstalledCheck = JudgeReadiness.liveInstalled

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
    ///   - substrate: which engine will generate. `.server` makes every local
    ///     entry a flagged row — the server route skips local judges.
    public static func compose(
        selected: String,
        candidates: [Candidate],
        openRouterKeyPresent: Bool,
        substrate: JudgeReadiness.Substrate = .local,
        capability: CapabilityCheck = liveCapability,
        installed: InstalledCheck = liveInstalled
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

        /// Why a LOCAL id cannot judge here, or nil when it can. The order is
        /// the Run gate's: the workspace first (a local judge is unreachable
        /// from a server route whatever its snapshot says), then presence,
        /// then — for ids with something on disk to inspect — capability.
        func localRefusal(_ model: String, capabilityChecked: Bool) -> String? {
            if substrate == .server { return localOnServerReason }
            guard installed(model) else { return notInstalledReason }
            guard capabilityChecked else { return nil }
            let verdict = capability(model)
            return verdict.isCapable
                ? nil : (verdict.reason ?? "it is not a text model")
        }

        // 1. The stored selection, first and unconditionally. It is always
        //    inspected as fully as a cache id, whatever list it came from:
        //    what the panel HOLDS is what a run would use.
        let selection = selected.trimmingCharacters(in: .whitespacesAndNewlines)
        if !selection.isEmpty {
            switch JudgeModelSpelling.parse(selection) {
            case .local(let model):
                if let reason = localRefusal(model, capabilityChecked: true) {
                    offers.selectionCaption = flaggedSelectionCaption(
                        model: model, reason: reason)
                    place(
                        Option(
                            id: selection,
                            label: reason == notInstalledReason
                                ? JudgeReadiness.notInstalledLabel(model)
                                : flaggedLabel(model),
                            caption: reason))
                } else {
                    place(Option(id: selection, label: selection))
                }
            default:
                // Claude and OpenRouter selections pass by construction —
                // there is no snapshot on this Mac to inspect.
                place(Option(id: selection, label: selection))
            }
        }

        // 2. The rest, in the caller's order.
        //
        //    Dropped vs flagged, and why they differ: a cache-scan id that
        //    fails the CAPABILITY test is dropped, because the scan is broad
        //    on purpose and its rejects are artifacts nobody chose to offer
        //    (a dictionary repo, a lens repo). Everything else that cannot
        //    judge is flagged and stays listed — a curated tier that is not
        //    installed is precisely the thing you would install, and in a
        //    server workspace every local entry is unreachable at once, which
        //    is a fact about the workspace, not about the models.
        for candidate in candidates {
            let trimmed = candidate.id.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            guard case .local(let model) = JudgeModelSpelling.parse(trimmed) else {
                place(Option(id: trimmed, label: trimmed))
                continue
            }
            // A cache-scan reject is dropped wherever the run would happen:
            // a dictionary repo is not a judge on any workspace, so this test
            // comes before the workspace one.
            if candidate.capabilityChecked, !capability(model).isCapable { continue }
            guard
                let reason = localRefusal(
                    model, capabilityChecked: candidate.capabilityChecked)
            else {
                place(Option(id: trimmed, label: trimmed))
                continue
            }
            place(
                Option(
                    id: trimmed,
                    label: reason == notInstalledReason
                        ? JudgeReadiness.notInstalledLabel(model)
                        : flaggedLabel(model),
                    caption: reason))
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
