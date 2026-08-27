import Foundation
import SteeringKit

/// How a SINGLE-STRING judge selector names its backend.
///
/// Two places in the app pin a judge as one plain string rather than as a
/// `JudgeRef` record — the Robustness Check's judge, and the ad-hoc judge the
/// engine synthesizes when a study pins no panel. Both used to be routed by
/// `ClaudePairedJudge.isClaudeModel` alone, a two-way split (claude / local)
/// that left the engine's third judge backend unreachable from either.
///
/// The spelling is NOT a new vocabulary. It is the study path's judge spec
/// (`ExperimentCLIRunner.parseJudges`, `<name>:<kind>[:<model>[:<provider>]]`)
/// with the name field dropped, because a single-string selector has no name
/// to carry: the same `openrouter` kind token, the same colon separator, the
/// same field ORDER, and the same rule that the provider field is
/// OpenRouter's alone. It also extends the `anthropic:<model>` prefix
/// `ClaudePairedJudge` already understands, rather than competing with it.
///
///     openrouter:<model>:<provider>   e.g. openrouter:vendor/model:together
///     anthropic:<model>  /  claude-…  the Claude API
///     <anything else>                 a locally cached model
///
/// A model slug contains `/` but never `:`, so splitting on `:` is safe.
public enum JudgeModelSpelling {

    /// The kind token, spelled exactly as `ExperimentStore.knownJudgeKinds`
    /// and the manifest's `JudgeRef.kind` spell it.
    public static let openRouterKind = "openrouter"
    public static let openRouterPrefix = openRouterKind + ":"

    public enum Selection: Sendable, Equatable {
        case local(model: String)
        case claude(model: String)
        case openRouter(model: String, provider: String)
        /// An `openrouter:` spelling that names no serving provider. Kept as
        /// its own case rather than folded into `local`: an unpinned provider
        /// is not a pinned judge, and the caller must say so instead of
        /// quietly loading the slug as a local model.
        case openRouterUnpinned(model: String)

        /// The manifest's kind vocabulary — what a report stamps.
        public var kind: String {
            switch self {
            case .local: "local"
            case .claude: "claude"
            case .openRouter, .openRouterUnpinned: JudgeModelSpelling.openRouterKind
            }
        }

        /// The MODEL, without the kind or provider fields.
        public var model: String {
            switch self {
            case .local(let model), .claude(let model),
                .openRouterUnpinned(let model):
                model
            case .openRouter(let model, _): model
            }
        }

        public var provider: String? {
            if case .openRouter(_, let provider) = self { return provider }
            return nil
        }
    }

    /// nil for a blank selection (no judge asked for). Never throws: a
    /// malformed OpenRouter spelling becomes `openRouterUnpinned`, which the
    /// routes refuse in the engine's own voice.
    public static func parse(_ raw: String?) -> Selection? {
        let trimmed = (raw ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        if trimmed.lowercased().hasPrefix(openRouterPrefix) {
            let fields = trimmed.dropFirst(openRouterPrefix.count)
                .split(separator: ":", omittingEmptySubsequences: false)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            let model = fields.first ?? ""
            let provider = fields.count > 1 ? fields[1] : ""
            guard !model.isEmpty, !provider.isEmpty else {
                return .openRouterUnpinned(model: model)
            }
            return .openRouter(model: model, provider: provider)
        }
        // The historical two-way split, unchanged: `claude-…`/`anthropic:…`
        // is the Claude API, anything else is a local repo id.
        return ClaudePairedJudge.isClaudeModel(trimmed)
            ? .claude(model: trimmed) : .local(model: trimmed)
    }

    /// The canonical spelling for an OpenRouter judge — the ONE writer, so a
    /// picker and a parser cannot drift.
    public static func spellOpenRouter(model: String, provider: String) -> String {
        let model = model.trimmingCharacters(in: .whitespacesAndNewlines)
        let provider = provider.trimmingCharacters(in: .whitespacesAndNewlines)
        return provider.isEmpty
            ? openRouterPrefix + model
            : openRouterPrefix + model + ":" + provider
    }

    /// The refusal an unpinned OpenRouter selection earns, worded like the
    /// study path's (`SweepObjectives.sweepJudgeRoute`,
    /// `OpenRouterCatalog.preflight`) so one mistake reads the same
    /// everywhere.
    public static func unpinnedProviderRefusal(model: String) -> String {
        "openrouter judge for '\(model)' has no pinned provider — an "
            + "unpinned provider is not a pinned judge"
    }
}
