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

    /// The warning the SERVER robustness route raises for a local judge —
    /// hoisted here (review round 7, finding 2) so the Run gate can refuse in
    /// the route's own words instead of paraphrasing them. The route stays
    /// the only place that EMITS it during a run; this is the one place it is
    /// written.
    public static func localJudgeSkippedInServerWorkspace(model: String) -> String {
        "coherence judge '\(model)' is a local model — judging skipped in a "
            + "server workspace; pick a Claude or OpenRouter judge"
    }
}

/// Whether a single-string judge is RUNNABLE — the one precondition list, so
/// the picker's flags, a Run button's readiness gate, and the executing route
/// cannot disagree about the same selection.
///
/// The three surfaces had drifted apart (review round 7, finding 5). The
/// Studies pane's `pairedJudgeDisabledReason` predates `JudgeModelSpelling`
/// and recognised one shape only — Claude, plus a key. It therefore passed an
/// `openrouter:` spelling with no pinned provider (execution throws),
/// OpenRouter with no key (the client refuses mid-run), and a local pick this
/// Mac cannot answer with (the loader either refuses or, worse, downloads it).
/// A green Run button followed by a refusal is the readiness gate lying.
///
/// Everything here is a value: key state arrives as a PRESENCE boolean and no
/// credential is ever read, held, or rendered.
public enum JudgeReadiness {

    /// Where the generations come from — which decides whether a LOCAL judge
    /// can be reached at all. On the server route nothing is loaded on this
    /// Mac, so a local judge is skipped; both API backends work either way.
    public enum Substrate: Sendable, Equatable {
        case local
        case server
    }

    /// Is this repo's snapshot on this Mac? Injectable; the default is the
    /// ONE local installed-model test, so a picker's badge, this gate, and a
    /// load refusal cannot disagree.
    public typealias InstalledCheck = @Sendable (String) -> Bool
    public typealias CapabilityCheck = @Sendable (String) -> LocalJudgeCapability.Verdict

    public static let liveInstalled: InstalledCheck = {
        SteeredContainerLoader.isCached(modelID: $0)
    }
    public static let liveCapability: CapabilityCheck = {
        LocalJudgeCapability.verdict(forModelID: $0)
    }

    // MARK: - Sentences

    /// Why an uninstalled repo cannot judge, in the voice the capability
    /// verdicts use — a clause, so the picker can drop it into its flagged
    /// caption and the gate into its refusal.
    public static let notInstalledReason =
        "it is not installed on this Mac — install it in Models first"

    /// The picker's short label suffix for an offered-but-absent tier.
    public static func notInstalledLabel(_ model: String) -> String {
        "\(model) (not installed)"
    }

    /// Why a local judge is unreachable from a server workspace, as a clause
    /// for the picker's flag. The RUN refusal uses the route's own sentence
    /// (`JudgeModelSpelling.localJudgeSkippedInServerWorkspace`) instead.
    public static let localOnServerReason =
        "local model — not runnable against a server workspace"

    public static func notInstalledRefusal(model: String) -> String {
        "judge '\(model)' is not installed on this Mac — install it in "
            + "Models first, or pick a judge that is"
    }

    public static func incapableRefusal(model: String, reason: String) -> String {
        "judge '\(model)' cannot judge: \(reason)"
    }

    /// Worded exactly as the Studies pane has always worded it.
    public static func claudeKeyRefusal(model: String) -> String {
        "judge '\(model)' needs a Claude API key — set ANTHROPIC_API_KEY or "
            + "save a key in the Compute section (stored in the macOS Keychain)"
    }

    /// Worded like the panel-judge twin beside it ("needs an external judge
    /// key — save one in the Compute section or set OPENROUTER_API_KEY").
    public static func openRouterKeyRefusal(model: String) -> String {
        "judge '\(model)' needs an external judge key — save one in the "
            + "Compute section or set OPENROUTER_API_KEY"
    }

    // MARK: - The gate

    /// Why this selection cannot be run, or nil when it can. A blank
    /// selection is nil: "no judge asked for" is a legal state everywhere,
    /// and the panes that DEFAULT a blank to the Claude judge pass that
    /// default in themselves.
    ///
    /// The order is the order execution hits them, so the first thing a
    /// researcher is told is the first thing that would have failed.
    public static func refusal(
        for raw: String?,
        substrate: Substrate = .local,
        claudeKeyPresent: Bool,
        openRouterKeyPresent: Bool,
        installed: InstalledCheck = liveInstalled,
        capability: CapabilityCheck = liveCapability
    ) -> String? {
        guard let selection = JudgeModelSpelling.parse(raw) else { return nil }
        switch selection {
        case .openRouterUnpinned(let model):
            // What `ExperimentTasks.resolvedJudges` throws, before anything
            // runs.
            return JudgeModelSpelling.unpinnedProviderRefusal(model: model)
        case .openRouter(let model, _):
            return openRouterKeyPresent ? nil : openRouterKeyRefusal(model: model)
        case .claude(let model):
            return claudeKeyPresent ? nil : claudeKeyRefusal(model: model)
        case .local(let model):
            if substrate == .server {
                return JudgeModelSpelling.localJudgeSkippedInServerWorkspace(
                    model: model)
            }
            // Installed BEFORE capable: `LocalJudgeCapability` answers "no
            // cached snapshot" for an absent repo, which is true but tells a
            // researcher looking at a listed model tier nothing about what to
            // do. The install test is also what stops a Run from becoming an
            // invisible multi-gigabyte download inside the loader.
            guard installed(model) else { return notInstalledRefusal(model: model) }
            let verdict = capability(model)
            guard verdict.isCapable else {
                return incapableRefusal(
                    model: model, reason: verdict.reason ?? "it is not a text model")
            }
            return nil
        }
    }
}
