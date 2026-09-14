import Foundation

/// What the app's "Run scientific diagnostic" form offers instead of blank
/// text fields (researcher request 2026-09-13): the batteries this workspace
/// holds, the workspace-relative spelling of a chosen file, and the reading
/// of the agent lines that decides whether a model pin is required.
///
/// The RULES live here, unit-tested, so the sheet only lays them out. The
/// agent grammar mirrors the engine's `battery_run.parse_agent` exactly —
/// `<name>=<reference>` splits at the first `=`, `baseline` is the base
/// model, `<concept>:<layer>:<alpha>` is a condition on it, and anything
/// else is a variant artifact that brings its own base model.
public enum ScientificDiagnosticInputs {

    /// Workspace-relative paths (`prompts/batteries/<name>.jsonl`) of every
    /// battery in the conventional directory, sorted by name. The same
    /// relative path is what the server resolves inside ITS workspace, so a
    /// seed battery lists here and exists there; a battery authored only on
    /// this Mac lists here and must be staged before the server can read it.
    public static func batteryFiles(root: URL) -> [String] {
        TaskPromptsStore.jsonlFiles(in: VectorCatalog.batteriesDirectory(root: root))
            .map { VectorCatalog.batteriesRelativeDirectory + "/" + $0.lastPathComponent }
    }

    /// A chosen file's path relative to `root`, or nil when it lies outside
    /// the workspace — an absolute Mac path can never mean anything to a
    /// server, so the caller refuses rather than sending it. Symlinked roots
    /// (`/var` → `/private/var`, where temporary workspaces live) are matched
    /// on both spellings.
    public static func workspaceRelativePath(_ url: URL, root: URL) -> String? {
        func tail(_ path: String, under rootPath: String) -> String? {
            let prefix = rootPath.hasSuffix("/") ? rootPath : rootPath + "/"
            guard path.hasPrefix(prefix), path.count > prefix.count else { return nil }
            return String(path.dropFirst(prefix.count))
        }
        let standardized = url.standardizedFileURL
        let candidates = [standardized, standardized.resolvingSymlinksInPath()]
        let standardizedRoot = root.standardizedFileURL
        let roots = [standardizedRoot, standardizedRoot.resolvingSymlinksInPath()]
        for candidate in candidates {
            for candidateRoot in roots {
                if let relative = tail(candidate.path, under: candidateRoot.path) {
                    return relative
                }
            }
        }
        return nil
    }

    /// The agent references in a free-text box, one per line: trimmed, with
    /// blank lines dropped. A trailing newline used to reach the server as
    /// an empty agent and be refused as "a nonempty agents array".
    public static func agentLines(_ text: String) -> [String] {
        text.split(omittingEmptySubsequences: true, whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }

    public enum AgentKind: Equatable, Sendable {
        /// The unmodified base model — a dose of nothing ON the pinned model.
        case baseline
        /// `<concept>:<layer>:<alpha>` — a steering condition ON the pinned model.
        case condition
        /// A variant artifact reference; it carries its own base model.
        case artifact
    }

    /// One line's kind, by the engine's grammar. A condition needs a numeric
    /// layer and alpha; a malformed one is reported as an artifact here and
    /// refused by name on the server, which is the honest reading (the
    /// server is the authority, this only decides which caption to show).
    public static func agentKind(_ line: String) -> AgentKind {
        let reference = splitName(line).reference
        if reference == "baseline" { return .baseline }
        let parts = reference.split(separator: ":", omittingEmptySubsequences: false)
        if parts.count == 3,
            !parts[0].trimmingCharacters(in: .whitespaces).isEmpty,
            Int(parts[1].trimmingCharacters(in: .whitespaces)) != nil,
            Double(parts[2].trimmingCharacters(in: .whitespaces)) != nil
        {
            return .condition
        }
        return .artifact
    }

    /// The agents that are a dose ON a model of the form's choosing — the
    /// ones for which the engine requires `--model` and refuses a guessed
    /// pin. Empty means every listed agent brings its own base model.
    public static func agentsNeedingModel(_ lines: [String]) -> [String] {
        lines.filter { agentKind($0) != .artifact }
    }

    /// The caption under the model picker, worded for the agents typed so
    /// far: the engine's rule, before the engine has to state it as a refusal.
    public static func modelCaption(agents: [String]) -> String {
        if agents.isEmpty {
            return "list at least one agent below — a baseline or a "
                + "concept:layer:alpha condition runs on the model chosen here"
        }
        let needing = agentsNeedingModel(agents)
        if needing.isEmpty {
            return "leave blank: every agent listed brings its own base model, "
                + "and a differing choice here is refused rather than overriding it"
        }
        return "required for \(needing.joined(separator: ", ")) — a baseline "
            + "or condition is a dose on this model, and a reading whose model "
            + "was guessed is not a pin"
    }

    /// `<name>=<reference>` at the FIRST `=`, the engine's rule; an unnamed
    /// line keeps its whole text as the reference.
    static func splitName(_ raw: String) -> (name: String?, reference: String) {
        let trimmed = raw.trimmingCharacters(in: .whitespaces)
        guard let separator = trimmed.firstIndex(of: "=") else { return (nil, trimmed) }
        let name = trimmed[..<separator].trimmingCharacters(in: .whitespaces)
        let reference = trimmed[trimmed.index(after: separator)...]
            .trimmingCharacters(in: .whitespaces)
        guard !name.isEmpty, !reference.isEmpty else { return (nil, trimmed) }
        return (name, reference)
    }
}
