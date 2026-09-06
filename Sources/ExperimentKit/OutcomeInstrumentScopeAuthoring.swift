import Foundation

/// Derive a scope before publishing its study. The declaration command and
/// design instantiation use the same vocabulary, item selection and refusals.
/// An instantiation can supply the exact prompt bytes whose template pin it
/// already checked, avoiding a second read of a potentially changed input.
enum OutcomeInstrumentScopeAuthoring {
    static func apply(responseFormats: [String], into manifest: inout ExperimentManifest,
                      workspaceRoot: URL, reviewedPrompts: Data? = nil) throws {
        let known = ExperimentStore.knownResponseFormats.joined(separator: "|")
        if let unknown = responseFormats.first(where: { ResponseFormat(rawValue: $0) == nil }) {
            throw ExperimentError.malformed(
                "unknown responseFormat '\(unknown)' — known: " + ExperimentStore.knownResponseFormats.joined(separator: ", "),
                repair: "steerlab-cli experiment set-instrument-scope \(manifest.name) <\(known)>[,…]  (\"\" clears the declaration)")
        }
        // Removing a stale declaration must not require its old input file.
        guard !responseFormats.isEmpty else {
            manifest.outcomeInstrumentScope = nil
            return
        }
        guard let file = manifest.taskPromptsFile, !file.isEmpty else {
            throw ExperimentError(
                reason: "declare the task prompts first ('steerlab-cli experiment pin-prompts \(manifest.name) prompts/…/file.jsonl') — the scope pins which of THEIR rows the instrument reads")
        }
        let data = try reviewedPrompts ?? Data(contentsOf: ExperimentStore.resolveProjectPath(file, root: workspaceRoot))
        let document = try TaskPromptsDocument.load(data)
        let pin = ResponseFormat.Scope.pin(responseFormats: responseFormats, items: document.responseFormatItems)
        guard pin.itemCount > 0 else {
            throw ExperimentError.malformed(
                "the declared outcomeInstrumentScope selects zero task items of '\(file)' — the instruments would run on nothing and silently produce zero records",
                repair: "steerlab-cli experiment set-instrument-scope \(manifest.name) <\(known)>[,…]  (a format the pinned items actually declare), or \"\" to clear the declaration")
        }
        manifest.outcomeInstrumentScope = pin
    }
}
