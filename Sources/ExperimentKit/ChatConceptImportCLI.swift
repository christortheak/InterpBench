import Foundation

/// `steerlab-cli concept import-chat` — the `concept` family's one verb on
/// this client. The Python client spells it identically (`steerlab concept
/// import-chat`); both read the flags here and leave every rule, sentence, and
/// byte to their twin of `ChatConceptImport`.
enum ChatConceptImportCLI {

    static let usage =
        "steerlab-cli concept import-chat <name> --positive <chat.jsonl> "
        + "--negative <chat.jsonl> [--validation-pairs <n> --seed <n>]"

    static func run(
        _ invocation: ExperimentCLIInvocation, workspaceRoot: URL, now: Date,
        sink: ExperimentCLISink
    ) throws -> ExperimentCLIResult {
        let args = invocation.args
        guard args.first == "import-chat" else {
            throw ExperimentError(reason: "verbs: import-chat")
        }
        var positionals: [String] = []
        var values: [String: String] = [:]
        var index = 1
        while index < args.count {
            let token = args[index]
            if token.hasPrefix("--"), index + 1 < args.count {
                values[token] = args[index + 1]
                index += 2
            } else {
                positionals.append(token)
                index += 1
            }
        }
        guard positionals.count == 1, let positive = values["--positive"],
            let negative = values["--negative"]
        else {
            throw ChatConceptImport.Refusal(
                code: "usage", reason: "usage: \(usage)", repairAction: usage,
                state: .blocked)
        }
        let validationPairs = try values["--validation-pairs"].map(
            ChatConceptImport.parseCount) ?? 0
        let seed = try values["--seed"].map(ChatConceptImport.parseSeed)
        let outcome = try ChatConceptImport.importConcept(
            name: positionals[0],
            positive: URL(fileURLWithPath: positive),
            negative: URL(fileURLWithPath: negative),
            validationPairs: validationPairs, seed: seed,
            root: workspaceRoot, now: now)
        let lines = outcome.summaryLines
        for line in lines { sink.out(line) }
        return ExperimentCLIResult(
            message: lines[0], changed: true, payload: outcome.payload,
            nextAction: .init(
                verb: "experiment attach <name> \(outcome.prepared.name)",
                detail: "pin the new concept into a draft study"))
    }
}
