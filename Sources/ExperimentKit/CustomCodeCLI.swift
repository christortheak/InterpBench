import Foundation

/// The Mac command line's custom-code notice and acknowledgement verb.
/// The owner is `CustomCodeNotice`; this adapts it to `pack apply`,
/// `experiment attach-agent`, and `experiment acknowledge-custom-code`.
enum CustomCodeCLI {
    static let client = "steerlab-cli"

    /// Add the `customCode` block to a payload and print the notice, for a
    /// surface that has already written the study. A damaged acknowledgement
    /// record does not turn that success into a failure: every provider is
    /// shown as unacknowledged and the problem rides along as `recordProblem`.
    static func attachNotice(
        to payload: inout [String: JSONValue], study: String, workspaceRoot: URL,
        sink: ExperimentCLISink
    ) {
        let review: CustomCodeNotice.Review?
        var problem: String?
        do {
            review = try CustomCodeNotice.review(study: study, workspaceRoot: workspaceRoot)
        } catch {
            problem = "\(error)"
            let url = ExperimentRepository(workspaceRoot: workspaceRoot).manifestURL(study)
            let document = (try? JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: url))) ?? .null
            let found = CustomCodeNotice.providers(in: document, workspaceRoot: workspaceRoot)
            review = found.isEmpty ? nil : CustomCodeNotice.Review(study: study, providers: found.map {
                .init(sha256: $0.sha256, policyNames: $0.policyNames, sourceText: $0.sourceText,
                      acknowledgedAt: nil, acknowledgedBy: nil)
            })
        }
        guard let review else { return }
        var block = review.block
        if let problem, case .object(var object) = block {
            object["recordProblem"] = .string(problem)
            block = .object(object)
        }
        payload["customCode"] = block
        for line in review.lines { sink.out(line) }
    }

    static func acknowledge(
        _ invocation: ExperimentCLIInvocation, workspaceRoot: URL, sink: ExperimentCLISink
    ) throws -> ExperimentCLIResult {
        let args = invocation.args
        var positionals: [String] = []
        var hashes: [String] = []
        var index = 1
        while index < args.count {
            if args[index] == "--sha256", index + 1 < args.count {
                hashes.append(args[index + 1])
                index += 2
            } else {
                positionals.append(args[index])
                index += 1
            }
        }
        guard positionals.count == 1 else {
            throw ExperimentError.malformed(
                "experiment acknowledge-custom-code needs one study name.",
                repair: "steerlab-cli experiment acknowledge-custom-code <name> [--sha256 <hash>]…")
        }
        let name = positionals[0]
        // Standard study resolution, so a mistyped name answers not-found.
        _ = try ExperimentStore.load(name: name)
        let url = ExperimentRepository(workspaceRoot: workspaceRoot).manifestURL(name)
        let document = try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: url))
        guard !hashes.isEmpty else {
            let review = try CustomCodeNotice.review(of: document, study: name, workspaceRoot: workspaceRoot)
            let rows = review?.providers ?? []
            var payload: [String: JSONValue] = [
                "study": .string(name),
                "providers": .array(rows.map { row in
                    guard case .object(var object) = row.json else { return row.json }
                    object["sourceText"] = .string(row.sourceText)
                    return .object(object)
                }),
                "recordFile": .string(CustomCodeNotice.fileName),
            ]
            let message: String
            if rows.isEmpty {
                message = "'\(name)' carries no custom code; nothing to acknowledge."
            } else if let review, review.needsAcknowledgement {
                payload["notice"] = .string(CustomCodeNotice.notice)
                payload["acknowledgeCommand"] = .string(review.acknowledgeCommand)
                message = "'\(name)' carries \(rows.count) piece(s) of custom code; "
                    + "\(review.pending.count) not yet acknowledged."
            } else {
                message = "'\(name)' carries \(rows.count) piece(s) of custom code, all acknowledged."
            }
            sink.out(message)
            for row in rows {
                let state = row.acknowledged
                    ? "acknowledged \(row.acknowledgedAt ?? "") by \(row.acknowledgedBy ?? "")"
                    : "not acknowledged"
                let named = row.policyNames.isEmpty ? "unnamed" : row.policyNames.joined(separator: ", ")
                sink.out("--- custom code SHA-256 \(row.sha256) (policy: \(named); \(state))")
                sink.out(row.sourceText)
            }
            if let review, review.needsAcknowledgement {
                sink.out(CustomCodeNotice.notice)
                sink.out("If you trust the source: \(review.acknowledgeCommand)")
            }
            return ExperimentCLIResult(message: message, payload: payload)
        }
        let result = try CustomCodeNotice.acknowledge(
            hashes, in: document, study: name, workspaceRoot: workspaceRoot, client: client)
        let review = try CustomCodeNotice.review(of: document, study: name, workspaceRoot: workspaceRoot)
        let message = result.added.isEmpty
            ? "every named piece of custom code in '\(name)' was already acknowledged"
            : "recorded \(result.added.count) acknowledgement(s) for '\(name)' in \(CustomCodeNotice.fileName)"
        sink.out(message)
        return ExperimentCLIResult(
            message: message, changed: !result.added.isEmpty,
            payload: [
                "study": .string(name),
                "acknowledged": .array(result.added),
                "alreadyAcknowledged": .array(result.alreadyAcknowledged.map { .string($0) }),
                "recordFile": .string(CustomCodeNotice.fileName),
                "providers": .array((review?.providers ?? []).map(\.json)),
            ])
    }
}
