import Foundation

/// `results check-choices`: a run's stored choices read again with the
/// current choice parser, reported per condition, with nothing written.
///
/// The check is written once, in the Python client
/// (`steerlab_server.client.results_commands.check_choices`, over
/// `steerlab_server.experiment.choice_reread`), exactly as `results export`
/// is. This type only carries the request across the local bridge and reads
/// the answer back, so the Mac command line and the app report what the
/// Python client reports. The Python parser and this build's
/// `Judicial.parseChoice` are held to one fixture, so the answer is the
/// reading `analyze --reread-choices` would use here.
public enum ChoiceCheck {

    /// The bridge action. Python twin: `results_commands.CHECK_CHOICES_BRIDGE_ACTION`.
    public static let action = "results-check-choices"

    public static let purpose =
        "Read a run's stored choices again with the current choice parser and "
        + "report, per condition, how many would read differently: changed to "
        + "another option, changed to no answer, or gained an answer, with a "
        + "few records of each kind to look at. Uses the newest completed run "
        + "unless --run names another. Writes nothing and changes nothing under "
        + "runs/; to analyze with the current reading, analyze the run with "
        + "--reread-choices."

    /// The answer, as the Python client gave it.
    public struct Report: Sendable, Equatable {
        /// The run that was checked, relative to the workspace.
        public let run: String
        public let parserVersion: String
        public let total: ChoiceReread.Counts
        public let conditions: [String: ChoiceReread.Counts]
        /// A few records of each kind of change, by kind's raw name.
        public var examples: [String: [ChoiceReread.Example]] = [:]
        /// The Python client's whole answer, for the command line's envelope.
        public let response: [String: JSONValue]
    }

    /// The request as it crosses the bridge.
    public static func payload(
        workspaceRoot: URL, study: String, run: String? = nil, client: ResultsExport.Client
    ) -> [String: JSONValue] {
        var payload: [String: JSONValue] = [
            "workspaceRoot": .string(workspaceRoot.path),
            "study": .string(study),
            "client": .string(client.rawValue),
        ]
        if let run, !run.isEmpty { payload["run"] = .string(run) }
        return payload
    }

    /// Run the check. Throws `ResultsExport.Refusal` when the Python client
    /// declined, and the bridge's own error when it cannot run at all.
    public static func check(
        workspaceRoot: URL, study: String, run: String? = nil,
        client: ResultsExport.Client, python: URL? = nil, source: URL? = nil
    ) async throws -> Report {
        let answer = try await DiagnosticWorkspace.perform(
            action,
            payload: payload(workspaceRoot: workspaceRoot, study: study, run: run, client: client),
            python: python, source: source)
        return try report(from: answer)
    }

    /// Read the Python client's answer: a refusal arrives as data, in the
    /// shape `results export` uses.
    static func report(from answer: JSONValue) throws -> Report {
        guard case .object(let fields) = answer else {
            throw ExperimentError(reason: "The choice check returned no answer.")
        }
        if case .object(let refusal) = fields["refusal"] {
            throw ResultsExport.Refusal(
                code: string(refusal["code"]) ?? "refused",
                reason: string(refusal["reason"]) ?? "The choice check was refused.",
                repairAction: string(refusal["repairAction"]) ?? "",
                state: string(refusal["state"]).flatMap(SteerLabCLIState.init(rawValue:))
                    ?? .refused)
        }
        guard let run = string(fields["run"]),
            let version = string(fields["parserVersion"]),
            let total = counts(fields["total"])
        else {
            throw ExperimentError(reason: "The choice check answered without its counts.")
        }
        var conditions: [String: ChoiceReread.Counts] = [:]
        if case .object(let blocks) = fields["conditions"] {
            for (name, block) in blocks {
                if let parsed = counts(block) { conditions[name] = parsed }
            }
        }
        var examples: [String: [ChoiceReread.Example]] = [:]
        if case .object(let kinds) = fields["examples"] {
            for (kind, value) in kinds {
                guard case .array(let entries) = value else { continue }
                examples[kind] = entries.compactMap(example)
            }
        }
        return Report(
            run: run, parserVersion: version, total: total, conditions: conditions,
            examples: examples, response: fields)
    }

    private static func example(_ value: JSONValue) -> ChoiceReread.Example? {
        guard case .object(let fields) = value, case .number(let line)? = fields["line"]
        else { return nil }
        var sampleIndex: Int?
        if case .number(let index)? = fields["sampleIndex"] { sampleIndex = Int(index) }
        return ChoiceReread.Example(
            line: Int(line), condition: string(fields["condition"]) ?? "",
            promptID: string(fields["promptID"]) ?? "", sampleIndex: sampleIndex,
            stored: string(fields["stored"]), reread: string(fields["reread"]))
    }

    private static func counts(_ value: JSONValue?) -> ChoiceReread.Counts? {
        guard case .object(let fields) = value else { return nil }
        func number(_ key: String) -> Int {
            if case .number(let count) = fields[key] { return Int(count) }
            return 0
        }
        var counts = ChoiceReread.Counts()
        counts.storedChoices = number("storedChoices")
        counts.unchanged = number("unchanged")
        counts.changedToOtherOption = number("changedToOtherOption")
        counts.changedToNoAnswer = number("changedToNoAnswer")
        counts.gainedAnswer = number("gainedAnswer")
        counts.changed = number("changed")
        counts.noFinishReason = number("noFinishReason")
        counts.noFinishReasonChanged = number("noFinishReasonChanged")
        return counts
    }

    private static func string(_ value: JSONValue?) -> String? {
        if case .string(let text) = value { return text }
        return nil
    }

    /// What a person at a terminal reads after a check.
    static func summaryLines(study: String, report: Report) -> [String] {
        var lines = [
            "Checked \(study), \(report.run), with choice parser version \(report.parserVersion)"
        ]
        let blocks = [("all conditions", report.total)]
            + report.conditions.sorted { $0.key < $1.key }.map { ($0.key, $0.value) }
        for (name, counts) in blocks {
            lines.append(
                "  \(name): \(counts.storedChoices) stored, \(counts.unchanged) unchanged, "
                    + "\(counts.changedToOtherOption) to another option, "
                    + "\(counts.changedToNoAnswer) to no answer, "
                    + "\(counts.gainedAnswer) gained an answer")
            if counts.noFinishReason > 0 {
                lines.append(
                    "    \(counts.noFinishReason) from an older engine that did not record "
                        + "whether a reply was cut off (read as complete replies; "
                        + "\(counts.noFinishReasonChanged) changed)")
            }
        }
        let labels: [(ChoiceReread.Kind, String)] = [
            (.changedToOtherOption, "to another option"),
            (.changedToNoAnswer, "to no answer"),
            (.gainedAnswer, "gained an answer"),
        ]
        func said(_ value: String?) -> String { value ?? "no answer" }
        for (kind, label) in labels {
            for entry in report.examples[kind.rawValue] ?? [] {
                lines.append(
                    "  e.g. \(label): line \(entry.line), \(entry.condition), item \(entry.promptID)"
                        + (entry.sampleIndex.map { ", sample \($0)" } ?? "")
                        + ": \(said(entry.stored)) -> \(said(entry.reread))")
            }
        }
        return lines
    }

    /// One sentence for a person.
    static func message(study: String, report: Report) -> String {
        if report.total.changed == 0 {
            return "All \(report.total.storedChoices) stored choices in \(report.run) read the "
                + "same with choice parser version \(report.parserVersion). Nothing was written."
        }
        return "\(report.total.changed) of \(report.total.storedChoices) stored choices in "
            + "\(report.run) would read differently with choice parser version "
            + "\(report.parserVersion). Nothing was written. To analyze with the current "
            + "reading: steerlab-cli experiment analyze \(study) --reread-choices"
    }
}

/// `steerlab-cli results check-choices <study> [--run <run-dir>]`.
enum ChoiceCheckCLI {

    static let usage = "steerlab-cli results check-choices <study> [--run <run-dir>]"

    static func run(
        _ invocation: ExperimentCLIInvocation, sink: ExperimentCLISink,
        workspaceRoot: URL, python: URL? = nil, source: URL? = nil
    ) async throws -> ExperimentCLIResult {
        var positionals: [String] = []
        var run: String?
        var index = 1
        while index < invocation.args.count {
            let word = invocation.args[index]
            if word == "--run", index + 1 < invocation.args.count {
                run = invocation.args[index + 1]
                index += 2
            } else {
                positionals.append(word)
                index += 1
            }
        }
        guard positionals.count == 1, let study = positionals.first else {
            throw ExperimentError.malformed(
                "Name one study to check.",
                repair: usage
                    + "  (steerlab-cli experiment list shows the studies in this workspace)")
        }
        do {
            let report = try await ChoiceCheck.check(
                workspaceRoot: workspaceRoot, study: study, run: run, client: .commandLine,
                python: python, source: source)
            for line in ChoiceCheck.summaryLines(study: study, report: report) { sink.out(line) }
            return ExperimentCLIResult(
                message: ChoiceCheck.message(study: study, report: report), changed: false,
                payload: report.response)
        } catch let refusal as ResultsExport.Refusal {
            sink.err(
                "steerlab-cli results check-choices: \(refusal.reason)\n"
                    + (refusal.repairAction.isEmpty ? "" : "  \(refusal.repairAction)\n"))
            throw ExperimentCLIStop(
                exitCode: 1, state: refusal.state, code: refusal.code,
                reason: refusal.reason, repairAction: refusal.repairAction)
        }
    }
}
