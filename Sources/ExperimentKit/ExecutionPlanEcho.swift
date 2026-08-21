import Foundation

/// What the submit control is ABOUT to do, in one line, resolved from the
/// same inputs the submission itself uses.
///
/// Finding 11c. The Pipeline Composer declares a chain as manifest data, but
/// the unified Run button submits `panel.remoteVerb` — an independent
/// setting living inside a collapsed "Remote options" disclosure. On
/// 2026-07-26 a researcher declared `[extract, validate, sweep]`, pressed
/// Run, and got a plain `run`: the composer's selections were never
/// consulted, no refusal was possible (both states are individually legal),
/// and the mismatch was only discoverable hours later from the finished
/// job's stage list.
///
/// So the plan is echoed immediately above the control, before submission.
/// Pure and unit-tested, per the project rule that decisions live in
/// ExperimentKit and never in a view.
public enum ExecutionPlanEcho {

    public struct Plan: Sendable, Equatable {
        /// What will execute, stated positively. Always present.
        public var summary: String
        /// A declared intention this submission will NOT honor. Nil when the
        /// submission does what the visible declarations imply.
        public var mismatch: String?

        public init(summary: String, mismatch: String? = nil) {
            self.summary = summary
            self.mismatch = mismatch
        }
    }

    /// - Parameters:
    ///   - verb: the selected remote verb (ignored for local runs).
    ///   - target: where the submission goes.
    ///   - serverLabel: display name of the server, for the summary line.
    ///   - dryRun: the Remote options dry-run toggle.
    ///   - declaredPipelineStages: the manifest's declared chain, via
    ///     `ShardedSubmission.declaredPipelineStages` — nil when no pipeline
    ///     block exists.
    public static func describe(
        verb: String,
        target: SubstrateRouting.Substrate,
        serverLabel: String,
        dryRun: Bool,
        declaredPipelineStages: [String]?
    ) -> Plan {
        guard target == .server else {
            // The verb picker is a REMOTE control; saying otherwise here
            // would reproduce the confusion in mirror image.
            return Plan(
                summary: "runs the study in this app on this Mac — the "
                    + "Remote options verb applies to server submissions only")
        }

        let stages = declaredPipelineStages ?? []
        let prefix = dryRun
            ? "prepares the bundle and job script on \(serverLabel); NOTHING executes"
            : "submits to \(serverLabel)"

        if verb == "pipeline" {
            guard !stages.isEmpty else {
                return Plan(
                    summary: "\(prefix): pipeline",
                    mismatch: "no pipeline block is declared on this study — "
                        + "the server refuses a 'pipeline' submission without "
                        + "one. Declare the chain in Pipeline (chain runner).")
            }
            return Plan(
                summary: "\(prefix): pipeline → \(stages.joined(separator: " → ")) "
                    + "(\(stages.count) stage\(stages.count == 1 ? "" : "s"), "
                    + "one job, one model load)")
        }

        let summary = "\(prefix): \(verb) — this verb only"
        // A declared chain that this submission bypasses. Declaring
        // exactly the verb being submitted is not a mismatch.
        guard !stages.isEmpty, stages != [verb] else {
            return Plan(summary: summary)
        }
        return Plan(
            summary: summary,
            mismatch: "a pipeline is declared on this study "
                + "(\(stages.joined(separator: " → "))) but Verb is '\(verb)', "
                + "so only '\(verb)' will run and the other stages are "
                + "ignored. Set Verb to 'pipeline' in Remote options to run "
                + "the chain.")
    }
}
