import Foundation

public enum StudySubmissionPresentation {
    public static func bundleSubmittedStatus(
        study: String, verb: String, dryRun: Bool, substrate: String, jobID: String
    ) -> String {
        let mode = dryRun ? "\(verb) (dry run — prepared only, nothing executes)" : verb
        return "bundled study '\(study)' submitted: \(mode) on \(substrate) — "
            + "job \(jobID), following in the activity pane"
    }
}
