import Foundation

/// Present the owner's reviewed budgets and coverage without deriving a second
/// execution plan in the Mac client. Unknown values stay visibly unavailable.
public enum FittingReviewSummary {
    public static func lines(operation: String, draft: JSONValue) -> [String] {
        guard case .object(let document) = draft else { return [] }
        var result: [String] = []
        if case .object(let fitting) = document["fittingReview"],
           case .object(let pilot) = fitting["pilotMeasurement"] {
            result.append("Measured pilot: \(number(pilot["rowsPerHour"], decimals: 2)) fitted rows per hour; about \(number(pilot["extrapolatedHoursAtRowCap"], decimals: 2)) hours at this row cap.")
            result.append("This extrapolates the selected pilot, not the scheduler walltime. Row lengths, hardware, and contention can change it.")
        }
        guard case .object(let review) = document["operationReview"] else { return result }
        switch operation {
        case "jlens-fit-benchmark":
            result.append("Up to \(number(review["totalRowBudget"])) row evaluations across \(count(review["cases"])) benchmark cases, including rows that may be skipped.")
            result.append("Compare numerical agreement alongside speed and memory before choosing a larger fitting budget.")
        case "jlens-fit-round":
            result.append("\(number(review["globalRowBudget"])) total corpus rows, divided among \(count(review["shards"])) shards. This is the whole round’s budget, including skipped rows.")
            if case .number(let bytes) = review["minimumLensAndCheckpointBytes"], bytes.isFinite {
                result.append("Retained shard lenses and checkpoints need at least \(number(.number(bytes / 1_073_741_824), decimals: 2)) GiB, plus staging, exports, and scratch space.")
            } else {
                result.append("Storage estimate unavailable here. Review it on the prepared engine before submitting shards.")
            }
            result.append("Creating the round plan starts no GPU fitting. Review and submit its shard jobs separately.")
        case "jlens-fit-merge":
            result.append("\(number(review["promptsFitted"])) fitted rows across \(number(review["rowsConsidered"])) considered rows.")
            if review["partial"] == .bool(true) {
                result.append("Partial merge: \(count(review["missingRows"])) planned rows are missing. The merged lens will retain that limitation.")
            } else if review["partial"] == .bool(false) {
                result.append("The selected contributions cover their declared rows. Coverage alone does not establish readout quality.")
            } else {
                result.append("Coverage is unavailable. Review the merge inputs before proceeding.")
            }
        case "jlens-fit-assess":
            result.append("Up to \(number(review["rows"])) corpus rows across \(count(review["sourceLayers"])) source layers, using the selected position cap.")
            if case .object(let resources) = review["resources"],
               case .number(let staging) = resources["temporaryActivationBytesUpperBound"],
               case .number(let pair) = resources["float32LensPairBytes"], staging.isFinite, pair.isFinite {
                result.append("Budget up to \(number(.number(staging / 1_073_741_824), decimals: 2)) GiB of temporary tensor storage at float32 for selected activations. One lens layer pair uses \(number(.number(pair / 1_073_741_824), decimals: 2)) GiB at float32.")
                result.append("These are tensor sizes, not peak memory. Allow additional space for model weights, forward activations, vocabulary logits, transfers, and file overhead. Temporary activations are removed when assessment exits normally or with an error.")
            }
            result.append("This compares readouts. It does not prove that the text is independent of fitting data or qualify the lens.")
        default:
            break
        }
        return result
    }

    public static func capacityLines(_ plan: JSONValue) -> [String] {
        guard case .object(let document) = plan,
              case .object(let capacity) = document["capacity"],
              case .string(let summary) = capacity["summary"] else { return [] }
        var result = [summary]
        if case .array(let jobs) = capacity["activeJobs"] {
            for case .object(let job) in jobs {
                if case .string(let id) = job["jobID"], case .string(let status) = job["status"] {
                    let origin = job["belongsToThisRound"] == .bool(true) ? "this round" : "another scientific task"
                    result.append("\(id): \(status) (\(origin)).")
                }
            }
        }
        return result
    }

    private static func number(_ value: JSONValue?, decimals: Int = 0) -> String {
        guard case .number(let number) = value, number.isFinite else { return "unavailable" }
        return number.formatted(.number.precision(.fractionLength(decimals)))
    }

    private static func count(_ value: JSONValue?) -> String {
        guard case .array(let values) = value else { return "unavailable" }
        return values.count.formatted()
    }
}
