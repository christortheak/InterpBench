import ExperimentKit
import SwiftUI

struct ProbeMeasurementResultsView: View {
    let value: JSONValue
    @State private var filter = ""
    private var rows: [[String: JSONValue]] {
        guard case .object(let body) = value, case .array(let readings) = body["readings"] else { return [] }
        return readings.compactMap { if case .object(let row) = $0 { return row }; return nil }
            .filter { filter.isEmpty || text($0["measurementID"]).localizedCaseInsensitiveContains(filter) }
    }
    private func text(_ value: JSONValue?) -> String {
        switch value {
        case .string(let s): s
        case .number(let n): String(n)
        case .bool(let b): String(b)
        default: "—"
        }
    }
    var body: some View {
        DisclosureGroup("Probe readings") {
            VStack(alignment: .leading, spacing: 8) {
                Text("Each score reads the input token at the listed position. That activation predicts the next token; it does not read the token it predicts. Scores are not probabilities.").font(.caption)
                if case .object(let body) = value {
                    Text("Recording: \(text(body["status"])) · omitted by budget: \(text(body["omittedReadings"]))").font(.caption)
                    if case .array(let failures) = body["failures"] {
                        ForEach(failures.indices, id: \.self) { i in Text(text(failures[i])).foregroundStyle(.orange) }
                    }
                }
                TextField("Filter by measurement ID", text: $filter)
                ScrollView([.horizontal, .vertical]) {
                    LazyVStack(alignment: .leading, spacing: 8) {
                        ForEach(rows.indices, id: \.self) { i in
                            let row = rows[i]
                            VStack(alignment: .leading, spacing: 2) {
                                Text("\(text(row["probeLabel"])) · \(text(row["measurementID"])) · \(text(row["stage"])) · \(text(row["recordingStage"]))")
                                Text("Input position \(text(row["inputTokenPosition"])) (token \(text(row["inputTokenID"]))) → predicts token \(text(row["predictedTokenID"]))")
                                Text("\(text(row["status"])) · score \(text(row["score"])) · \(text(row["scoreKind"]))")
                                if case .string(let reason) = row["reason"] { Text(reason).foregroundStyle(.orange) }
                            }.font(.caption.monospaced()).textSelection(.enabled)
                        }
                    }
                }.frame(maxHeight: 300)
            }
        }
    }
}
