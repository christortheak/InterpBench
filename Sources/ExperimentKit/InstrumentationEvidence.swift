import Foundation

/// Structural evidence admission, mirrored by Python and exercised on its real output.
/// Hash verification precedes this check; scientific qualification remains separate.
public enum InstrumentationEvidence {
    public static func validate(_ value: JSONValue) throws {
        func refuse(_ reason: String) throws { throw ExperimentError(reason: reason + " Retain the source and recollect complete evidence.") }
        guard case .object(let record) = value else { try refuse("Each evidence row must be a JSON object."); return }
        if let value = record["instrumentationRequirements"] {
            guard case .array(let values) = value, values.allSatisfy({ if case .string = $0 { true } else { false } }) else { try refuse("Instrumentation requirements must be capability names."); return }
        }
        let required: [JSONValue] = if case .array(let values) = record["instrumentationRequirements"] { values } else { [] }
        for (capability, field, rowsKey) in [("policy-v1", "interventionDecisions", "decisions"), ("probe-readings-v1", "probeMeasurements", "readings")] {
            guard let blockValue = record[field] else {
                if required.contains(.string(capability)) { try refuse("A response is missing its declared \(field).") }
                continue
            }
            guard case .object(let block) = blockValue, let schema = block["schemaVersion"], [.number(1), .number(2)].contains(schema), case .array(let rows) = block[rowsKey],
                  case .array(let prompt) = block["promptTokenIDs"], case .array(let output) = block["outputTokenIDs"] else { try refuse("Unsupported or malformed instrumentation evidence."); return }
            if field == "probeMeasurements", schema != .number(1) { try refuse("Unsupported probe measurement evidence schema.") }
            if let recorded = record["outputTokenIDs"], recorded != .array(output) { try refuse("Instrumentation and response token IDs disagree.") }
            if let condition = record["condition"], let observed = block["condition"], condition != observed { try refuse("Instrumentation belongs to a different condition.") }
            let tokens = prompt + output
            for token in tokens {
                guard case .number(let x) = token, x.isFinite, x >= 0, x.rounded() == x else { try refuse("Invalid evidence token ID."); return }
            }
            var declarations: [String: [String: JSONValue]] = [:]
            if field == "interventionDecisions", schema == .number(2) {
                guard case .array(let docs) = block["declarations"], case .array(let policies) = block["policies"] else { try refuse("Policy evidence must bind its declarations and hashes."); return }
                for doc in docs {
                    guard case .object(let object) = doc, case .string(let sha) = object["sha256"], case .array = object["actions"] else { try refuse("Malformed policy evidence declaration."); return }
                    declarations[sha] = object
                }
                let policyNames = policies.compactMap { value -> String? in if case .string(let name) = value { name } else { nil } }
                if declarations.count != docs.count || policyNames.count != policies.count || declarations.keys.sorted() != policyNames.sorted() { try refuse("Policy declarations disagree with the executed hashes.") }
            }
            for rowValue in rows {
                guard case .object(let row) = rowValue, case .number(let position) = row["inputTokenPosition"], position.isFinite, position >= 0, position.rounded() == position,
                      case .number(let predicted) = row["predictsTokenPosition"], predicted == position + 1 else { try refuse("Invalid consumed/predicted token positions."); return }
                for (key, index) in [("inputTokenID", position), ("predictedTokenID", predicted)] {
                    if let actual = row[key] {
                        let expected = index < Double(tokens.count) ? tokens[Int(index)] : JSONValue.null
                        if actual != expected { try refuse("Evidence token alignment differs from the retained sequence.") }
                    }
                }
                if field == "probeMeasurements", row["status"] == .string("recorded") {
                    guard case .number(let score) = row["score"], score.isFinite else { try refuse("A recorded probe score must be finite."); return }
                }
                if field == "interventionDecisions", schema == .number(2) {
                    guard case .string(let policy) = row["policySHA256"], let declaration = declarations[policy], declaration["site"] == row["site"] else { try refuse("A decision names an undeclared policy or site."); return }
                    if block["status"] == .string("complete"), [JSONValue.string("requested"), .string("failed"), .string("skipped")].contains(row["status"] ?? .null) { try refuse("Incomplete decisions cannot be labelled complete evidence.") }
                    guard case .object(let requested) = row["strengths"], case .object(let applied) = row["appliedStrengths"], case .object(let outcomes) = row["actionOutcomes"] else { try refuse("Policy evidence needs requested and applied strengths and action outcomes."); return }
                    if case .array(let actions) = declaration["actions"] {
                        for (name, strength) in requested {
                            let spec = actions.compactMap { value -> [String: JSONValue]? in
                                if case .object(let object) = value, object["id"] == .string(name) { return object }; return nil
                            }.first
                            guard case .array(let bounds) = spec?["bounds"], bounds.count == 2, case .number(let low) = bounds[0], case .number(let high) = bounds[1], case .number(let number) = strength, low.isFinite, high.isFinite, number >= low, number <= high else { try refuse("A requested action is undeclared or outside its bounds."); return }
                        }
                    }
                    for value in Array(requested.values) + Array(applied.values) {
                        guard case .number(let x) = value, x.isFinite else { try refuse("Policy strengths must be finite."); return }
                    }
                    for value in outcomes.values {
                        guard case .object = value else { try refuse("Action outcomes must be acknowledgement objects."); return }
                    }
                    for (id, strength) in applied {
                        guard requested[id] == strength, case .object(let outcome) = outcomes[id], outcome["status"] == .string("applied") else { try refuse("An applied strength lacks a successful action acknowledgement."); return }
                    }
                    if row["status"] == .string("applied"), requested != applied { try refuse("An applied decision has unacknowledged actions.") }
                }
            }
        }
    }

    public static func validateFile(_ file: URL) throws {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        var pending = Data()
        func line(_ data: Data) throws {
            if data.allSatisfy({ [9, 13, 32].contains($0) }) { return }
            try validate(JSONDecoder().decode(JSONValue.self, from: data))
        }
        while let chunk = try handle.read(upToCount: 64 * 1024), !chunk.isEmpty {
            pending.append(chunk)
            while let end = pending.firstIndex(of: 10) {
                if pending.distance(from: pending.startIndex, to: end) > 64 * 1024 * 1024 { throw ExperimentError(reason: "An evidence row exceeds the 64 MiB reading limit.") }
                try line(pending.subdata(in: pending.startIndex..<end))
                pending.removeSubrange(pending.startIndex...end)
            }
            if pending.count > 64 * 1024 * 1024 { throw ExperimentError(reason: "An evidence row exceeds the 64 MiB reading limit.") }
        }
        if !pending.isEmpty { try line(pending) }
    }
}
