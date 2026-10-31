import CryptoKit
import Foundation

/// Structural evidence admission, mirrored by Python and exercised on its real output.
/// Hash verification precedes this check; scientific qualification remains separate.
public enum InstrumentationEvidence {
    private static func isText(_ value: JSONValue) -> Bool { if case .string = value { return true }; return false }
    private static func isFiniteNumber(_ value: JSONValue) -> Bool { if case .number(let x) = value { return x.isFinite }; return false }
    private static func isDigest(_ value: JSONValue?) -> Bool {
        guard case .string(let text) = value else { return false }
        return text.count == 64 && text.allSatisfy { "0123456789abcdef".contains($0) }
    }
    private static func number(_ value: JSONValue?) -> Double? { if case .number(let x) = value, x.isFinite { return x }; return nil }
    /// Execution computes in float32; recomputation here is float64.
    private static func close(_ a: Double, _ b: Double) -> Bool { abs(a - b) <= max(1e-9, 1e-5 * max(abs(a), abs(b))) }

    /// An evidence declaration's units for one action. Twin: `instrumentation_evidence.declared_units`.
    static func declaredUnits(_ action: [String: JSONValue]) -> String? {
        if action["kind"] == .string("add") { if case .string(let units) = action["strengthUnits"] { return units }; return "rawMultiplier" }
        guard case .string(let kind) = action["kind"] else { return nil }
        return ["ablate": "fraction", "logitBias": "logit", "allowTokens": "onOff", "forceToken": "onOff"][kind]
    }

    /// `promptTokensSHA256` of retained token IDs: Python's `json.dumps` of the integer list.
    static func tokenIDsSHA256(_ ids: [JSONValue]) -> String {
        let text = "[" + ids.map { value -> String in if case .number(let x) = value { String(Int(x)) } else { "" } }.joined(separator: ", ") + "]"
        return SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    /// The span entries of version-2 policy evidence, one per span policy, keyed by policy hash with the positions their
    /// ranges cover. Twin: `instrumentation_evidence.validate_spans`.
    private static func spans(_ block: [String: JSONValue], declarations: [String: [String: JSONValue]], prompt: [JSONValue],
                              refuse: (String) throws -> Void) throws -> [String: (positions: Set<Int>, entry: [String: JSONValue])] {
        var result: [String: (positions: Set<Int>, entry: [String: JSONValue])] = [:]
        var entries: [JSONValue] = []
        if let value = block["spans"] {
            guard case .array(let values) = value else { try refuse("Span evidence must be a list of resolved spans."); return [:] }
            entries = values
        }
        for value in entries {
            guard case .object(let entry) = value, case .string(let policy) = entry["policySHA256"], let declaration = declarations[policy],
                  declaration["positions"] == .string("span"), result[policy] == nil else { try refuse("A resolved span names a policy that does not act on a span, or names it twice."); return [:] }
            guard isDigest(entry["spanEvidenceSHA256"]), isDigest(entry["renderedTextSHA256"]), isDigest(entry["promptTokensSHA256"]) else { try refuse("A resolved span must name its resolution, rendered text, and prompt tokens by SHA-256."); return [:] }
            if entry["promptTokensSHA256"] != .string(tokenIDsSHA256(prompt)) { try refuse("A resolved span belongs to a different prompt than the retained prompt tokens.") }
            guard case .array(let ranges) = entry["tokenRanges"], !ranges.isEmpty else { try refuse("A resolved span needs its token ranges."); return [:] }
            var positions = Set<Int>(); var end = 0
            for range in ranges {
                guard case .array(let pair) = range, pair.count == 2, case .number(let low) = pair[0], case .number(let high) = pair[1],
                      low.rounded() == low, high.rounded() == high, Double(end) <= low, low < high, high <= Double(prompt.count) else { try refuse("Span token ranges must be ascending, disjoint, nonempty, and inside the prompt."); return [:] }
                positions.formUnion(Int(low)..<Int(high)); end = Int(high)
            }
            if entry["tokenCount"] != .number(Double(positions.count)) { try refuse("A span token count must equal the positions its ranges cover.") }
            guard case .string(let dose) = entry["doseMode"], ["perPosition", "fixedTotal"].contains(dose), declaration["spanDose"] == .string(dose) else { try refuse("A resolved span must state its policy’s declared dose mode."); return [:] }
            var specs: [String: [String: JSONValue]] = [:]
            if case .array(let actions) = declaration["actions"] {
                for action in actions { if case .object(let object) = action, case .string(let id) = object["id"] { specs[id] = object } }
            }
            guard case .object(let summaries) = entry["actions"], Set(summaries.keys) == Set(specs.keys) else { try refuse("A resolved span summarizes each of its policy’s actions once."); return [:] }
            for (name, value) in summaries {
                guard case .object(let summary) = value, let spec = specs[name], case .string(let units) = summary["strengthUnits"], units == declaredUnits(spec) else { try refuse("Span dose units must equal the action’s declared units."); return [:] }
                guard case .number(let count) = summary["appliedPositions"], count.rounded() == count, count >= 0, count <= Double(positions.count) else { try refuse("A span’s applied positions must count span tokens."); return [:] }
                let strength = number(summary["perPositionStrength"])
                if summary["perPositionStrength"] != .null, strength == nil { try refuse("A per-position span strength must be finite.") }
                if spec["kind"] == .string("add") {
                    guard let summed = number(summary["summedDose"]), strength.map({ close(summed, $0 * count) }) ?? true, count > 0 || summed == 0 else { try refuse("A summed span dose must add what each applied position received."); return [:] }
                } else if summary["summedDose"] != .null { try refuse("Only an addition has a summed span dose.") }
            }
            result[policy] = (positions, entry)
        }
        if declarations.contains(where: { $0.value["positions"] == .string("span") && result[$0.key] == nil }) { try refuse("A span policy’s evidence must name the span it resolved for this response.") }
        return result
    }

    /// A span policy's decision acts only at its resolved positions, and each scale is the declared strength converted by the
    /// action's units and, for a fixedTotal dose, shared over the span. Twin: `instrumentation_evidence.validate_span_row`.
    private static func validateSpanRow(_ row: [String: JSONValue], declaration: [String: JSONValue],
                                        span: (positions: Set<Int>, entry: [String: JSONValue]), refuse: (String) throws -> Void) throws {
        guard case .number(let position) = row["inputTokenPosition"], span.positions.contains(Int(position)) else { try refuse("A span policy decision lies outside its resolved span."); return }
        let share = span.entry["doseMode"] == .string("fixedTotal") ? number(span.entry["tokenCount"]) ?? 1 : 1
        guard case .object(let scales) = row["appliedScales"] ?? .object([:]), case .object(let strengths) = row["strengths"] else { return }
        for (name, value) in scales {
            var spec: [String: JSONValue] = [:]
            if case .array(let actions) = declaration["actions"] {
                for action in actions { if case .object(let object) = action, object["id"] == .string(name) { spec = object } }
            }
            guard let scale = number(value), let requested = number(strengths[name]) else { try refuse("A span decision’s applied scale differs from its declared strength, units, and dose."); return }
            let strength = requested / share
            var expected = strength
            if declaredUnits(spec) == "residualNorm", let norm = number(spec["residualNorm"]), let length = number(spec["vectorNorm"]), length != 0 { expected = strength * norm / length }
            if !close(scale, expected) { try refuse("A span decision’s applied scale differs from its declared strength, units, and dose.") }
        }
    }

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
            var spanEntries: [String: (positions: Set<Int>, entry: [String: JSONValue])] = [:]
            if field == "interventionDecisions", schema == .number(2) {
                guard case .array(let docs) = block["declarations"], case .array(let policies) = block["policies"] else { try refuse("Policy evidence must bind its declarations and hashes."); return }
                for doc in docs {
                    guard case .object(let object) = doc, case .string(let sha) = object["sha256"], case .array = object["actions"] else { try refuse("Malformed policy evidence declaration."); return }
                    declarations[sha] = object
                }
                let policyNames = policies.compactMap { value -> String? in if case .string(let name) = value { name } else { nil } }
                if declarations.count != docs.count || policyNames.count != policies.count || declarations.keys.sorted() != policyNames.sorted() { try refuse("Policy declarations disagree with the executed hashes.") }
                spanEntries = try spans(block, declarations: declarations, prompt: prompt, refuse: refuse)
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
                            // Execution bounds strengths in float32, so a strength at a fractional bound records that bound's float32 value.
                            guard case .array(let bounds) = spec?["bounds"], bounds.count == 2, case .number(let low) = bounds[0], case .number(let high) = bounds[1], case .number(let number) = strength, low.isFinite, high.isFinite, number >= Double(Float(low)), number <= Double(Float(high)) else { try refuse("A requested action is undeclared or outside its bounds."); return }
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
                    // Units and absolute scales are additive: rows written before they existed have neither.
                    if let value = row["strengthUnits"] {
                        guard case .object(let units) = value, units.allSatisfy({ requested[$0.key] != nil && isText($0.value) }) else { try refuse("Strength units must name requested actions."); return }
                    }
                    if let value = row["appliedScales"] {
                        guard case .object(let scales) = value, scales.allSatisfy({ applied[$0.key] != nil && isFiniteNumber($0.value) }) else { try refuse("Applied scales must be finite and belong to applied actions."); return }
                    }
                    if let span = spanEntries[policy] { try validateSpanRow(row, declaration: declaration, span: span, refuse: refuse) }
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
