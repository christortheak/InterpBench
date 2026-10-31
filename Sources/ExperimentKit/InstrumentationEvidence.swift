import CryptoKit
import Foundation

/// Structural evidence admission, mirrored by Python and exercised on its real output.
/// Hash verification precedes this check; scientific qualification remains separate.
public enum InstrumentationEvidence {
    private static func isDigest(_ value: JSONValue?) -> Bool {
        guard case .string(let text) = value else { return false }
        return text.count == 64 && text.allSatisfy { "0123456789abcdef".contains($0) }
    }
    private static func number(_ value: JSONValue?) -> Double? { if case .number(let x) = value, x.isFinite { return x }; return nil }

    /// `promptTokensSHA256` of retained token IDs: Python's `json.dumps` of the integer list.
    static func tokenIDsSHA256(_ ids: [JSONValue]) -> String {
        let text = "[" + ids.map { value -> String in if case .number(let x) = value { String(Int(x)) } else { "" } }.joined(separator: ", ") + "]"
        return SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    /// A fixedTotal span's strength at each of its n tokens: float32 strength / n, as execution
    /// applies it. Twin: `policy_artifacts.span_share`.
    static func spanShare(_ strength: Double, _ count: Double) -> Double { Double(Float(strength) / Float(count)) }

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
                for case .object(let action) in actions { if case .string(let id) = action["id"] { specs[id] = action } }
            }
            guard case .object(let summaries) = entry["actions"], Set(summaries.keys) == Set(specs.keys) else { try refuse("A resolved span summarizes each of its policy’s actions once."); return [:] }
            for (name, value) in summaries {
                guard case .object(let summary) = value, let spec = specs[name], case .string(let units) = summary["strengthUnits"],
                      units == declaredUnits(spec, version: declaration["schemaVersion"]) else { try refuse("Span dose units must be the declared units of each action."); return [:] }
                guard case .number(let count) = summary["appliedPositions"], count.rounded() == count, count >= 0, count <= Double(positions.count) else { try refuse("A span’s applied positions must count span tokens."); return [:] }
                let strength = number(summary["perPositionStrength"])
                if summary["perPositionStrength"] != .null, strength == nil { try refuse("A per-position span strength must be finite.") }
                if spec["kind"] == .string("add") {
                    // Recomputable whenever every position received one strength: the sum of what each applied.
                    guard let summed = number(summary["summedDose"]), strength.map({ summed == $0 * count }) ?? true, count > 0 || summed == 0 else { try refuse("A summed span dose must add what each applied position received."); return [:] }
                } else if summary["summedDose"] != .null { try refuse("Only an addition has a summed span dose.") }
            }
            result[policy] = (positions, entry)
        }
        if declarations.contains(where: { $0.value["positions"] == .string("span") && result[$0.key] == nil }) { try refuse("A span policy’s evidence must name the span it resolved for this response.") }
        return result
    }

    /// Units of the action kinds no policy version changes. Twin: `policy_artifacts.FIXED_UNITS`.
    private static let fixedUnits = ["ablate": "fraction", "logitBias": "logit", "allowTokens": "onOff", "forceToken": "onOff"]

    /// An evidence declaration's units for one action, or nil when the declaration contradicts
    /// its version. Twin: `instrumentation_evidence.declared_units`.
    static func declaredUnits(_ action: [String: JSONValue], version: JSONValue?) -> String? {
        guard case .string(let kind) = action["kind"] else { return nil }
        guard kind == "add" else { return fixedUnits[kind] }
        let declared = action["strengthUnits"]
        let units: String? = if case .string(let text) = declared { text } else { nil }
        if version == .number(1) { return declared == nil ? "rawMultiplier" : nil }
        if version == .number(2) { return units == "residualNorm" || units == "rawMultiplier" ? units : nil }
        return declared == nil ? "rawMultiplier" : units  // recorded before declarations stated a version
    }

    /// What an applied addition's `appliedScales` entry must be, recomputed from its declaration.
    /// Residual-norm units use `vector_math.norm_unit_scale`'s float32 order, (strength·R)/‖v‖; Python
    /// forms the same correctly rounded float32 values. Twin: `instrumentation_evidence.expected_scale`.
    static func expectedScale(_ strength: Double, _ action: [String: JSONValue], version: JSONValue?) -> Double? {
        switch declaredUnits(action, version: version) {
        case "rawMultiplier": return strength
        case "residualNorm":
            guard case .number(let norm) = action["residualNorm"], case .number(let length) = action["vectorNorm"],
                  norm.isFinite, length.isFinite, norm > 0, length > 0 else { return nil }
            let product = Float(strength) * Float(norm)
            return Double(product / Float(length))
        default: return nil
        }
    }

    private static func unitsAgree(_ value: JSONValue, requested: [String: JSONValue], specs: [String: [String: JSONValue]], version: JSONValue?) -> Bool {
        guard case .object(let units) = value, Set(units.keys) == Set(requested.keys) else { return false }
        return units.allSatisfy { entry in
            guard let spec = specs[entry.key], let declared = declaredUnits(spec, version: version) else { return false }
            return entry.value == .string(declared)
        }
    }

    /// `share` is what a recorded strength applies at each token: itself, or a fixedTotal span's share.
    private static func scalesAgree(_ value: JSONValue, applied: [String: JSONValue], specs: [String: [String: JSONValue]], version: JSONValue?,
                                    share: (Double) -> Double = { $0 }) -> Bool {
        let additions = Set(applied.keys.filter { specs[$0]?["kind"] == .string("add") })
        guard case .object(let scales) = value, Set(scales.keys) == additions else { return false }
        return scales.allSatisfy { entry in
            guard case .number(let recorded) = entry.value, recorded.isFinite, case .number(let strength)? = applied[entry.key],
                  let spec = specs[entry.key] else { return false }
            return expectedScale(share(strength), spec, version: version) == recorded
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
                    guard case .object(let object) = doc, case .string(let sha) = object["sha256"], case .array(let actions) = object["actions"] else { try refuse("Malformed policy evidence declaration."); return }
                    // Additive: declarations recorded before it existed state no schemaVersion.
                    if let version = object["schemaVersion"] {
                        let additionsAgree = actions.allSatisfy { value in
                            guard case .object(let action) = value, action["kind"] == .string("add") else { return true }
                            return declaredUnits(action, version: version) != nil
                        }
                        guard InterventionPolicyLibrary.schemaVersions.contains(version), additionsAgree else { try refuse("A policy declaration’s schema version and its additions’ units disagree."); return }
                    }
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
                    // Present, they must be the declaration's own: a row cannot restate what its policy meant.
                    var specs: [String: [String: JSONValue]] = [:]
                    if case .array(let actions) = declaration["actions"] {
                        for case .object(let action) in actions { if case .string(let id) = action["id"] { specs[id] = action } }
                    }
                    let version = declaration["schemaVersion"]
                    if let value = row["strengthUnits"], !unitsAgree(value, requested: requested, specs: specs, version: version) {
                        try refuse("Strength units must be the declared units of each requested action.")
                    }
                    // A span policy acts only at its resolved tokens, and a fixedTotal scale converts strength / n.
                    let span = spanEntries[policy]
                    if let span, !span.positions.contains(Int(position)) { try refuse("A span policy decision lies outside its resolved span.") }
                    let count = span.flatMap { $0.entry["doseMode"] == .string("fixedTotal") ? number($0.entry["tokenCount"]) : nil }
                    if let value = row["appliedScales"], !scalesAgree(value, applied: applied, specs: specs, version: version,
                                                                       share: { strength in count.map { spanShare(strength, $0) } ?? strength }) {
                        try refuse("Applied scales must be each applied addition’s declared multiplier of its stored direction.")
                    }
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
