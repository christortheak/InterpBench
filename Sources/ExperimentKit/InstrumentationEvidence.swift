import Foundation

/// Structural evidence admission, mirrored by Python and exercised on its real output.
/// Hash verification precedes this check; scientific qualification remains separate.
public enum InstrumentationEvidence {
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

    private static func scalesAgree(_ value: JSONValue, applied: [String: JSONValue], specs: [String: [String: JSONValue]], version: JSONValue?) -> Bool {
        let additions = Set(applied.keys.filter { specs[$0]?["kind"] == .string("add") })
        guard case .object(let scales) = value, Set(scales.keys) == additions else { return false }
        return scales.allSatisfy { entry in
            guard case .number(let recorded) = entry.value, recorded.isFinite, case .number(let strength)? = applied[entry.key],
                  let spec = specs[entry.key] else { return false }
            return expectedScale(strength, spec, version: version) == recorded
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
                    if let value = row["appliedScales"], !scalesAgree(value, applied: applied, specs: specs, version: version) {
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
