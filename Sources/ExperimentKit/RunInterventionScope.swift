import Foundation
import SteeringKit

/// Run-start description of the full condition matrix. The existing resolvers
/// supply the same dosed vectors that generation will arm. Description failures
/// are recorded; the execution loop remains responsible for execution errors.
public enum RunInterventionScope {
    public static let filename = "intervention-scope.json"
    public static let promptCountPerItem =
        "supplied per item at generation time — the rendered prompt's token count"

    public struct Entry: Codable, Sendable {
        public var condition: String
        public var interventionState: [String: JSONValue]
        public var scopes: [InterventionScope]
        public var neutralPCBasisPath: String?
        public var unresolved: String?
    }

    public struct Document: Codable, Sendable {
        public var schemaVersion = 1
        public var experiment: String
        public var conditions: [Entry]
    }

    static func entry(name: String, state: [String: JSONValue] = [:],
                      neutralPCBasisPath: String? = nil,
                      resolve: () throws -> [ExperimentTasks.CellInjection]) -> Entry {
        do {
            var scopes = try InterventionPlan.scopeInventory(resolve().map(\.planEdit), promptTokenCount: 1)
            for index in scopes.indices where scopes[index].detail["promptTokenCount"] == .integer(1) {
                scopes[index].detail["promptTokenCount"] = .string(promptCountPerItem)
            }
            return .init(condition: name, interventionState: state, scopes: scopes,
                         neutralPCBasisPath: neutralPCBasisPath)
        } catch {
            return .init(condition: name, interventionState: state, scopes: [], unresolved: String(describing: error))
        }
    }

    public static func write(experiment: String, entries: [Entry], to directory: URL) throws {
        let destination = directory.appending(component: filename)
        guard !FileManager.default.fileExists(atPath: destination.path) else { return }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted, .withoutEscapingSlashes]
        let bytes = try encoder.encode(Document(experiment: experiment, conditions: entries))
        // No replacement, including a concurrent writer or resumed invocation.
        try bytes.write(to: destination, options: .withoutOverwriting)
    }

    static func state(_ condition: ExperimentManifest.Condition) -> [String: JSONValue] {
        var result: [String: JSONValue] = [
            "slots": .array(condition.slots.map { slot in
                var value: [String: JSONValue] = ["concept": .string(slot.concept),
                    "layer": .number(Double(slot.layer)), "alpha": .number(slot.alpha)]
                if slot.effectiveMode == .ablate { value["mode"] = .string("ablate") }
                return .object(value)
            }),
            "bandWidth": .number(Double(condition.bandWidth)),
            "alphaInNormUnits": .bool(condition.alphaInNormUnits),
            "controlType": condition.controlType.map(JSONValue.string) ?? .null]
        if condition.controlType == "randomMatchedNorm" {
            result["randomVectorAlgorithm"] = .string(SteeringVectorMath.randomVectorAlgorithm)
        }
        return result
    }

    static func state(name: String, variant: ModelVariantArtifact) -> [String: JSONValue] {
        ["slots": .array(variant.injections.map { injection in
            var value: [String: JSONValue] = ["concept": .string(injection.concept),
                "layer": .number(Double(injection.layer)), "alpha": .number(injection.alpha)]
            if injection.effectiveMode == .ablate { value["mode"] = .string("ablate") }
            return .object(value)
        }), "bandWidth": .number(Double(variant.bandWidth)),
         "alphaInNormUnits": .bool(variant.alphaInNormUnits), "controlType": .null,
         "variant": .string(name), "adapters": .array(variant.adapters.map {
             .object(["adapterDirectory": .string($0.adapterDirectory),
                      "adapterHash": $0.adapterHash.map(JSONValue.string) ?? .null])
         })]
    }
}
