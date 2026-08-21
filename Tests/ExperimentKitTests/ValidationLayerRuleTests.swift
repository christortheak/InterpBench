import Foundation
import Testing

@testable import ExperimentKit

/// D4 — the validation read layer as a DECLARED measurement decision.
///
/// The historical rule made it a side effect of the injection conditions
/// ("the layer a condition steers this concept at, else mid-network"), so
/// moving the validation read meant editing steering conditions — a different
/// decision entirely. On 2026-07-26 a researcher testing a readout-layer
/// hypothesis had to hand-write nine conditions at layer 41 because the
/// manifest offered no way to say the thing they meant.
struct ValidationLayerRuleTests {

    private func resolve(
        layer: Int? = nil, fraction: Double? = nil, condition: Int? = nil,
        layerCount: Int = 62
    ) -> ValidationLayerRule.Resolution {
        ValidationLayerRule.resolve(
            concept: "fear", declaredLayer: layer, declaredFraction: fraction,
            conditionLayer: condition, layerCount: layerCount)
    }

    // MARK: precedence

    @Test func aDeclaredIndexBeatsEverything() {
        let result = resolve(layer: 41, condition: 7)
        #expect(result.layer == 41)
        #expect(result.source == .declaredIndex)
    }

    @Test func aDeclaredFractionBeatsTheLegacyRule() {
        let result = resolve(fraction: 0.66, condition: 7)
        #expect(result.layer == 40)
        #expect(result.source == .declaredFraction)
    }

    /// The legacy rule must be preserved EXACTLY, or existing manifests
    /// change their numbers — and their content hashes — on upgrade.
    @Test func theLegacyRuleIsUnchangedWhenNothingIsDeclared() {
        #expect(resolve(condition: 7).layer == 7)
        #expect(resolve(condition: 7).source == .steeringCondition)
        #expect(resolve().layer == 31)
        #expect(resolve().source == .midNetwork)
    }

    // MARK: clamping

    @Test func anOutOfRangeDeclarationClampsRatherThanCrashing() {
        #expect(resolve(layer: 999).layer == 61)
        #expect(resolve(layer: 0).layer == 0)
        #expect(resolve(fraction: 1.0).layer == 61)
        #expect(resolve(fraction: 0.0).layer == 0)
        #expect(resolve(fraction: 0.5, layerCount: 1).layer == 0)
    }

    // MARK: the report says both numbers

    @Test func theSummaryCarriesIndexDepthAndReason() {
        // An index means nothing without the depth it represents, and a
        // depth means nothing without the model it resolved against.
        let summary = resolve(layer: 41).summary
        #expect(summary.contains("layer 41 of 62"))
        #expect(summary.contains("0.67 depth"))
        #expect(summary.contains("declared validationLayer"))
    }

    @Test func theLegacySummarySaysHowToSayItDirectly() {
        // A researcher reading "inherited from a steering condition" should
        // learn there is a way to declare it instead.
        #expect(resolve(condition: 7).summary.contains("declare validationLayer"))
        #expect(resolve().summary.contains("declare validationLayer"))
    }

    // MARK: the ambiguity refusal

    @Test func declaringBothAnIndexAndAFractionIsAViolation() throws {
        let problem = try #require(
            ValidationLayerRule.violation(declaredLayer: 41, declaredFraction: 0.66))
        #expect(problem.contains("both"))
        // Declaring exactly one is fine.
        #expect(
            ValidationLayerRule.violation(
                declaredLayer: 41, declaredFraction: nil) == nil)
        #expect(
            ValidationLayerRule.violation(
                declaredLayer: nil, declaredFraction: 0.66) == nil)
        #expect(
            ValidationLayerRule.violation(
                declaredLayer: nil, declaredFraction: nil) == nil)
    }

    @Test func nonsenseDeclarationsAreRefused() {
        #expect(
            ValidationLayerRule.violation(
                declaredLayer: -1, declaredFraction: nil) != nil)
        #expect(
            ValidationLayerRule.violation(
                declaredLayer: nil, declaredFraction: 1.5) != nil)
        #expect(
            ValidationLayerRule.violation(
                declaredLayer: nil, declaredFraction: .nan) != nil)
    }

    // MARK: depth lists (validate-at-the-sweep-layers, 2026-08-01)

    @Test func aDepthListResolvesEveryEntryInDeclaredOrder() throws {
        let resolutions = try ValidationLayerRule.resolveAll(
            concept: "fear", declaredFractions: [0.5, 0.6, 0.7, 0.8],
            layerCount: 62)
        #expect(resolutions.map(\.layer) == [31, 37, 43, 49])
        #expect(resolutions.allSatisfy { $0.source == .declaredFraction })
    }

    @Test func theScalarPathIsAOneElementList() throws {
        // Every existing manifest keeps its single resolution — same layer,
        // same source, through the same function the loop consumes.
        let resolutions = try ValidationLayerRule.resolveAll(
            concept: "fear", conditionLayer: 7, layerCount: 62)
        #expect(resolutions.map(\.layer) == [7])
        #expect(resolutions.first?.source == .steeringCondition)
    }

    @Test func twoEntriesResolvingToOneLayerThrow() {
        // 0.60 and 0.61 of 62 layers are both layer 37: silently collapsing
        // them would misreport what was declared.
        #expect(throws: ExperimentError.self) {
            try ValidationLayerRule.resolveAll(
                concept: "fear", declaredFractions: [0.6, 0.61], layerCount: 62)
        }
    }

    @Test func anOutOfRangeIndexInAListThrows() {
        #expect(throws: ExperimentError.self) {
            try ValidationLayerRule.resolveAll(
                concept: "fear", declaredLayers: [100], layerCount: 62)
        }
    }

    @Test func pluralDeclarationsAreExclusiveWithEverythingElse() throws {
        let both = try #require(ValidationLayerRule.violation(
            declaredLayer: 41, declaredFraction: nil,
            declaredLayers: [1, 2]))
        #expect(both.contains("exactly one"))
        #expect(ValidationLayerRule.violation(
            declaredLayer: nil, declaredFraction: nil,
            declaredFractions: [0.5, 0.6]) == nil)
    }

    @Test func pluralDeclarationsRefuseEmptyDuplicatesAndNonsense() throws {
        func violation(
            layers: [Int]? = nil, fractions: [Double]? = nil
        ) -> String? {
            ValidationLayerRule.violation(
                declaredLayer: nil, declaredFraction: nil,
                declaredLayers: layers, declaredFractions: fractions)
        }
        #expect(try #require(violation(layers: [])).contains("non-empty"))
        #expect(try #require(violation(layers: [3, 3])).contains("duplicate"))
        #expect(try #require(violation(layers: [-1])).contains("non-negative"))
        #expect(
            try #require(violation(fractions: [0.5, 0.5])).contains("duplicate"))
        #expect(try #require(violation(fractions: [1.5])).contains("[0, 1]"))
    }

    /// Cross-engine: the Python suite consumes the same fixture, so both
    /// engines resolve a declared list identically.
    @Test func theDepthListResolutionMatchesThePythonTwin() throws {
        let url = CodeResources.compiledCheckoutPath.appending(
            components: "Tests", "Fixtures", "cross-engine",
            "validation-depth-lists.json")
        let cases = try #require(
            try JSONSerialization.jsonObject(with: try Data(contentsOf: url))
                as? [[String: Any]])
        #expect(cases.count >= 6)
        for entry in cases {
            let label = entry["label"] as? String ?? "?"
            let input = try #require(entry["input"] as? [String: Any])
            func run() throws -> [ValidationLayerRule.Resolution] {
                try ValidationLayerRule.resolveAll(
                    concept: "fear",
                    declaredLayers: input["declared_layers"] as? [Int],
                    declaredFractions: input["declared_fractions"] as? [Double],
                    declaredLayer: input["declared_layer"] as? Int,
                    declaredFraction: input["declared_fraction"] as? Double,
                    conditionLayer: input["condition_layer"] as? Int,
                    layerCount: try #require(input["layer_count"] as? Int))
            }
            if let refusal = entry["refusal"] as? String {
                do {
                    _ = try run()
                    Issue.record("'\(label)' should have refused")
                } catch let error as ExperimentError {
                    #expect(
                        error.reason.contains(refusal),
                        "refusal drift on '\(label)': \(error.reason)")
                }
                continue
            }
            let expect = try #require(entry["expect"] as? [[String: Any]])
            let resolutions = try run()
            #expect(
                resolutions.map(\.layer) == expect.map { $0["layer"] as? Int ?? -1 },
                "layer drift on '\(label)'")
            #expect(
                resolutions.map(\.source.rawValue)
                    == expect.map { $0["source"] as? String ?? "?" },
                "source drift on '\(label)'")
        }
    }

    // MARK: the manifest wiring

    @Test func aLegacyManifestKeepsItsContentHash() throws {
        var manifest = ExperimentManifest(
            name: "vl", description: "", modelID: "test/model")
        manifest.validationLayer = nil
        manifest.validationLayerFraction = nil
        let before = ExperimentStore.manifestHash(manifest)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let text = String(decoding: try encoder.encode(manifest), as: UTF8.self)
        #expect(!text.contains("validationLayer"))
        let decoded = try JSONDecoder().decode(
            ExperimentManifest.self, from: try encoder.encode(manifest))
        #expect(ExperimentStore.manifestHash(decoded) == before)
    }

    // MARK: cross-engine agreement

    /// A study validated on one substrate must read the same depth as the
    /// same study on the other. Generated by the Python twin.
    @Test func theResolutionMatchesThePythonTwin() throws {
        let url = CodeResources.compiledCheckoutPath.appending(
            components: "Tests", "Fixtures", "cross-engine",
            "validation-layers.json")
        let cases = try #require(
            try JSONSerialization.jsonObject(with: try Data(contentsOf: url))
                as? [[String: Any]])
        #expect(cases.count >= 8)
        for entry in cases {
            let label = entry["label"] as? String ?? "?"
            let input = try #require(entry["input"] as? [String: Any])
            let result = ValidationLayerRule.resolve(
                concept: "fear",
                declaredLayer: input["declared_layer"] as? Int,
                declaredFraction: input["declared_fraction"] as? Double,
                conditionLayer: input["condition_layer"] as? Int,
                layerCount: try #require(input["layer_count"] as? Int))
            #expect(
                result.layer == entry["layer"] as? Int,
                "layer drift on '\(label)'")
            #expect(
                result.source.rawValue == entry["source"] as? String,
                "source drift on '\(label)'")
            let depth = try #require(entry["depthFraction"] as? Double)
            #expect(
                abs(result.depthFraction - depth) < 1e-9,
                "depth drift on '\(label)'")
        }
    }
}
