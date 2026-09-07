import Foundation
import MLX
import Testing
@testable import SteeringKit

@Suite(.serialized) struct ScientificScopeStabilityTests {
    private func python(_ script: String, input: Data = Data("null".utf8)) throws -> Data {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        let local = repository.appending(path: "Server/.venv.nosync/bin/python").path
        let executable = ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"]
            ?? (FileManager.default.isExecutableFile(atPath: local) ? local : "python3")
        process.arguments = [executable, "-c", "import json,sys\nx=json.loads(sys.argv[1])\n" + script,
            String(decoding: input, as: UTF8.self)]
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONPATH"] = repository.appending(path: "Server").path
        process.environment = environment
        let output = Pipe(), errors = Pipe()
        process.standardOutput = output; process.standardError = errors
        try process.run()
        let data = output.fileHandleForReading.readDataToEndOfFile()
        let diagnostics = errors.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        try #require(process.terminationStatus == 0, "\(String(decoding: diagnostics, as: UTF8.self))")
        return data
    }

    @Test func scopesMatchPythonAndTheActualEditedPositions() throws {
        let edits: [InterventionPlan.Edit] = [
            .init(layer: 1, vector: [1, 0], strength: 0.5, mode: .ablate, concept: "a", centering: "neutralMean"),
            .init(layer: 1, vector: [2, 0], strength: 0.5, mode: .ablate, concept: "b"),
            .init(layer: 1, vector: [1, 0], strength: 2, mode: .add, concept: "c")]
        let scopes = try InterventionPlan.scopeInventory(edits, promptTokenCount: 4)
        let pythonBytes = try python("""
        from steerlab_server.steering.plan import Edit,Mode,scope_inventory
        edits=[Edit(1,[1,0],.5,Mode.ABLATE,'a','neutralMean'),
               Edit(1,[2,0],.5,Mode.ABLATE,'b'),Edit(1,[1,0],2,Mode.ADD,'c')]
        print(json.dumps([s.to_dict() for s in scope_inventory(edits,prompt_token_count=4)]))
        """)
        let native = try JSONSerialization.jsonObject(with: JSONEncoder().encode(scopes)) as! NSArray
        let foreign = try JSONSerialization.jsonObject(with: pythonBytes) as! NSArray
        #expect(native == foreign)
        #expect(scopes[0].centering == "mixed(neutralMean,none)")
        #expect(scopes[0].detail["rankPerLayer"] == .object(["1": .integer(1)]))
        let chain = try InterventionPlan.interventions(edits, promptTokenCount: 4)
        let h = MLXArray([Float(4), 3, 4, 3]).reshaped([1, 2, 2])
        func apply(_ offset: Int) -> [Float] {
            chain.reduce(h) { $1.apply($0, layer: 1, offset: offset) }.asArray(Float.self)
        }
        #expect(apply(0) == [2, 3, 2, 3]) // ablation at every mid-prefill position; no addition
        #expect(apply(2) == [2, 3, 4, 3]) // final prompt position gets the addition
        #expect(apply(4) == [2, 3, 4, 3]) // last position on decode too
        #expect(VectorInjector(layer: 1, vector: [1, 0], alpha: 2).scope().detail["promptTokenCount"] == .null)
    }

    @Test func stabilityMatchesPythonDrawsAndStatisticsForAllThreeRecipes() throws {
        let positive: [[Float]] = [[1,2,0],[3,1,2],[2,-2,1],[-1,2,3],[4,0,1],[0,3,-1],[2,2,2],[3,-1,0]]
        let negative: [[Float]] = [[0,1,0],[1,0,1],[0,1,0],[1,0,2],[0,0,1],[1,0,0],[0,1,2],[1,0,1]]
        let input = try JSONSerialization.data(withJSONObject: ["p": positive, "n": negative])
        let bytes = try python("""
        from steerlab_server.steering.vector_math import direction_stability,ExtractionMethod
        print(json.dumps([direction_stability(x['p'],x['n'] if m!='designatedReference' else x['n'][:6],
            ExtractionMethod(m),resamples=16,fraction=.6,seed=18446744073709551615,order_shuffles=8).to_dict()
            for m in ('meanDifference','lat','designatedReference')]))
        """, input: input)
        let foreign = try JSONDecoder().decode([DirectionStability].self, from: bytes)
        for (index, method) in [ExtractionMethod.meanDifference, .pairedDifferencePCA, .designatedReference].enumerated() {
            let local = try SteeringVectorMath.directionStability(positive: positive,
                negative: method == .designatedReference ? Array(negative.prefix(6)) : negative,
                method: method, resamples: 16, fraction: 0.6, seed: UInt64.max, orderShuffles: 8)
            let expected = foreign[index]
            #expect(local.seed == UInt64.max)
            #expect(local.resampleSeeds == expected.resampleSeeds)
            #expect(local.orderShuffleSeeds == expected.orderShuffleSeeds)
            #expect(local.paired == expected.paired && local.subsampleSize == expected.subsampleSize)
            #expect(local.signFlips == expected.signFlips)
            #expect(local.degenerateDraws.count == expected.degenerateDraws.count)
            for (a, b) in zip(local.resampleCosines + local.orderShuffleCosines,
                              expected.resampleCosines + expected.orderShuffleCosines) {
                #expect(abs(a - b) < 2e-5)
            }
            #expect(abs(local.meanCosine - expected.meanCosine) < 2e-5)
            #expect(abs(local.percentile5Cosine - expected.percentile5Cosine) < 2e-5)
            let sorted = local.resampleCosines.sorted()
            #expect(local.medianCosine == (sorted[7] + sorted[8]) / 2)
        }
    }

    @Test func degenerateDrawsKeepTheirSeedsAndDoNotEnterSummaries() throws {
        let p: [[Float]] = [[1,0],[2,1],[0,0],[0,0],[3,-1],[0,0]]
        let n = Array(repeating: [Float(0),0], count: p.count)
        let input = try JSONSerialization.data(withJSONObject: ["p": p, "n": n])
        let expected = try JSONDecoder().decode(DirectionStability.self, from: python("""
        from steerlab_server.steering.vector_math import direction_stability,ExtractionMethod
        print(json.dumps(direction_stability(x['p'],x['n'],ExtractionMethod('lat'),
            resamples=32,fraction=.5,seed=17,order_shuffles=4).to_dict()))
        """, input: input))
        let actual = try SteeringVectorMath.directionStability(positive: p, negative: n,
            method: .pairedDifferencePCA, resamples: 32, fraction: 0.5, seed: 17, orderShuffles: 4)
        #expect(!actual.degenerateDraws.isEmpty)
        #expect(actual.degenerateDraws.map(\.seed) == expected.degenerateDraws.map(\.seed))
        #expect(actual.degenerateDraws.map(\.kind) == expected.degenerateDraws.map(\.kind))
        #expect(actual.resampleCosines.count == expected.resampleCosines.count)
        #expect(abs(actual.meanCosine - expected.meanCosine) < 2e-5)
        #expect(actual.resampleCosines.count + actual.degenerateDraws.filter { $0.kind == "resample" }.count == 32)
    }

    @Test func independentConstantContrastAndBadRequests() throws {
        let p: [[Float]] = [[2,0],[3,1],[4,-1],[1,2],[5,3]]
        let n = p.map { [$0[0] - 2, $0[1]] }
        for method in [ExtractionMethod.meanDifference, .pairedDifferencePCA] {
            let value = try SteeringVectorMath.directionStability(positive: p, negative: n,
                method: method, resamples: 10, fraction: 0.5, seed: 0, orderShuffles: 4)
            #expect(value.subsampleSize == 3) // half-up for 2.5
            #expect(value.resampleCosines.allSatisfy { abs($0 - 1) < 1e-6 })
            #expect(value.orderShuffleCosines.allSatisfy { abs($0 - 1) < 1e-6 })
            #expect(value.signFlips == 0)
        }
        for fraction in [0.0, -1, Double.nan, Double.infinity, 1.01] {
            #expect(throws: (any Error).self) {
                try SteeringVectorMath.directionStability(positive: p, negative: n,
                    method: .meanDifference, resamples: 4, fraction: fraction, seed: 0)
            }
        }
        #expect(throws: (any Error).self) {
            try SteeringVectorMath.directionStability(positive: p, negative: p,
                method: .meanDifference, resamples: 4, fraction: 0.5, seed: 0)
        }
    }
}
