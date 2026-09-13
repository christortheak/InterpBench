import Foundation
import MLX

/// Shared site vocabulary. Logit selection belongs to the P5 sampling adapter.
public enum ResidualSite: Hashable, Sendable {
    case pre(Int)
    case post(Int)

    public var layer: Int {
        switch self { case .pre(let layer), .post(let layer): layer }
    }
}

/// A response-scoped native observer runtime around the unchanged action chain.
/// Access is serialized; model execution remains confined to its existing owner.
/// Callbacks use tensors, not JSON, and must not mutate their input or use the
/// model's random stream. They may retain bounded provider state explicitly.
public final class ResidualRuntime: @unchecked Sendable {
    public enum Stage: Sendable { case preAction, postAction }
    public struct Context: Sendable {
        public let site: ResidualSite
        public let positions: Range<Int>
        public let promptTokenCount: Int?
        public let identity: [String: String]
        /// Separate state for every row, even if a caller forwards a batch.
        public let batchIndex: Int
    }
    public struct Reading: Sendable {
        public let id: String
        public let site: ResidualSite
        public let providerID: String
        public let stage: Stage
        public let observe: @Sendable (MLXArray, Context, inout [String: MLXArray]) -> Void
        public init(id: String, site: ResidualSite, stage: Stage, providerID: String? = nil,
                    observe: @escaping @Sendable (MLXArray, Context, inout [String: MLXArray]) -> Void) {
            self.id = id; self.site = site; self.stage = stage; self.observe = observe
            self.providerID = providerID ?? id
        }
    }
    public enum ConfigurationError: Error { case invalidReadings, reusedScope }
    private let lock = NSRecursiveLock()
    private var used = false
    private var readings: [Reading]
    private var states: [String: [Int: [String: MLXArray]]] = [:]
    private let identity: [String: String]
    private let promptTokenCount: Int?

    public init(readings: [Reading], identity: [String: String] = [:], promptTokenCount: Int? = nil) throws {
        guard Set(readings.map(\.id)).count == readings.count,
              readings.allSatisfy({ !$0.id.isEmpty && !$0.providerID.isEmpty && $0.site.layer >= 0 }),
              promptTokenCount.map({ $0 >= 0 }) ?? true else { throw ConfigurationError.invalidReadings }
        self.readings = readings; self.identity = identity; self.promptTokenCount = promptTokenCount
    }

    public func validate(layerCount: Int) throws {
        lock.lock(); defer { lock.unlock() }
        guard readings.allSatisfy({ $0.site.layer < layerCount }) else { throw ConfigurationError.invalidReadings }
    }

    fileprivate func begin(layerCount: Int) throws {
        lock.lock(); defer { lock.unlock() }
        guard !used else { throw ConfigurationError.reusedScope }
        try validate(layerCount: layerCount)
        used = true
    }

    public func close() {
        lock.lock(); defer { lock.unlock() }
        used = true
        readings.removeAll(); states.removeAll()
    }

    /// Identical owner calls in identical order. No vector folding or dtype change.
    public static func applyLegacy(_ h: MLXArray, interventions: [any LayerIntervention],
                                   layer: Int, offset: Int) -> MLXArray {
        var result = h
        for intervention in interventions {
            result = intervention.apply(result, layer: layer, offset: offset)
        }
        return result
    }

    public func apply(_ h: MLXArray, site: ResidualSite, offset: Int,
                      interventions: [any LayerIntervention] = []) -> MLXArray {
        lock.lock(); defer { lock.unlock() }
        observe(h, site: site, offset: offset, stage: .preAction)
        let result = Self.applyLegacy(h, interventions: interventions, layer: site.layer, offset: offset)
        observe(result, site: site, offset: offset, stage: .postAction)
        return result
    }

    private func observe(_ h: MLXArray, site: ResidualSite, offset: Int, stage: Stage) {
        for reading in readings where reading.site == site && reading.stage == stage {
            // Native decoder inputs are [batch, sequence, hidden]. Padding masks
            // are not inferred here; no padded-study measurement claim is made.
            for index in 0..<h.dim(0) {
                var state = states[reading.providerID]?[index] ?? [:]
                let context = Context(site: site, positions: offset..<(offset + h.dim(1)),
                    promptTokenCount: promptTokenCount, identity: identity, batchIndex: index)
                reading.observe(h[index..<(index + 1)], context, &state)
                states[reading.providerID, default: [:]][index] = state
            }
        }
    }
}

public protocol ResidualRuntimeHookable: InterventionHookable, ResidualShapeProviding {
    var residualRuntime: ResidualRuntime? { get set }
}

extension ResidualRuntimeHookable {
    /// Scoped installation validates sites and clears state on success or error.
    /// Existing action arming remains owned by the caller's model-container scope.
    public func withResidualRuntime<T>(_ runtime: ResidualRuntime, _ body: () throws -> T) throws -> T {
        try runtime.begin(layerCount: residualBlockCount)
        let previous = residualRuntime
        residualRuntime = runtime
        defer { residualRuntime = previous; runtime.close() }
        return try body()
    }
}
