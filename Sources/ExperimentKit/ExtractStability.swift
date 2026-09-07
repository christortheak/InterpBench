import Foundation
import SteeringKit

/// Native diagnostic owner. Captures each class once, then resamples in memory.
/// It writes diagnostics/, never modifies a manifest or creates a study run.
public enum ExtractStability {
    public static let diagnosticNote = "This is a STABILITY DIAGNOSTIC of an extracted direction under resampling of ITS OWN contrast population — the same stimuli, redrawn. A high cosine means the direction is determined by the contrast rather than by which particular rows were drawn. It is NOT evidence that the direction is the concept: a stable direction can encode a confound the two classes share, and resampling cannot see a confound that is present in every draw. It is NOT behavioral validation: nothing here touches generation, and a direction that is perfectly stable may steer nothing. Any recipe choice made because of these numbers is a SELECTION DECISION and belongs in the study's selection provenance, declared before the evidence run, not discovered after it."
    static let layerKeys: Set<String> = ["degenerateDraws", "meanCosine", "medianCosine", "minCosine",
        "orderShuffleCosines", "percentile5Cosine", "resampleCosines", "signFlips"]

    public struct Failure: Error, Sendable {
        public let code: String
        public let reason: String
        public let repairAction: String
        public var state: SteerLabCLIState = .refused
    }

    public struct Prepared: Sendable {
        public let manifest: ExperimentManifest
        public let ref: ExperimentManifest.ConceptRef
        public let positive: [String]
        public let negative: [String]
        public let stimulus: [String: JSONValue]
    }
    public struct Result: Sendable {
        public let directory: URL
        public let path: URL
        public let summary: [String: JSONValue]
    }

    public static func preflight(experiment: String, concept: String,
        resamples: Int = 32, fraction: Double = 0.5) throws -> Prepared {
        let manifest = try ExperimentStore.load(name: experiment)
        guard let ref = manifest.concepts.first(where: { $0.name == concept }) else {
            throw Failure(code: "notFound", reason: "Study '\(experiment)' has no concept '\(concept)'.", repairAction: "Choose one of: \(manifest.concepts.map(\.name).sorted().joined(separator: ", ")).", state: .notFound)
        }
        guard ref.options.method.isPaired || ref.options.method == .designatedReference else {
            throw Failure(code: "unsupportedMethod", reason: "\(ref.options.method.rawValue) has no two-population direction to resample.", repairAction: "Use this diagnostic on meanDifference, lat, or designatedReference; use the Python science workflow for other method-specific diagnostics.")
        }
        let positive: [String], negative: [String]
        var stimulus: [String: JSONValue] = ["referenceName": .null,
            "referenceStimulusHashLive": .null, "referenceStimulusHashPinned": .null,
            "stimulusHashPinned": .string(ref.stimulusSetHash)]
        if ref.options.method == .designatedReference {
            guard let pin = ref.designatedReference else {
                throw Failure(code: "unpinnedReference", reason: "The designated reference is not pinned.", repairAction: "Re-attach '\(concept)' with a pinned reference before diagnosing it.")
            }
            positive = try ExperimentStore.loadStoriesTexts(for: concept)
            negative = try ExperimentStore.loadStoriesTexts(for: pin.name)
            stimulus["referenceName"] = .string(pin.name)
            stimulus["referenceStimulusHashLive"] = ExperimentStore.storiesHash(for: pin.name).map(JSONValue.string) ?? .null
            stimulus["referenceStimulusHashPinned"] = .string(pin.hash)
            stimulus["stimulusHashLive"] = ExperimentStore.storiesHash(for: concept).map(JSONValue.string) ?? .null
        } else {
            let rows = try StimulusSet(directory: VectorCatalog.conceptsDirectory.appending(component: concept))
            positive = rows.positive; negative = rows.negative
            stimulus["stimulusHashLive"] = .string(rows.hash)
        }
        do {
            try SteeringVectorMath.checkStabilityRequest(method: ref.options.method,
                positiveCount: positive.count, negativeCount: negative.count, resamples: resamples, fraction: fraction)
        } catch {
            throw Failure(code: "usage", reason: String(describing: error),
                repairAction: "Use at least 2 resamples and a fraction in (0, 1] that selects at least 2 rows; paired-difference PCA requires equal class sizes.", state: .blocked)
        }
        return .init(manifest: manifest, ref: ref, positive: positive, negative: negative, stimulus: stimulus)
    }

    public static func run(experiment: String, concept: String, resamples: Int = 32,
        fraction: Double = 0.5, seed: UInt64 = 0, orderShuffles: Int = 8) async throws -> Result {
        let prepared = try preflight(experiment: experiment, concept: concept, resamples: resamples, fraction: fraction)
        let manifest = prepared.manifest
        // A diagnostic observes the recipe; it must not pin a draft as a side effect.
        let container = try await SteeredContainerLoader.load(modelID: manifest.modelID, revision: manifest.modelRevision)
        _ = await ExperimentTasks.ensureModelCapabilities(modelID: manifest.modelID, revision: manifest.modelRevision)
        let options = prepared.ref.options
        let positive = try await ConceptExtractor.activations(container: container, texts: prepared.positive,
            position: options.readingPosition, rendering: options.resolvedExtractionRendering)
        let negative = try await ConceptExtractor.activations(container: container, texts: prepared.negative,
            position: options.readingPosition, rendering: options.resolvedExtractionRendering)
        return try finish(prepared: prepared, positive: positive, negative: negative,
            modelRevision: manifest.modelRevision ?? SteeredContainerLoader.cachedRevision(for: manifest.modelID),
            resamples: resamples, fraction: fraction, seed: seed, orderShuffles: orderShuffles)
    }

    /// Capture injection seam for fixtures; production calls it only with the
    /// activations above. UInt64 seeds remain integers in the report's JSON.
    static func finish(prepared: Prepared, positive: StimulusActivations, negative: StimulusActivations,
        modelRevision: String?, resamples: Int, fraction: Double, seed: UInt64, orderShuffles: Int,
        observedAt: Date = Date(), diagnosticsRoot: URL? = nil) throws -> Result {
        let manifest = prepared.manifest, ref = prepared.ref, options = ref.options
        guard let count = positive.values.first?.count, count > 0,
              positive.values.count == prepared.positive.count,
              negative.values.count == prepared.negative.count,
              (positive.values + negative.values).allSatisfy({ $0.count == count }) else {
            throw ExperimentError(reason: "Capture returned an incomplete layer/row matrix; check the model's activation hooks.")
        }
        let rows = Dictionary(uniqueKeysWithValues: (0 ..< count).map { layer in
            (layer, (positive: positive.values.map { $0[layer] }, negative: negative.values.map { $0[layer] }))
        })
        let readings: [Int: DirectionStability]
        do {
            readings = try SteeringVectorMath.stabilityByLayer(rows, method: options.method,
                resamples: resamples, fraction: fraction, seed: seed, orderShuffles: orderShuffles)
        } catch {
            throw Failure(code: "degenerateData", reason: "Stability could not be measured: \(error)",
                repairAction: "Check that the contrast has nondegenerate stimuli, then rerun the diagnostic.")
        }
        func object<T: Encodable>(_ value: T) throws -> Any {
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(value), options: .fragmentsAllowed)
        }
        let first = try object(readings[0]!) as! [String: Any]
        let layers: [[String: Any]] = try readings.keys.sorted().map { layer in
            var fields = (try object(readings[layer]!) as! [String: Any]).filter { layerKeys.contains($0.key) }
            fields["layer"] = layer
            return fields
        }
        var identity: String?, identityProblem: String?
        do { identity = RecipeIdentity.hash(try RecipeIdentity.required(manifest: manifest, ref: ref)) }
        catch { identityProblem = String(describing: error) }
        let reading = RecipeIdentity.canonicalReading(options.readingPosition)
        let resolution = ReadingPositionResolutionReport.make(position: options.readingPosition,
            rendering: options.resolvedExtractionRendering, resolutions: positive.resolutions + negative.resolutions)
        let document: [String: Any] = [
            "buildCommit": SteerLabVersion.current.split(separator: "+").dropFirst().first.map(String.init) as Any? ?? NSNull(),
            "concept": ref.name, "diagnosticNote": diagnosticNote, "engineVersion": SteerLabVersion.current,
            "experiment": manifest.name, "experimentStatus": manifest.status.rawValue,
            "extractionMethod": options.method.rawValue, "extractionRendering": try object(options.resolvedExtractionRendering),
            "layerCount": count, "layers": layers, "modelID": manifest.modelID,
            "modelRevision": modelRevision as Any? ?? NSNull(), "neutralProjectionApplied": false,
            "observedAt": ISO8601DateFormatter().string(from: observedAt),
            "readingPosition": options.readingPosition.label, "readingPositionMode": reading.0,
            "readingPositionParameter": reading.1 as Any? ?? NSNull(),
            "readingPositionResolution": try object(resolution),
            "recipeIdentityHash": identity as Any? ?? NSNull(), "recipeIdentityUnprovable": identityProblem as Any? ?? NSNull(),
            "resample": first.filter { !layerKeys.contains($0.key) }, "schema": 1,
            "stimulus": try object(prepared.stimulus), "verb": "experiment extract-stability"]
        let parent = diagnosticsRoot ?? ExperimentStore.workspaceRoot.appending(component: "diagnostics")
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        // UUID makes this publication independent of wall-clock resolution.
        let directory = parent.appending(component: "extract-stability-" + UUID().uuidString.lowercased())
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        let path = directory.appending(component: "stability.json")
        let bytes = try JSONSerialization.data(withJSONObject: document, options: [.sortedKeys, .prettyPrinted, .withoutEscapingSlashes])
        try bytes.write(to: path, options: .withoutOverwriting)
        let worst = readings.keys.sorted().min { readings[$0]!.minCosine < readings[$1]!.minCosine }!
        return .init(directory: directory, path: path, summary: [
            "concept": .string(ref.name), "diagnosticNote": .string(diagnosticNote),
            "directory": .string(directory.path), "path": .string(path.path), "experiment": .string(manifest.name),
            "extractionMethod": .string(options.method.rawValue), "layerCount": .number(Double(count)),
            "minCosine": .number(readings[worst]!.minCosine), "worstLayer": .number(Double(worst)),
            "signFlips": .number(Double(readings.values.reduce(0) { $0 + $1.signFlips })),
            "recipeIdentityHash": identity.map(JSONValue.string) ?? .null,
            "stimulusDrift": .bool(prepared.stimulus["stimulusHashLive"] != prepared.stimulus["stimulusHashPinned"])])
    }
}
