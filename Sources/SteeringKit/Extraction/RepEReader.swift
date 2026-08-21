import CryptoKit
import Foundation
import MLXLMCommon

/// Faithful RepE reader LAT (Zou et al., arXiv:2310.01405 §3.1, App. C.1) —
/// the Swift twin of `Server/steerlab_server/steering/repe_reader.py`, which
/// is the schema/math source of truth. A reader is a fitted **measurement
/// instrument**, not a steering vector: task template + LAT token position +
/// PCA direction with training normalization + held-out scalar accuracy (see
/// `docs/REPE-IMPLEMENTATION-BRIEF.md`). Pipeline:
///
///     stimulus → render task template → capture hidden state at the LAT token
///     → normalized pair differences → centered PCA → PC1 oriented by labels
///     → ScalarProbe fitted on the TRAIN activations
///
/// Inference renders the *same* template, captures the *same* token position,
/// and scores through the stored probe (training center/scale) — never a raw
/// cosine-to-vector shortcut. A reader can *derive* a steering vector
/// (`deriveSteeringVector`), but the derived artifact stamps its reader
/// provenance so "reading-vector activation addition" is never conflated with
/// the paper's full control experiments.
///
/// Concept-agnostic by design: concepts, templates, and stimuli enter as data
/// (`prompts/templates/`, reader pairs JSONL). Reader artifacts are
/// substrate-specific (`RepEReader.substrate`) — activations do not transfer
/// between engines, so a Python/HF reader must be re-fitted here (and vice
/// versa); the shared inputs are the template/dataset bytes, whose SHA-256
/// hashes are identical across engines.
public enum RepEReader {

    public static let artifactType = "repe-reader-lat"
    /// This engine's substrate stamp (the server's is
    /// "python-hf-transformers"). Scoring and manifest verification refuse
    /// readers fitted elsewhere.
    public static let substrate = "swift-mlx"
    /// Honest control-mode label stamped on derived steering vectors.
    public static let controlMode = "reading-vector activation addition"
    /// Rendering contract stamped into every artifact (server twin:
    /// `READER_RENDERING_CONVENTION`; identical semantics, this engine's
    /// extraction path named).
    public static let renderingConvention =
        "rawCompletion scaffold: no chat template, no system role, no family "
        + "thinking suffix; tokenized by the extraction path "
        + "(ConceptExtractor.activations) with the tokenizer's default special "
        + "tokens — single BOS added by the tokenizer, LAT token = final "
        + "scaffold token"

    public struct ReaderError: Error, CustomStringConvertible, Equatable {
        public let reason: String
        public var description: String { reason }

        public init(reason: String) {
            self.reason = reason
        }
    }

    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - Task templates (prompts/templates/<id>.json — shared data files)

    /// One registry entry from `prompts/templates/<id>.json`.
    ///
    /// `hash` is the SHA-256 of the file's raw bytes (stimulus-set
    /// convention): changing the template changes every artifact fitted
    /// through it. `divergence` marks deliberate departures from the paper
    /// (e.g. the unnamed clean-room scaffold, which never names the concept).
    /// JSON keys match the server's `TaskTemplate.to_dict` exactly.
    public struct TaskTemplate: Codable, Sendable, Equatable {
        public var id: String
        public var conceptSlot: Bool
        public var text: String
        public var latToken: String
        /// Not present in the template *file*; carried in embedded copies.
        public var hash: String
        public var divergence: String?

        enum CodingKeys: String, CodingKey {
            case id, conceptSlot, text, latToken, hash, divergence
        }

        public init(
            id: String, conceptSlot: Bool, text: String,
            latToken: String = "final", hash: String = "",
            divergence: String? = nil
        ) {
            self.id = id
            self.conceptSlot = conceptSlot
            self.text = text
            self.latToken = latToken
            self.hash = hash
            self.divergence = divergence
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            id = try container.decode(String.self, forKey: .id)
            conceptSlot = try container.decodeIfPresent(Bool.self, forKey: .conceptSlot) ?? false
            text = try container.decode(String.self, forKey: .text)
            latToken = try container.decodeIfPresent(String.self, forKey: .latToken) ?? "final"
            hash = try container.decodeIfPresent(String.self, forKey: .hash) ?? ""
            divergence = try container.decodeIfPresent(String.self, forKey: .divergence)
        }

        public func readingPosition() throws -> ReadingPosition {
            guard latToken == "final" else {
                throw ReaderError(
                    reason: "template '\(id)': unsupported latToken '\(latToken)' "
                        + "(only 'final' is implemented)")
            }
            return .lastToken
        }

        /// Pure slot substitution; the family-aware scaffold pass happens in
        /// `renderScaffold`.
        public func render(stimulus: String, concept: String? = nil) throws -> String {
            guard text.contains("{{stimulus}}") else {
                throw ReaderError(reason: "template '\(id)' has no {{stimulus}} slot")
            }
            var rendered = text
            if conceptSlot {
                guard let concept, !concept.isEmpty else {
                    throw ReaderError(
                        reason: "template '\(id)' names the concept but none was given")
                }
                rendered = rendered.replacingOccurrences(of: "{{concept}}", with: concept)
            } else if rendered.contains("{{concept}}") {
                throw ReaderError(
                    reason: "template '\(id)' declares conceptSlot=false but contains "
                        + "a {{concept}} slot")
            }
            return rendered.replacingOccurrences(of: "{{stimulus}}", with: stimulus)
        }
    }

    /// Loads one template JSON; hash = SHA-256 over the file's raw bytes.
    /// The registry is one file per id: the id must match the filename.
    public static func loadTemplate(url: URL) throws -> TaskTemplate {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ReaderError(reason: "missing template file: \(url.path)")
        }
        let data = try Data(contentsOf: url)
        var template: TaskTemplate
        do {
            template = try JSONDecoder().decode(TaskTemplate.self, from: data)
        } catch {
            throw ReaderError(reason: "\(url.path): invalid template JSON: \(error)")
        }
        template.hash = sha256Hex(data)
        let expected = url.deletingPathExtension().lastPathComponent
        guard url.pathExtension != "json" || template.id == expected else {
            throw ReaderError(
                reason: "\(url.path): template id '\(template.id)' does not match "
                    + "filename '\(expected)' — the registry is one file per id")
        }
        return template
    }

    // MARK: - Family-aware scaffold pass (server twin: render_reader)

    /// Chat-template / special-token markers that must never appear in a
    /// reader scaffold: the scaffold is tokenized with the tokenizer's
    /// defaults, so an embedded BOS or turn marker is the double-BOS /
    /// hand-tokenized-template hazard this guard exists to prevent.
    static let forbiddenScaffoldMarkers = [
        "<bos>", "<eos>", "<start_of_turn>", "<end_of_turn>",  // Gemma
        "<|im_start|>", "<|im_end|>", "<|endoftext|>",  // Qwen/ChatML
        "</s>",
    ]

    /// Family-aware pass for a RepE reader scaffold (measurement, not chat).
    ///
    /// Reader scaffolds are deliberately **rawCompletion-style plain text**:
    /// the LAT reader reads the hidden state at the scaffold's final token, so
    /// the rendered string must reach the model through exactly the tokenizer
    /// conventions the extractor uses (`ConceptExtractor.activations`
    /// tokenizes with the tokenizer defaults — single BOS, no chat template).
    /// Family notes:
    ///
    /// - **Gemma**: no system role exists and the tokenizer adds BOS itself; a
    ///   scaffold that embeds `<bos>`/turn markers would double them.
    /// - **Qwen**: no ` /no_think` suffix is appended (unlike the
    ///   rawCompletion *generation* path) — nothing is generated, and
    ///   appending it would move the LAT token off the scaffold's final
    ///   position.
    ///
    /// The convention is stamped into every reader artifact as
    /// `renderingConvention`.
    public static func renderReaderScaffold(_ text: String, modelID: String) throws -> String {
        _ = modelID  // both families share the plain-text contract today
        for marker in forbiddenScaffoldMarkers where text.contains(marker) {
            throw ReaderError(
                reason: "reader scaffold embeds special/chat-template marker '\(marker)' — "
                    + "scaffolds must be plain text; the tokenizer adds special tokens "
                    + "exactly once (double-BOS / hand-tokenized-template hazard)")
        }
        if text.drop(while: \.isWhitespace).hasPrefix("<s>") {
            throw ReaderError(
                reason: "reader scaffold embeds a manual '<s>' BOS — the tokenizer adds it")
        }
        return text
    }

    /// Template substitution + the family-aware scaffold pass.
    public static func renderScaffold(
        template: TaskTemplate, stimulus: String, concept: String?, modelID: String
    ) throws -> String {
        try renderReaderScaffold(
            template.render(stimulus: stimulus, concept: concept), modelID: modelID)
    }

    // MARK: - Reader dataset (pairs.jsonl)

    /// One paired row: unrendered stimuli + the template that will render them
    /// (brief §2). Storing the raw stimulus keeps the corpus re-renderable per
    /// model family while the hash pins the bytes. JSON keys match the
    /// server's `pairs.jsonl` contract exactly.
    public struct Pair: Codable, Sendable, Equatable {
        public var id: String?
        public var concept: String
        public var positiveStimulus: String
        public var negativeStimulus: String
        public var topic: String?
        public var split: String
        public var templateID: String

        public init(
            id: String? = nil, concept: String,
            positiveStimulus: String, negativeStimulus: String,
            topic: String? = nil, split: String = "train", templateID: String
        ) {
            self.id = id
            self.concept = concept
            self.positiveStimulus = positiveStimulus
            self.negativeStimulus = negativeStimulus
            self.topic = topic
            self.split = split
            self.templateID = templateID
        }
    }

    public struct Dataset: Sendable, Equatable {
        public let concept: String
        public let pairs: [Pair]
        /// SHA-256 over the file's raw bytes (stimulus-set convention).
        public let hash: String

        public var train: [Pair] { pairs.filter { $0.split == "train" } }
        public var heldOut: [Pair] { pairs.filter { $0.split != "train" } }

        public init(concept: String, pairs: [Pair], hash: String) {
            self.concept = concept
            self.pairs = pairs
            self.hash = hash
        }
    }

    /// Loads `pairs.jsonl`; hash = SHA-256 over raw bytes. Rows must share one
    /// concept — a reader measures exactly one.
    public static func loadPairs(url: URL) throws -> Dataset {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ReaderError(reason: "missing reader pairs file: \(url.path)")
        }
        let data = try Data(contentsOf: url)
        return try parsePairs(data, source: url.path)
    }

    /// Split from file IO so the parse is unit-testable (and reusable by the
    /// Concept Lab editor).
    public static func parsePairs(_ data: Data, source: String) throws -> Dataset {
        struct Row: Decodable {
            let id: String?
            let concept: String?
            let positiveStimulus: String?
            let negativeStimulus: String?
            let topic: String?
            let split: String?
            let templateID: String?
        }
        let decoder = JSONDecoder()
        var pairs: [Pair] = []
        let lines = String(decoding: data, as: UTF8.self)
            .split(separator: "\n", omittingEmptySubsequences: true)
        for raw in lines {
            let line = raw.trimmingCharacters(in: .whitespaces)
            guard !line.isEmpty else { continue }
            guard let row = try? decoder.decode(Row.self, from: Data(line.utf8)) else {
                throw ReaderError(reason: "\(source): invalid JSON line")
            }
            for (key, value) in [
                ("positiveStimulus", row.positiveStimulus),
                ("negativeStimulus", row.negativeStimulus),
                ("concept", row.concept),
                ("templateID", row.templateID),
            ] where value == nil {
                throw ReaderError(reason: "\(source): row missing '\(key)'")
            }
            let split = (row.split?.isEmpty == false ? row.split! : "train").lowercased()
            pairs.append(
                Pair(
                    id: row.id, concept: row.concept!,
                    positiveStimulus: row.positiveStimulus!,
                    negativeStimulus: row.negativeStimulus!,
                    topic: row.topic, split: split, templateID: row.templateID!))
        }
        guard !pairs.isEmpty else {
            throw ReaderError(reason: "empty reader pairs file: \(source)")
        }
        let concepts = Set(pairs.map(\.concept))
        guard concepts.count == 1 else {
            throw ReaderError(
                reason: "\(source): mixed concepts \(concepts.sorted()) — one reader "
                    + "dataset per concept")
        }
        return Dataset(concept: pairs[0].concept, pairs: pairs, hash: sha256Hex(data))
    }

    // MARK: - Reader artifact

    /// The fitted instrument (brief §4): one per concept × layer × template ×
    /// model × substrate. The full template record is embedded so inference is
    /// standalone and drift-proof; `templateID`/`templateHash` remain the
    /// registry pins. The encoded JSON shape matches the server's
    /// `ReaderArtifact.to_dict` byte-for-byte in schema (keys, nested "probe",
    /// nil omission) so artifacts are cross-engine *readable* — but never
    /// cross-engine *usable*: `substrate` gates scoring and verification.
    public struct Artifact: Codable, Sendable, Equatable {
        public var modelID: String
        public var revision: String?
        public var concept: String
        public var layer: Int
        public var template: TaskTemplate
        public var datasetHash: String
        public var probe: SteeringVectorMath.ScalarProbe
        public var pc1ExplainedVariance: Float
        public var trainAccuracy: Float
        public var heldOutAccuracy: Float?
        public var trainPairCount: Int
        public var heldOutPairCount: Int
        public var substrate: String
        public var renderingConvention: String
        public var extractionDate: String

        public var templateID: String { template.id }
        public var templateHash: String { template.hash }
        public var latTokenPosition: String { template.latToken }

        public func readingPosition() throws -> ReadingPosition {
            try template.readingPosition()
        }

        enum CodingKeys: String, CodingKey {
            case artifactType, schemaVersion
            case modelID, revision, substrate, concept, layer
            case templateID, templateHash, template, templateDivergence
            case datasetHash, latTokenPosition, readingPosition
            case probe, pc1ExplainedVariance, trainAccuracy, heldOutAccuracy
            case trainPairCount, heldOutPairCount
            case renderingConvention, extractionDate
        }

        public init(
            modelID: String, revision: String?, concept: String, layer: Int,
            template: TaskTemplate, datasetHash: String,
            probe: SteeringVectorMath.ScalarProbe,
            pc1ExplainedVariance: Float, trainAccuracy: Float,
            heldOutAccuracy: Float?, trainPairCount: Int, heldOutPairCount: Int,
            substrate: String = RepEReader.substrate,
            renderingConvention: String = RepEReader.renderingConvention,
            extractionDate: String = Artifact.timestamp()
        ) {
            self.modelID = modelID
            self.revision = revision
            self.concept = concept
            self.layer = layer
            self.template = template
            self.datasetHash = datasetHash
            self.probe = probe
            self.pc1ExplainedVariance = pc1ExplainedVariance
            self.trainAccuracy = trainAccuracy
            self.heldOutAccuracy = heldOutAccuracy
            self.trainPairCount = trainPairCount
            self.heldOutPairCount = heldOutPairCount
            self.substrate = substrate
            self.renderingConvention = renderingConvention
            self.extractionDate = extractionDate
        }

        public static func timestamp(_ date: Date = Date()) -> String {
            let formatter = ISO8601DateFormatter()
            formatter.formatOptions = [.withInternetDateTime]
            return formatter.string(from: date)
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let kind = try container.decodeIfPresent(String.self, forKey: .artifactType)
            guard kind == RepEReader.artifactType else {
                throw ReaderError(
                    reason: "not a \(RepEReader.artifactType) artifact "
                        + "(artifactType=\(kind ?? "nil"))")
            }
            modelID = try container.decode(String.self, forKey: .modelID)
            revision = try container.decodeIfPresent(String.self, forKey: .revision)
            concept = try container.decode(String.self, forKey: .concept)
            layer = try container.decode(Int.self, forKey: .layer)
            var decodedTemplate = try container.decode(TaskTemplate.self, forKey: .template)
            // The registry pin wins over the embedded copy's own hash field
            // (server `ReaderArtifact.from_dict` behavior).
            if let pinned = try container.decodeIfPresent(String.self, forKey: .templateHash) {
                decodedTemplate.hash = pinned
            }
            template = decodedTemplate
            datasetHash = try container.decode(String.self, forKey: .datasetHash)
            probe = try container.decode(
                SteeringVectorMath.ScalarProbe.self, forKey: .probe)
            pc1ExplainedVariance = try container.decode(
                Float.self, forKey: .pc1ExplainedVariance)
            trainAccuracy = try container.decode(Float.self, forKey: .trainAccuracy)
            heldOutAccuracy = try container.decodeIfPresent(
                Float.self, forKey: .heldOutAccuracy)
            trainPairCount = try container.decodeIfPresent(
                Int.self, forKey: .trainPairCount) ?? 0
            heldOutPairCount = try container.decodeIfPresent(
                Int.self, forKey: .heldOutPairCount) ?? 0
            substrate = try container.decodeIfPresent(String.self, forKey: .substrate) ?? ""
            renderingConvention = try container.decodeIfPresent(
                String.self, forKey: .renderingConvention) ?? ""
            extractionDate = try container.decodeIfPresent(
                String.self, forKey: .extractionDate) ?? ""
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(RepEReader.artifactType, forKey: .artifactType)
            try container.encode(1, forKey: .schemaVersion)
            try container.encode(modelID, forKey: .modelID)
            try container.encodeIfPresent(revision, forKey: .revision)
            try container.encode(substrate, forKey: .substrate)
            try container.encode(concept, forKey: .concept)
            try container.encode(layer, forKey: .layer)
            try container.encode(template.id, forKey: .templateID)
            try container.encode(template.hash, forKey: .templateHash)
            try container.encode(template, forKey: .template)
            try container.encodeIfPresent(template.divergence, forKey: .templateDivergence)
            try container.encode(datasetHash, forKey: .datasetHash)
            try container.encode(template.latToken, forKey: .latTokenPosition)
            try container.encode(try readingPosition().label, forKey: .readingPosition)
            try container.encode(probe, forKey: .probe)
            try container.encode(pc1ExplainedVariance, forKey: .pc1ExplainedVariance)
            try container.encode(trainAccuracy, forKey: .trainAccuracy)
            try container.encodeIfPresent(heldOutAccuracy, forKey: .heldOutAccuracy)
            try container.encode(trainPairCount, forKey: .trainPairCount)
            try container.encode(heldOutPairCount, forKey: .heldOutPairCount)
            try container.encode(renderingConvention, forKey: .renderingConvention)
            try container.encode(extractionDate, forKey: .extractionDate)
        }
    }

    /// Writes `<name>.json` into a run directory; returns the file URL.
    @discardableResult
    public static func saveArtifact(
        _ artifact: Artifact, to directory: URL, name: String? = nil
    ) throws -> URL {
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let name = name ?? "reader-\(artifact.concept)-layer\(artifact.layer)"
        let url = directory.appending(component: "\(name).json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(artifact).write(to: url)
        return url
    }

    public static func loadArtifact(url: URL) throws -> Artifact {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ReaderError(reason: "missing reader artifact: \(url.path)")
        }
        let data = try Data(contentsOf: url)
        do {
            return try JSONDecoder().decode(Artifact.self, from: data)
        } catch let error as ReaderError {
            throw error
        } catch {
            throw ReaderError(reason: "\(url.path): unreadable reader artifact: \(error)")
        }
    }

    // MARK: - Fit

    /// RepE App. C.1 direction: L2-normalized pair differences → centered PCA
    /// → PC1, sign oriented by the paired labels (majority of normalized diffs
    /// project positive; ties fall back to the class-mean criterion). Mirrors
    /// the inner LAT math of `SteeringVectorMath.direction` but returns the
    /// *unit* PC plus its explained variance — the reader does not norm-match
    /// to CAA.
    static func pc1Oriented(
        positive: [[Float]], negative: [[Float]]
    ) throws -> (component: [Float], explainedVariance: Float) {
        guard positive.count == negative.count,
            positive.first?.count == negative.first?.count
        else {
            throw ReaderError(
                reason: "unpaired activations: positive \(positive.count) vs "
                    + "negative \(negative.count)")
        }
        var diffs: [[Float]] = []
        for (p, n) in zip(positive, negative) {
            let d = zip(p, n).map(-)
            let norm = SteeringVectorMath.l2Norm(d)
            if norm > 0 { diffs.append(d.map { $0 / norm }) }
        }
        guard diffs.count >= 2 else {
            throw ReaderError(reason: "need at least 2 non-degenerate train pairs")
        }
        // Alternate orientation before PCA so the shared concept direction is
        // not centered out of normalized labeled pairs (same convention as
        // SteeringVectorMath.direction).
        let oriented = diffs.enumerated().map { index, d in
            index.isMultiple(of: 2) ? d : d.map { -$0 }
        }
        let result: SteeringVectorMath.PrincipalComponentsResult
        do {
            result = try SteeringVectorMath.principalComponentsWithVariance(
                of: oriented, count: 1)
        } catch {
            throw ReaderError(reason: "degenerate pair differences (no PC1)")
        }
        guard var pc = result.components.first,
            let explained = result.explainedVariance.first
        else {
            throw ReaderError(reason: "degenerate pair differences (no PC1)")
        }
        let scores = diffs.map { SteeringVectorMath.dot($0, pc) }
        let positiveScores = scores.count { $0 > 0 }
        let flip: Bool
        if positiveScores * 2 == scores.count {
            let meanDiff = try SteeringVectorMath.meanDifference(
                positive: positive, negative: negative)
            flip = SteeringVectorMath.dot(pc, meanDiff) < 0
        } else {
            flip = positiveScores * 2 < scores.count
        }
        if flip { pc = pc.map { -$0 } }
        return (pc, explained)
    }

    static func pairAccuracy(
        probe: SteeringVectorMath.ScalarProbe,
        positive: [[Float]], negative: [[Float]]
    ) throws -> Float? {
        let total = positive.count + negative.count
        guard total > 0 else { return nil }
        var correct = 0
        for row in positive {
            if try probe.classifiesPositive(row) { correct += 1 }
        }
        for row in negative {
            if try !probe.classifiesPositive(row) { correct += 1 }
        }
        return Float(correct) / Float(total)
    }

    /// Pure fit over pre-captured LAT-token activations — the unit-testable
    /// math half. `capturedValues[textIndex][layer]` follows the rendered-text
    /// order the async `fit` produces: train pairs then held-out pairs,
    /// positive before negative within each pair.
    public static func fit(
        dataset: Dataset,
        template: TaskTemplate,
        capturedValues: [[[Float]]],
        modelID: String,
        revision: String?,
        layers: [Int]? = nil
    ) throws -> [Artifact] {
        for pair in dataset.pairs where pair.templateID != template.id {
            throw ReaderError(
                reason: "pair '\(pair.id ?? "?")' pins template '\(pair.templateID)' "
                    + "but the fit uses '\(template.id)'")
        }
        let train = dataset.train
        let held = dataset.heldOut
        guard train.count >= 2 else {
            throw ReaderError(
                reason: "need at least 2 train pairs, have \(train.count) "
                    + "(rows default to split 'train')")
        }
        guard capturedValues.count == 2 * (train.count + held.count) else {
            throw ReaderError(
                reason: "captured \(capturedValues.count) activations for "
                    + "\(train.count + held.count) pairs — expected two per pair")
        }
        let layerCount = capturedValues.first?.count ?? 0
        guard layerCount > 0 else {
            throw ReaderError(reason: "no layers captured")
        }
        let chosen = layers ?? Array(0 ..< layerCount)

        let nTrain = train.count
        var artifacts: [Artifact] = []
        for layer in chosen {
            guard (0 ..< layerCount).contains(layer) else {
                throw ReaderError(
                    reason: "layer \(layer) out of range 0..\(layerCount - 1)")
            }
            let posTrain = (0 ..< nTrain).map { capturedValues[2 * $0][layer] }
            let negTrain = (0 ..< nTrain).map { capturedValues[2 * $0 + 1][layer] }
            let posHeld = (0 ..< held.count).map { capturedValues[2 * (nTrain + $0)][layer] }
            let negHeld = (0 ..< held.count).map { capturedValues[2 * (nTrain + $0) + 1][layer] }

            let (pc, explained) = try pc1Oriented(positive: posTrain, negative: negTrain)
            // Training normalization: center at the train activation mean,
            // score scale/center from the train projections — the "fit
            // params" the paper's inference reuses on new text.
            let center = try SteeringVectorMath.mean(posTrain + negTrain)
            let probe = try SteeringVectorMath.scalarProbe(
                direction: pc, positive: posTrain, negative: negTrain,
                activationCenter: center)
            let trainAccuracy = try pairAccuracy(
                probe: probe, positive: posTrain, negative: negTrain)
            let heldAccuracy = try pairAccuracy(
                probe: probe, positive: posHeld, negative: negHeld)
            artifacts.append(
                Artifact(
                    modelID: modelID, revision: revision,
                    concept: dataset.concept, layer: layer, template: template,
                    datasetHash: dataset.hash, probe: probe,
                    pc1ExplainedVariance: explained,
                    trainAccuracy: trainAccuracy ?? 0,
                    heldOutAccuracy: heldAccuracy,
                    trainPairCount: nTrain, heldOutPairCount: held.count))
        }
        return artifacts
    }

    /// Fits one reader per layer from the dataset's train split; held-out rows
    /// (any other `split` value) score the instrument they did not fit.
    ///
    /// Rendering goes through `renderScaffold` (family-aware); activations are
    /// captured by the same extraction path every other recipe uses, at the
    /// template's LAT token position.
    public static func fit(
        container: ModelContainer,
        modelID: String,
        revision: String?,
        dataset: Dataset,
        template: TaskTemplate,
        layers: [Int]? = nil
    ) async throws -> [Artifact] {
        let ordered = dataset.train + dataset.heldOut
        var texts: [String] = []
        texts.reserveCapacity(ordered.count * 2)
        for pair in ordered {
            for stimulus in [pair.positiveStimulus, pair.negativeStimulus] {
                texts.append(
                    try renderScaffold(
                        template: template, stimulus: stimulus,
                        concept: dataset.concept, modelID: modelID))
            }
        }
        let captured = try await ConceptExtractor.activations(
            container: container, texts: texts, position: template.readingPosition())
        return try fit(
            dataset: dataset, template: template, capturedValues: captured.values,
            modelID: modelID, revision: revision, layers: layers)
    }

    // MARK: - Exact inference

    /// Pure probe scoring for a pre-captured LAT-token activation.
    public static func scoreActivation(
        _ reader: Artifact, activation: [Float]
    ) throws -> Float {
        try reader.probe.score(activation)
    }

    /// Preconditions for scoring texts through a fitted reader. Internal but
    /// test-visible (pure CPU, no model) so the guard itself is unit-tested:
    /// a reader is fitted per substrate AND per model — its probe reads a
    /// specific model's residual geometry, so scoring through the wrong model
    /// would return silently meaningless numbers, not an error.
    static func validateForScoring(reader: Artifact, modelID: String) throws {
        guard reader.substrate == substrate else {
            throw ReaderError(
                reason: "reader was fitted on substrate '\(reader.substrate)'; this "
                    + "engine is '\(substrate)' — reader artifacts are "
                    + "substrate-specific, re-fit here")
        }
        guard reader.modelID == modelID else {
            throw ReaderError(
                reason: "reader was fitted on model '\(reader.modelID)'; the loaded "
                    + "model is '\(modelID)' — a reader is a per-model measurement "
                    + "instrument, re-fit it for this model")
        }
    }

    /// The paper's inference, exactly: render the SAME template around each
    /// new stimulus, capture the LAT token at the reader's layer, normalize
    /// with the training parameters (inside the probe), and project. Not
    /// cosine-to-vector.
    public static func scoreTexts(
        container: ModelContainer, modelID: String,
        reader: Artifact, texts: [String]
    ) async throws -> [Float] {
        try validateForScoring(reader: reader, modelID: modelID)
        let rendered = try texts.map {
            try renderScaffold(
                template: reader.template, stimulus: $0,
                concept: reader.concept, modelID: modelID)
        }
        let captured = try await ConceptExtractor.activations(
            container: container, texts: rendered,
            position: reader.readingPosition())
        return try captured.values.map { values in
            guard reader.layer < values.count else {
                throw ReaderError(
                    reason: "reader layer \(reader.layer) out of range for a "
                        + "\(values.count)-layer capture — wrong model?")
            }
            return try scoreActivation(reader, activation: values[reader.layer])
        }
    }

    public static func scoreText(
        container: ModelContainer, modelID: String,
        reader: Artifact, text: String
    ) async throws -> Float {
        try await scoreTexts(
            container: container, modelID: modelID, reader: reader, texts: [text])[0]
    }

    // MARK: - Derive-steering conversion (brief §6)

    /// Pure half of the reader → steering-vector conversion: the unit reading
    /// direction placed at the reader's layer (zeros below, Gemma-Scope import
    /// convention) plus a sidecar stamping `source: repe-reader-lat`, the
    /// reader file id + SHA-256, and the honest `controlMode` — because "we
    /// steered with a RepE reader direction" is a different claim from "we
    /// reproduced RepE control".
    public static func deriveSteeringArtifact(
        from reader: Artifact, readerFileName: String, readerBytes: Data
    ) throws -> (vectors: ConceptVectors, sidecar: SteeringVectorSidecar) {
        let direction = reader.probe.direction
        guard !direction.isEmpty else {
            throw ReaderError(reason: "reader probe has an empty direction")
        }
        let hidden = direction.count
        var perLayer = [[Float]](
            repeating: [Float](repeating: 0, count: hidden), count: reader.layer)
        perLayer.append(direction)
        let vectors = ConceptVectors(perLayer: perLayer)
        var sidecar = SteeringVectorSidecar(
            modelID: reader.modelID,
            revision: reader.revision,
            concept: reader.concept,
            stimulusSetHash: reader.datasetHash,
            vectors: vectors,
            appliedReadingPosition: try reader.readingPosition())
        sidecar.extractionMethod = "repeReaderLAT"
        sidecar.source = artifactType
        sidecar.readerID = readerFileName
        sidecar.readerHash = sha256Hex(readerBytes)
        sidecar.controlMode = controlMode
        return (vectors, sidecar)
    }

    /// Converts a saved reader into a standard steering-vector artifact in
    /// `runDirectory` (`<name>.safetensors` + `<name>.json`). Returns the
    /// artifact base URL (`<runDirectory>/<name>`).
    @discardableResult
    public static func deriveSteeringVector(
        readerURL: URL, into runDirectory: URL, name: String? = nil
    ) throws -> URL {
        guard FileManager.default.fileExists(atPath: readerURL.path) else {
            throw ReaderError(reason: "missing reader artifact: \(readerURL.path)")
        }
        let bytes = try Data(contentsOf: readerURL)
        let reader: Artifact
        do {
            reader = try JSONDecoder().decode(Artifact.self, from: bytes)
        } catch let error as ReaderError {
            throw error
        } catch {
            throw ReaderError(
                reason: "\(readerURL.path): unreadable reader artifact: \(error)")
        }
        let (vectors, sidecar) = try deriveSteeringArtifact(
            from: reader, readerFileName: readerURL.lastPathComponent, readerBytes: bytes)
        let name = name ?? "\(reader.concept)-repe-reader"
        try SteeringVectorStore.save(
            vectors: vectors, sidecar: sidecar, to: runDirectory, name: name)
        return runDirectory.appending(component: name)
    }
}
