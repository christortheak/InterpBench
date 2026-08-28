import CryptoKit
import Foundation
import SteeringKit

/// POLE MIRRORING for steering-vector artifacts — the other end of a
/// contrastive direction, minted as a properly provenanced artifact of its own.
///
/// A CAA direction points from its negative file's pole toward its positive
/// file's pole (`mean(pos) − mean(neg)`). A researcher who wants to inject the
/// OTHER pole has, today, nothing but a negative α — which every downstream
/// surface (sweep grids, alpha ladders, the norm-unit dose policy, the
/// playground) treats as "less of the concept" rather than "the opposite
/// concept", and which no artifact records. Mirroring writes the negation
/// down: the tensors multiplied by −1 at every layer, under a NEW concept
/// name, with a derivation stamp naming the bytes it came from.
///
/// Three decisions carry the honesty of the result.
///
/// **The negation is a SIGN-BIT FLIP, not arithmetic.** The `.safetensors`
/// bytes are copied and each float's IEEE-754 sign bit is XORed, so nothing is
/// decoded and re-encoded through a lossy path, `-0.0` round-trips as `-0.0`,
/// and the transform is an INVOLUTION: mirroring a mirror returns the parent's
/// tensor bytes byte-for-byte (`PoleMirrorTests`). Only the `layer_<i>`
/// tensors are flipped — `neutral_mean_layer_<i>` is the residual stream's own
/// mean at that layer, an absolute activation statistic that has nothing to do
/// with which pole the concept vector points at, and negating it would corrupt
/// the ablation mean-centring that reads it.
///
/// **A new concept name is REQUIRED.** Two artifacts under one concept name
/// pointing in opposite directions is a hazard nothing downstream can detect:
/// every selector, pin, and promotion matcher addresses a direction by
/// concept.
///
/// **`stimulusSetHash` is PRESERVED and qualified.** The mirrored concept's
/// stimuli are the same two files as the source's with the positive/negative
/// roles swapped. Minting a fresh hash would claim different bytes were read;
/// carrying the source's hash silently would claim the same recipe. So the
/// hash travels and `polesSwappedFromSource: true` says what changed about its
/// meaning.
///
/// Server twin: `Server/steerlab_server/steering/pole_mirror.py`.
public enum PoleMirror {

    /// A typed mirroring refusal: which gate declined, why, and the repair.
    ///
    /// `Kind.rawValue` is the stable machine code the CLI puts in
    /// `error.code`; the server twin's `PoleMirrorError.kind` carries the same
    /// strings. Deliberately NOT a `LifecycleGate`: that vocabulary describes a
    /// STUDY's state, and these describe an artifact transform (the same
    /// reasoning `vectors compare`'s non-gate `notFound` follows).
    public struct MirrorError: Error, CustomStringConvertible, Equatable {
        public enum Kind: String, Sendable, Equatable {
            /// The named base path is not a vector artifact pair.
            case sourceNotFound
            /// `--concept` absent, blank, or equal to the source's concept.
            case conceptRequired
            /// The destination already holds an artifact under that name.
            case destinationOccupied
            /// The source is already the mirror of the requested concept.
            case doubleMirror
            /// The pair exists and cannot be read as an artifact.
            case unreadableArtifact
            /// The source's extraction method has no swapped-pole semantics.
            case unmirrorableMethod
        }

        public let kind: Kind
        public let reason: String
        public let repairAction: String
        public var description: String { reason }

        public init(kind: Kind, reason: String, repairAction: String) {
            self.kind = kind
            self.reason = reason
            self.repairAction = repairAction
        }
    }

    // MARK: - Which methods have a mirrored pole at all

    /// The extraction methods a mirrored pole is DEFINED for, decided from the
    /// methods' own properties rather than from a list that drifts as methods
    /// are added: PAIRED (two authored stimulus files, so swapping their roles
    /// is what the negation means) AND source-concept-bearing (so the swapped
    /// files, and a validation.jsonl over them, exist somewhere a study could
    /// pin). Today that is the CAA family, `meanDifference` and
    /// `pairedDifferencePCA`.
    ///
    /// Why the OTHER source-concept-bearing methods are excluded — the
    /// question this restriction had to answer (external review round 8,
    /// finding 2):
    ///
    /// - `designatedReference` is source-concept-bearing but UNPAIRED. Its
    ///   direction is mean(concept stories) − mean(a designated REFERENCE
    ///   corpus's stories), so its negation is "the reference corpus minus the
    ///   concept" — a different comparison, not the concept's opposite pole.
    ///   The reference is a baseline the study designated, not a pole a
    ///   researcher authored as the concept's other end, and nothing in the
    ///   sidecar's `designatedReference {name, hash}` schema can even express
    ///   a swap: the `stimulusSetHash` is the concept's own stories hash, and
    ///   a mirrored artifact would have to claim the reference corpus is now a
    ///   concept with held-out scenarios of its own. Not obviously yes, so:
    ///   excluded, and the refusal says why.
    /// - `emotionGrandMean` negates to "the population mean minus the
    ///   concept", which is generic negation with no second pole anywhere.
    /// - `optvec`, `gemmaScopeSAE` and `repeReaderLAT` have no source concept
    ///   at all — no stimulus files to swap, and a `validation.jsonl` under
    ///   the mirrored name is a file `attachArtifact` pins EXPLICITLY ABSENT,
    ///   so the success message used to promise a workflow attach forbids.
    ///
    /// Server twin: `pole_mirror.mirrorable_methods`.
    public static var mirrorableMethods: [ExtractionMethod] {
        ExtractionMethod.allCases.filter { $0.isPaired && $0.hasSourceConcept }
    }

    /// Those methods' raw values, sorted — the vocabulary the refusal names.
    public static var mirrorableMethodList: String {
        mirrorableMethods.map(\.rawValue).sorted().joined(separator: ", ")
    }

    /// What a successful mint prints, and the reason this type writes NOTHING
    /// into `prompts/concepts/`: the mirrored pole's held-out evidence is a
    /// file only the researcher can author, and an engine that invented it
    /// would be manufacturing the very evidence the validate gate exists to
    /// demand. Server twin: `pole_mirror.validation_authoring_note`.
    public static func validationAuthoringNote(concept: String) -> String {
        "to validate the mirrored pole, author prompts/concepts/\(concept)/"
            + "validation.jsonl — the source concept's rows with every "
            + "expresses label inverted are the natural starting point"
    }

    /// The result of one mint.
    public struct MirrorResult: Sendable, Equatable {
        public let runDirectory: URL
        /// `<runDir>/<name>`, no extension — the catalog id.
        public let artifact: URL
        public let concept: String
        public let sourceArtifact: URL
        public let sourceConcept: String
        public let sourceVectorsHash: String
        public let sourceSidecarHash: String
        public let layerCount: Int
        /// Carried through so the run directory's `config.json` can record the
        /// model these bytes belong to without re-reading the sidecar.
        public let modelID: String?
        public let revision: String?
    }

    // MARK: - Bit-exact negation

    /// `layer_<i>` and nothing else. `neutral_mean_layer_<i>` deliberately
    /// fails this match (see the type's doc comment).
    static func isLayerTensorKey(_ key: String) -> Bool {
        guard key.hasPrefix("layer_") else { return false }
        let index = key.dropFirst("layer_".count)
        return !index.isEmpty && index.allSatisfy(\.isNumber)
    }

    /// Element widths in bytes for the FLOAT dtypes safetensors names.
    /// Integer dtypes are absent on purpose: two's-complement negation is not
    /// a sign-bit flip, so an integer tensor is refused rather than silently
    /// mangled.
    static let floatElementBytes: [String: Int] = [
        "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    ]

    /// One tensor's header entry, decoded through `SidecarJSON` so the header
    /// never has to become `[String: Any]`.
    private struct HeaderEntry: Codable {
        let dtype: String
        let dataOffsets: [Int]

        enum CodingKeys: String, CodingKey {
            case dtype
            case dataOffsets = "data_offsets"
        }
    }

    /// Returns `payload` with every `layer_<i>` float's sign bit flipped.
    ///
    /// Bit-exact and involutive by construction: no float is decoded, so no
    /// rounding, no NaN-payload rewrite, and `-0.0` survives as `-0.0`. Every
    /// other byte — the 8-byte header length, the JSON header, and any
    /// non-`layer_` tensor (notably `neutral_mean_layer_<i>`) — is copied
    /// unchanged. Server twin: `pole_mirror.negate_layer_tensors`.
    public static func negatedTensorBytes(_ payload: Data) throws -> Data {
        var bytes = [UInt8](payload)
        guard bytes.count >= 8 else {
            throw MirrorError(
                kind: .unreadableArtifact,
                reason: "safetensors payload is shorter than its 8-byte header length",
                repairAction: truncatedRepair)
        }
        var headerLength = 0
        for offset in (0 ..< 8).reversed() {
            headerLength = (headerLength << 8) | Int(bytes[offset])
        }
        let headerStart = 8
        guard headerLength >= 0, headerLength <= bytes.count - headerStart else {
            throw MirrorError(
                kind: .unreadableArtifact,
                reason: "safetensors header claims \(headerLength) bytes but only "
                    + "\(bytes.count - headerStart) follow",
                repairAction: truncatedRepair)
        }
        let dataStart = headerStart + headerLength
        let header: [String: SidecarJSON]
        do {
            header = try JSONDecoder().decode(
                [String: SidecarJSON].self,
                from: Data(bytes[headerStart ..< dataStart]))
        } catch {
            throw MirrorError(
                kind: .unreadableArtifact,
                reason: "safetensors header is not readable JSON: \(error)",
                repairAction: truncatedRepair)
        }

        var flipped = 0
        for (key, value) in header.sorted(by: { $0.key < $1.key })
        where isLayerTensorKey(key) {
            guard case .object = value else { continue }
            let entry: HeaderEntry
            do {
                entry = try JSONDecoder().decode(
                    HeaderEntry.self, from: JSONEncoder().encode(value))
            } catch {
                throw MirrorError(
                    kind: .unreadableArtifact,
                    reason: "tensor '\(key)' has no readable dtype/data_offsets",
                    repairAction: truncatedRepair)
            }
            guard let width = floatElementBytes[entry.dtype] else {
                throw MirrorError(
                    kind: .unreadableArtifact,
                    reason: "tensor '\(key)' has dtype '\(entry.dtype)' — mirroring "
                        + "flips IEEE sign bits and is defined only for float "
                        + "tensors (\(floatElementBytes.keys.sorted().joined(separator: ", ")))",
                    repairAction: "re-extract the source artifact as float32")
            }
            guard entry.dataOffsets.count == 2 else {
                throw MirrorError(
                    kind: .unreadableArtifact,
                    reason: "tensor '\(key)' has no readable dtype/data_offsets",
                    repairAction: truncatedRepair)
            }
            let start = entry.dataOffsets[0]
            let stop = entry.dataOffsets[1]
            guard start >= 0, stop >= start, (stop - start) % width == 0,
                stop <= bytes.count - dataStart
            else {
                throw MirrorError(
                    kind: .unreadableArtifact,
                    reason: "tensor '\(key)' data_offsets [\(start), \(stop)] do not "
                        + "describe whole \(entry.dtype) elements inside the payload",
                    repairAction: truncatedRepair)
            }
            // The sign bit is the MSB of the LAST byte of each little-endian
            // element, for every IEEE width safetensors carries.
            var offset = dataStart + start + width - 1
            while offset < dataStart + stop {
                bytes[offset] ^= 0x80
                offset += width
            }
            flipped += 1
        }
        guard flipped > 0 else {
            throw MirrorError(
                kind: .unreadableArtifact,
                reason: "safetensors payload carries no layer_<i> tensors — it is "
                    + "not a steering-vector artifact",
                repairAction: "pass the base path of a vector artifact written by "
                    + "`steerlab-cli experiment extract <name>`")
        }
        return Data(bytes)
    }

    // MARK: - Sidecar

    /// The mirrored artifact's sidecar: the source's, field for field, with
    /// the concept renamed and the derivation stamped.
    ///
    /// The transform runs over the RAW decoded JSON rather than through
    /// `SteeringVectorSidecar`, so blocks this engine does not model survive
    /// untouched — the failure mode `SteeringVectorSidecar.optvec`'s doc
    /// comment records, where a Swift decode→re-encode silently dropped an
    /// artifact's whole OptVec provenance block.
    ///
    /// Everything sign-invariant is preserved verbatim, and most of the
    /// sidecar IS sign-invariant:
    ///
    /// - `normsPerLayer` / `residualNormPerLayer` and the whole
    ///   `residualNorm*` denominator family — an L2 norm does not change when
    ///   the vector it measures is negated, so a mirrored artifact's α in norm
    ///   units means exactly the dose the source's did;
    /// - `readingPosition` / `readingPositionResolution` /
    ///   `extractionRendering` — where and how the activations were read;
    /// - `coversModelDepth`, `layerCount`, `hiddenSize` — the shape is
    ///   untouched;
    /// - `modelID` / `revision` / `substrate` — the same bytes on the same
    ///   model;
    /// - `extractionMethod` / `recipeMethod` / `signConvention` and the
    ///   reader-, SAE- and OptVec-provenance blocks — these describe how the
    ///   SOURCE direction was produced, which is still the true answer to
    ///   "where did these numbers come from"; `negatedFrom` names the artifact
    ///   they describe.
    ///
    /// Exactly one field is DROPPED: `recipeIdentityHash`. That hash is an
    /// identity claim about THESE bytes ("running this recipe produces this
    /// artifact"), its canonical form includes the concept name, and promotion
    /// matches candidates on it (`AgentPromotion`). Carrying the source's hash
    /// onto a renamed, negated artifact would let the matcher treat the mirror
    /// as the source recipe's output — a wrong answer rather than a missing
    /// one.
    public static func mirroredSidecar(
        _ original: [String: SidecarJSON], concept: String,
        sourceArtifact: String, sourceVectorsHash: String,
        sourceSidecarHash: String, date: Date = Date()
    ) -> [String: SidecarJSON] {
        var sidecar = original
        sidecar["concept"] = .string(concept)
        let sourceConcept = stringValue(original["concept"]) ?? ""
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        formatter.timeZone = TimeZone(identifier: "UTC")
        sidecar["negatedFrom"] = .object([
            "path": .string(sourceArtifact),
            "sha256TensorHash": .string(sourceVectorsHash),
            "sha256SidecarHash": .string(sourceSidecarHash),
            "concept": .string(sourceConcept),
            "date": .string(formatter.string(from: date)),
        ])
        // The inherited stimulusSetHash, qualified rather than reminted: same
        // two files, roles swapped (the type's doc comment).
        sidecar["polesSwappedFromSource"] = .bool(true)
        sidecar["recipeIdentityHash"] = nil
        return sidecar
    }

    // MARK: - Mint

    /// Mint the mirrored pole of `artifact` (`<runDir>/<name>`, no extension)
    /// into `runDirectory`. The source is never modified (run directories are
    /// immutable). Returns the new artifact's identity.
    @discardableResult
    public static func mirrorPoles(
        artifact: URL, concept: String, into runDirectory: URL,
        outputName: String? = nil, date: Date = Date()
    ) throws -> MirrorResult {
        let name = artifact.lastPathComponent
        let vectorsSource = artifact.appendingPathExtension("safetensors")
        let sidecarSource = artifact.appendingPathExtension("json")
        let fm = FileManager.default
        guard !name.isEmpty, fm.fileExists(atPath: vectorsSource.path),
            fm.fileExists(atPath: sidecarSource.path)
        else {
            throw MirrorError(
                kind: .sourceNotFound,
                reason: sourceNotFoundReason(base: artifact.path),
                repairAction: sourceNotFoundRepair(program: program))
        }
        let concept = concept.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !concept.isEmpty else {
            throw MirrorError(
                kind: .conceptRequired,
                reason: conceptRequiredReason(sourceConcept: ""),
                repairAction: conceptRequiredRepair(
                    program: program, base: artifact.path))
        }

        let vectorBytes = try Data(contentsOf: vectorsSource)
        let sidecarBytes = try Data(contentsOf: sidecarSource)
        let original: [String: SidecarJSON]
        do {
            original = try JSONDecoder().decode(
                [String: SidecarJSON].self, from: sidecarBytes)
        } catch {
            throw MirrorError(
                kind: .unreadableArtifact,
                reason: "unreadable sidecar \(sidecarSource.path): \(error)",
                repairAction: basePathRepair)
        }
        // `layerCount` is read as a NUMBER here, before a single byte is
        // written. It used to be enough that the key existed, and the value
        // was converted only AFTER both files had landed — so a sidecar
        // carrying a non-numeric layerCount produced a complete artifact pair
        // whose recorded layer count was silently 0. Server twin: the same
        // numeric guard in `pole_mirror.mirror_poles`.
        guard let layerCountValue = original["layerCount"],
            case .number(let layerCountNumber) = layerCountValue
        else {
            throw MirrorError(
                kind: .unreadableArtifact,
                reason: "'\(artifact.path)' is not a steering-vector artifact",
                repairAction: basePathRepair)
        }
        // A NUMBER is not yet a layer count (review round 10, finding 9).
        // `Int(2.5)` truncated to 2 and stamped a mirror claiming a depth its
        // source never had; `Int(0)`/`Int(-3)` stamped an impossible one; and
        // `Int(Double.nan)` / `Int(.infinity)` TRAP — a crash, not a refusal,
        // on a hostile or corrupt sidecar. Finiteness and integer-exactness
        // are therefore checked BEFORE the conversion, and nothing here can
        // trap. The only upper bound is what an `Int` can hold — no other
        // sidecar reader on either engine bounds this key above, and a gate
        // this file made up alone would refuse artifacts every other reader
        // accepts. Server twin: the same conditions in
        // `pole_mirror.mirror_poles`.
        if let problem = layerCountProblem(
            layerCountNumber, path: artifact.path)
        {
            throw MirrorError(
                kind: .unreadableArtifact, reason: problem,
                repairAction: basePathRepair)
        }
        // The conversion is written as `Int(exactly:)` — the same gate the
        // predicate just applied — so the trap is not merely unreachable, it
        // does not exist. `Int(_:)` stood here and traps on a value outside
        // `Int`'s range (review round 11, finding 3).
        guard let layerCount = Int(exactly: layerCountNumber) else {
            throw MirrorError(
                kind: .unreadableArtifact,
                reason: layerCountReason(
                    path: artifact.path, value: layerCountNumber),
                repairAction: basePathRepair)
        }
        // Which methods HAVE an opposite pole (see `mirrorableMethods`). This
        // gate is the reason the success message's validation-authoring note
        // is now always honest: it can only be printed for a method whose
        // mirrored concept can pin a validation.jsonl at attach.
        let recordedMethod = (stringValue(original["extractionMethod"]) ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let sourceMethod = ExtractionMethod(rawValue: recordedMethod)
        guard let sourceMethod, sourceMethod.isPaired,
            sourceMethod.hasSourceConcept
        else {
            throw MirrorError(
                kind: .unmirrorableMethod,
                reason: unmirrorableMethodReason(
                    base: artifact.path, method: recordedMethod,
                    label: sourceMethod?.label),
                repairAction: unmirrorableMethodRepair)
        }
        let sourceConcept = stringValue(original["concept"]) ?? ""
        guard sourceConcept != concept else {
            throw MirrorError(
                kind: .conceptRequired,
                reason: conceptRequiredReason(sourceConcept: sourceConcept),
                repairAction: conceptRequiredRepair(
                    program: program, base: artifact.path))
        }
        // Double mirror: this artifact is ALREADY the negation of the concept
        // being asked for, so the thing being requested exists and has a name.
        if let stamp = original["negatedFrom"], case .object(let existing) = stamp,
            stringValue(existing["concept"]) == concept
        {
            let parent = stringValue(existing["path"]) ?? ""
            throw MirrorError(
                kind: .doubleMirror,
                reason: doubleMirrorReason(
                    base: artifact.path, concept: concept, parent: parent),
                repairAction: doubleMirrorRepair(parent: parent))
        }

        let outName = outputName ?? concept
        guard !outName.isEmpty, !outName.contains("/"), outName != ".",
            outName != ".."
        else {
            throw MirrorError(
                kind: .conceptRequired,
                reason: "--output-name '\(outName)' must be a plain file-name "
                    + "component",
                repairAction: "pass --output-name <name> with no path separators, "
                    + "or omit it and the mirrored concept name is used")
        }

        try fm.createDirectory(at: runDirectory, withIntermediateDirectories: true)
        let vectorsPath = runDirectory.appending(component: "\(outName).safetensors")
        let sidecarPath = runDirectory.appending(component: "\(outName).json")
        // No-replace, the house rule for artifacts: a mirror that overwrote one
        // would destroy provenance nothing else records.
        for path in [vectorsPath, sidecarPath] where fm.fileExists(atPath: path.path) {
            throw MirrorError(
                kind: .destinationOccupied,
                reason: destinationOccupiedReason(path: path.path),
                repairAction: destinationOccupiedRepair)
        }

        let mirroredBytes = try negatedTensorBytes(vectorBytes)
        let vectorsHash = sha256Hex(vectorBytes)
        let sidecarHash = sha256Hex(sidecarBytes)
        let sidecar = mirroredSidecar(
            original, concept: concept, sourceArtifact: artifact.path,
            sourceVectorsHash: vectorsHash, sourceSidecarHash: sidecarHash,
            date: date)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let sidecarBytesOut = try encoder.encode(sidecar)

        // An artifact is a PAIR, so it is written as one. Both files land
        // under temporary names in the destination directory and are promoted
        // by rename only once both are on disk: a failure between the two
        // writes — a full disk, a permission change, an interrupt — used to
        // strand a tensor with no sidecar, which the catalog reads as an
        // unreadable artifact and which the `destinationOccupied` rule then
        // refuses to replace. The cleanup removes exactly the two temporary
        // names on EVERY failure path, not only the typed refusals: an
        // untyped write error is precisely the failure this exists for.
        // Server twin: `pole_mirror.mirror_poles`.
        let token = UUID().uuidString
        let vectorsTemp = runDirectory.appending(
            component: "\(outName).safetensors.\(token).partial")
        let sidecarTemp = runDirectory.appending(
            component: "\(outName).json.\(token).partial")
        //
        // The occupancy check above is a PREFLIGHT — it runs before the
        // tensors are negated and both temporaries are written — so a
        // destination can appear in the window between it and the promotion.
        // `FileManager.moveItem` cannot overwrite (it throws
        // `NSFileWriteFileExists` and leaves both files exactly as they were,
        // verified 2026-08-27), so unlike the server's `os.replace` this
        // engine never destroyed a thing; what it did was answer the race
        // with a raw Cocoa error instead of the refusal the same collision
        // gets one moment earlier. Named here, so both engines say the same
        // sentence to the same event (review round 9, finding 7).
        do {
            try mirroredBytes.write(to: vectorsTemp)
            try sidecarBytesOut.write(to: sidecarTemp)
            try commitNoReplace(from: vectorsTemp, to: vectorsPath)
            do {
                try commitNoReplace(from: sidecarTemp, to: sidecarPath)
            } catch {
                // The tensor is already promoted; take it back out so a failed
                // mint never leaves half an artifact under the final names.
                // A name THIS call created — whatever beat us to the sidecar
                // is untouched.
                try? fm.removeItem(at: vectorsPath)
                throw error
            }
        } catch {
            try? fm.removeItem(at: vectorsTemp)
            try? fm.removeItem(at: sidecarTemp)
            throw error
        }

        return MirrorResult(
            runDirectory: runDirectory,
            artifact: runDirectory.appending(component: outName),
            concept: concept, sourceArtifact: artifact,
            sourceConcept: sourceConcept, sourceVectorsHash: vectorsHash,
            sourceSidecarHash: sidecarHash, layerCount: layerCount,
            modelID: stringValue(original["modelID"]),
            revision: stringValue(original["revision"]))
    }

    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// One string out of an opaque sidecar value, or nil.
    static func stringValue(_ value: SidecarJSON?) -> String? {
        guard let value, case .string(let text) = value else { return nil }
        return text
    }

    // MARK: - Refusal texts (cross-engine literals; server twin: pole_mirror)
    //
    // One function per refusal, on both engines, because these sentences are
    // the product: an agent reading two engines' logs must read one sentence,
    // and a sentence duplicated at its two throw sites drifts.

    static let program = "steerlab-cli"

    static let artifactShape =
        "a vector artifact is <runDir>/<name>.safetensors PLUS its "
        + "<runDir>/<name>.json sidecar"

    static let truncatedRepair =
        "re-extract the source artifact; its .safetensors is corrupt"

    static let basePathRepair =
        "pass the base path of a vector artifact — <runDir>/<name> with no "
        + "extension"

    public static func sourceNotFoundReason(base: String) -> String {
        "no vector artifact at '\(base)' — \(artifactShape), and the reference "
            + "is the base path they share, with no extension"
    }

    public static func sourceNotFoundRepair(program: String) -> String {
        "pass <runDir>/<name> as `\(program) vectors mirror-poles` prints it, "
            + "or list the run directories under runs/"
    }

    public static func conceptRequiredReason(sourceConcept: String) -> String {
        let same =
            sourceConcept.isEmpty
            ? ""
            : " (you passed '\(sourceConcept)', which is the source's own name)"
        return "--concept <newName> is required and must differ from the "
            + "source's concept\(same) — the mirrored pole is a DIFFERENT "
            + "concept. A contrastive direction points from its negative "
            + "file's pole toward its positive file's pole, so its negation "
            + "points at the opposite pole; writing that under the source's "
            + "name would leave two artifacts with one concept name pointing "
            + "opposite ways, and every selector, pin, and promotion matcher "
            + "addresses a direction by concept"
    }

    public static func conceptRequiredRepair(program: String, base: String) -> String {
        "\(program) vectors mirror-poles \(base) --concept <a name for the "
            + "opposite pole>"
    }

    /// Promote one staged file, incapable of overwriting what stands at the
    /// destination — and, when something does, answering with the same typed
    /// refusal the preflight gives rather than a raw Cocoa error.
    ///
    /// `FileManager.moveItem` is already a no-replace primitive on this
    /// platform: an occupied destination throws `NSFileWriteFileExists` (516)
    /// and leaves both the destination and the staged file untouched. That is
    /// what this engine has always relied on — it is the reason a mirror
    /// could never destroy an artifact the way the server's `os.replace`
    /// could — so the fix here is the SENTENCE, not the primitive. Server
    /// twin: `pole_mirror._commit_no_replace`, where the primitive had to
    /// change too.
    static func commitNoReplace(from staged: URL, to destination: URL) throws {
        do {
            try FileManager.default.moveItem(at: staged, to: destination)
        } catch let error as CocoaError where error.code == .fileWriteFileExists {
            throw MirrorError(
                kind: .destinationOccupied,
                reason: destinationOccupiedReason(path: destination.path),
                repairAction: destinationOccupiedRepair)
        }
    }

    /// Whether a sidecar's numeric `layerCount` is a layer count, as the
    /// refusal sentence or nil.
    ///
    /// Pure, and separated from the read for two reasons: `Int(_:)` TRAPS on
    /// a non-finite Double, so nothing may convert before this runs; and
    /// Foundation's `JSONDecoder` refuses the `NaN`/`Infinity` literals
    /// outright (that sidecar is "unreadable" one gate earlier), which makes
    /// the non-finite branch unreachable through a JSON file on this engine
    /// and testable only here. The Python twin's decoder DOES accept them, so
    /// the branch is load-bearing there — one rule, both engines, whichever
    /// gate happens to see it first.
    ///
    /// No upper bound beyond what `Int` can hold is invented: no other
    /// sidecar reader on either engine bounds this key above. That bound is
    /// asked of `Int` itself, via `Int(exactly:)`. It used to be written as
    /// `value <= Double(Int.max)`, which admits exactly 2^63 — `Double` cannot
    /// hold `Int.max`, so the conversion rounds UP to 2^63 and the comparison
    /// passes a value `Int(_:)` then TRAPS on (review round 11, finding 3).
    /// `Int(exactly:)` answers nil instead, which is this typed refusal.
    static func layerCountProblem(_ value: Double, path: String) -> String? {
        guard value.isFinite, value == value.rounded(.towardZero), value >= 1,
            Int(exactly: value) != nil
        else { return layerCountReason(path: path, value: value) }
        return nil
    }

    /// A sidecar `layerCount` that is a number but not a layer count. The
    /// value is NAMED — a refusal about a field has to say what it read.
    /// Server twin: `pole_mirror._layer_count_reason`.
    public static func layerCountReason(path: String, value: Double) -> String {
        let spelled: String
        if value.isNaN {
            spelled = "NaN"
        } else if value.isInfinite {
            spelled = value < 0 ? "-Infinity" : "Infinity"
        } else if value == value.rounded(.towardZero),
            abs(value) < 1e15
        {
            spelled = String(Int(value))
        } else {
            spelled = String(value)
        }
        return "'\(path)' records layerCount \(spelled) — a steering-vector "
            + "artifact's layer count is a whole number of layers, 1 or more"
    }

    public static func destinationOccupiedReason(path: String) -> String {
        "'\(path)' already exists — mirroring never replaces an artifact (run "
            + "directories are immutable)"
    }

    public static let destinationOccupiedRepair =
        "write the mirror into a fresh run directory, or pass --output-name "
        + "<name> to give it a name that is free there"

    public static func doubleMirrorReason(
        base: String, concept: String, parent: String
    ) -> String {
        "'\(base)' is ITSELF the mirror of concept '\(concept)' — its "
            + "negatedFrom names '\(parent)'. Mirroring it back would mint a "
            + "third copy of a direction that already exists on disk"
    }

    public static func doubleMirrorRepair(parent: String) -> String {
        "use the original artifact at '\(parent)' instead of mirroring this one"
    }

    public static func unmirrorableMethodReason(
        base: String, method: String, label: String?
    ) -> String {
        let recorded: String
        if method.isEmpty {
            recorded = "records no extractionMethod"
        } else if let label {
            recorded = "records extractionMethod '\(method)' (\(label))"
        } else {
            recorded = "records extractionMethod '\(method)', which this "
                + "engine does not know"
        }
        return "'\(base)' \(recorded) — mirror-poles mints the opposite pole "
            + "only for a PAIRED, source-concept-bearing contrast "
            + "(\(mirrorableMethodList)), where the two poles ARE two authored "
            + "stimulus files and swapping their roles is exactly what the "
            + "negation MEANS. Every other direction negates GENERICALLY, with "
            + "no method-specific evidence semantics: a grand-mean or "
            + "class-vs-reference direction negates to 'the population (or the "
            + "designated reference corpus) minus the concept', which is a "
            + "different comparison and not the concept's opposite pole, and a "
            + "direction with no source concept has no stimulus files to swap "
            + "at all — so the validation.jsonl a mirrored pole is told to "
            + "author would be a file attach pins as EXPLICITLY ABSENT, and "
            + "authoring it later is drift"
    }

    public static let unmirrorableMethodRepair =
        "inject the opposite end of this direction with a NEGATIVE α in a "
        + "study condition — the sign flip is available there, needs no new "
        + "artifact, and claims no evidence the method cannot supply"
}
