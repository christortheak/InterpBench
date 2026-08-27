import CryptoKit
import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// POLE MIRRORING — `PoleMirror` and its `vectors mirror-poles` verb.
///
/// What has to be true for a mirrored artifact to be citable:
///
/// - the tensors are the parent's, negated at EVERY layer, BIT-EXACTLY —
///   asserted at the byte level via the double-negation involution, not merely
///   by comparing decoded floats, because "close enough after a round trip" is
///   exactly the claim a provenance stamp must not launder;
/// - the residual-mean tensors are NOT negated (they are an absolute
///   activation statistic, not a direction);
/// - everything sign-invariant survives, `negatedFrom` +
///   `polesSwappedFromSource` are stamped, and `recipeIdentityHash` — the one
///   field that would be a false identity claim about the new bytes — is
///   dropped;
/// - the result is an ordinary catalog artifact;
/// - and the refusals are typed, with the `--concept` one explaining WHY.
///
/// Every expectation here is duplicated in `Server/tests/test_pole_mirror.py`.
/// Two literals, one contract — the parity idiom this repo uses everywhere the
/// engines cannot import each other.
@Suite struct PoleMirrorTests {

    /// Neutral fixture concepts — a made-up contrast with no study attached.
    static let sourceConcept = "brightness"
    static let mirrorConcept = "dimness"

    /// The float payload deliberately carries the values a lossy negation path
    /// would betray: signed zeros (which must come back as the OTHER signed
    /// zero, and back again) and a tiny magnitude.
    static let layer0: [Float] = [0.0, -0.0, 1.5, -2.25, 1e-8]
    static let layer1: [Float] = [3.0, -4.0, 0.0, 0.5, -0.5]

    /// A two-layer artifact written through the STANDARD save path, plus the
    /// sidecar extras a mirror has to preserve (including a field this engine
    /// does not model).
    @discardableResult
    static func writeArtifact(
        into directory: URL, name: String = sourceConcept,
        neutralMean: Bool = true
    ) throws -> URL {
        let vectors = ConceptVectors(perLayer: [layer0, layer1])
        var sidecar = SteeringVectorSidecar(
            modelID: "org/m", revision: "abc", concept: sourceConcept,
            stimulusSetHash: "stim-hash", vectors: vectors,
            residualNormPerLayer: [11.0, 12.0],
            residualNormSource: "neutral-corpus",
            residualNormConvention: ResidualNormConvention.current)
        sidecar.extractionMethod = "meanDifference"
        sidecar.recipeMethod = VectorExtractionRecipe.Method.caaMeanDifference.rawValue
        sidecar.coversModelDepth = true
        sidecar.recipeIdentityHash = String(repeating: "d", count: 64)
        try SteeringVectorStore.save(
            vectors: vectors, sidecar: sidecar, to: directory, name: name,
            neutralMeanPerLayer: neutralMean
                ? [[Float](repeating: 9, count: 5), [Float](repeating: -8, count: 5)]
                : nil)
        return directory.appending(component: name)
    }

    static func rawSidecar(at base: URL) throws -> [String: SidecarJSON] {
        try JSONDecoder().decode(
            [String: SidecarJSON].self,
            from: Data(contentsOf: base.appendingPathExtension("json")))
    }

    static func tensorBytes(at base: URL) throws -> Data {
        try Data(contentsOf: base.appendingPathExtension("safetensors"))
    }

    func withTempDirectory<T>(_ body: (URL) throws -> T) rethrows -> T {
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "pole-mirror-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: temp) }
        return try body(temp)
    }

    // MARK: - The bytes

    @Test func everyLayerIsTheParentsNegation() throws {
        try withTempDirectory { temp in
            let source = try Self.writeArtifact(into: temp.appending(component: "src"))
            let result = try PoleMirror.mirrorPoles(
                artifact: source, concept: Self.mirrorConcept,
                into: temp.appending(component: "out"))

            let parent = try SteeringVectorStore.load(
                from: source.deletingLastPathComponent(),
                name: source.lastPathComponent)
            let mirrored = try SteeringVectorStore.load(
                from: result.runDirectory, name: Self.mirrorConcept)
            #expect(mirrored.vectors.layerCount == parent.vectors.layerCount)
            for layer in 0 ..< parent.vectors.layerCount {
                let before = parent.vectors.perLayer[layer]
                let after = mirrored.vectors.perLayer[layer]
                #expect(after == before.map { -$0 })
                // Signed zero is a SIGN, and it flips like every other one.
                #expect(
                    after.map { $0.sign } != before.map { $0.sign }
                        || before.isEmpty)
            }
        }
    }

    @Test func neutralMeanTensorsAreNotNegated() throws {
        // The residual mean is the stream's own centre at that layer — an
        // absolute activation statistic that has nothing to do with which pole
        // the concept vector points at. Negating it would corrupt the ablation
        // mean-centring that reads it.
        try withTempDirectory { temp in
            let source = try Self.writeArtifact(into: temp.appending(component: "src"))
            let result = try PoleMirror.mirrorPoles(
                artifact: source, concept: Self.mirrorConcept,
                into: temp.appending(component: "out"))

            let before = try SteeringVectorStore.loadNeutralMean(
                from: source.deletingLastPathComponent(),
                name: source.lastPathComponent)
            let after = try SteeringVectorStore.loadNeutralMean(
                from: result.runDirectory, name: Self.mirrorConcept)
            #expect(after == before)
            #expect(before != nil)
        }
    }

    @Test func doubleNegationReturnsTheParentsTensorBytes() throws {
        // THE bit-exactness assertion, and it is at the BYTE level on purpose:
        // a sign-bit flip is an involution, so anything that decoded and
        // re-encoded the floats (or normalised a NaN payload, or coerced −0.0
        // to 0.0) shows up here as a byte diff even when every decoded value
        // compares equal.
        try withTempDirectory { temp in
            let source = try Self.writeArtifact(into: temp.appending(component: "src"))
            let first = try PoleMirror.mirrorPoles(
                artifact: source, concept: Self.mirrorConcept,
                into: temp.appending(component: "one"))
            let second = try PoleMirror.mirrorPoles(
                artifact: first.artifact, concept: "brightness-again",
                into: temp.appending(component: "two"))

            let original = try Self.tensorBytes(at: source)
            #expect(try Self.tensorBytes(at: second.artifact) == original)
            // …and the single negation genuinely changed them.
            #expect(try Self.tensorBytes(at: first.artifact) != original)
        }
    }

    @Test func theHeaderBytesAreUntouched() throws {
        try withTempDirectory { temp in
            let source = try Self.writeArtifact(into: temp.appending(component: "src"))
            let result = try PoleMirror.mirrorPoles(
                artifact: source, concept: Self.mirrorConcept,
                into: temp.appending(component: "out"))
            let original = [UInt8](try Self.tensorBytes(at: source))
            let mirrored = [UInt8](try Self.tensorBytes(at: result.artifact))
            #expect(mirrored.count == original.count)
            var headerLength = 0
            for offset in (0 ..< 8).reversed() {
                headerLength = (headerLength << 8) | Int(original[offset])
            }
            #expect(
                Array(mirrored.prefix(8 + headerLength))
                    == Array(original.prefix(8 + headerLength)))
        }
    }

    @Test func aPayloadWithoutLayerTensorsIsRefused() throws {
        // A header carrying only `neutral_mean_layer_0` has nothing to mirror,
        // and a silent no-op would be a "mirror" identical to its parent.
        let header = #"{"neutral_mean_layer_0":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#
        var bytes = [UInt8]()
        let headerBytes = [UInt8](header.utf8)
        var length = UInt64(headerBytes.count)
        for _ in 0 ..< 8 {
            bytes.append(UInt8(length & 0xFF))
            length >>= 8
        }
        bytes.append(contentsOf: headerBytes)
        bytes.append(contentsOf: [0, 0, 0, 0])
        #expect(throws: PoleMirror.MirrorError.self) {
            try PoleMirror.negatedTensorBytes(Data(bytes))
        }
    }

    // MARK: - The sidecar

    @Test func sidecarPreservesEverySignInvariantField() throws {
        try withTempDirectory { temp in
            let source = try Self.writeArtifact(into: temp.appending(component: "src"))
            let result = try PoleMirror.mirrorPoles(
                artifact: source, concept: Self.mirrorConcept,
                into: temp.appending(component: "out"))

            let parent = try Self.rawSidecar(at: source)
            let mirrored = try Self.rawSidecar(at: result.artifact)
            // Norms are SIGN-INVARIANT: ‖−v‖ = ‖v‖, so a mirrored artifact's α
            // in norm units means exactly the dose the parent's did, and the
            // whole residualNorm* denominator family travels verbatim.
            for key in [
                "normsPerLayer", "residualNormPerLayer", "residualNormSource",
                "residualNormConvention", "readingPosition", "coversModelDepth",
                "modelID", "revision", "substrate", "extractionMethod",
                "recipeMethod", "extractionDate", "hiddenSize", "layerCount",
                "neutralMeanSource", "stimulusSetHash",
            ] {
                #expect(mirrored[key] == parent[key], "\(key)")
            }
            // Every key the parent had, minus the one identity claim that would
            // now be false, plus the two stamps.
            #expect(
                Set(mirrored.keys).subtracting(parent.keys)
                    == ["negatedFrom", "polesSwappedFromSource"])
            #expect(
                Set(parent.keys).subtracting(mirrored.keys)
                    == ["recipeIdentityHash"])
        }
    }

    @Test func theDerivationBlockNamesTheParentBytes() throws {
        try withTempDirectory { temp in
            let source = try Self.writeArtifact(into: temp.appending(component: "src"))
            let result = try PoleMirror.mirrorPoles(
                artifact: source, concept: Self.mirrorConcept,
                into: temp.appending(component: "out"))

            let sidecar = try JSONDecoder().decode(
                SteeringVectorSidecar.self,
                from: Data(
                    contentsOf: result.artifact.appendingPathExtension("json")))
            #expect(sidecar.concept == Self.mirrorConcept)
            #expect(sidecar.polesSwappedFromSource == true)
            // The mirrored pole's stimuli ARE the parent's two files with the
            // roles swapped: a fresh hash would claim different bytes were
            // read, and the parent's hash carried silently would claim the same
            // recipe.
            #expect(sidecar.stimulusSetHash == "stim-hash")
            let stamp = try #require(sidecar.negatedFrom)
            #expect(stamp.concept == Self.sourceConcept)
            #expect(stamp.path == source.path)
            // The hashes are the PARENT's bytes, which is what makes the stamp
            // checkable: negate the named bytes and you get these bytes back.
            #expect(
                stamp.sha256TensorHash
                    == PoleMirror.sha256Hex(try Self.tensorBytes(at: source)))
            #expect(
                stamp.sha256SidecarHash
                    == PoleMirror.sha256Hex(
                        try Data(
                            contentsOf: source.appendingPathExtension("json"))))
            #expect(stamp.date.hasSuffix("Z"))
        }
    }

    @Test func recipeIdentityHashIsDropped() throws {
        // It is an identity claim about THESE bytes ("this recipe produces this
        // artifact"), its canonical form includes the concept name, and
        // promotion matches candidates on it — so carrying the parent's onto a
        // renamed, negated artifact is a wrong answer rather than a missing one.
        try withTempDirectory { temp in
            let source = try Self.writeArtifact(into: temp.appending(component: "src"))
            let result = try PoleMirror.mirrorPoles(
                artifact: source, concept: Self.mirrorConcept,
                into: temp.appending(component: "out"))
            let sidecar = try JSONDecoder().decode(
                SteeringVectorSidecar.self,
                from: Data(
                    contentsOf: result.artifact.appendingPathExtension("json")))
            #expect(sidecar.recipeIdentityHash == nil)
        }
    }

    @Test func theSourceIsNeverModified() throws {
        try withTempDirectory { temp in
            let source = try Self.writeArtifact(into: temp.appending(component: "src"))
            let before = (
                try Self.tensorBytes(at: source),
                try Data(contentsOf: source.appendingPathExtension("json"))
            )
            _ = try PoleMirror.mirrorPoles(
                artifact: source, concept: Self.mirrorConcept,
                into: temp.appending(component: "out"))
            #expect(try Self.tensorBytes(at: source) == before.0)
            #expect(
                try Data(contentsOf: source.appendingPathExtension("json"))
                    == before.1)
        }
    }

    // MARK: - Refusals

    private func refusal(
        _ body: () throws -> Void
    ) throws -> PoleMirror.MirrorError {
        do {
            try body()
        } catch let error as PoleMirror.MirrorError {
            return error
        }
        Issue.record("expected a mirroring refusal")
        throw PoleMirror.MirrorError(
            kind: .unreadableArtifact, reason: "unreached", repairAction: "")
    }

    @Test func aMissingSourceRefusesWithTheArtifactShape() throws {
        try withTempDirectory { temp in
            let error = try refusal {
                try PoleMirror.mirrorPoles(
                    artifact: temp.appending(components: "nowhere", "x"),
                    concept: Self.mirrorConcept,
                    into: temp.appending(component: "out"))
            }
            #expect(error.kind == .sourceNotFound)
            #expect(error.reason.contains(".safetensors PLUS its"))
        }
    }

    @Test func theSourceConceptIsNotAnAcceptableNewName() throws {
        try withTempDirectory { temp in
            let source = try Self.writeArtifact(into: temp.appending(component: "src"))
            let error = try refusal {
                try PoleMirror.mirrorPoles(
                    artifact: source, concept: Self.sourceConcept,
                    into: temp.appending(component: "out"))
            }
            #expect(error.kind == .conceptRequired)
            #expect(error.reason.contains("pointing opposite ways"))
        }
    }

    @Test func aBlankConceptRefusesWithTheReasonItIsRequired() throws {
        try withTempDirectory { temp in
            let source = try Self.writeArtifact(into: temp.appending(component: "src"))
            let error = try refusal {
                try PoleMirror.mirrorPoles(
                    artifact: source, concept: "   ",
                    into: temp.appending(component: "out"))
            }
            #expect(error.kind == .conceptRequired)
        }
    }

    @Test func mirroringNeverReplacesAnArtifact() throws {
        try withTempDirectory { temp in
            let source = try Self.writeArtifact(into: temp.appending(component: "src"))
            let destination = temp.appending(component: "out")
            _ = try PoleMirror.mirrorPoles(
                artifact: source, concept: Self.mirrorConcept, into: destination)
            let error = try refusal {
                try PoleMirror.mirrorPoles(
                    artifact: source, concept: Self.mirrorConcept,
                    into: destination)
            }
            #expect(error.kind == .destinationOccupied)
            #expect(error.reason.contains("never replaces an artifact"))
        }
    }

    @Test func aDoubleMirrorNamesTheOriginal() throws {
        try withTempDirectory { temp in
            let source = try Self.writeArtifact(into: temp.appending(component: "src"))
            let first = try PoleMirror.mirrorPoles(
                artifact: source, concept: Self.mirrorConcept,
                into: temp.appending(component: "one"))
            let error = try refusal {
                try PoleMirror.mirrorPoles(
                    artifact: first.artifact, concept: Self.sourceConcept,
                    into: temp.appending(component: "two"))
            }
            #expect(error.kind == .doubleMirror)
            #expect(error.repairAction.contains(source.path))
        }
    }

    @Test func anOutputNameMustBeAFileNameComponent() throws {
        try withTempDirectory { temp in
            let source = try Self.writeArtifact(into: temp.appending(component: "src"))
            let error = try refusal {
                try PoleMirror.mirrorPoles(
                    artifact: source, concept: Self.mirrorConcept,
                    into: temp.appending(component: "out"), outputName: "a/b")
            }
            #expect(error.kind == .conceptRequired)
        }
    }
}

/// The verb, the catalog, and the one lifecycle question a mirrored artifact
/// raises: it has no `validation.jsonl` of its own, and the machinery that says
/// so honestly is the machinery that already exists.
///
/// Serialized and holding `ExperimentRootOverrideLock`: `WorkspaceRoot`'s
/// override is a process-global seam shared with every other lifecycle suite.
@Suite(.serialized) struct PoleMirrorVerbTests {

    func withWorkspace<T>(_ body: (URL) async throws -> T) async throws -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "mirror-verb-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        WorkspaceRoot.programmaticOverride = temp
        ExperimentStore.rootOverride = temp
        defer {
            ExperimentStore.rootOverride = nil
            WorkspaceRoot.programmaticOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try await body(temp)
    }

    private func run(_ args: [String]) async -> ExperimentCLIOutcome {
        await ExperimentCLIRunner(sink: .discarding).run(
            namespace: "vectors", args)
    }

    private func mirrorRunDirectory(_ root: URL, slug: String) throws -> URL {
        let runs = root.appending(component: "runs")
        let hits = try FileManager.default.contentsOfDirectory(
            at: runs, includingPropertiesForKeys: nil
        ).filter { $0.lastPathComponent.hasSuffix(slug) }
        #expect(hits.count == 1)
        return try #require(hits.first)
    }

    @Test func theVerbMintsIntoAFreshRunDirectoryAndTheCatalogListsIt() async throws {
        try await withWorkspace { root in
            let source = try PoleMirrorTests.writeArtifact(
                into: root.appending(components: "runs", "src"))
            _ = source
            let outcome = await run([
                "mirror-poles", "runs/src/\(PoleMirrorTests.sourceConcept)",
                "--concept", PoleMirrorTests.mirrorConcept, "--json",
            ])
            #expect(outcome.envelope.exitCode == 0)
            let result = try #require(outcome.envelope.result)
            #expect(result["concept"] == .string(PoleMirrorTests.mirrorConcept))
            #expect(
                result["sourceConcept"] == .string(PoleMirrorTests.sourceConcept))
            #expect(result["polesSwappedFromSource"] == .bool(true))
            // The success message names the file a researcher must author, and
            // the verb writes NOTHING into prompts/concepts/ itself.
            #expect(
                result["validationAuthoring"]
                    == .string(
                        PoleMirror.validationAuthoringNote(
                            concept: PoleMirrorTests.mirrorConcept)))
            #expect(
                !FileManager.default.fileExists(
                    atPath: root.appending(
                        components: "prompts", "concepts",
                        PoleMirrorTests.mirrorConcept
                    ).path))

            // A mirrored artifact is an ORDINARY catalog artifact.
            let listed = VectorCatalog.scan(
                runsDirectory: root.appending(component: "runs"))
            #expect(
                listed.contains {
                    $0.sidecar.concept == PoleMirrorTests.mirrorConcept
                        && $0.sidecar.negatedFrom != nil
                })
            #expect(
                listed.contains {
                    $0.sidecar.concept == PoleMirrorTests.sourceConcept
                })
            // …in a run directory stamped like every other one this engine
            // writes.
            let runDirectory = try mirrorRunDirectory(
                root, slug: "mirror-\(PoleMirrorTests.mirrorConcept)")
            #expect(
                FileManager.default.fileExists(
                    atPath: runDirectory.appending(component: "config.json").path))
        }
    }

    @Test func aMissingConceptRefusesWithTheReasonAndLeavesNoRunDirectory()
        async throws
    {
        try await withWorkspace { root in
            _ = try PoleMirrorTests.writeArtifact(
                into: root.appending(components: "runs", "src"))
            let outcome = await run([
                "mirror-poles", "runs/src/\(PoleMirrorTests.sourceConcept)",
                "--json",
            ])
            #expect(outcome.envelope.exitCode == SteerLabCLIState.blocked.exitCode)
            let error = try #require(outcome.envelope.error)
            #expect(error.code == "usage")
            #expect(error.reason.contains("pointing opposite ways"))
            #expect(error.repairAction.contains("--concept"))
            let runs = try FileManager.default.contentsOfDirectory(
                at: root.appending(component: "runs"), includingPropertiesForKeys: nil)
            #expect(!runs.contains { $0.lastPathComponent.contains("mirror-") })
        }
    }

    @Test func aMissingSourceIsNotFoundAndLeavesNoRunDirectory() async throws {
        try await withWorkspace { root in
            _ = try PoleMirrorTests.writeArtifact(
                into: root.appending(components: "runs", "src"))
            let outcome = await run([
                "mirror-poles", "runs/src/absent", "--concept",
                PoleMirrorTests.mirrorConcept, "--json",
            ])
            #expect(outcome.envelope.exitCode == SteerLabCLIState.notFound.exitCode)
            #expect(try #require(outcome.envelope.error).code == "notFound")
            let runs = try FileManager.default.contentsOfDirectory(
                at: root.appending(component: "runs"), includingPropertiesForKeys: nil)
            #expect(!runs.contains { $0.lastPathComponent.contains("mirror-") })
        }
    }

    @Test func aDoubleMirrorIsRefused() async throws {
        try await withWorkspace { root in
            _ = try PoleMirrorTests.writeArtifact(
                into: root.appending(components: "runs", "src"))
            #expect(
                await run([
                    "mirror-poles", "runs/src/\(PoleMirrorTests.sourceConcept)",
                    "--concept", PoleMirrorTests.mirrorConcept,
                ]).exitCode == 0)
            let minted = try mirrorRunDirectory(
                root, slug: "mirror-\(PoleMirrorTests.mirrorConcept)")
            let outcome = await run([
                "mirror-poles",
                "runs/\(minted.lastPathComponent)/\(PoleMirrorTests.mirrorConcept)",
                "--concept", PoleMirrorTests.sourceConcept, "--json",
            ])
            #expect(outcome.envelope.exitCode == SteerLabCLIState.refused.exitCode)
            let error = try #require(outcome.envelope.error)
            #expect(error.code == "doubleMirror")
            #expect(error.reason.contains("already exists on disk"))
        }
    }

    /// The lifecycle answer: a mirrored pole has NO validation.jsonl, and
    /// nothing invents one. Mirroring a source-concept-less artifact (a
    /// reader-derived direction) attaches cleanly and the absence is pinned
    /// EXPLICITLY — the existing explicit-null machinery, saying so honestly,
    /// rather than a new gate.
    @Test func aMirroredReaderDirectionAttachesWithItsValidationPinnedAbsent()
        async throws
    {
        try await withWorkspace { root in
            try WorkspaceCompute.declare(.localMLX, root: root)
            _ = try ExperimentStore.create(
                name: "mirror-study", description: "",
                modelID: "mlx-community/gemma-3-4b-it-4bit")

            let reader = RepEReader.Artifact(
                modelID: "mlx-community/gemma-3-4b-it-4bit", revision: "abc123",
                concept: "candour", layer: 1,
                template: RepEReader.TaskTemplate(
                    id: "unnamed-scenario-v1", conceptSlot: false,
                    text: "S: {{stimulus}} q", latToken: "final", hash: "th",
                    divergence: "unnamed-clean-room"),
                datasetHash: "reader-dataset-hash",
                probe: SteeringVectorMath.ScalarProbe(
                    direction: [1, 0, 0], activationCenter: [0, 0, 0],
                    projectionCenter: 0, projectionScale: 1, orientation: 1,
                    positiveMean: 1, negativeMean: -1),
                differenceCloudExplainedVariance: 0.6,
                trainAccuracy: 1, heldOutAccuracy: 0.9,
                trainPairCount: 8, heldOutPairCount: 4,
                signConvention: .heldOutPairAgreement, signHeldOutAccuracy: 1)
            var (vectors, sidecar) = try RepEReader.deriveSteeringArtifact(
                from: reader, readerFileName: "reader-candour-layer1.json",
                readerBytes: Data("reader-bytes".utf8))
            sidecar.residualNormPerLayer = [7.0, 7.5]
            sidecar.residualNormSource = "neutral-corpus"
            let sourceDirectory = root.appending(components: "runs", "derived")
            try SteeringVectorStore.save(
                vectors: vectors, sidecar: sidecar, to: sourceDirectory,
                name: "candour-repe-reader")

            let mirrored = try PoleMirror.mirrorPoles(
                artifact: sourceDirectory.appending(component: "candour-repe-reader"),
                concept: "reticence",
                into: root.appending(components: "runs", "mirrored"))

            let manifest = try ExperimentStore.attachArtifact(
                "reticence", artifact: "runs/mirrored/reticence",
                experimentName: "mirror-study")
            let ref = try #require(manifest.concepts.first { $0.name == "reticence" })
            #expect(ref.effectiveMethod == .repeReaderLAT)
            // The reader's dataset hash still travels verbatim — the mirrored
            // pole's stimuli are the same bytes.
            #expect(ref.stimulusSetHash == "reader-dataset-hash")
            // The absence of a validation.jsonl is pinned EXPLICITLY, not
            // merely missing: authoring one later is drift the existing
            // machinery names.
            #expect(ref.validationHash == nil)
            #expect(ref.validationHashPinnedAbsent)
            #expect(ExperimentStore.verify(manifest).isEmpty)
            #expect(mirrored.sourceConcept == "candour")
        }
    }
}
