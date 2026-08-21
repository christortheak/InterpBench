import CryptoKit
import Foundation
import Testing

@testable import SteeringKit

/// Pure-CPU tests for the faithful RepE reader recipe. Fixture semantics are
/// ported from the server's `Server/tests/test_repe_reader.py` (the source of
/// truth): same rendered strings, same synthetic fit fixture, same artifact
/// schema — cross-engine parity is asserted on tiny committed fixtures, never
/// live model output (CLAUDE.md › Conventions).
@Suite struct RepEReaderTests {

    /// Repo root derived from this file's compile-time path — the committed
    /// templates under `prompts/templates/` are shared data files.
    private var repoRoot: URL {
        URL(filePath: #filePath)  // …/Tests/SteeringKitTests/RepEReaderTests.swift
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private var templatesDirectory: URL {
        repoRoot.appending(components: "prompts", "templates")
    }

    private func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func tempDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(component: "repe-reader-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func expectReaderError(
        containing substring: String,
        _ body: () throws -> Void
    ) {
        do {
            try body()
            Issue.record("expected ReaderError containing '\(substring)'")
        } catch let error as RepEReader.ReaderError {
            #expect(error.reason.contains(substring), "got: \(error.reason)")
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }

    // Matches the server test's AMOUNT_TEMPLATE_TEXT and the committed
    // prompts/templates/amount-in-scenario-v1.json text.
    static let amountTemplateText =
        "Consider the amount of {{concept}} in the following scenario:\n"
        + "Scenario: {{stimulus}}\n"
        + "The amount of {{concept}} in the scenario is"

    private func writeTemplate(
        in directory: URL, id: String = "amount-in-scenario-v1",
        text: String = RepEReaderTests.amountTemplateText, conceptSlot: Bool = true,
        latToken: String = "final"
    ) throws -> URL {
        let object: [String: Any] = [
            "id": id, "conceptSlot": conceptSlot, "text": text, "latToken": latToken,
        ]
        let url = directory.appending(component: "\(id).json")
        try JSONSerialization.data(withJSONObject: object).write(to: url)
        return url
    }

    // MARK: - Templates (shared committed data)

    @Test func committedTemplatesLoadAndRenderParityStrings() throws {
        let named = try RepEReader.loadTemplate(
            url: templatesDirectory.appending(component: "amount-in-scenario-v1.json"))
        #expect(named.conceptSlot)
        #expect(named.latToken == "final")
        #expect(named.divergence == nil)
        #expect(try named.readingPosition() == .lastToken)
        // Cross-engine rendered-string fixture: the SAME template + stimulus
        // must produce this exact string on both engines (hard-coded from the
        // Python fixture inputs: stimulus "It rains.", concept "fear").
        let rendered = try named.render(stimulus: "It rains.", concept: "fear")
        #expect(
            rendered
                == "Consider the amount of fear in the following scenario:\n"
                + "Scenario: It rains.\n"
                + "The amount of fear in the scenario is")
        #expect(!rendered.contains("{{"))

        let unnamed = try RepEReader.loadTemplate(
            url: templatesDirectory.appending(component: "unnamed-scenario-v1.json"))
        #expect(!unnamed.conceptSlot)
        #expect(unnamed.divergence == "unnamed-clean-room")  // stamped divergence
        let cleanRoom = try unnamed.render(stimulus: "It rains.")  // no concept named
        #expect(
            cleanRoom
                == "Consider the following scenario:\n"
                + "Scenario: It rains.\n"
                + "The intensity of the described state is")
        #expect(!cleanRoom.contains("fear"))
    }

    @Test func templateHashIsRawBytesAndStable() throws {
        let directory = try tempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = try writeTemplate(in: directory)
        let expected = sha256Hex(try Data(contentsOf: url))
        let first = try RepEReader.loadTemplate(url: url)
        let second = try RepEReader.loadTemplate(url: url)
        #expect(first.hash == expected)
        #expect(second.hash == expected)

        // The committed registry files hash identically on both engines: the
        // pin is over raw file bytes, no re-serialization.
        let committedURL = templatesDirectory.appending(
            component: "amount-in-scenario-v1.json")
        let committed = try RepEReader.loadTemplate(url: committedURL)
        #expect(committed.hash == sha256Hex(try Data(contentsOf: committedURL)))
    }

    @Test func templateValidation() throws {
        let directory = try tempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        // id must match filename (one file per id).
        let path = try writeTemplate(in: directory, id: "a-v1")
        let renamed = directory.appending(component: "b-v1.json")
        try FileManager.default.moveItem(at: path, to: renamed)
        expectReaderError(containing: "does not match filename") {
            _ = try RepEReader.loadTemplate(url: renamed)
        }

        // Named template requires a concept at render time.
        let named = try RepEReader.loadTemplate(url: writeTemplate(in: directory))
        expectReaderError(containing: "names the concept") {
            _ = try named.render(stimulus: "x")
        }

        // A {{stimulus}} slot is mandatory.
        let noSlot = try RepEReader.loadTemplate(
            url: writeTemplate(
                in: directory, id: "no-slot-v1", text: "no slot here",
                conceptSlot: false))
        expectReaderError(containing: "stimulus") {
            _ = try noSlot.render(stimulus: "x")
        }

        // Only latToken "final" is implemented (honesty: refuse, don't guess).
        let other = try RepEReader.loadTemplate(
            url: writeTemplate(in: directory, id: "mid-v1", latToken: "penultimate"))
        expectReaderError(containing: "latToken") {
            _ = try other.readingPosition()
        }
    }

    @Test func scaffoldGuardRejectsEmbeddedChatMarkers() throws {
        // The #1 tokenizer risk: a hand-tokenized scaffold smuggling template
        // markers (double-BOS hazard). Both families' markers are refused.
        for marker in ["<bos>", "<start_of_turn>", "<|im_start|>"] {
            expectReaderError(containing: "special/chat-template") {
                _ = try RepEReader.renderReaderScaffold(
                    "\(marker) scenario text", modelID: "org/gemma-3-4b-it")
            }
        }
        expectReaderError(containing: "BOS") {
            _ = try RepEReader.renderReaderScaffold("<s> scenario", modelID: "org/m")
        }
        #expect(
            try RepEReader.renderReaderScaffold("plain scaffold", modelID: "Qwen/Qwen3-4B")
                == "plain scaffold")
    }

    // MARK: - Dataset (pairs.jsonl)

    private func pairRow(
        _ index: Int, _ positive: String, _ negative: String,
        split: String? = "train", concept: String = "fear",
        templateID: String = "amount-in-scenario-v1"
    ) -> [String: Any] {
        var row: [String: Any] = [
            "id": "\(concept)-pair-\(index)", "concept": concept,
            "positiveStimulus": positive, "negativeStimulus": negative,
            "topic": "t", "templateID": templateID,
        ]
        if let split { row["split"] = split }
        return row
    }

    private func writePairs(_ rows: [[String: Any]], to url: URL) throws {
        let lines = try rows.map {
            String(decoding: try JSONSerialization.data(withJSONObject: $0), as: UTF8.self)
        }
        try (lines.joined(separator: "\n") + "\n").write(
            to: url, atomically: true, encoding: .utf8)
    }

    @Test func pairsLoaderHashSplitAndErrors() throws {
        let directory = try tempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let url = directory.appending(component: "pairs.jsonl")
        try writePairs(
            [
                pairRow(0, "p0", "n0"),
                pairRow(1, "p1", "n1", split: "TEST"),
                pairRow(2, "p2", "n2", split: nil),
            ], to: url)
        let expected = sha256Hex(try Data(contentsOf: url))
        let dataset = try RepEReader.loadPairs(url: url)
        #expect(dataset.hash == expected)
        #expect(dataset.concept == "fear")
        #expect(dataset.pairs.map(\.split) == ["train", "test", "train"])
        #expect(dataset.train.count == 2)
        #expect(dataset.heldOut.count == 1)

        let mixed = directory.appending(component: "mixed.jsonl")
        try writePairs(
            [pairRow(0, "p0", "n0"), pairRow(1, "p1", "n1", concept: "calm")],
            to: mixed)
        expectReaderError(containing: "mixed concepts") {
            _ = try RepEReader.loadPairs(url: mixed)
        }

        let missing = directory.appending(component: "missing.jsonl")
        try #"{"positiveStimulus": "p", "concept": "fear"}"#.write(
            to: missing, atomically: true, encoding: .utf8)
        expectReaderError(containing: "negativeStimulus") {
            _ = try RepEReader.loadPairs(url: missing)
        }
    }

    // MARK: - Fit math on synthetic activations (server fixture port)

    // One layer, two dims; the concept lives on the x axis. Held-out pair has
    // one deliberately misclassified positive (x below the train center).
    static let acts: [String: [[Float]]] = [
        "p0": [[2.0, 0.5]], "n0": [[0.0, 0.5]],
        "p1": [[2.0, -0.5]], "n1": [[0.0, -0.5]],
        "hp": [[0.5, 0.0]], "hn": [[0.0, 0.0]],
    ]

    private func fitFixture(swapLabels: Bool = false)
        -> (dataset: RepEReader.Dataset, template: RepEReader.TaskTemplate,
            captured: [[[Float]]])
    {
        let template = RepEReader.TaskTemplate(
            id: "amount-in-scenario-v1", conceptSlot: true,
            text: Self.amountTemplateText, latToken: "final", hash: "th")
        let keys: [(String, String, String)] = [
            (swapLabels ? "n0" : "p0", swapLabels ? "p0" : "n0", "train"),
            (swapLabels ? "n1" : "p1", swapLabels ? "p1" : "n1", "train"),
            (swapLabels ? "hn" : "hp", swapLabels ? "hp" : "hn", "test"),
        ]
        let pairs = keys.enumerated().map { index, row in
            RepEReader.Pair(
                id: "fear-pair-\(index)", concept: "fear",
                positiveStimulus: row.0, negativeStimulus: row.1,
                topic: "t", split: row.2, templateID: template.id)
        }
        let dataset = RepEReader.Dataset(concept: "fear", pairs: pairs, hash: "dh")
        var captured: [[[Float]]] = []
        for pair in pairs {
            captured.append(Self.acts[pair.positiveStimulus]!)
            captured.append(Self.acts[pair.negativeStimulus]!)
        }
        return (dataset, template, captured)
    }

    private func approx(_ values: [Float], _ expected: [Float], tolerance: Float = 1e-5) -> Bool {
        values.count == expected.count
            && zip(values, expected).allSatisfy { abs($0 - $1) <= tolerance }
    }

    @Test func fitMathOrientationAndAccuracy() throws {
        let (dataset, template, captured) = fitFixture()
        let readers = try RepEReader.fit(
            dataset: dataset, template: template, capturedValues: captured,
            modelID: "org/m", revision: "abc")
        #expect(readers.count == 1)
        let reader = readers[0]
        // PC1 of the normalized pair differences is the +x concept axis,
        // oriented so the positive class scores positive.
        #expect(approx(reader.probe.direction, [1, 0]))
        #expect(abs(reader.pc1ExplainedVariance - 1) <= 1e-5)
        // Training normalization: centered at the train activation mean.
        #expect(approx(reader.probe.activationCenter ?? [], [1, 0]))
        #expect(reader.trainAccuracy == 1)
        // Held-out pair: positive at x=0.5 falls below the center →
        // misclassified; negative at x=0 is correct → 0.5.
        #expect(reader.heldOutAccuracy == 0.5)
        #expect(reader.trainPairCount == 2)
        #expect(reader.heldOutPairCount == 1)
        #expect(reader.substrate == "swift-mlx")
        #expect(reader.renderingConvention == RepEReader.renderingConvention)
        #expect(reader.modelID == "org/m")
        #expect(reader.revision == "abc")
        // Positive train stimulus scores positive through the fitted probe.
        #expect(try RepEReader.scoreActivation(reader, activation: Self.acts["p0"]![0]) > 0)
        #expect(try RepEReader.scoreActivation(reader, activation: Self.acts["n0"]![0]) < 0)
    }

    @Test func fitOrientsByPairedLabels() throws {
        let (dataset, template, captured) = fitFixture(swapLabels: true)
        let reader = try RepEReader.fit(
            dataset: dataset, template: template, capturedValues: captured,
            modelID: "org/m", revision: "abc")[0]
        // With labels swapped the reading direction flips: −x is now "positive".
        #expect(approx(reader.probe.direction, [-1, 0]))
        #expect(try RepEReader.scoreActivation(reader, activation: Self.acts["n0"]![0]) > 0)
    }

    @Test func fitGuards() throws {
        let (dataset, _, captured) = fitFixture()
        let other = RepEReader.TaskTemplate(
            id: "other-v1", conceptSlot: true,
            text: Self.amountTemplateText, latToken: "final", hash: "oh")
        expectReaderError(containing: "pins template") {
            _ = try RepEReader.fit(
                dataset: dataset, template: other, capturedValues: captured,
                modelID: "org/m", revision: nil)
        }
        let (fullDataset, template, _) = fitFixture()
        let onePair = RepEReader.Dataset(
            concept: "fear", pairs: [fullDataset.pairs[0]], hash: "dh")
        expectReaderError(containing: "at least 2 train") {
            _ = try RepEReader.fit(
                dataset: onePair, template: template,
                capturedValues: [Self.acts["p0"]!, Self.acts["n0"]!],
                modelID: "org/m", revision: nil)
        }
    }

    // MARK: - Artifact round-trip + schema (brief §4 contract)

    @Test func artifactRoundTripAndShape() throws {
        let (dataset, template, captured) = fitFixture()
        let reader = try RepEReader.fit(
            dataset: dataset, template: template, capturedValues: captured,
            modelID: "org/m", revision: "abc")[0]

        let directory = try tempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = try RepEReader.saveArtifact(reader, to: directory)
        #expect(url.lastPathComponent == "reader-fear-layer0.json")
        let loaded = try RepEReader.loadArtifact(url: url)
        #expect(loaded == reader)
        let probeInput: [Float] = [0.3, 9.9]
        #expect(
            try RepEReader.scoreActivation(loaded, activation: probeInput)
                == RepEReader.scoreActivation(reader, activation: probeInput))

        // Brief §4 artifact contract: exact key names shared with the server.
        let object = try #require(
            try JSONSerialization.jsonObject(with: Data(contentsOf: url))
                as? [String: Any])
        #expect(object["artifactType"] as? String == "repe-reader-lat")
        #expect(object["schemaVersion"] as? Int == 1)
        #expect(object["substrate"] as? String == "swift-mlx")
        #expect(object["templateID"] as? String == "amount-in-scenario-v1")
        #expect(object["templateHash"] as? String == template.hash)
        #expect(object["datasetHash"] as? String == dataset.hash)
        #expect(object["latTokenPosition"] as? String == "final")
        #expect(object["readingPosition"] as? String == "last token")
        let probe = try #require(object["probe"] as? [String: Any])
        // Probe JSON keys are the cross-engine ScalarProbe contract.
        #expect(
            Set(probe.keys).isSuperset(of: [
                "direction", "projectionCenter", "projectionScale",
                "orientation", "activationCenter",
            ]))
        #expect(object["renderingConvention"] as? String == RepEReader.renderingConvention)
        #expect(object["trainAccuracy"] != nil)
        #expect(object["heldOutAccuracy"] != nil)
        #expect(object["pc1ExplainedVariance"] != nil)
        let embedded = try #require(object["template"] as? [String: Any])
        #expect(embedded["id"] as? String == "amount-in-scenario-v1")
        #expect(embedded["hash"] as? String == template.hash)

        expectReaderError(containing: "not a repe-reader-lat") {
            _ = try JSONDecoder().decode(
                RepEReader.Artifact.self, from: Data(#"{"artifactType": "other"}"#.utf8))
        }
    }

    /// Cross-engine READABILITY: the server's artifact JSON (shape verbatim
    /// from `test_repe_reader.py`'s `_manual_reader().to_dict()`) decodes
    /// here, the foreign substrate survives the round trip — and it is the
    /// manifest verify gate (ExperimentStoreTests) that rejects it for local
    /// use. Readable, never silently usable.
    @Test func serverArtifactDecodesWithForeignSubstratePreserved() throws {
        let serverConvention =
            "rawCompletion scaffold: no chat template, no system role, no family "
            + "thinking suffix; tokenized by the extraction path "
            + "(extractor.activations) with the tokenizer's default special tokens "
            + "— single BOS added by the tokenizer, LAT token = final scaffold token"
        let json = """
            {
              "artifactType": "repe-reader-lat",
              "schemaVersion": 1,
              "modelID": "org/m",
              "revision": "abc",
              "substrate": "python-hf-transformers",
              "concept": "fear",
              "layer": 0,
              "templateID": "unnamed-scenario-v1",
              "templateHash": "th",
              "template": {
                "conceptSlot": false,
                "divergence": "unnamed-clean-room",
                "hash": "th",
                "id": "unnamed-scenario-v1",
                "latToken": "final",
                "text": "S: {{stimulus}} q"
              },
              "templateDivergence": "unnamed-clean-room",
              "datasetHash": "dh",
              "latTokenPosition": "final",
              "readingPosition": "last token",
              "probe": {
                "direction": [1.0, 0.0],
                "projectionCenter": 0.5,
                "projectionScale": 2.0,
                "orientation": 1.0,
                "positiveMean": 1.0,
                "negativeMean": -1.0,
                "activationCenter": [1.0, 0.0]
              },
              "pc1ExplainedVariance": 0.9,
              "trainAccuracy": 1.0,
              "heldOutAccuracy": 0.8,
              "trainPairCount": 4,
              "heldOutPairCount": 2,
              "renderingConvention": "\(serverConvention)",
              "extractionDate": "2026-07-03T00:00:00Z"
            }
            """
        let reader = try JSONDecoder().decode(
            RepEReader.Artifact.self, from: Data(json.utf8))
        #expect(reader.substrate == "python-hf-transformers")  // preserved, not rewritten
        #expect(reader.concept == "fear")
        #expect(reader.template.divergence == "unnamed-clean-room")
        #expect(reader.templateHash == "th")
        #expect(reader.renderingConvention == serverConvention)
        // Pure probe math is engine-independent: the training normalization
        // reproduces the server's hand-computed 0.75.
        #expect(
            abs(try RepEReader.scoreActivation(reader, activation: [3, 7]) - 0.75) <= 1e-6)
    }

    // MARK: - Exact inference (pure probe scoring)

    private func manualReader(
        layer: Int = 0, substrate: String = RepEReader.substrate
    ) -> RepEReader.Artifact {
        RepEReader.Artifact(
            modelID: "org/m", revision: "abc", concept: "fear", layer: layer,
            template: RepEReader.TaskTemplate(
                id: "unnamed-scenario-v1", conceptSlot: false,
                text: "S: {{stimulus}} q", latToken: "final", hash: "th",
                divergence: "unnamed-clean-room"),
            datasetHash: "dh",
            probe: SteeringVectorMath.ScalarProbe(
                direction: [1, 0], activationCenter: [1, 0],
                projectionCenter: 0.5, projectionScale: 2, orientation: 1,
                positiveMean: 1, negativeMean: -1),
            pc1ExplainedVariance: 0.9, trainAccuracy: 1, heldOutAccuracy: 0.8,
            trainPairCount: 4, heldOutPairCount: 2, substrate: substrate)
    }

    @Test func validateForScoringGuardsSubstrateAndModel() throws {
        // Right substrate + right model: scoring preconditions pass.
        try RepEReader.validateForScoring(reader: manualReader(), modelID: "org/m")

        // Model mismatch throws: a reader is a per-model measurement
        // instrument, and the message names both models so the operator can
        // act on it.
        do {
            try RepEReader.validateForScoring(reader: manualReader(), modelID: "org/other")
            Issue.record("expected model-mismatch ReaderError")
        } catch let error as RepEReader.ReaderError {
            #expect(error.reason.contains("per-model measurement instrument"))
            #expect(error.reason.contains("'org/m'"))
            #expect(error.reason.contains("'org/other'"))
            #expect(error.reason.contains("re-fit"))
        }

        // Foreign substrate still refuses, even with a matching model id
        // (the substrate guard fires first — activations never transfer).
        expectReaderError(containing: "substrate-specific") {
            try RepEReader.validateForScoring(
                reader: manualReader(substrate: "python-hf-transformers"),
                modelID: "org/m")
        }
    }

    @Test func scoreActivationMatchesHandComputation() throws {
        let reader = manualReader()
        // Training normalization: (([3,7]−[1,0])·[1,0] − 0.5) / 2 = 0.75.
        #expect(abs(try RepEReader.scoreActivation(reader, activation: [3, 7]) - 0.75) <= 1e-6)
        // Same rendered scaffold contract as the server fixture.
        #expect(
            try RepEReader.renderScaffold(
                template: reader.template, stimulus: "hello",
                concept: reader.concept, modelID: "org/m") == "S: hello q")
    }

    // MARK: - Derive-steering conversion (brief §6)

    @Test func derivedSidecarCarriesReaderProvenance() throws {
        let reader = manualReader(layer: 1)
        let bytes = try JSONEncoder().encode(reader)
        let (vectors, sidecar) = try RepEReader.deriveSteeringArtifact(
            from: reader, readerFileName: "reader-fear-layer1.json", readerBytes: bytes)
        // Unit reading direction at the reader's layer, zeros below
        // (single-layer import convention) — honest provenance on the sidecar.
        #expect(vectors.perLayer.count == 2)
        #expect(approx(vectors.perLayer[1], [1, 0]))
        #expect(vectors.perLayer[0] == [0, 0])
        #expect(sidecar.source == "repe-reader-lat")
        #expect(sidecar.readerID == "reader-fear-layer1.json")
        #expect(sidecar.readerHash == sha256Hex(bytes))
        #expect(sidecar.controlMode == "reading-vector activation addition")
        #expect(sidecar.extractionMethod == "repeReaderLAT")
        #expect(sidecar.stimulusSetHash == "dh")
        #expect(sidecar.modelID == "org/m")
        #expect(sidecar.revision == "abc")
        #expect(sidecar.concept == "fear")
        #expect(sidecar.readingPosition == "last token")
        #expect(sidecar.layerCount == 2)
        #expect(sidecar.normsPerLayer.first == 0)

        // The provenance fields survive the sidecar's JSON round trip (the
        // cross-engine on-disk contract).
        let encoded = try JSONEncoder().encode(sidecar)
        let decoded = try JSONDecoder().decode(SteeringVectorSidecar.self, from: encoded)
        #expect(decoded.source == "repe-reader-lat")
        #expect(decoded.readerHash == sidecar.readerHash)
        #expect(decoded.controlMode == sidecar.controlMode)
    }
}
