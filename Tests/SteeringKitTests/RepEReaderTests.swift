import CryptoKit
import Foundation
import Testing

@testable import SteeringKit

/// Pure-CPU tests for the template-mediated RepE reader recipe. Fixture semantics are
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
        // The two train differences are identical after normalization, so
        // the DIFFERENCE CLOUD has no variance to apportion — absent, not 0
        // (which would read as "PC1 explains nothing"). The pre-2026-08-27
        // number here was 1.0, measured over the ± alternated copies PCA is
        // fitted on.
        #expect(reader.differenceCloudExplainedVariance == nil)
        #expect(reader.explainedVarianceBasis == "degenerateDifferenceCloud")
        #expect(reader.contrastMode == .supervisedContent)
        // ONE held-out pair is below the minimum that may decide a sign, so
        // the fit falls back to the reference implementation's train-label
        // majority and says so out loud. `heldOutSplitFixesTheSign` covers the
        // branch where the held-out split does decide.
        #expect(reader.signConvention == .trainMajority)
        #expect(reader.signHeldOutAccuracy == nil)
        #expect(reader.signFallbackReason?.contains("below the minimum 2") == true)
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

    // MARK: - Artifact round-trip + schema (REPE-IMPLEMENTATION-BRIEF §4 contract)

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
        // Schema 2 (2026-08-27): contrast mode, sign convention, layer
        // recommendation, difference-cloud variance basis, extraction
        // rendering. Every one of them decodes with a stamped LEGACY default,
        // so a schema-1 artifact still loads — `legacyArtifactDecodesWith…`.
        #expect(object["schemaVersion"] as? Int == 2)
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
        // The legacy `pc1ExplainedVariance` key is gone: its basis changed,
        // and writing the old key with the new semantics would make every
        // pre-existing consumer silently wrong instead of visibly out of date.
        #expect(object["pc1ExplainedVariance"] == nil)
        #expect(
            object["pc1ExplainedVarianceBasis"] as? String
                == "degenerateDifferenceCloud")
        #expect(object["contrastMode"] as? String == "supervisedContent")
        #expect(object["signConvention"] as? String == "trainMajority")
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
            differenceCloudExplainedVariance: 0.9, trainAccuracy: 1, heldOutAccuracy: 0.8,
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

    // MARK: - Derive-steering conversion (REPE-IMPLEMENTATION-BRIEF §6)

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

    // MARK: - Paper step 4: held-out sign and layer selection

    /// Two held-out pairs whose differences point the OPPOSITE way from the
    /// train majority: the held-out split wins, which is the paper's rule and
    /// not `get_signs`'.
    @Test func heldOutSplitFixesTheSign() throws {
        // Train: +x is "positive" by label. Held-out: the labelled positive
        // sits BELOW its negative on x, twice — a held-out set that says the
        // reading direction is −x.
        let acts: [String: [[Float]]] = [
            "p0": [[2.0, 0.5]], "n0": [[0.0, 0.5]],
            "p1": [[2.0, -0.5]], "n1": [[0.0, -0.5]],
            "hp0": [[0.0, 0.2]], "hn0": [[1.0, 0.2]],
            "hp1": [[0.1, -0.2]], "hn1": [[1.1, -0.2]],
        ]
        let template = RepEReader.TaskTemplate(
            id: "amount-in-scenario-v1", conceptSlot: true,
            text: Self.amountTemplateText, latToken: "final", hash: "th")
        let rows: [(String, String, String)] = [
            ("p0", "n0", "train"), ("p1", "n1", "train"),
            ("hp0", "hn0", "test"), ("hp1", "hn1", "test"),
        ]
        let pairs = rows.enumerated().map { index, row in
            RepEReader.Pair(
                id: "fear-pair-\(index)", concept: "fear",
                positiveStimulus: row.0, negativeStimulus: row.1,
                split: row.2, templateID: template.id)
        }
        let dataset = RepEReader.Dataset(concept: "fear", pairs: pairs, hash: "dh")
        let captured = pairs.flatMap {
            [acts[$0.positiveStimulus]!, acts[$0.negativeStimulus]!]
        }
        let reader = try RepEReader.fit(
            dataset: dataset, template: template, capturedValues: captured,
            modelID: "org/m", revision: "abc")[0]
        #expect(reader.signConvention == .heldOutPairAgreement)
        #expect(reader.signHeldOutAccuracy == 1)
        #expect(reader.signFallbackReason == nil)
        // Train majority alone would have chosen +x; the held-out split flips
        // it. This is the whole behavioural difference between the paper's
        // step 4 and the reference implementation's get_signs.
        #expect(approx(reader.probe.direction, [-1, 0]))
    }

    /// A held-out set that splits evenly says nothing, and the fit says THAT
    /// rather than treating a tie as evidence.
    @Test func evenlySplitHeldOutFallsBackLoudly() throws {
        let acts: [String: [[Float]]] = [
            "p0": [[2.0, 0.5]], "n0": [[0.0, 0.5]],
            "p1": [[2.0, -0.5]], "n1": [[0.0, -0.5]],
            "ha": [[1.0, 0.0]], "hb": [[0.0, 0.0]],
            "hc": [[0.0, 0.1]], "hd": [[1.0, 0.1]],
        ]
        let template = RepEReader.TaskTemplate(
            id: "amount-in-scenario-v1", conceptSlot: true,
            text: Self.amountTemplateText, latToken: "final", hash: "th")
        let rows: [(String, String, String)] = [
            ("p0", "n0", "train"), ("p1", "n1", "train"),
            ("ha", "hb", "test"), ("hc", "hd", "test"),
        ]
        let pairs = rows.enumerated().map { index, row in
            RepEReader.Pair(
                id: "fear-pair-\(index)", concept: "fear",
                positiveStimulus: row.0, negativeStimulus: row.1,
                split: row.2, templateID: template.id)
        }
        let dataset = RepEReader.Dataset(concept: "fear", pairs: pairs, hash: "dh")
        let captured = pairs.flatMap {
            [acts[$0.positiveStimulus]!, acts[$0.negativeStimulus]!]
        }
        let reader = try RepEReader.fit(
            dataset: dataset, template: template, capturedValues: captured,
            modelID: "org/m", revision: "abc")[0]
        #expect(reader.signConvention == .trainMajority)
        #expect(reader.signHeldOutAccuracy == nil)
        #expect(reader.signFallbackReason?.contains("split evenly") == true)
        #expect(approx(reader.probe.direction, [1, 0]))
    }

    /// `recommendedLayer` is the argmax of held-out accuracy across the layers
    /// fitted TOGETHER, stamped on every artifact of the set — and it is a
    /// recommendation, which the artifact says in its own words.
    @Test func layerRecommendationIsStampedOnTheWholeSet() throws {
        let template = RepEReader.TaskTemplate(
            id: "amount-in-scenario-v1", conceptSlot: true,
            text: Self.amountTemplateText, latToken: "final", hash: "th")
        // Two layers, both with non-degenerate differences: layer 0 separates
        // the held-out pair, layer 1 puts it on the wrong side.
        func row(_ x0: Float, _ x1: Float) -> [[Float]] { [[x0, 0.5], [x1, 0.5]] }
        let acts: [String: [[Float]]] = [
            "p0": row(2, 1), "n0": row(0, 0),
            "p1": row(3, 2), "n1": row(0.5, 0.5),
            "hp": row(2, 0), "hn": row(0, 3),
        ]
        let rows: [(String, String, String)] = [
            ("p0", "n0", "train"), ("p1", "n1", "train"), ("hp", "hn", "test"),
        ]
        let pairs = rows.enumerated().map { index, r in
            RepEReader.Pair(
                id: "fear-pair-\(index)", concept: "fear",
                positiveStimulus: r.0, negativeStimulus: r.1,
                split: r.2, templateID: template.id)
        }
        let dataset = RepEReader.Dataset(concept: "fear", pairs: pairs, hash: "dh")
        let captured = pairs.flatMap {
            [acts[$0.positiveStimulus]!, acts[$0.negativeStimulus]!]
        }
        let readers = try RepEReader.fit(
            dataset: dataset, template: template, capturedValues: captured,
            modelID: "org/m", revision: "abc")
        #expect(readers.count == 2)
        let recommended = readers.map(\.recommendedLayer)
        // Every artifact of the set carries the SAME recommendation — a reader
        // opened alone must not have to re-derive it from its siblings.
        #expect(Set(recommended.compactMap { $0 }).count == 1)
        #expect(readers.allSatisfy { $0.layerRecommendationBasis == "heldOutAccuracy" })
        let best = try #require(readers.first?.recommendedLayer)
        let bestArtifact = try #require(readers.first { $0.layer == best })
        #expect(
            (bestArtifact.heldOutAccuracy ?? 0)
                >= (readers.map { $0.heldOutAccuracy ?? 0 }.max() ?? 0))
        // The note travels with the number, in the artifact bytes.
        let directory = try tempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = try RepEReader.saveArtifact(readers[0], to: directory)
        let object = try #require(
            try JSONSerialization.jsonObject(with: Data(contentsOf: url))
                as? [String: Any])
        #expect(object["recommendedLayer"] as? Int == best)
        #expect(
            (object["layerRecommendationNote"] as? String)?
                .contains("never selected automatically") == true)
    }

    // MARK: - Paper step 1b: T+/T− template pairs (unsupervised LAT)

    static let stancePairTemplate = RepEReader.TaskTemplate(
        id: "instructed-stance-pair-v1", conceptSlot: false,
        text: "{{instruction}}\nScenario: {{stimulus}}\nThe described state is",
        latToken: "final", hash: "sh",
        instructionPair: .init(experimental: "T-plus.", reference: "T-minus."))

    @Test func committedTemplatePairTemplateLoadsAndRendersBothInstructions() throws {
        let template = try RepEReader.loadTemplate(
            url: templatesDirectory.appending(
                component: "instructed-stance-pair-v1.json"))
        #expect(template.isTemplatePair)
        let pair = try #require(template.instructionPair)
        #expect(pair.experimental != pair.reference)
        // Hygiene: the shipped example never names a concept, so a reader
        // fitted through it cannot become a concept-word detector.
        #expect(!template.conceptSlot)
        #expect(template.divergence?.contains("synthetic-neutral") == true)
        let plus = try template.render(
            stimulus: "the room went quiet", instruction: pair.experimental)
        let minus = try template.render(
            stimulus: "the room went quiet", instruction: pair.reference)
        #expect(plus != minus)
        #expect(plus.hasSuffix("The described state is"))
        #expect(!plus.contains("{{"))
    }

    @Test func templatePairSchemaRefusals() throws {
        let directory = try tempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        func write(_ id: String, _ object: [String: Any]) throws -> URL {
            let url = directory.appending(component: "\(id).json")
            try JSONSerialization.data(withJSONObject: object).write(to: url)
            return url
        }
        let noSlot = try write(
            "no-slot-v1",
            [
                "id": "no-slot-v1", "conceptSlot": false,
                "text": "S: {{stimulus}} q", "latToken": "final",
                "instructionPair": ["experimental": "a", "reference": "b"],
            ])
        expectReaderError(containing: "no {{instruction}} slot") {
            _ = try RepEReader.loadTemplate(url: noSlot)
        }
        let orphanSlot = try write(
            "orphan-slot-v1",
            [
                "id": "orphan-slot-v1", "conceptSlot": false,
                "text": "{{instruction}} S: {{stimulus}} q", "latToken": "final",
            ])
        expectReaderError(containing: "declares no instructionPair") {
            _ = try RepEReader.loadTemplate(url: orphanSlot)
        }
        let identical = try write(
            "identical-v1",
            [
                "id": "identical-v1", "conceptSlot": false,
                "text": "{{instruction}} S: {{stimulus}} q", "latToken": "final",
                "instructionPair": ["experimental": "same", "reference": "same"],
            ])
        expectReaderError(containing: "identical") {
            _ = try RepEReader.loadTemplate(url: identical)
        }
    }

    @Test func datasetShapeAndTemplateMustAgree() throws {
        let contentPairs = RepEReader.Dataset(
            concept: "fear",
            pairs: [
                RepEReader.Pair(
                    concept: "fear", positiveStimulus: "p", negativeStimulus: "n",
                    templateID: "instructed-stance-pair-v1")
            ],
            hash: "dh")
        expectReaderError(containing: "a second stimulus would be a confound") {
            _ = try RepEReader.resolveContrastMode(
                dataset: contentPairs, template: Self.stancePairTemplate)
        }
        let single = RepEReader.Dataset(
            concept: "fear",
            pairs: [
                RepEReader.Pair.templatePair(
                    concept: "fear", stimulus: "s", templateID: "amount-in-scenario-v1")
            ],
            hash: "dh")
        let plain = RepEReader.TaskTemplate(
            id: "amount-in-scenario-v1", conceptSlot: true,
            text: Self.amountTemplateText, latToken: "final", hash: "th")
        expectReaderError(containing: "nothing to contrast the stimulus against") {
            _ = try RepEReader.resolveContrastMode(dataset: single, template: plain)
        }
    }

    @Test func mixedRowShapesAreRefused() throws {
        let jsonl = #"""
            {"concept": "fear", "positiveStimulus": "p", "negativeStimulus": "n", "templateID": "t"}
            {"concept": "fear", "stimulus": "s", "templateID": "t"}
            """#
        expectReaderError(containing: "one file cannot mean both") {
            _ = try RepEReader.parsePairs(Data(jsonl.utf8), source: "mixed")
        }
        let both = #"""
            {"concept": "fear", "stimulus": "s", "positiveStimulus": "p", "templateID": "t"}
            """#
        expectReaderError(containing: "declares both 'stimulus'") {
            _ = try RepEReader.parsePairs(Data(both.utf8), source: "both")
        }
    }

    /// The unsupervised fit: one stimulus, two instructions, differences
    /// H(T+) − H(T−) in SEEDED random orientation, no per-difference
    /// normalization (the reference implementation's own construction).
    @Test func unsupervisedTemplatePairFitIsSeededAndStamped() throws {
        let template = Self.stancePairTemplate
        let stimuli = ["s0", "s1", "s2", "s3", "s4"]
        let pairs = stimuli.enumerated().map { index, stimulus in
            RepEReader.Pair.templatePair(
                id: "fear-row-\(index)", concept: "fear", stimulus: stimulus,
                split: index < 3 ? "train" : "test", templateID: template.id)
        }
        let dataset = RepEReader.Dataset(concept: "fear", pairs: pairs, hash: "dh")
        // T+ sits further along +x than T− for every stimulus, by varying
        // amounts so the difference cloud has real variance.
        var captured: [[[Float]]] = []
        for (index, _) in stimuli.enumerated() {
            let magnitude = Float(index + 1)
            captured.append([[magnitude, 0.1 * magnitude]])  // T+
            captured.append([[0, 0.1 * magnitude]])  // T−
        }
        let reader = try RepEReader.fit(
            dataset: dataset, template: template, capturedValues: captured,
            modelID: "org/m", revision: "abc")[0]
        #expect(reader.contrastMode == .unsupervisedTemplatePair)
        #expect(reader.orientationSeed == RepEReader.defaultOrientationSeed)
        #expect(reader.signConvention == .heldOutPairAgreement)
        // The direction is +x: H(T+) − H(T−) points that way for every row,
        // and the held-out rows agree.
        #expect(abs(reader.probe.direction[0]) > 0.99)
        #expect(try RepEReader.scoreActivation(reader, activation: [5, 0]) > 0)
        // Explained variance now MEANS the difference cloud's, and this cloud
        // has variance (unlike the supervised fixture's identical rows).
        let explained = try #require(reader.differenceCloudExplainedVariance)
        #expect(explained > 0 && explained <= 1 + 1e-5)
        #expect(reader.explainedVarianceBasis == "differenceCloud")

        // Seeded: the same seed reproduces the direction, a different one is
        // still a legitimate sample of the same equivalence class.
        let again = try RepEReader.fit(
            dataset: dataset, template: template, capturedValues: captured,
            modelID: "org/m", revision: "abc",
            orientationSeed: RepEReader.defaultOrientationSeed)[0]
        #expect(approx(again.probe.direction, reader.probe.direction))
        let other = try RepEReader.fit(
            dataset: dataset, template: template, capturedValues: captured,
            modelID: "org/m", revision: "abc", orientationSeed: 7)[0]
        #expect(other.orientationSeed == 7)
    }

    /// The orientation draw is the paper's `random.shuffle(d)` made
    /// reproducible: same seed, same ± sequence, on both engines.
    @Test func orientationSignsAreDeterministicAndBalancedEnough() throws {
        let a = RepEReader.orientationSigns(count: 16, seed: 231_001_405)
        let b = RepEReader.orientationSigns(count: 16, seed: 231_001_405)
        #expect(a == b)
        #expect(a.allSatisfy { $0 == 1 || $0 == -1 })
        #expect(RepEReader.orientationSigns(count: 16, seed: 1) != a)
        // Not a statistical claim — just that the stream is not constant.
        #expect(Set(a).count == 2)
    }

    // MARK: - Declarable rendering (paper's user_tag/assistant_tag analogue)

    @Test func rawRenderingRemainsTheDefaultAndTheStampIsAbsent() throws {
        let (dataset, template, captured) = fitFixture()
        let reader = try RepEReader.fit(
            dataset: dataset, template: template, capturedValues: captured,
            modelID: "org/m", revision: "abc")[0]
        #expect(reader.extractionRendering == nil)
        #expect(reader.renderingConvention == RepEReader.renderingConvention)
        #expect(reader.resolvedExtractionRendering.isRaw)
        let directory = try tempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = try RepEReader.saveArtifact(reader, to: directory)
        let object = try #require(
            try JSONSerialization.jsonObject(with: Data(contentsOf: url))
                as? [String: Any])
        // Absent, not {"mode":"raw"} — a raw fit's bytes stay what they were.
        #expect(object["extractionRendering"] == nil)
    }

    @Test func chatTemplateRenderingIsStampedAndChangesTheConvention() throws {
        let (dataset, template, captured) = fitFixture()
        let rendering = ExtractionRendering.chatTemplate()
        let reader = try RepEReader.fit(
            dataset: dataset, template: template, capturedValues: captured,
            modelID: "org/m", revision: "abc", extractionRendering: rendering)[0]
        #expect(reader.extractionRendering != nil)
        #expect(reader.renderingConvention == RepEReader.chatTemplateRenderingConvention)
        #expect(!reader.resolvedExtractionRendering.isRaw)
        let directory = try tempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = try RepEReader.saveArtifact(reader, to: directory)
        let reloaded = try RepEReader.loadArtifact(url: url)
        #expect(reloaded.extractionRendering == reader.extractionRendering)
        #expect(!reloaded.resolvedExtractionRendering.isRaw)
    }

    /// The marker guard's REASON depends on the rendering: under raw it is the
    /// double-BOS hazard; under a chat template the template supplies the
    /// markers, so what is left is turn-boundary forgery inside the content.
    /// The manual-`<s>` check is raw-only for the same reason.
    @Test func markerGuardReasonFollowsTheRendering() throws {
        expectReaderError(containing: "double-BOS") {
            _ = try RepEReader.renderReaderScaffold("a <bos> b", modelID: "org/m")
        }
        expectReaderError(containing: "forges a turn boundary") {
            _ = try RepEReader.renderReaderScaffold(
                "a <start_of_turn> b", modelID: "org/m",
                rendering: .chatTemplate())
        }
        expectReaderError(containing: "manual '<s>' BOS") {
            _ = try RepEReader.renderReaderScaffold("<s> hello", modelID: "org/m")
        }
        // Under a chat template a leading "<s>" in CONTENT is ordinary text.
        #expect(
            try RepEReader.renderReaderScaffold(
                "<s> hello", modelID: "org/m", rendering: .chatTemplate())
                == "<s> hello")
    }

    // MARK: - Legacy artifacts stay decodable

    /// A pre-2026-08-27 artifact: `pc1ExplainedVariance` with no basis, no
    /// contrastMode, no signConvention, no rendering. Every absent field means
    /// its documented legacy value, and the basis stamp says which number this
    /// artifact actually carries rather than relabelling it.
    @Test func legacyArtifactDecodesWithStampedLegacySemantics() throws {
        let json = """
            {
              "artifactType": "repe-reader-lat",
              "schemaVersion": 1,
              "modelID": "org/m",
              "revision": "abc",
              "substrate": "swift-mlx",
              "concept": "fear",
              "layer": 3,
              "templateID": "unnamed-scenario-v1",
              "templateHash": "th",
              "template": {
                "conceptSlot": false, "hash": "th", "id": "unnamed-scenario-v1",
                "latToken": "final", "text": "S: {{stimulus}} q"
              },
              "datasetHash": "dh",
              "latTokenPosition": "final",
              "readingPosition": "last token",
              "probe": {
                "direction": [1, 0], "projectionCenter": 0.5,
                "projectionScale": 2, "orientation": 1,
                "positiveMean": 1, "negativeMean": -1
              },
              "pc1ExplainedVariance": 0.87,
              "trainAccuracy": 1,
              "heldOutAccuracy": 0.8,
              "trainPairCount": 4,
              "heldOutPairCount": 2,
              "renderingConvention": "rawCompletion scaffold",
              "extractionDate": "2026-07-03T00:00:00Z"
            }
            """
        let reader = try JSONDecoder().decode(
            RepEReader.Artifact.self, from: Data(json.utf8))
        #expect(reader.differenceCloudExplainedVariance == 0.87)
        #expect(reader.explainedVarianceBasis == "alternatedRows")
        #expect(reader.contrastMode == .supervisedContent)
        #expect(reader.signConvention == .trainMajority)
        #expect(reader.signHeldOutAccuracy == nil)
        #expect(reader.orientationSeed == nil)
        #expect(reader.recommendedLayer == nil)
        #expect(reader.extractionRendering == nil)
        #expect(reader.resolvedExtractionRendering.isRaw)
        #expect(try RepEReader.scoreActivation(reader, activation: [3, 7]) == 1.25)
    }

    // MARK: - Derive-steering conversion applies the probe orientation

    /// Audit finding 1. `ScalarProbe.score` is `orientation · (a·d − centre)`,
    /// so a reader whose PC1 came out anti-aligned with the positive class
    /// stores a direction pointing AWAY from the concept. Shipping those bytes
    /// as a steering vector injected the concept backwards while every
    /// provenance stamp said forwards.
    @Test func derivedVectorAppliesTheProbeOrientation() throws {
        var reader = manualReader(layer: 1)
        reader.probe = SteeringVectorMath.ScalarProbe(
            direction: [1, 0], activationCenter: [1, 0],
            projectionCenter: 0.5, projectionScale: 2, orientation: -1,
            positiveMean: -1, negativeMean: 1)
        let bytes = try JSONEncoder().encode(reader)
        let (vectors, sidecar) = try RepEReader.deriveSteeringArtifact(
            from: reader, readerFileName: "reader-fear-layer1.json", readerBytes: bytes)
        // The stored probe direction is +x; the READING direction is −x.
        #expect(approx(vectors.perLayer[1], [-1, 0]))
        #expect(sidecar.readerProbeOrientation == -1)
        // A concept-positive activation must score positive through the probe
        // AND project positive onto the derived vector — the two agreeing is
        // the whole point of the fix.
        let conceptPositive: [Float] = [-4, 0]
        #expect(try RepEReader.scoreActivation(reader, activation: conceptPositive) > 0)
        #expect(SteeringVectorMath.dot(conceptPositive, vectors.perLayer[1]) > 0)

        // A +1 reader is unchanged: the fix is a sign, not a rewrite.
        let forward = manualReader(layer: 1)
        let (forwardVectors, forwardSidecar) = try RepEReader.deriveSteeringArtifact(
            from: forward, readerFileName: "r.json", readerBytes: Data())
        #expect(approx(forwardVectors.perLayer[1], [1, 0]))
        #expect(forwardSidecar.readerProbeOrientation == 1)
    }

    /// The derived sidecar carries the reader's identity AND its method, so an
    /// attached reader-derived concept can be verified without re-opening the
    /// reader file (which a bundle may not carry).
    @Test func derivedSidecarCarriesReaderMethodAndInstrumentPins() throws {
        let reader = manualReader(layer: 1)
        let (_, sidecar) = try RepEReader.deriveSteeringArtifact(
            from: reader, readerFileName: "reader-fear-layer1.json",
            readerBytes: Data())
        #expect(sidecar.extractionMethod == ExtractionMethod.repeReaderLAT.rawValue)
        #expect(sidecar.recipeMethod == "repeReaderLAT")
        #expect(sidecar.readerLayer == 1)
        #expect(sidecar.readerTemplateID == "unnamed-scenario-v1")
        #expect(sidecar.readerTemplateHash == "th")
        #expect(sidecar.readerContrastMode == "supervisedContent")
        #expect(sidecar.readerSignConvention == "trainMajority")
        #expect(sidecar.signConvention == "trainMajority")
        // The method resolves in the ExtractionMethod vocabulary — without
        // that, attaching the artifact was refused as an unknown method.
        let method = try #require(
            ExtractionMethod(rawValue: sidecar.extractionMethod ?? ""))
        #expect(method == .repeReaderLAT)
        #expect(!method.hasSourceConcept)
        #expect(method.sourceConceptAbsence?.kind.contains("RepE reader") == true)
    }

    // MARK: - Cross-engine twin literals for the new refusals

    /// The twin-literal idiom: this table and its Python counterpart
    /// (`test_repe_reader.REFUSAL_TWINS`) are two INDEPENDENT copies of the
    /// same words. Neither engine can reword a refusal — or quietly drop the
    /// repair off the end of one — without the other engine's suite going red.
    /// A refusal a researcher meets on the Mac and on the cluster has to be
    /// the same instruction, or "follow the repair" is advice about one
    /// machine.
    static let refusalTwins: [String: String] = [
        "chatTemplateMarker":
            "reader scaffold embeds turn marker '<start_of_turn>' inside the user "
            + "turn's content — under a chatTemplate rendering the template supplies "
            + "the markers, so an embedded one forges a turn boundary and moves the "
            + "LAT token off the generation prompt. Repair: write the scaffold as "
            + "plain content and let the template do the framing",
        "mixedShapes":
            "S: mixes content-pair rows (positiveStimulus/negativeStimulus) with "
            + "template-pair rows (stimulus) — the two produce different "
            + "differences, so one file cannot mean both. Repair: split them into "
            + "two datasets",
        "bothShapes":
            "S: row declares both 'stimulus' and a positive/negative pair — a "
            + "template-pair row holds ONE stimulus (the T+/T− instructions carry "
            + "the contrast). Repair: drop 'stimulus' for a content pair, or drop "
            + "'positiveStimulus'/'negativeStimulus' for a template pair",
        "contentPairsUnderTemplatePair":
            "template 'tp' declares a T+/T− instructionPair but the dataset holds "
            + "content pairs (positiveStimulus/negativeStimulus) — under a template "
            + "pair the contrast is the INSTRUCTION and a second stimulus would be a "
            + "confound. Repair: fit these pairs through a single-template reader "
            + "template, or rewrite the dataset as one-stimulus ('stimulus') rows",
        "singleStimulusUnderPlainTemplate":
            "the dataset holds one-stimulus ('stimulus') rows but template 'pl' "
            + "declares no instructionPair — there is nothing to contrast the "
            + "stimulus against. Repair: choose a template-pair template (one with "
            + "'instructionPair'), or rewrite the dataset as "
            + "positiveStimulus/negativeStimulus content pairs",
        "instructionPairWithoutSlot":
            "template 'x' declares an instructionPair but its text has no "
            + "{{instruction}} slot — the T+/T− instructions would never reach the "
            + "model. Repair: add {{instruction}} to the text, or drop "
            + "instructionPair to make this a single-template reader",
        "slotWithoutInstructionPair":
            "template 'x' has a {{instruction}} slot but declares no "
            + "instructionPair — nothing would fill it. Repair: add an "
            + "instructionPair with 'experimental' and 'reference', or remove the "
            + "slot",
        "emptyInstruction":
            "template 'x': instructionPair needs both 'experimental' (T+) and "
            + "'reference' (T−) — one empty instruction makes the contrast a "
            + "rendering artifact. Repair: write both instructions",
        "identicalInstructions":
            "template 'x': instructionPair's experimental and reference "
            + "instructions are identical — every difference would be exactly zero. "
            + "Repair: write two instructions that differ in the quality under study",
        "signFallbackNoHeldOut":
            "no held-out pairs (every row is split 'train'): the sign follows "
            + "train-label majority, the reference implementation's get_signs. "
            + "Repair: mark some rows with a non-'train' split so the paper's "
            + "held-out sign selection can run",
        "signFallbackTooFew":
            "1 held-out pair(s) projected off zero, below the minimum 2: a "
            + "one-pair vote is a coin flip wearing a validation split's authority, "
            + "so the sign follows train-label majority instead",
        "signFallbackTied":
            "held-out pairs split evenly (2 for, 2 against): the held-out set does "
            + "not discriminate at this layer, so the sign follows train-label "
            + "majority. Read this layer's heldOutAccuracy before trusting its "
            + "direction",
    ]

    /// Every new refusal, PRODUCED by the engine rather than transcribed.
    private func producedRefusals() -> [String: String] {
        var out: [String: String] = [:]
        func capture(_ label: String, _ body: () throws -> Void) {
            do {
                try body()
                Issue.record("\(label): expected a refusal")
            } catch let error as RepEReader.ReaderError {
                out[label] = error.reason
            } catch {
                Issue.record("\(label): unexpected error type: \(error)")
            }
        }
        capture("chatTemplateMarker") {
            _ = try RepEReader.renderReaderScaffold(
                "a <start_of_turn> b", modelID: "m", rendering: .chatTemplate())
        }
        capture("mixedShapes") {
            _ = try RepEReader.parsePairs(
                Data(
                    (#"{"concept":"c","positiveStimulus":"p","negativeStimulus":"n","templateID":"t"}"#
                        + "\n"
                        + #"{"concept":"c","stimulus":"s","templateID":"t"}"# + "\n")
                        .utf8),
                source: "S")
        }
        capture("bothShapes") {
            _ = try RepEReader.parsePairs(
                Data(
                    (#"{"concept":"c","stimulus":"s","positiveStimulus":"p","templateID":"t"}"#
                        + "\n").utf8),
                source: "S")
        }
        let pairTemplate = RepEReader.TaskTemplate(
            id: "tp", conceptSlot: false, text: "{{instruction}} S: {{stimulus}}",
            latToken: "final", hash: "h",
            instructionPair: .init(experimental: "a", reference: "b"))
        let plainTemplate = RepEReader.TaskTemplate(
            id: "pl", conceptSlot: false, text: "S: {{stimulus}}",
            latToken: "final", hash: "h")
        let contentPairs = RepEReader.Dataset(
            concept: "c",
            pairs: [
                RepEReader.Pair(
                    concept: "c", positiveStimulus: "p", negativeStimulus: "n",
                    templateID: "tp")
            ], hash: "h")
        let single = RepEReader.Dataset(
            concept: "c",
            pairs: [
                RepEReader.Pair.templatePair(
                    concept: "c", stimulus: "s", templateID: "pl")
            ], hash: "h")
        capture("contentPairsUnderTemplatePair") {
            _ = try RepEReader.resolveContrastMode(
                dataset: contentPairs, template: pairTemplate)
        }
        capture("singleStimulusUnderPlainTemplate") {
            _ = try RepEReader.resolveContrastMode(
                dataset: single, template: plainTemplate)
        }
        func template(
            _ text: String, _ pair: RepEReader.TaskTemplate.InstructionPair?
        ) -> RepEReader.TaskTemplate {
            RepEReader.TaskTemplate(
                id: "x", conceptSlot: false, text: text, latToken: "final",
                hash: "h", instructionPair: pair)
        }
        capture("instructionPairWithoutSlot") {
            try template("S: {{stimulus}}", .init(experimental: "a", reference: "b"))
                .validateInstructionSlot()
        }
        capture("slotWithoutInstructionPair") {
            try template("{{instruction}} S: {{stimulus}}", nil)
                .validateInstructionSlot()
        }
        capture("emptyInstruction") {
            try template(
                "{{instruction}} S: {{stimulus}}",
                .init(experimental: "a", reference: "")
            ).validateInstructionSlot()
        }
        capture("identicalInstructions") {
            try template(
                "{{instruction}} S: {{stimulus}}",
                .init(experimental: "same", reference: "same")
            ).validateInstructionSlot()
        }
        out["signFallbackNoHeldOut"] = RepEReader.heldOutSignFallbackReason(
            heldOutPairCount: 0, decided: 0, agree: 0, disagree: 0)
        out["signFallbackTooFew"] = RepEReader.heldOutSignFallbackReason(
            heldOutPairCount: 1, decided: 1, agree: 1, disagree: 0)
        out["signFallbackTied"] = RepEReader.heldOutSignFallbackReason(
            heldOutPairCount: 4, decided: 4, agree: 2, disagree: 2)
        return out
    }

    @Test func newRefusalsAreCrossEngineTwinLiterals() {
        let produced = producedRefusals()
        #expect(
            Set(produced.keys) == Set(Self.refusalTwins.keys),
            "a new refusal must be added to BOTH engines' twin-literal tables")
        for (label, expected) in Self.refusalTwins {
            #expect(produced[label] == expected, "\(label): \(produced[label] ?? "nil")")
        }
    }
}
