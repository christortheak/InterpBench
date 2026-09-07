import Foundation
import Testing
@testable import SteeringKit

@Suite struct ReaderEvidenceRoleTests {
    func fixture() -> (RepEReader.Dataset, RepEReader.TaskTemplate, [[[Float]]]) {
        let template = RepEReader.TaskTemplate(id: "example", conceptSlot: true,
            text: "{{concept}}: {{stimulus}}", latToken: "final", hash: "template")
        let pairs = ["train", "TRAIN", "test", "test", "FinalTest"].enumerated().map { i, split in
            RepEReader.Pair(id: "row-\(i)", concept: "signal", positiveStimulus: "positive-\(i)",
                negativeStimulus: "negative-\(i)", split: split, templateID: template.id)
        }
        let values: [Float] = [2, 0, 3, 0, 0, 2, 0, 3, 2, 0]
        let captured = values.map { [[$0, 0], [$0 * 2, 0]] }
        return (.init(concept: "signal", pairs: pairs, hash: "dataset"), template, captured)
    }

    @Test func finalRowsNeverChangeFitSignNormalizationOrRecommendedLayer() throws {
        let (dataset, template, captured) = fixture()
        #expect(dataset.train.count == 2 && dataset.heldOut.count == 2 && dataset.finalTest.count == 1)
        let fit = try RepEReader.fit(dataset: dataset, template: template,
            capturedValues: captured, modelID: "example/model", revision: nil)
        var changed = captured
        changed.swapAt(8, 9)
        let again = try RepEReader.fit(dataset: dataset, template: template,
            capturedValues: changed, modelID: "example/model", revision: nil)
        for (a, b) in zip(fit, again) {
            #expect(a.probe == b.probe)
            #expect(a.recommendedLayer == b.recommendedLayer)
            #expect(a.signConvention == .heldOutPairAgreement)
            // The classifier's orientation/threshold are train-fitted, even when
            // the reading direction's sign was selected by held-out pairs.
            #expect(a.signHeldOutAccuracy == 1)
            #expect(a.heldOutAccuracy == 0)
            #expect(a.finalTestAccuracy == 1 && b.finalTestAccuracy == 0)
            #expect(a.evidenceRoles?.heldOutAccuracy == "selection")
            #expect(a.evidenceRoles?.finalTestAccuracy == "finalEvaluation")
            #expect(a.evidenceRoles?.splitCounts == ["train": 2, "heldOut": 2, "finalTest": 1])
        }
        #expect(try RepEReader.fitTexts(dataset: dataset, template: template, modelID: "example/model").count == 10)
    }

    @Test func sharedRolesMatchPythonOwnerAndDerivedVector() throws {
        let (dataset, template, captured) = fixture()
        let artifact = try #require(RepEReader.fit(dataset: dataset, template: template,
            capturedValues: captured, modelID: "example/model", revision: nil).first)
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        let local = repository.appending(path: "Server/.venv.nosync/bin/python").path
        let executable = ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"]
            ?? (FileManager.default.isExecutableFile(atPath: local) ? local : "python3")
        process.arguments = [executable, "-c", """
        import json, sys
        from steerlab_server.steering.repe_reader import ReaderArtifact
        a = ReaderArtifact.from_dict(json.loads(sys.argv[1]))
        print(json.dumps(a.to_dict()))
        """, String(decoding: try JSONEncoder().encode(artifact), as: UTF8.self)]
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONPATH"] = repository.appending(path: "Server").path
        process.environment = environment
        let output = Pipe(), errors = Pipe()
        process.standardOutput = output; process.standardError = errors
        try process.run()
        let data = output.fileHandleForReading.readDataToEndOfFile()
        let diagnostics = errors.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        #expect(process.terminationStatus == 0, "\(String(decoding: diagnostics, as: UTF8.self))")
        let returned = try JSONDecoder().decode(RepEReader.Artifact.self, from: data)
        #expect(returned.evidenceRoles == artifact.evidenceRoles)
        #expect(returned.evidenceRoleNote == artifact.evidenceRoleNote)
        #expect(returned.finalTestAccuracy == artifact.finalTestAccuracy)
        #expect(returned.finalTestPairCount == artifact.finalTestPairCount)
        #expect(returned.splitOverlap == artifact.splitOverlap)
        let (_, sidecar) = try RepEReader.deriveSteeringArtifact(from: artifact,
            readerFileName: "reader.json", readerBytes: JSONEncoder().encode(artifact))
        #expect(sidecar.readerEvidenceRoles == artifact.evidenceRoles)
    }

    @Test func absentFinalTestIsNotZeroAndLegacyEncodingDoesNotInventEvidence() throws {
        let (all, template, captured) = fixture()
        let dataset = RepEReader.Dataset(concept: all.concept, pairs: Array(all.pairs.prefix(4)), hash: all.hash)
        var artifact = try #require(RepEReader.fit(dataset: dataset, template: template,
            capturedValues: Array(captured.prefix(8)), modelID: "example/model", revision: nil).first)
        #expect(artifact.finalTestAccuracy == nil && artifact.finalTestPairCount == nil)
        artifact.evidenceRoles = nil; artifact.evidenceRolesBasis = nil
        artifact.evidenceRoleNote = nil; artifact.splitOverlap = nil
        let data = try JSONEncoder().encode(artifact)
        let decoded = try JSONDecoder().decode(RepEReader.Artifact.self, from: data)
        let object = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(decoded)) as? [String: Any])
        for key in ["evidenceRoles", "evidenceRolesBasis", "evidenceRoleNote", "splitOverlap", "finalTestAccuracy", "finalTestPairCount"] {
            #expect(object[key] == nil)
        }
        #expect(decoded.resolvedEvidenceRoles.heldOutAccuracy == "selection")
    }

    @Test func overlapUsesCasefoldWhitespaceButNotCanonicalUnicodeEquality() throws {
        let (original, _, _) = fixture()
        var pairs = Array(original.pairs.prefix(2))
        pairs[0].positiveStimulus = "Straße\u{1f}Σ"; pairs[0].negativeStimulus = "n"
        pairs[1].positiveStimulus = "STRASSE σ"; pairs[1].negativeStimulus = "N"
        pairs[1].split = "finalTest"
        #expect(throws: RepEReader.ReaderError.self) {
            try RepEReader.checkSplitOverlap(.init(concept: "signal", pairs: pairs, hash: "fixture"))
        }
        pairs[0].positiveStimulus = "é"; pairs[1].positiveStimulus = "e\u{301}"
        try RepEReader.checkSplitOverlap(.init(concept: "signal", pairs: pairs, hash: "fixture"))
    }

    @Test func exactTrainOverlapIsCheckedForProgrammaticFitsAndParsedInputs() throws {
        let (original, template, captured) = fixture()
        var pairs = original.pairs
        pairs[3].positiveStimulus = "  POSITIVE-0 \n"
        pairs[3].negativeStimulus = "NEGATIVE-0"
        let dataset = RepEReader.Dataset(concept: original.concept, pairs: pairs, hash: "changed")
        #expect(throws: RepEReader.ReaderError.self) {
            try RepEReader.fit(dataset: dataset, template: template, capturedValues: captured,
                               modelID: "example/model", revision: nil)
        }
        let bytes = try pairs.map { String(decoding: try JSONEncoder().encode($0), as: UTF8.self) }.joined(separator: "\n")
        #expect(throws: RepEReader.ReaderError.self) {
            try RepEReader.parsePairs(Data(bytes.utf8), source: "fixture")
        }
    }
}
