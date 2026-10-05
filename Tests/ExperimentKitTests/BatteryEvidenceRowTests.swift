import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit

/// The battery rows that carry no score: an ERROR row and a NOT-APPLICABLE
/// row.
///
/// When an agent cannot be loaded during validation, the Python engine
/// records an `error` row for its condition — the condition, the battery
/// hash, the engine's account of what went wrong, and no score keys — and
/// goes on to score the other conditions. This side's decoder used to reject
/// the whole evidence file over that one row, so a study with one broken
/// agent appeared to have no validation evidence at all. The row is now read
/// as what it is, every other row is kept, and the error is shown against
/// its condition: in the freeze refusal, and in the validation report view.
///
/// The same view showed a not-applicable condition as a bare name with
/// nothing where the number would be. It now says "not applicable" and gives
/// the reason, in the sentence the freeze gates already use.
///
/// The evidence fixture is written by the Python engine's own validation
/// code (`Server/tests/test_battery_error_row_fixture.py`), not by hand.
extension ExperimentStoreTests {

    private static let engineError =
        "variant adapters need peft: pip install -e .[lora]"

    private func pythonEvidenceWithOneErrorRow() throws -> Data {
        let url = URL(filePath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appending(
                path: "Fixtures/cross-engine/validation-evidence-error-row.json")
        return try Data(contentsOf: url)
    }

    /// A model-output study whose variant conditions are plain agents — no
    /// adapters, no injections, no policies — written to the workspace the
    /// way a promoted agent is.
    private func makeAgentStudy(
        name: String, conditions: [String]
    ) throws -> ExperimentManifest {
        let root = try #require(ExperimentStore.rootOverride)
        var manifest = try ExperimentStore.create(
            name: name, description: "", modelID: "test/model",
            modelRevision: "abc123")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        for condition in conditions {
            let artifact = ModelVariantArtifact(
                name: condition, baseModelID: "test/model", adapters: [],
                injections: [], promptMode: "chatAssistant",
                qwenThinkingEnabled: false, temperature: 0, systemPrompt: "")
            let relativePath = "runs/model-variants/\(condition)/model-variant.json"
            let url = root.appending(path: relativePath)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            let data = try encoder.encode(artifact)
            try data.write(to: url)
            let digest = SHA256.hash(data: data)
                .map { String(format: "%02x", $0) }.joined()
            manifest.variantConditions.append(
                .init(
                    name: condition, artifactPath: relativePath,
                    artifactHash: digest, artifact: artifact))
        }
        try ExperimentStore.save(manifest)
        return manifest
    }

    /// An imported Python-engine validate run for `manifest`, carrying the
    /// engine-written evidence file.
    private func plantPythonValidateRun(
        for manifest: ExperimentManifest, evidence: Data
    ) throws -> URL {
        let directory = ExperimentStore.runsDirectory.appending(
            component: "20260721T000000000Z-exp-\(manifest.name)-validate")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try JSONEncoder().encode(manifest).write(
            to: directory.appending(component: "experiment.json"))
        try #"{"experiment":"\#(manifest.name)","concepts":{}}"#.write(
            to: directory.appending(component: "validation-report.json"),
            atomically: true, encoding: .utf8)
        try evidence.write(
            to: directory.appending(component: "validation-evidence.json"))
        return directory
    }

    // MARK: - The decoder

    @Test func aPythonWrittenEvidenceFileWithOneErrorRowKeepsEveryRow() throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "battery-rows-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try pythonEvidenceWithOneErrorRow().write(
            to: directory.appending(component: "validation-evidence.json"))

        // Before: nil — one row the decoder could not read hid all three.
        let rows = try #require(
            ExperimentStore.validationEvidenceBatteryResults(at: directory))

        #expect(rows.map(\.condition) == ["baseline", "agent-ok", "agent-broken"])
        // The scored rows are read exactly as they were written.
        for row in rows.prefix(2) {
            #expect(row.isScored)
            #expect(row.total == 2 && row.correct == 2 && row.accuracy == 1)
            #expect(row.batteryFormat == 1 && row.armingIsolated == false)
            #expect(row.error == nil && row.notApplicable == nil)
        }
        // The error row is an error row: the engine's own words, and no
        // reading. Its score fields are placeholders, never a measured zero.
        let broken = rows[2]
        #expect(broken.error == Self.engineError)
        #expect(!broken.isScored)
        #expect(broken.notApplicable == nil)
        #expect(broken.batteryHash == rows[0].batteryHash)
    }

    @Test func anErrorRowIsWrittenWithoutScoreKeysAndAMalformedRowStillFails() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let row = CapabilityBatteryConditionResult(
            condition: "agent-broken", batteryHash: "bh", error: "did not load")
        let written = String(decoding: try encoder.encode(row), as: UTF8.self)
        // No `accuracy: 0` for either engine to read as a measured failure.
        #expect(
            written
                == #"{"batteryHash":"bh","condition":"agent-broken","error":"did not load"}"#)
        #expect(
            try JSONDecoder().decode(
                CapabilityBatteryConditionResult.self, from: Data(written.utf8))
                == row)

        // A row that is neither a reading nor a stated reason is malformed,
        // and still fails: nothing here loosens what counts as a score.
        let neither = Data(#"{"condition":"x","batteryHash":"bh"}"#.utf8)
        #expect(throws: (any Error).self) {
            try JSONDecoder().decode(
                CapabilityBatteryConditionResult.self, from: neither)
        }
    }

    // MARK: - The freeze gate

    /// One broken agent: the validate run is still recognised as evidence,
    /// and the refusal is about that one condition and says why.
    @Test func oneBrokenAgentIsNamedWithItsErrorInsteadOfHidingTheEvidence() throws {
        try withTempRoot {
            let manifest = try makeAgentStudy(
                name: "one-broken-agent", conditions: ["agent-ok", "agent-broken"])
            _ = try plantPythonValidateRun(
                for: manifest, evidence: try pythonEvidenceWithOneErrorRow())

            // The run is evidence. Before, the evidence file failed to
            // decode, so no validate run matched at all.
            #expect(
                ExperimentStore.validationEvidence(
                    for: manifest, runSubstrate: WorkspaceScoping.serverSubstrate)
                    != nil)

            var refusal: FreezeRefusal?
            do {
                _ = try ExperimentStore.freeze(
                    name: "one-broken-agent",
                    runSubstrate: WorkspaceScoping.serverSubstrate)
                Issue.record("a study with an unscored required agent froze")
            } catch let error as ExperimentError {
                refusal = error.freezeRefusal
            }
            let refused = try #require(refusal)
            // The battery gate, and only it: validation evidence exists.
            #expect(refused.gate == .batteryEvidence)
            #expect(!refused.gates.contains(.validateEvidence))
            #expect(refused.reason.contains("condition(s): agent-broken."))
            #expect(!refused.reason.contains("agent-ok"))
            // The error, against its condition, in the engine's own words.
            #expect(
                refused.reason.contains(
                    "The capability battery could not run for agent-broken: "
                        + Self.engineError + "."))
        }
    }

    /// The rest of the evidence is kept: a study that does not require the
    /// broken condition freezes on the very same evidence file.
    @Test func theOtherRowsOfThatEvidenceFileStillSatisfyTheGate() throws {
        try withTempRoot {
            let manifest = try makeAgentStudy(
                name: "the-agent-that-loaded", conditions: ["agent-ok"])
            _ = try plantPythonValidateRun(
                for: manifest, evidence: try pythonEvidenceWithOneErrorRow())

            let frozen = try ExperimentStore.freeze(
                name: "the-agent-that-loaded",
                runSubstrate: WorkspaceScoping.serverSubstrate)

            #expect(frozen.status == .frozen)
            #expect(frozen.freezeForced == nil)
        }
    }

    /// An error row never stands in for a score, and without one the
    /// refusal reads exactly as it always has.
    @Test func theGateTreatsAnErrorRowAsMissingEvidence() throws {
        var manifest = ExperimentManifest(
            name: "gate", description: "", modelID: "test/model")
        manifest.variantConditions = [
            .init(
                name: "agent", artifactPath: "a.json", artifactHash: "h",
                artifact: ModelVariantArtifact(
                    name: "agent", baseModelID: "test/model", adapters: [],
                    injections: [], promptMode: "chatAssistant",
                    qwenThinkingEnabled: false, temperature: 0, systemPrompt: ""))
        ]
        let baseline = CapabilityBatteryConditionResult(
            condition: "baseline", batteryHash: "bh", total: 2, correct: 2,
            accuracy: 1)

        func refusal(_ rows: [CapabilityBatteryConditionResult]) -> String? {
            do {
                try FreezePolicy.checkVariantBatteryEvidence(
                    manifest,
                    facts: .init(hasMatchingEvidence: true, results: rows))
                return nil
            } catch { return "\(error)" }
        }

        let withError = try #require(
            refusal([
                baseline,
                .init(condition: "agent", batteryHash: "bh", error: "did not load."),
            ]))
        #expect(withError.contains("condition(s): agent."))
        // One full stop, whether or not the engine's message ended with one.
        #expect(withError.contains("could not run for agent: did not load. Repair"))

        // No row at all for the condition: the historical sentence, verbatim.
        #expect(
            refusal([baseline])
                == "cannot freeze 'gate': matching validate evidence has no "
                + "capability-battery results for condition(s): agent — re-run "
                + "'steerlab-cli experiment validate gate' (each variant "
                + "condition runs the pinned battery), or freeze --force")

        // And a scored row passes.
        #expect(
            refusal([
                baseline,
                .init(
                    condition: "agent", batteryHash: "bh", total: 2, correct: 1,
                    accuracy: 0.5),
            ]) == nil)
    }

    // MARK: - The validation report view's rows

    /// A Mac-engine validation report with a not-applicable row: the row
    /// says "not applicable" and gives the reason sentence.
    @Test func aNotApplicableConditionIsLabelledWithItsReason() throws {
        let json = """
            {"experiment":"policy-study","validation":{},
             "capabilityBattery":[
               {"condition":"baseline","batteryHash":"bh","total":20,
                "correct":19,"accuracy":0.95},
               {"condition":"policy-agent","batteryHash":"bh",
                "notApplicable":"interventionPolicy"}]}
            """
        let report = try #require(
            RunResults.validationReport(fromJSON: Data(json.utf8)))
        let baseline = try #require(report.capabilityBattery.first)
        // A scored row shows its number and nothing else.
        #expect(baseline.unscoredLabel == nil && baseline.unscoredExplanation == nil)

        let policy = try #require(report.capabilityBattery.last)
        #expect(policy.accuracy == nil)
        #expect(policy.unscoredLabel == "not applicable")
        // Word for word the sentence freeze and the checklist use.
        #expect(
            policy.unscoredExplanation
                == "The capability battery was not applied to policy-agent, "
                + "because its agent uses an intervention policy, which the "
                + "battery cannot run. This study has no capability control "
                + "for that agent.")
    }

    /// A Python-engine validate run keeps its battery rows in the evidence
    /// file, not in its report. The results model reads them from there, so
    /// the error is shown against its condition.
    @Test func aPythonValidateRunShowsItsErrorRowAgainstTheCondition() throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "validate-run-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try #"{"experiment":"one-broken-agent","concepts":{"formal":{"layer":10,"scenarioCount":8,"labeled":true,"scenarioAccuracy":0.75}}}"#
            .write(
                to: directory.appending(component: "validation-report.json"),
                atomically: true, encoding: .utf8)
        try pythonEvidenceWithOneErrorRow().write(
            to: directory.appending(component: "validation-evidence.json"))

        let model = RunResults.load(runDirectory: directory)

        let rows = try #require(model.validationReport).capabilityBattery
        #expect(rows.map(\.condition) == ["baseline", "agent-ok", "agent-broken"])
        #expect(rows[0].accuracy == 1 && rows[0].unscoredLabel == nil)
        #expect(rows[1].correct == 2 && rows[1].total == 2)
        let broken = rows[2]
        #expect(broken.accuracy == nil)
        #expect(broken.unscoredLabel == "could not run")
        #expect(
            broken.unscoredExplanation
                == "The capability battery could not run for agent-broken: "
                + Self.engineError + ".")
        // The remote loader fetches this file too, by the same name list.
        #expect(RunResults.ArtifactBytes.fileNames.contains("validation-evidence.json"))
    }

    /// A report that carries its own battery rows is not overridden by the
    /// evidence file beside it.
    @Test func aReportWithItsOwnBatteryRowsKeepsThem() throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "validate-run-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try #"{"validation":{},"capabilityBattery":[{"condition":"baseline","batteryHash":"bh","total":4,"correct":3,"accuracy":0.75}]}"#
            .write(
                to: directory.appending(component: "validation-report.json"),
                atomically: true, encoding: .utf8)
        try pythonEvidenceWithOneErrorRow().write(
            to: directory.appending(component: "validation-evidence.json"))

        let model = RunResults.load(runDirectory: directory)

        let rows = try #require(model.validationReport).capabilityBattery
        #expect(rows.map(\.condition) == ["baseline"])
        #expect(rows[0].accuracy == 0.75)
    }
}
