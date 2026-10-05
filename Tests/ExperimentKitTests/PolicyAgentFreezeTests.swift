import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit

/// A study whose agent carries an intervention policy freezes without force.
///
/// The capability battery cannot run an intervention policy, so the
/// `batteryEvidence` freeze gate could never be satisfied for such a
/// condition and `--force` — which stamps the whole study non-citable — was
/// the only way to freeze it. The gate now exempts exactly those conditions
/// and nothing else:
///
/// * the frozen manifest carries `capabilityBatteryNotApplied` (a lifecycle
///   stamp, outside the content hash) naming each exempted condition and the
///   reason, and is NOT marked forced;
/// * baseline and every condition without a policy owe battery evidence
///   exactly as before, and a not-applicable row never stands in for a score;
/// * the forced path and its two stamps are unchanged;
/// * the not-applicable battery row is written by this engine and read from
///   the Python engine's evidence.
///
/// Server twin: `Server/tests/test_policy_agent_freeze.py`.
extension ExperimentStoreTests {

    private static let policyAgent = "policy-agent"
    private static let plainAgent = "plain-agent"
    private static let policyReason =
        ExperimentManifest.BatteryNotApplied.interventionPolicyReason
    private static let policyStamp = [
        ExperimentManifest.BatteryNotApplied(
            condition: policyAgent, reason: policyReason)
    ]
    /// Word for word the server's sentence (`SENTENCE` in the twin suite).
    private static let policySentence =
        "The capability battery was not applied to policy-agent, because its "
        + "agent uses an intervention policy, which the battery cannot run. "
        + "This study has no capability control for that agent."

    private func hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// The smallest attachment the decoder accepts: exact text plus its
    /// digest, declaring a version-1 policy.
    private func policyAttachments() -> [JSONValue] {
        let text = "{\"schemaVersion\":1}"
        return [
            .object([
                "json": .string(text), "sha256": .string(hex(Data(text.utf8))),
            ])
        ]
    }

    /// A model-output study whose variant conditions are the named agents;
    /// `policy-agent` carries an intervention policy.
    private func makePolicyStudy(
        name: String, conditions: [String] = [policyAgent]
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
                qwenThinkingEnabled: false, temperature: 0, systemPrompt: "",
                interventionPolicies: condition == Self.policyAgent
                    ? policyAttachments() : nil)
            let relativePath = "runs/model-variants/\(condition)/model-variant.json"
            let url = root.appending(path: relativePath)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            let data = try encoder.encode(artifact)
            try data.write(to: url)
            manifest.variantConditions.append(
                .init(
                    name: condition, artifactPath: relativePath,
                    artifactHash: hex(data), artifact: artifact))
        }
        try ExperimentStore.save(manifest)
        return manifest
    }

    private func scored(_ condition: String) -> CapabilityBatteryConditionResult {
        .init(
            condition: condition, batteryHash: "bh", total: 12, correct: 12,
            accuracy: 1)
    }

    private func notApplicable(_ condition: String) -> CapabilityBatteryConditionResult {
        .init(
            condition: condition, batteryHash: "bh",
            notApplicable: Self.policyReason)
    }

    private func freezeRefusal(_ name: String) -> FreezeRefusal? {
        do {
            _ = try ExperimentStore.freeze(name: name)
            return nil
        } catch let error as ExperimentError {
            return error.freezeRefusal
        } catch {
            return nil
        }
    }

    // MARK: - The gate and the stamp

    @Test func policyAgentStudyFreezesWithoutForceAndCarriesTheStamp() throws {
        try withTempRoot {
            let manifest = try makePolicyStudy(name: "policy-study")
            try fabricateValidationEvidence(
                for: manifest,
                capabilityBattery: [
                    scored("baseline"), notApplicable(Self.policyAgent),
                ])

            let frozen = try ExperimentStore.freeze(name: "policy-study")

            #expect(frozen.status == .frozen)
            // Not forced: neither force stamp exists, so nothing marks the
            // study non-citable.
            #expect(frozen.freezeForced == nil)
            #expect(frozen.forcedGatesSkipped == nil)
            // The honest stamp: which condition, and why.
            #expect(frozen.capabilityBatteryNotApplied == Self.policyStamp)
            // On disk, verifying clean, with the stamp outside the hash.
            let reloaded = try ExperimentStore.load(name: "policy-study")
            #expect(reloaded.capabilityBatteryNotApplied == Self.policyStamp)
            #expect(ExperimentStore.verify(reloaded) == [])
            var unstamped = reloaded
            unstamped.capabilityBatteryNotApplied = nil
            #expect(
                ExperimentStore.manifestHash(unstamped)
                    == ExperimentStore.manifestHash(reloaded))
            #expect(ExperimentStore.manifestHash(reloaded) == frozen.freezeHash)
        }
    }

    /// The exemption is decided by the manifest, not by what the evidence
    /// happens to say: evidence with no row at all for the policy agent
    /// freezes too.
    @Test func theGateDoesNotNeedARowForTheExemptCondition() throws {
        try withTempRoot {
            let manifest = try makePolicyStudy(name: "policy-norow")
            try fabricateValidationEvidence(
                for: manifest, capabilityBattery: [scored("baseline")])
            let frozen = try ExperimentStore.freeze(name: "policy-norow")
            #expect(frozen.capabilityBatteryNotApplied == Self.policyStamp)
            #expect(frozen.freezeForced == nil)
        }
    }

    @Test func aPlainAgentBesideAPolicyAgentStillNeedsBatteryEvidence() throws {
        try withTempRoot {
            let manifest = try makePolicyStudy(
                name: "policy-mixed",
                conditions: [Self.policyAgent, Self.plainAgent])
            try fabricateValidationEvidence(
                for: manifest,
                capabilityBattery: [
                    scored("baseline"), notApplicable(Self.policyAgent),
                ])
            let refusal = try #require(freezeRefusal("policy-mixed"))
            #expect(refusal.gate == .batteryEvidence)
            // The refusal names the condition that owes evidence, and only
            // that one.
            #expect(refusal.reason.contains(Self.plainAgent))
            #expect(!refusal.reason.contains(Self.policyAgent))

            try fabricateValidationEvidence(
                for: manifest,
                capabilityBattery: [
                    scored("baseline"), notApplicable(Self.policyAgent),
                    scored(Self.plainAgent),
                ])
            let frozen = try ExperimentStore.freeze(name: "policy-mixed")
            #expect(frozen.capabilityBatteryNotApplied == Self.policyStamp)
            #expect(frozen.freezeForced == nil)
        }
    }

    /// Validate evidence is matched by scope, which does not cover
    /// conditions: a row written while a condition's agent carried a policy
    /// must not satisfy the gate for an agent the battery CAN run.
    @Test func aNotApplicableRowNeverStandsInForAScore() throws {
        try withTempRoot {
            let manifest = try makePolicyStudy(
                name: "plain-na", conditions: [Self.plainAgent])
            try fabricateValidationEvidence(
                for: manifest,
                capabilityBattery: [
                    scored("baseline"), notApplicable(Self.plainAgent),
                ])
            let refusal = try #require(freezeRefusal("plain-na"))
            #expect(refusal.gate == .batteryEvidence)
            #expect(refusal.reason.contains(Self.plainAgent))
        }
    }

    @Test func baselineEvidenceIsStillRequired() throws {
        try withTempRoot {
            let manifest = try makePolicyStudy(name: "policy-nobase")
            // No validate run at all: refused exactly as any variant study.
            #expect(freezeRefusal("policy-nobase")?.gate == .batteryEvidence)
            // A validate run that scored nothing for baseline: refused.
            try fabricateValidationEvidence(
                for: manifest,
                capabilityBattery: [notApplicable(Self.policyAgent)])
            let refusal = try #require(freezeRefusal("policy-nobase"))
            #expect(refusal.gate == .batteryEvidence)
            #expect(refusal.reason.contains("baseline"))
            #expect(!refusal.reason.contains(Self.policyAgent))
        }
    }

    // MARK: - The forced path is unchanged

    @Test func forcedFreezeOfAPlainVariantStudyIsUnchanged() throws {
        try withTempRoot {
            _ = try makePolicyStudy(
                name: "plain-forced", conditions: [Self.plainAgent])
            let frozen = try ExperimentStore.freeze(
                name: "plain-forced", force: true)
            #expect(frozen.freezeForced == true)
            #expect(frozen.forcedGatesSkipped == ["batteryEvidence"])
            // No policy agent, so no stamp: this study's frozen bytes are
            // what they always were.
            #expect(frozen.capabilityBatteryNotApplied == nil)
            let stored = try #require(
                ExperimentStore.manifestData(name: "plain-forced"))
            #expect(
                !String(decoding: stored, as: UTF8.self)
                    .contains("capabilityBatteryNotApplied"))
        }
    }

    /// Force still skips, and still stamps, the gate that would have failed
    /// — here baseline's missing battery evidence. The not-applied stamp is
    /// written beside the force stamps, not instead of them.
    @Test func forceKeepsItsMeaningOnAPolicyAgentStudy() throws {
        try withTempRoot {
            _ = try makePolicyStudy(name: "policy-forced")
            let frozen = try ExperimentStore.freeze(
                name: "policy-forced", force: true)
            #expect(frozen.freezeForced == true)
            #expect(frozen.forcedGatesSkipped == ["batteryEvidence"])
            #expect(frozen.capabilityBatteryNotApplied == Self.policyStamp)
        }
    }

    // MARK: - Additive schema

    @Test func theStampIsAdditiveAndClearedOnDuplicate() throws {
        try withTempRoot {
            let manifest = try makePolicyStudy(name: "policy-schema")
            // A manifest without the key decodes, and does not write it.
            let draftBytes = try #require(
                ExperimentStore.manifestData(name: "policy-schema"))
            #expect(
                !String(decoding: draftBytes, as: UTF8.self)
                    .contains("capabilityBatteryNotApplied"))
            #expect(manifest.capabilityBatteryNotApplied == nil)

            try fabricateValidationEvidence(
                for: manifest, capabilityBattery: [scored("baseline")])
            let frozen = try ExperimentStore.freeze(name: "policy-schema")

            // One with it round-trips through the model without loss.
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            let bytes = try encoder.encode(frozen)
            let decoded = try JSONDecoder().decode(
                ExperimentManifest.self, from: bytes)
            #expect(decoded == frozen)
            #expect(try encoder.encode(decoded) == bytes)
            let object = try #require(
                try JSONSerialization.jsonObject(with: bytes) as? [String: Any])
            let stamp = try #require(
                object["capabilityBatteryNotApplied"] as? [[String: String]])
            #expect(
                stamp == [
                    ["condition": Self.policyAgent, "reason": "interventionPolicy"]
                ])

            // A reason this engine does not know is carried, not refused.
            var future = object
            future["capabilityBatteryNotApplied"] = [
                ["condition": "x", "reason": "aLaterReason"]
            ]
            let carried = try JSONDecoder().decode(
                ExperimentManifest.self,
                from: try JSONSerialization.data(withJSONObject: future))
            #expect(carried.capabilityBatteryNotApplied?.first?.reason == "aLaterReason")
            #expect(
                ExperimentStore.manifestHash(carried)
                    == ExperimentStore.manifestHash(decoded))

            // Duplicate starts a fresh draft with no stamp.
            let copy = try ExperimentStore.duplicate(
                name: "policy-schema", as: "policy-schema-v2")
            #expect(copy.status == .draft)
            #expect(copy.capabilityBatteryNotApplied == nil)
        }
    }

    /// The battery gate is not asked of a multi-agent study at all, so there
    /// is nothing to exempt and nothing to stamp; and only an attached,
    /// non-empty policy list on a concrete agent is an exemption.
    @Test func theExemptionIsExactlyThePolicyCondition() throws {
        try withTempRoot {
            var manifest = try makePolicyStudy(
                name: "policy-scope",
                conditions: [Self.policyAgent, Self.plainAgent])
            #expect(FreezePolicy.batteryNotApplied(manifest) == Self.policyStamp)
            #expect(
                FreezePolicy.batteryExemptionReason(manifest.variantConditions[1])
                    == nil)

            var emptyList = manifest.variantConditions[0]
            emptyList.artifact.interventionPolicies = []
            #expect(FreezePolicy.batteryExemptionReason(emptyList) == nil)

            var forward = manifest.variantConditions[0]
            forward.fromPromotion = .init(concept: "c")
            #expect(FreezePolicy.batteryExemptionReason(forward) == nil)

            manifest.studyKind = .multiAgent
            #expect(FreezePolicy.batteryNotApplied(manifest).isEmpty)
        }
    }

    // MARK: - The evidence rows

    @Test func theNotApplicableRowIsWrittenWithoutScoresAndReadBack() throws {
        try withTempRoot {
            let manifest = try makePolicyStudy(name: "policy-rows")
            try fabricateValidationEvidence(
                for: manifest,
                capabilityBattery: [
                    scored("baseline"), notApplicable(Self.policyAgent),
                ])
            let directory = try #require(
                ExperimentStore.validationEvidence(for: manifest))
            let written = try #require(
                try JSONSerialization.jsonObject(
                    with: Data(
                        contentsOf: directory.appending(
                            component: "validation-evidence.json")))
                    as? [String: Any])
            let rows = try #require(written["batteryResults"] as? [[String: Any]])
            // The cross-engine row: a reason and NO score keys — a zero
            // accuracy would read as a measured failure on either engine.
            #expect(Set(rows[1].keys) == ["condition", "batteryHash", "notApplicable"])
            #expect(rows[1]["notApplicable"] as? String == "interventionPolicy")
            // A scored row is byte-for-byte what it always was.
            #expect(
                Set(rows[0].keys)
                    == ["condition", "batteryHash", "total", "correct", "accuracy"])

            let read = try #require(
                ExperimentStore.validationEvidenceBatteryResults(at: directory))
            #expect(read == [scored("baseline"), notApplicable(Self.policyAgent)])
            #expect(read[1].notApplicable == Self.policyReason)
        }
    }

    /// The Python engine's row, as `validation_workflow._battery_results`
    /// writes it, decodes here. Its `error` row for an agent that failed to
    /// load decodes too, as an error row and never as a score
    /// (`BatteryEvidenceRowTests` holds the rest of that behaviour).
    @Test func thePythonEnginesNotApplicableRowDecodes() throws {
        let python = Data(
            #"{"condition":"policy-agent","batteryHash":"bh","notApplicable":"interventionPolicy"}"#
                .utf8)
        let row = try JSONDecoder().decode(
            CapabilityBatteryConditionResult.self, from: python)
        #expect(row == notApplicable(Self.policyAgent))

        let scoredRow = Data(
            #"{"condition":"baseline","batteryHash":"bh","total":12,"correct":12,"accuracy":1,"batteryFormat":2,"armingIsolated":true}"#
                .utf8)
        let decoded = try JSONDecoder().decode(
            CapabilityBatteryConditionResult.self, from: scoredRow)
        #expect(decoded.notApplicable == nil)
        #expect(decoded.batteryFormat == 2 && decoded.armingIsolated == true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        #expect(
            String(decoding: try encoder.encode(decoded), as: UTF8.self)
                == #"{"accuracy":1,"armingIsolated":true,"batteryFormat":2,"batteryHash":"bh","condition":"baseline","correct":12,"total":12}"#)

        let errorRow = Data(
            #"{"condition":"plain-agent","batteryHash":"bh","error":"adapter directory not found"}"#
                .utf8)
        let failed = try JSONDecoder().decode(
            CapabilityBatteryConditionResult.self, from: errorRow)
        #expect(failed.error == "adapter directory not found")
        #expect(!failed.isScored && failed.notApplicable == nil)
    }

    /// An imported Python-engine validate run whose battery rows include the
    /// not-applicable row satisfies a server-bound freeze on the Mac.
    @Test func anImportedPythonValidateRunFreezesAPolicyStudyHere() throws {
        try withTempRoot {
            let manifest = try makePolicyStudy(name: "policy-imported")
            let directory = ExperimentStore.runsDirectory.appending(
                component: "20260721T000000000Z-exp-policy-imported-validate")
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            try JSONEncoder().encode(manifest).write(
                to: directory.appending(component: "experiment.json"))
            try #"{"experiment":"policy-imported","concepts":{}}"#.write(
                to: directory.appending(component: "validation-report.json"),
                atomically: true, encoding: .utf8)
            try #"""
            {"schemaVersion":1,"task":"validate","experiment":"policy-imported",
             "substrate":"python-hf-transformers","reportFile":"validation-report.json",
             "validationScopeHash":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
             "batteryResults":[
               {"condition":"baseline","batteryHash":"bh","total":4,"correct":4,
                "accuracy":1.0,"batteryFormat":1,"armingIsolated":false},
               {"condition":"policy-agent","batteryHash":"bh",
                "notApplicable":"interventionPolicy"}]}
            """#.write(
                to: directory.appending(component: "validation-evidence.json"),
                atomically: true, encoding: .utf8)

            let frozen = try ExperimentStore.freeze(
                name: "policy-imported",
                runSubstrate: WorkspaceScoping.serverSubstrate)
            #expect(frozen.status == .frozen)
            #expect(frozen.freezeForced == nil)
            #expect(frozen.capabilityBatteryNotApplied == Self.policyStamp)
        }
    }

    // MARK: - Where it shows

    @Test func readinessAdvisoriesAndTheSettingsSummarySayWhy() throws {
        try withTempRoot {
            let manifest = try makePolicyStudy(name: "policy-shown")
            try fabricateValidationEvidence(
                for: manifest, capabilityBattery: [scored("baseline")])

            // The sentence, word for word the server's.
            #expect(
                ExperimentStore.batteryNotAppliedSentences(manifest)
                    == [Self.policySentence])
            #expect(
                ExperimentStore.freezeAdvisories(for: manifest)
                    .contains(Self.policySentence))

            // Freeze readiness: ready, with the gate shown as not applicable
            // — its own list, not an unmet gate and not repeated among the
            // advisories.
            let readiness = ExperimentStore.freezeReadiness(for: manifest)
            #expect(readiness.ready, "unexpected gates: \(readiness.unmetGates)")
            #expect(readiness.notApplicable == [Self.policySentence])
            #expect(!readiness.advisories.contains(Self.policySentence))

            // The readiness checklist: a not-applicable battery row for the
            // condition, with the reason, and never a blocker.
            let root = try #require(ExperimentStore.rootOverride)
            let requirements = StudyDataReadiness.requirements(
                for: manifest, workspaceRoot: root)
            let row = try #require(
                requirements.first { $0.status == .notApplicable })
            #expect(row.kind == .capabilityBattery)
            #expect(row.title == "capability battery — policy-agent")
            #expect(row.detail == Self.policySentence)
            let summary = StudyDataReadiness.summary(requirements)
            #expect(summary.notApplicableCount == 1)
            #expect(!summary.blockers.contains(row))
            #expect(summary.line.hasSuffix(" · 1 not applicable"))

            // The generated settings summary names the exempted condition.
            let frozen = try ExperimentStore.freeze(name: "policy-shown")
            let markdown = ExperimentStore.preregistrationMarkdown(frozen)
            #expect(markdown.contains("## Capability battery\n\n- \(Self.policySentence)"))
            let written = try String(
                contentsOf: ExperimentStore.directory.appending(
                    components: "policy-shown",
                    ExperimentStore.preregistrationFilename),
                encoding: .utf8)
            #expect(written.contains("- \(Self.policySentence)"))
        }
    }

    /// A study with no policy agent reads exactly as it did on every surface.
    @Test func aStudyWithoutAPolicyAgentShowsNothingNew() throws {
        try withTempRoot {
            let manifest = try makePolicyStudy(
                name: "plain-shown", conditions: [Self.plainAgent])
            #expect(ExperimentStore.batteryNotAppliedSentences(manifest).isEmpty)
            #expect(ExperimentStore.freezeReadiness(for: manifest).notApplicable.isEmpty)
            let root = try #require(ExperimentStore.rootOverride)
            let requirements = StudyDataReadiness.requirements(
                for: manifest, workspaceRoot: root)
            #expect(!requirements.contains { $0.status == .notApplicable })
            let summary = StudyDataReadiness.summary(requirements)
            #expect(summary.notApplicableCount == 0)
            #expect(!summary.line.contains("not applicable"))
            #expect(
                !ExperimentStore.preregistrationMarkdown(manifest)
                    .contains("## Capability battery"))
        }
    }
}
