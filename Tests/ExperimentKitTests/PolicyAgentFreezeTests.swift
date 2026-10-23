import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit

/// A study whose agent carries an intervention policy freezes without force.
///
/// The capability battery cannot run an intervention policy, so the
/// `batteryEvidence` freeze gate could never be satisfied for such a
/// condition and `--force` — which stamps the whole study forced — was
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
    /// A policy agent that ALSO carries static steering (a vector injection).
    private static let mixedAgent = "mixed-agent"
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
        + "This study has no capability control for that agent as it runs. "
        + "Any reading of its static steering alone, labelled \"policy-agent "
        + "without its policy\", describes that steering, not the agent as it "
        + "runs."

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
                injections: condition == Self.mixedAgent
                    ? [.init(concept: "c", vectorArtifactID: "runs/x/c", layer: 1, alpha: 0.1)]
                    : [],
                promptMode: "chatAssistant",
                qwenThinkingEnabled: false, temperature: 0, systemPrompt: "",
                interventionPolicies: [Self.policyAgent, Self.mixedAgent].contains(condition)
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
            // study forced.
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

    // MARK: - The static part of a policy agent (ruling 2026-10-23)

    private func withoutPolicy(
        _ condition: String, accuracy: Double = 0.75, error: String? = nil
    ) -> CapabilityBatteryConditionResult {
        .init(
            condition: condition, batteryHash: "bh",
            notApplicable: Self.policyReason,
            withoutPolicy: error.map { .init(error: $0) }
                ?? .init(total: 4, correct: 3, accuracy: accuracy))
    }

    private func batteryGateRefusal(
        _ manifest: ExperimentManifest, _ rows: [CapabilityBatteryConditionResult],
        expectedHash: String? = nil
    ) -> String? {
        do {
            try FreezePolicy.checkVariantBatteryEvidence(
                manifest,
                facts: .init(
                    hasMatchingEvidence: true, results: rows,
                    expectedHash: expectedHash))
            return nil
        } catch let error as ExperimentError {
            return error.reason
        } catch {
            return "\(error)"
        }
    }

    @Test func onlyAPolicyAgentWithStaticSteeringOwesAStaticPartReading() throws {
        try withTempRoot {
            let manifest = try makePolicyStudy(
                name: "owes",
                conditions: [Self.policyAgent, Self.mixedAgent, Self.plainAgent])
            let byName = Dictionary(
                uniqueKeysWithValues: manifest.variantConditions.map { ($0.name, $0) })
            let mixed = try #require(byName[Self.mixedAgent])
            let policyOnly = try #require(byName[Self.policyAgent])
            let plain = try #require(byName[Self.plainAgent])
            #expect(FreezePolicy.owesStaticPartReading(mixed))
            #expect(!FreezePolicy.owesStaticPartReading(policyOnly))
            #expect(!FreezePolicy.owesStaticPartReading(plain))
            #expect(
                FreezePolicy.withoutPolicyLabel(Self.mixedAgent)
                    == "mixed-agent without its policy")
        }
    }

    @Test func theGateRequiresTheStaticPartReading() throws {
        try withTempRoot {
            let manifest = try makePolicyStudy(
                name: "static-gate", conditions: [Self.mixedAgent])
            // The not-applicable row alone no longer suffices for an agent
            // that also carries static steering.
            let missing = try #require(
                batteryGateRefusal(
                    manifest, [scored("baseline"), notApplicable(Self.mixedAgent)]))
            #expect(
                missing.hasPrefix(
                    "cannot freeze 'static-gate': matching validate evidence has "
                        + "no capability-battery reading of mixed-agent without "
                        + "its policy (its static steering alone)"))
            #expect(missing.contains(FreezePolicy.batteryScopePhrase))
            // Evidence from an older engine never carries the reading: the
            // refusal says which validate takes it, word for word the
            // server's sentence.
            #expect(
                missing.hasSuffix(", or freeze --force. " + FreezePolicy.staticPartReadingNote))
            #expect(
                FreezePolicy.staticPartReadingNote
                    == "Only validate from SteerLab 0.9.16 or later takes a policy "
                    + "agent's static-part reading, so evidence from an older engine "
                    + "never carries it: a runner on an older engine must be updated "
                    + "before it validates again")
            // A reading that could not be taken says why, and is missing.
            let failed = try #require(
                batteryGateRefusal(
                    manifest,
                    [scored("baseline"), withoutPolicy(Self.mixedAgent, error: "no vector")]))
            #expect(
                failed.contains(
                    "The capability battery could not run for mixed-agent "
                        + "without its policy: no vector."))
            // With the reading inside the row, the gate passes.
            #expect(
                batteryGateRefusal(
                    manifest, [scored("baseline"), withoutPolicy(Self.mixedAgent)]) == nil)
            // Drift is checked for the static-part reading too.
            let baselineOther = CapabilityBatteryConditionResult(
                condition: "baseline", batteryHash: "other", total: 12, correct: 12,
                accuracy: 1)
            let drifted = try #require(
                batteryGateRefusal(
                    manifest, [baselineOther, withoutPolicy(Self.mixedAgent)],
                    expectedHash: "other"))
            #expect(drifted.contains("drifted"))
            #expect(drifted.contains("mixed-agent without its policy"))

            // A policy agent with no static steering owes nothing more.
            let policyOnly = try makePolicyStudy(name: "policy-only-gate")
            #expect(
                batteryGateRefusal(
                    policyOnly, [scored("baseline"), notApplicable(Self.policyAgent)]) == nil)
            // The static part never stands in for a condition's own score.
            let plain = try makePolicyStudy(
                name: "plain-gate", conditions: [Self.plainAgent])
            #expect(
                batteryGateRefusal(
                    plain, [scored("baseline"), withoutPolicy(Self.plainAgent, accuracy: 1)])
                    != nil)
        }
    }

    /// The Python engine's evidence file, written by its own validation code
    /// (`Server/tests/test_battery_without_policy_fixture.py`), decodes here:
    /// the static-part reading lands in `withoutPolicy`, the row itself stays
    /// unscored, and the row satisfies this engine's gate.
    @Test func thePythonEnginesStaticPartRowDecodesAndSatisfiesTheGate() throws {
        let url = URL(filePath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appending(path: "Fixtures/cross-engine/validation-evidence-without-policy.json")
        struct Evidence: Decodable {
            let batteryResults: [CapabilityBatteryConditionResult]
        }
        let data = try Data(contentsOf: url)
        let evidence = try JSONDecoder().decode(Evidence.self, from: data)
        let rows = Dictionary(
            uniqueKeysWithValues: evidence.batteryResults.map { ($0.condition, $0) })
        let mixed = try #require(rows["agent-policy-and-vector"])
        #expect(!mixed.isScored)
        #expect(mixed.notApplicable == Self.policyReason)
        let reading = try #require(mixed.withoutPolicy)
        #expect(reading.isScored)
        #expect(reading.correct == 1 && reading.total == 2 && reading.accuracy == 0.5)
        #expect(reading.batteryFormat == 1 && reading.armingIsolated == false)
        #expect(rows["agent-policy-only"]?.withoutPolicy == nil)
        #expect(rows["agent-plain"]?.withoutPolicy == nil)

        // Re-encoded by this engine, the row keeps the shape: no score keys
        // on the row, the reading in its own object.
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        #expect(
            String(decoding: try encoder.encode(mixed), as: UTF8.self)
                == #"{"batteryHash":"d4167d6a465c561cf2b1764139ed05a362fb3b961821764825139001025de985","condition":"agent-policy-and-vector","notApplicable":"interventionPolicy","withoutPolicy":{"accuracy":0.5,"armingIsolated":false,"batteryFormat":1,"correct":1,"total":2}}"#)

        // The validation report box shows it under its own label.
        let display = RunResults.batteryRows(fromEvidenceJSON: data)
        let shown = try #require(display.first { $0.condition == "agent-policy-and-vector" })
        #expect(shown.unscoredLabel == "not applicable")
        #expect(
            shown.withoutPolicyLine
                == "agent-policy-and-vector without its policy: 50% (1/2). It "
                + "describes the static steering, not the agent as it runs.")
        #expect(display.first { $0.condition == "agent-policy-only" }?.withoutPolicyLine == nil)

        // The same row, under this study's condition name, satisfies the gate.
        try withTempRoot {
            let manifest = try makePolicyStudy(
                name: "python-static", conditions: [Self.mixedAgent])
            var renamed = mixed
            renamed.condition = Self.mixedAgent
            var baseline = try #require(rows["baseline"])
            baseline.batteryHash = mixed.batteryHash
            #expect(batteryGateRefusal(manifest, [baseline, renamed]) == nil)
        }
    }

    /// A key that means nothing on a scored row is never read there: a scored
    /// row cannot carry a second score.
    @Test func aScoredRowNeverCarriesAStaticPartReading() throws {
        let row = Data(
            #"{"condition":"baseline","batteryHash":"bh","total":2,"correct":2,"accuracy":1,"withoutPolicy":{"total":2,"correct":0,"accuracy":0}}"#
                .utf8)
        let decoded = try JSONDecoder().decode(
            CapabilityBatteryConditionResult.self, from: row)
        #expect(decoded.isScored && decoded.withoutPolicy == nil)
    }

    // MARK: - Facts, not verdicts (ruling 2026-10-23)

    @Test func theSettingsSummaryOfAForcedFreezeSaysSo() throws {
        try withTempRoot {
            _ = try makePolicyStudy(name: "forced-summary", conditions: [Self.plainAgent])
            let frozen = try ExperimentStore.freeze(name: "forced-summary", force: true)
            #expect(frozen.freezeForced == true)
            let skipped = (frozen.forcedGatesSkipped ?? []).map { "`\($0)`" }
                .joined(separator: ", ")
            let markdown = ExperimentStore.preregistrationMarkdown(frozen)
            #expect(
                markdown.contains(
                    "\n## Freeze\n\n- Frozen with force (`--force`). Gates skipped: "
                        + "\(skipped).\n- The manifest records this permanently, "
                        + "in `freezeForced` and `forcedGatesSkipped`.\n"))
            #expect(!markdown.contains("citable"))
            // An unforced study gets nothing new: its summary is what it was.
            var unforced = frozen
            unforced.freezeForced = nil
            unforced.forcedGatesSkipped = nil
            #expect(FreezePolicy.forcedFreezeSummaryLines(unforced).isEmpty)
            #expect(!ExperimentStore.preregistrationMarkdown(unforced).contains("## Freeze"))
            // The advisory states the skipped gates, not a verdict.
            let advisory = try #require(
                ExperimentStore.freezeAdvisories(for: frozen)
                    .first { $0.hasPrefix("forced freeze") })
            #expect(advisory.hasSuffix("; the manifest records this permanently"))
        }
    }

    @Test func theRunHeaderStatesFreezeFactsAndBatteryExemptions() throws {
        // A clean freeze whose snapshot stamps one exempt condition.
        let stamped = RunResults.manifestSnapshotReading(
            from: Data(
                #"{"name":"s","status":"frozen","capabilityBatteryNotApplied":[{"condition":"guided","reason":"interventionPolicy"}]}"#
                    .utf8))
        let clean = RunResults.withSnapshotFacts(
            RunResults.classify(
                manifestStatus: stamped.statusRaw, freezeForced: stamped.freezeForced,
                hasMatchingValidateEvidence: nil, stampedHashMatchesLive: true,
                manifestRevision: "abc", recordRevisions: []),
            reading: stamped)
        #expect(clean.runClass == .frozenEvidenceGrade)
        #expect(clean.label == "Frozen — every freeze check that applies passed")
        #expect(
            clean.batteryNotAppliedLines == [
                FreezePolicy.batteryNotAppliedSentence(
                    .init(condition: "guided", reason: Self.policyReason))
            ])

        // A legacy forced snapshot, shaped like a policy study frozen with
        // force before the exemption existed: forced, no stamp.
        let legacy = RunResults.manifestSnapshotReading(
            from: Data(
                #"{"name":"p","status":"frozen","freezeForced":true,"forcedGatesSkipped":["batteryEvidence"],"appVersion":"0.9.6"}"#
                    .utf8))
        let forced = RunResults.withSnapshotFacts(
            RunResults.classify(
                manifestStatus: legacy.statusRaw, freezeForced: legacy.freezeForced,
                hasMatchingValidateEvidence: nil, stampedHashMatchesLive: true,
                manifestRevision: "abc", recordRevisions: []),
            reading: legacy)
        #expect(forced.runClass == .forcedFreeze)
        #expect(forced.label == "Frozen with force")
        #expect(
            forced.detail
                == "frozen with --force; gates skipped: batteryEvidence (recorded "
                + "in forcedGatesSkipped)")
        // The two other cases, each stated as what it is: an empty list
        // (nothing would have failed) and no list at all.
        var skippedNothing = forced
        skippedNothing.forcedGatesSkipped = []
        #expect(
            skippedNothing.detail
                == "frozen with --force; gates skipped: none (every gate would "
                + "have passed)")
        var unlisted = forced
        unlisted.forcedGatesSkipped = nil
        #expect(
            unlisted.detail
                == "frozen with --force; the snapshot does not list which gates "
                + "were skipped")
        #expect(!unlisted.detail.contains("records this permanently"))
        // Nothing credits it with an exemption its manifest does not stamp.
        #expect(forced.batteryNotAppliedLines.isEmpty)
        for text in [clean.label, clean.detail, forced.label, forced.detail] {
            #expect(!text.contains("citable") && !text.contains("evidence-grade"))
        }
    }

    @Test func theBatteryColumnSaysWhyACellHasNoScore() {
        var model = RunResults.Model(runDirectory: URL(filePath: "/tmp/no-run"))
        var report = RunResults.Report()
        var baseline = RunResults.ConditionSummary(generations: 4)
        baseline.capabilityBatteryAccuracy = 0.6
        baseline.capabilityBatteryItemCount = 10
        report.conditions = [
            "baseline": baseline, "guided": .init(generations: 4),
            "stamped": .init(generations: 4), "legacy": .init(generations: 4),
        ]
        report.capabilityBatteryNotApplied = [
            .init(condition: "guided", reason: Self.policyReason)
        ]
        model.report = report
        model.classification.batteryNotApplied = [
            .init(condition: "stamped", reason: Self.policyReason)
        ]
        #expect(model.batteryCell("baseline") == .init(text: "60% of 10", help: nil))
        // Recorded in the run's report, or in the snapshot's stamp.
        for condition in ["guided", "stamped"] {
            #expect(
                model.batteryCell(condition)
                    == .init(
                        text: "not applicable",
                        help: FreezePolicy.batteryNotAppliedSentence(
                            .init(condition: condition, reason: Self.policyReason))))
        }
        // Recorded nowhere: said as such, never a bare dash.
        #expect(
            model.batteryCell("legacy")
                == .init(
                    text: "not recorded",
                    help: "No capability battery reading was recorded for legacy."))

        // The run report's own record of what its battery did not score.
        let parsed = RunResults.report(
            fromJSON: Data(
                #"{"conditions":{},"capabilityBatteryNotApplied":[{"condition":"guided","reason":"interventionPolicy"}]}"#
                    .utf8))
        #expect(
            parsed?.capabilityBatteryNotApplied
                == [.init(condition: "guided", reason: Self.policyReason)])
    }

    // MARK: - Review follow-ups (2026-10-23)

    /// An in-memory study whose one agent uses a policy and carries only an
    /// adapter as static steering.
    private func adapterOnlyPolicyStudy() -> ExperimentManifest {
        var manifest = ExperimentManifest(
            name: "adapter-policy", description: "", modelID: "test/model")
        manifest.modelRevision = String(repeating: "a", count: 40)
        manifest.variantConditions = [
            .init(
                name: "adapter-agent", artifactPath: "runs/model-variants/a.json",
                artifactHash: String(repeating: "c", count: 64),
                artifact: .init(
                    name: "adapter-agent", baseModelID: "test/model",
                    adapters: [
                        .init(
                            name: "a1", artifactPath: "runs/a", adapterDirectory: "runs/a",
                            adapterHash: String(repeating: "d", count: 64))
                    ],
                    injections: [], promptMode: "chatAssistant",
                    qwenThinkingEnabled: false, temperature: 0, systemPrompt: "",
                    interventionPolicies: policyAttachments()))
        ]
        return manifest
    }

    @Test func anAdapterAloneIsStaticSteeringThatOwesAReading() throws {
        let manifest = adapterOnlyPolicyStudy()
        let variant = try #require(manifest.variantConditions.first)
        #expect(FreezePolicy.hasStaticSteering(variant))
        #expect(FreezePolicy.owesStaticPartReading(variant))
        let refusal = try #require(
            batteryGateRefusal(
                manifest, [scored("baseline"), notApplicable("adapter-agent")]))
        #expect(refusal.contains("capability-battery reading of adapter-agent without its policy"))
        #expect(
            batteryGateRefusal(manifest, [scored("baseline"), withoutPolicy("adapter-agent")])
                == nil)
    }

    /// The gate's repair names the engines that take the static-part reading
    /// only when a policy agent owes one (server twin: `freeze_gate_repair`).
    @Test func theBatteryRepairNamesTheEngineVersionOnlyWhenAReadingIsOwed() throws {
        try withTempRoot {
            func repair(_ manifest: ExperimentManifest) -> String? {
                ExperimentStore.freezeGateTable(name: manifest.name, autoCommit: false)
                    .compactMap { $0.evaluate(manifest) }
                    .first { $0.gate == .batteryEvidence }?.repairAction
            }
            let owing = adapterOnlyPolicyStudy()
            let owed = repair(owing)
            #expect(owed?.hasSuffix(". " + FreezePolicy.staticPartReadingNote) == true)
            var policyOnly = owing
            policyOnly.variantConditions[0].artifact.adapters = []
            let notOwed = try #require(repair(policyOnly))
            #expect(!notOwed.contains(FreezePolicy.staticPartReadingNote))
        }
    }

    /// The data readiness checklist shows the static-part reading as its own
    /// row: owed until matching validate evidence records it, then recorded.
    @Test func theReadinessChecklistShowsTheStaticPartReadingOwedThenRecorded() throws {
        try withTempRoot {
            let manifest = try makePolicyStudy(
                name: "static-ready", conditions: [Self.mixedAgent, Self.policyAgent])
            let root = try #require(ExperimentStore.rootOverride)
            func row() -> DataRequirement? {
                StudyDataReadiness.requirements(for: manifest, workspaceRoot: root)
                    .first { $0.id == "capabilityBattery:withoutPolicy:mixed-agent" }
            }
            // A policy agent with no static steering gets no such row.
            #expect(
                !StudyDataReadiness.requirements(for: manifest, workspaceRoot: root)
                    .contains { $0.id == "capabilityBattery:withoutPolicy:policy-agent" })
            let owed = try #require(row())
            #expect(owed.title == "capability battery — mixed-agent without its policy")
            #expect(owed.kind == .capabilityBattery)
            #expect(owed.status != .notApplicable && owed.status != .present)
            #expect(owed.detail.hasPrefix("owed: the freeze gate requires"))
            #expect(owed.detail.contains(FreezePolicy.staticPartReadingNote))
            // The agent as it runs is still recorded as not applicable.
            #expect(
                StudyDataReadiness.requirements(for: manifest, workspaceRoot: root)
                    .contains {
                        $0.id == "capabilityBattery:notApplicable:mixed-agent"
                            && $0.status == .notApplicable
                    })

            try fabricateValidationEvidence(
                for: manifest,
                capabilityBattery: [
                    scored("baseline"), withoutPolicy(Self.mixedAgent),
                    notApplicable(Self.policyAgent),
                ])
            let recorded = try #require(row())
            #expect(recorded.status == .present)
            #expect(
                recorded.detail
                    == "recorded in the matching validate evidence: 75% (3/4). It "
                    + "describes the static steering, not the agent as it runs.")
        }
    }
}
