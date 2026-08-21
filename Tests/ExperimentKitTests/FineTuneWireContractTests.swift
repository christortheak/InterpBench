import Foundation
import Testing

@testable import ExperimentKit

/// The LoRA-readiness wire contract, client side
/// (`docs/CLUSTER-LORA-READINESS.md` §3 + the implementation contract §6).
///
/// These tests exist because the Swift app and the Python server were built
/// against the same written contract by different hands: a key spelled
/// `base_model_id` here or `instruction_chat` on the wire would fail only at
/// run time on a cluster node, minutes into a queued allocation. So the
/// assertions are deliberately literal — exact key SETS, exact spellings,
/// explicit nulls — rather than "it round-trips".
@Suite struct FineTuneWireContractTests {

    private func object(_ value: some Encodable) throws -> [String: Any] {
        let data = try JSONEncoder().encode(value)
        let parsed = try JSONSerialization.jsonObject(with: data)
        return try #require(parsed as? [String: Any])
    }

    private func sampleRequest(
        evidenceGrade: Bool = false,
        expectedPlanHash: String? = nil,
        controlArm: RemoteFineTuneRequest.ControlArm? = .init(
            kind: "shuffledAssistantPairing")
    ) -> RemoteFineTuneRequest {
        var hyperparameters = RemoteFineTuneRequest.Hyperparameters()
        hyperparameters.rank = 16
        hyperparameters.alpha = 32
        hyperparameters.epochs = 3
        return RemoteFineTuneRequest(
            baseModelID: "google/gemma-3-27b-it",
            revision: "0123456789abcdef0123456789abcdef01234567",
            name: "stance-crit-lora-v1",
            trainingMode: RemoteFineTuneRequest.wireTrainingMode(.instructionChat),
            evidenceGrade: evidenceGrade,
            dataset: RemoteFineTuneRequest.Dataset(
                bundleID: "stance-lora-family-v1",
                manifestPath: "adapters/stance-lora-family-v1-manifest.json",
                manifestHash: "f0f0",
                files: [
                    RemoteFineTuneRequest.DatasetFile(
                        role: "train",
                        path: "adapters/x/training/train.jsonl",
                        sha256: "aaaa",
                        content: #"{"user":"q","assistant":"a"}"# + "\n"),
                    RemoteFineTuneRequest.DatasetFile(
                        role: "validation",
                        path: "adapters/x/validation/validation.jsonl",
                        sha256: "bbbb",
                        content: nil),
                ]),
            hyperparameters: hyperparameters,
            selectionMetric: "validationLoss",
            controlArm: controlArm,
            expectedPlanHash: expectedPlanHash)
    }

    // MARK: - v2 request body

    @Test func v2TrainBodyEncodesTheContractKeysExactly() throws {
        let body = try object(sampleRequest())

        #expect(
            Set(body.keys) == [
                "schemaVersion", "baseModelID", "revision", "name",
                "trainingMode", "evidenceGrade", "dataset", "hyperparameters",
                "selectionMetric", "controlArm", "expectedPlanHash",
            ])
        #expect(body["schemaVersion"] as? Int == 2)
        #expect(body["baseModelID"] as? String == "google/gemma-3-27b-it")
        // camelCase on the wire; the Python side maps it to instruction_chat.
        // The app's own enum raw value is instruction_chat — sending THAT is
        // the bug this asserts against.
        #expect(body["trainingMode"] as? String == "instructionChat")
        #expect(FineTuneTrainingMode.instructionChat.rawValue == "instruction_chat")
        #expect(
            RemoteFineTuneRequest.wireTrainingMode(.document) == "document")
        #expect(body["evidenceGrade"] as? Bool == false)
        #expect(body["selectionMetric"] as? String == "validationLoss")
        // Unset optionals are explicit nulls, not omitted keys.
        #expect(body["expectedPlanHash"] is NSNull)
    }

    @Test func v2DatasetAndFilesEncodeTheContractKeys() throws {
        let body = try object(sampleRequest())
        let dataset = try #require(body["dataset"] as? [String: Any])
        #expect(
            Set(dataset.keys) == ["bundleID", "manifestPath", "manifestHash", "files"])
        #expect(dataset["bundleID"] as? String == "stance-lora-family-v1")
        #expect(
            dataset["manifestPath"] as? String
                == "adapters/stance-lora-family-v1-manifest.json")

        let files = try #require(dataset["files"] as? [[String: Any]])
        #expect(files.count == 2)
        #expect(Set(files[0].keys) == ["role", "path", "sha256", "content"])
        #expect(files[0]["role"] as? String == "train")
        #expect(files[0]["path"] as? String == "adapters/x/training/train.jsonl")
        #expect(files[0]["sha256"] as? String == "aaaa")
        #expect(files[0]["content"] as? String == #"{"user":"q","assistant":"a"}"# + "\n")
        // content: null = a server-resident path the server resolves itself.
        #expect(files[1]["content"] is NSNull)
        #expect(files[1]["role"] as? String == "validation")
    }

    @Test func v2HyperparametersEncodeEveryContractKey() throws {
        let body = try object(sampleRequest())
        let hyperparameters = try #require(body["hyperparameters"] as? [String: Any])
        #expect(
            Set(hyperparameters.keys) == [
                "rank", "alpha", "dropout", "learningRate", "epochs", "maxSteps",
                "batchSize", "gradientAccumulation", "warmupSteps", "lrSchedule",
                "maxGradNorm", "weightDecay", "seed", "maxSequenceTokens",
                "longDocumentPolicy", "chunkOverlapTokens", "evalIntervalSteps",
                "checkpointIntervalSteps", "targetModules", "dtype",
            ])
        #expect(hyperparameters["rank"] as? Int == 16)
        #expect(hyperparameters["alpha"] as? Double == 32)
        #expect(hyperparameters["epochs"] as? Int == 3)
        #expect(hyperparameters["lrSchedule"] as? String == "linear")
        #expect(hyperparameters["longDocumentPolicy"] as? String == "split")
        #expect(hyperparameters["chunkOverlapTokens"] as? Int == 64)
        #expect(hyperparameters["maxSequenceTokens"] as? Int == 512)
        #expect(hyperparameters["dtype"] as? String == "auto")
        #expect(
            hyperparameters["targetModules"] as? [String]
                == ["q_proj", "k_proj", "v_proj", "o_proj"])
        #expect(hyperparameters["maxSteps"] is NSNull)
        #expect(hyperparameters["evalIntervalSteps"] is NSNull)
        #expect(hyperparameters["checkpointIntervalSteps"] is NSNull)
    }

    @Test func controlArmEncodesKindAndOnlyDeclaresAgainstWhenDeclared() throws {
        let shuffled = try object(sampleRequest())
        let arm = try #require(shuffled["controlArm"] as? [String: Any])
        #expect(Set(arm.keys) == ["kind"])
        #expect(arm["kind"] as? String == "shuffledAssistantPairing")

        let declared = try object(
            sampleRequest(
                controlArm: .init(
                    kind: "declaredNeutralizedDataset",
                    declaredAgainst: "stance-crit-lora-v1")))
        let declaredArm = try #require(declared["controlArm"] as? [String: Any])
        #expect(Set(declaredArm.keys) == ["kind", "declaredAgainst"])
        #expect(declaredArm["declaredAgainst"] as? String == "stance-crit-lora-v1")

        let none = try object(sampleRequest(controlArm: nil))
        #expect(none["controlArm"] is NSNull)
    }

    // MARK: - submit body (the v2 body FLATTENED + resources + force)

    @Test func submitBodyFlattensTheRequestAndAddsResourcesAndForce() throws {
        let body = try object(
            FineTuneSubmitBody(
                request: sampleRequest(
                    evidenceGrade: true, expectedPlanHash: "c0ffee"),
                resources: ["gpus": .string("1"), "walltime": .string("08:00:00")],
                force: false))

        #expect(
            Set(body.keys) == [
                "schemaVersion", "baseModelID", "revision", "name",
                "trainingMode", "evidenceGrade", "dataset", "hyperparameters",
                "selectionMetric", "controlArm", "expectedPlanHash",
                "resources", "force",
            ])
        #expect(body["evidenceGrade"] as? Bool == true)
        #expect(body["expectedPlanHash"] as? String == "c0ffee")
        #expect(body["force"] as? Bool == false)
        let resources = try #require(body["resources"] as? [String: Any])
        #expect(resources["gpus"] as? String == "1")
        #expect(resources["walltime"] as? String == "08:00:00")
    }

    // MARK: - Route refusals made client-side

    @Test func daemonRouteRefusesEvidenceGradeBeforeSending() async throws {
        let client = ClusterClient(
            profile: ClusterConnectionProfile(
                baseURL: try #require(URL(string: "http://127.0.0.1:9"))))
        await #expect(throws: RemoteFineTuneError.self) {
            _ = try await client.fineTuneTrainV2(sampleRequest(evidenceGrade: true))
        }
    }

    @Test func slurmRouteRefusesEvidenceGradeWithoutAConfirmedPlan() async throws {
        let client = ClusterClient(
            profile: ClusterConnectionProfile(
                baseURL: try #require(URL(string: "http://127.0.0.1:9"))))
        await #expect(throws: RemoteFineTuneError.self) {
            _ = try await client.fineTuneSubmit(
                sampleRequest(evidenceGrade: true, expectedPlanHash: nil))
        }
    }

    // MARK: - Capability gating

    private func capabilities(_ json: String) throws -> ClusterCapabilities {
        try JSONDecoder().decode(
            ClusterCapabilities.self, from: Data(json.utf8))
    }

    @Test func remoteFineTuneBlockDecodesAndGatesEvidenceGrade() throws {
        let full = try capabilities(
            """
            {"serverVersion": "1.0", "remoteFineTune": {
              "schemaVersion": 2, "explicitSplits": true, "documentRows": true,
              "instructionChatAssistantMask": true, "checkpointResume": true,
              "revisionPinRequired": true, "walltimePreflight": true,
              "planEndpoint": true, "slurmSubmission": false}}
            """)
        #expect(full.fineTuneSchemaVersion == 2)
        #expect(full.supportsEvidenceGradeFineTune)
        #expect(full.missingEvidenceGradeFineTuneCapabilities.isEmpty)
        #expect(full.supportsStructuredFineTuneUpload)
        #expect(full.supportsFineTunePlanEndpoint)
        #expect(full.supportsFineTuneWalltimePreflight)
        // Slurm is announced false here: the block is satisfied, but the
        // evidence-grade EXECUTION path is not available on this deployment.
        #expect(!full.supportsFineTuneSlurmSubmission)
    }

    @Test func oneMissingFlagFailsTheGateAndIsNamed() throws {
        let noMask = try capabilities(
            """
            {"remoteFineTune": {
              "schemaVersion": 2, "explicitSplits": true, "documentRows": true,
              "checkpointResume": true, "revisionPinRequired": true}}
            """)
        #expect(!noMask.supportsEvidenceGradeFineTune)
        #expect(!noMask.supportsFineTuneInstructionChat)
        #expect(
            noMask.missingEvidenceGradeFineTuneCapabilities
                == ["instructionChatAssistantMask"])
        // Structured upload is still available — a weaker, honest route.
        #expect(noMask.supportsStructuredFineTuneUpload)

        let v1Schema = try capabilities(
            """
            {"remoteFineTune": {
              "schemaVersion": 1, "explicitSplits": true, "documentRows": true,
              "instructionChatAssistantMask": true, "checkpointResume": true,
              "revisionPinRequired": true}}
            """)
        #expect(!v1Schema.supportsEvidenceGradeFineTune)
        #expect(!v1Schema.supportsStructuredFineTuneUpload)
        #expect(
            v1Schema.missingEvidenceGradeFineTuneCapabilities == ["schemaVersion>=2"])
    }

    @Test func absentBlockMeansTheOldContract() throws {
        let old = try capabilities(#"{"serverVersion": "0.9", "engine": "python"}"#)
        #expect(old.remoteFineTune == nil)
        #expect(old.fineTuneSchemaVersion == 0)
        #expect(!old.supportsEvidenceGradeFineTune)
        #expect(!old.supportsStructuredFineTuneUpload)
        #expect(
            old.missingEvidenceGradeFineTuneCapabilities.contains("schemaVersion>=2"))
        #expect(
            old.missingEvidenceGradeFineTuneCapabilities.contains("explicitSplits"))
    }

    /// Same rule as `supportsWorkspaceSwitch`: the gate survives a key move
    /// into the flat `features` object.
    @Test func dottedFeatureSpellingSatisfiesTheGate() throws {
        let dotted = try capabilities(
            """
            {"features": {
              "remoteFineTune.schemaVersion": 2,
              "remoteFineTune.explicitSplits": true,
              "remoteFineTune.documentRows": true,
              "remoteFineTune.instructionChatAssistantMask": true,
              "remoteFineTune.checkpointResume": true,
              "remoteFineTune.revisionPinRequired": true,
              "remoteFineTune.planEndpoint": true}}
            """)
        #expect(dotted.remoteFineTune == nil)
        #expect(dotted.fineTuneSchemaVersion == 2)
        #expect(dotted.supportsEvidenceGradeFineTune)
        #expect(dotted.supportsFineTunePlanEndpoint)
        #expect(!dotted.supportsFineTuneSlurmSubmission)
    }

    // MARK: - Plan decoding + the confirmation summary

    @Test func planDecodesTolerantlyAndSummarizesWhatTheResearcherConfirms() throws {
        let json = """
            {"plan": {
              "resolvedRevision": "0123456789abcdef0123456789abcdef01234567",
              "trainingMode": "instruction_chat", "evidenceGrade": false,
              "selectionMetric": "validationLoss", "dtype": "bfloat16",
              "schedule": {"totalSteps": 120, "epochs": 3,
                           "effectiveBatchSize": 4, "warmupSteps": 10,
                           "lrSchedule": "linear"},
              "dataset": {"bundleID": "stance-lora-family-v1",
                          "manifestHash": "f0f0",
                          "files": [
                            {"role": "train", "path": "a.jsonl", "sha256": "aaaa",
                             "rowsRoot": "r1", "rows": 40},
                            {"role": "validation", "path": "v.jsonl",
                             "sha256": "bbbb", "rowsRoot": "r2", "rows": 8}]},
              "somethingTheAppDoesNotModel": {"nested": [1, 2, 3]}},
             "planHash": "beef"}
            """
        let response = try JSONDecoder().decode(
            RemoteFineTunePlanResponse.self, from: Data(json.utf8))
        #expect(response.planHash == "beef")
        #expect(
            response.plan.resolvedRevision
                == "0123456789abcdef0123456789abcdef01234567")
        #expect(response.plan.schedule?.totalSteps == 120)
        #expect(response.plan.dataset?.rows(role: "train") == 40)
        #expect(response.plan.dataset?.rows(role: "validation") == 8)
        // Unmodelled keys survive in `raw` instead of failing the decode.
        if case .object(let raw) = response.plan.raw {
            #expect(raw["somethingTheAppDoesNotModel"] != nil)
        } else {
            Issue.record("the plan's raw JSON must be kept for display")
        }

        let summary = FineTuningPanel.serverTrainingPlanSummary(
            plan: response.plan, planHash: response.planHash,
            trainFileCount: 1, validationFileCount: 1)
        #expect(summary.contains { $0.contains("0123456789abcdef") })
        #expect(summary.contains { $0.hasPrefix("train: 1 file (40 rows)") })
        #expect(summary.contains { $0.hasPrefix("validation: 1 file (8 rows)") })
        #expect(summary.contains { $0.contains("120 steps over 3 epoch(s)") })
        #expect(summary.contains { $0.contains("bfloat16") })
        #expect(summary.contains { $0.contains("plan hash: beef") })
    }
}
