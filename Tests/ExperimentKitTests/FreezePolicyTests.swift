import Testing

@testable import ExperimentKit

@Suite struct FreezePolicyTests {
    @Test func carriedModelOutputConditionsDoNotCreatePanelEvidenceGates() {
        var manifest = ExperimentManifest(name: "policy", description: "", modelID: "model")
        manifest.conditions = [.init(name: "carried", slots: [])]
        manifest.studyKind = .multiAgent
        #expect(
            FreezePolicy.validation(
                manifest, name: manifest.name, runSubstrate: "python-hf-transformers",
                facts: .init(hasMatchingEvidence: false)) == nil)
    }

    @Test func matchingEvidenceStillRefusesWhenItProbedNothing() throws {
        var manifest = ExperimentManifest(name: "policy", description: "", modelID: "model")
        manifest.conditions = [.init(name: "arm", slots: [])]
        let result = try #require(
            FreezePolicy.validation(
                manifest, name: manifest.name, runSubstrate: "python-hf-transformers",
                facts: .init(
                    hasMatchingEvidence: true,
                    vacuousProblem: "no held-out probe", vacuousRepair: "attach probe")))
        #expect(result.gate == .validateEvidence)
        #expect(result.repairAction == "attach probe")
        #expect(
            FreezePolicy.validation(
                manifest, name: manifest.name, runSubstrate: "python-hf-transformers",
                facts: .init(hasMatchingEvidence: true)) == nil)
    }

    @Test func missingEvidenceRepairNamesTheExecutionSubstrate() throws {
        var manifest = ExperimentManifest(name: "policy", description: "", modelID: "model")
        manifest.conditions = [.init(name: "arm", slots: [])]
        let result = try #require(
            FreezePolicy.validation(
                manifest, name: manifest.name, runSubstrate: "python-hf-transformers",
                facts: .init(hasMatchingEvidence: false)))
        #expect(result.repairAction.contains("steerlab-server"))
    }

    @Test func unexpectedEvidenceErrorsRemainTheOriginalError() throws {
        enum ProbeFailure: Error { case unreadable }
        let result = try #require(
            FreezePolicy.checkedOutcome(
                .gitClean, name: "policy", repairAction: "repair",
                result: .failure(ProbeFailure.unreadable)))
        #expect(FreezePolicy.freezeRefusal([result]) is ProbeFailure)
    }

    @Test func draftFieldEditsNeedNoWorkspaceAndPreserveClearSemantics() throws {
        var manifest = ExperimentManifest(name: "policy", description: "", modelID: "model")
        try ManifestDraftEdits.setSamplingPolicy(
            samplesPerItem: 3, seedPolicy: "derivedSHA256", experimentName: "policy",
            manifest: &manifest)
        #expect(manifest.samplesPerItem == 3)
        try ManifestDraftEdits.setSamplingPolicy(
            samplesPerItem: nil, seedPolicy: nil, experimentName: "policy", manifest: &manifest)
        #expect(manifest.samplesPerItem == nil)
        #expect(manifest.seedPolicy == nil)
        manifest.status = .frozen
        #expect(throws: ExperimentError.self) {
            try ManifestMutationPolicy.admitDraftEdit(manifest)
        }
        #expect(throws: ExperimentError.self) { try ManifestMutationPolicy.admitFreeze(manifest) }
    }
}
