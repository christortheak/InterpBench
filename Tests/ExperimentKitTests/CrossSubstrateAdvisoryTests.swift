import Foundation
import Testing

@testable import ExperimentKit

/// WS7.1 cross-substrate validate-evidence advisory: NON-BLOCKING, loud, and
/// only when the evidence's engine is KNOWN to differ — legacy runs are never
/// accused, and same-engine evidence stays silent. Fires in two places: the
/// freeze-advisory list (readiness surface) and study-run start (stdout + a
/// durable advisories.txt in the run directory).
///
/// Extends the serialized `ExperimentStoreTests` suite for its temp-root seam
/// and evidence fixtures. Wording is byte-identical to the server's
/// `cross_substrate_validation_advisory` modulo engine names.
extension ExperimentStoreTests {

    private static let thisEngine = "swift-mlx"
    private static let otherEngine = "python-hf-transformers"

    private func makeConceptStudy(name: String = "xsub") throws -> ExperimentManifest {
        var manifest = try ExperimentStore.create(
            name: name, description: "", modelID: "test/model",
            modelRevision: "abc123")
        // Fully pinned modern attach (validation + markers) so the
        // measurement-pin advisory stays silent and these tests isolate the
        // CROSS-SUBSTRATE advisory.
        manifest.concepts.append(
            ExperimentStore.makeConceptRef(
                name: "french", stimulusSetHash: "ff", options: .init()))
        manifest.markersHash = ExperimentStore.liveMarkersHash(manifest)
        try ExperimentStore.save(manifest)
        return manifest
    }

    /// Plants a canonical config.json into a fabricated validate run
    /// directory (the pre-substrate-evidence fallback the advisory consults).
    private func plantRunConfig(
        experiment: ExperimentManifest, stamp: String, substrate: String
    ) throws {
        let dir = ExperimentStore.runsDirectory.appending(
            component: "\(stamp)-exp-\(experiment.name)-validate")
        let config = #"{"schemaVersion": 2, "runType": "validate", "substrate": "\#(substrate)"}"#
        try Data(config.utf8).write(to: dir.appending(component: "config.json"))
    }

    // MARK: The advisory predicate

    @Test func advisoryPresentWhenOnlyForeignEvidenceExists() throws {
        try withTempRoot {
            let manifest = try makeConceptStudy()
            try fabricateValidationEvidence(
                for: manifest, substrate: Self.otherEngine)
            let advisory = ExperimentStore.crossSubstrateValidationAdvisory(
                for: manifest)
            #expect(
                advisory
                    == "validation evidence was produced on \(Self.otherEngine); "
                    + "runs on \(Self.thisEngine) should re-validate on-substrate")
        }
    }

    @Test func advisoryAbsentWhenSameEngineEvidenceExists() throws {
        try withTempRoot {
            let manifest = try makeConceptStudy()
            try fabricateValidationEvidence(
                for: manifest, substrate: Self.thisEngine)
            #expect(ExperimentStore.crossSubstrateValidationAdvisory(for: manifest) == nil)
            // …even when foreign evidence ALSO exists — the gate relies on
            // the same-engine run, so there is nothing to advise about.
            try fabricateValidationEvidence(
                for: manifest, substrate: Self.otherEngine,
                stamp: "20260611T000000000Z")
            #expect(ExperimentStore.crossSubstrateValidationAdvisory(for: manifest) == nil)
        }
    }

    @Test func advisoryAbsentWhenEngineUnknowable() throws {
        try withTempRoot {
            let manifest = try makeConceptStudy()
            // Legacy evidence: no substrate stamp, no config.json — could be
            // either engine, and legacy is treated as THIS engine by the
            // gate. Never accuse.
            try fabricateValidationEvidence(for: manifest, substrate: nil)
            #expect(ExperimentStore.crossSubstrateValidationAdvisory(for: manifest) == nil)
        }
    }

    @Test func advisoryAbsentWhenNoEvidenceAtAll() throws {
        try withTempRoot {
            let manifest = try makeConceptStudy()
            #expect(ExperimentStore.crossSubstrateValidationAdvisory(for: manifest) == nil)
        }
    }

    @Test func advisoryUsesRunConfigWhenEvidencePredatesSubstrateStamp() throws {
        try withTempRoot {
            // Unstamped (legacy-shaped) evidence whose run carries a
            // canonical config.json naming the OTHER engine: the gate would
            // count it as legacy this-engine evidence — exactly the sneaky
            // case the advisory catches.
            let manifest = try makeConceptStudy()
            try fabricateValidationEvidence(for: manifest, substrate: nil)
            try plantRunConfig(
                experiment: manifest, stamp: "20260610T000000000Z",
                substrate: Self.otherEngine)
            let advisory = ExperimentStore.crossSubstrateValidationAdvisory(
                for: manifest)
            #expect(advisory?.contains(Self.otherEngine) == true)
            #expect(advisory?.contains("re-validate on-substrate") == true)
        }
    }

    // MARK: The perspective parameter (server-routed freeze, paired tree)

    @Test func perspectiveShiftEvaluatesTheSameTreeFromTheServersViewpoint() throws {
        try withTempRoot {
            // Local (swift-mlx) evidence only: coherent for a LOCAL freeze,
            // incoherent for a SERVER-routed freeze of the same paired tree —
            // exactly the validate-locally-then-freeze-on-the-server case the
            // panel warns about at decision time.
            let manifest = try makeConceptStudy()
            try fabricateValidationEvidence(
                for: manifest, substrate: Self.thisEngine)
            #expect(ExperimentStore.crossSubstrateValidationAdvisory(for: manifest) == nil)
            let advisory = ExperimentStore.crossSubstrateValidationAdvisory(
                for: manifest, perspective: Self.otherEngine)
            #expect(
                advisory
                    == "validation evidence was produced on \(Self.thisEngine); "
                    + "runs on \(Self.otherEngine) should re-validate on-substrate")
            // And the inverse: server evidence is coherent from the server's
            // perspective.
            try fabricateValidationEvidence(
                for: manifest, substrate: Self.otherEngine,
                stamp: "20260611T000000000Z")
            #expect(
                ExperimentStore.crossSubstrateValidationAdvisory(
                    for: manifest, perspective: Self.otherEngine) == nil)
        }
    }

    // MARK: Freeze surface (the existing advisory list)

    @Test func freezeAdvisoriesIncludeCrossSubstrateEvidence() throws {
        try withTempRoot {
            let manifest = try makeConceptStudy()
            try fabricateValidationEvidence(
                for: manifest, substrate: Self.otherEngine)
            let advisories = ExperimentStore.freezeAdvisories(for: manifest)
            #expect(advisories.count == 1)
            #expect(advisories.first?.contains("re-validate on-substrate") == true)
            // Readiness carries the same list (the panel's freeze surface).
            let readiness = ExperimentStore.freezeReadiness(for: manifest)
            #expect(
                readiness.advisories.contains { $0.contains("re-validate on-substrate") })
        }
    }

    @Test func freezeAdvisoriesSilentForSameEngineOrNoEvidence() throws {
        try withTempRoot {
            let manifest = try makeConceptStudy()
            #expect(ExperimentStore.freezeAdvisories(for: manifest).isEmpty)
            try fabricateValidationEvidence(
                for: manifest, substrate: Self.thisEngine)
            #expect(ExperimentStore.freezeAdvisories(for: manifest).isEmpty)
        }
    }

    // MARK: Study-run-start surface

    @Test func studyRunStartStampsAdvisoriesFileWhenForeign() throws {
        try withTempRoot {
            let manifest = try makeConceptStudy()
            try fabricateValidationEvidence(
                for: manifest, substrate: Self.otherEngine)
            let runDirectory = try ExperimentTasks.makeRunDirectory(
                experiment: manifest, task: "run")
            let advisories = try String(
                contentsOf: runDirectory.appending(component: "advisories.txt"),
                encoding: .utf8)
            #expect(advisories.contains("re-validate on-substrate"))
            #expect(advisories.contains(Self.otherEngine))

            // Non-study tasks never stamp the advisory file.
            let extractDirectory = try ExperimentTasks.makeRunDirectory(
                experiment: manifest, task: "extract")
            #expect(
                !FileManager.default.fileExists(
                    atPath: extractDirectory.appending(component: "advisories.txt").path))
        }
    }

    @Test func studyRunStartSilentWhenSameEngine() throws {
        try withTempRoot {
            let manifest = try makeConceptStudy()
            try fabricateValidationEvidence(
                for: manifest, substrate: Self.thisEngine)
            let runDirectory = try ExperimentTasks.makeRunDirectory(
                experiment: manifest, task: "run")
            #expect(
                !FileManager.default.fileExists(
                    atPath: runDirectory.appending(component: "advisories.txt").path))
        }
    }
}
