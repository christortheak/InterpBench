import Foundation
import Testing

@testable import ExperimentKit

/// Mac-authority mode (2026-07-21): the Mac freezes; the cluster is a
/// runner. The local freeze gate's evidence matcher therefore keys on the
/// substrate the study will RUN on (`runSubstrate`), not on the substrate
/// performing the freeze — an IMPORTED server validate run (evidence bundle
/// brought home) satisfies a local freeze of a server-bound study, and no
/// false cross-substrate advisory fires for server-evidence + server-run.
///
/// Extends the serialized `ExperimentStoreTests` suite for its temp-root
/// seam. The imported-run fixture mirrors what the server's
/// `tasks._validate_impl` writes and `package_evidence` ships home:
/// `experiment.json` snapshot, `validation-evidence.json` with
/// substrate "python-hf-transformers" and the SERVER's engine-local scope
/// hash (NOT comparable to Swift's), and `validation-report.json` with the
/// server's top-level "concepts" shape.
extension ExperimentStoreTests {

    private static let serverEngine = WorkspaceScoping.serverSubstrate
    private static let localEngine = ExperimentStore.evidenceSubstrate

    /// Fabricates the run directory an imported server evidence bundle
    /// leaves under this workspace's runs/ tree.
    private func fabricateImportedServerValidateRun(
        for manifest: ExperimentManifest,
        stamp: String = "20260721T000000000Z",
        battery: [CapabilityBatteryConditionResult]? = nil
    ) throws {
        let dir = ExperimentStore.runsDirectory.appending(
            component: "\(stamp)-exp-\(manifest.name)-validate")
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        try JSONEncoder().encode(manifest).write(
            to: dir.appending(component: "experiment.json"))
        try #"{"experiment":"\#(manifest.name)","concepts":{"french":{"layer":18}}}"#
            .write(
                to: dir.appending(component: "validation-report.json"),
                atomically: true, encoding: .utf8)
        var evidence: [String: Any] = [
            "schemaVersion": 1,
            "task": "validate",
            "experiment": manifest.name,
            "substrate": Self.serverEngine,
            "reportFile": "validation-report.json",
            // The server's engine-local canonicalization — deliberately NOT
            // Swift's hash of the same scope. The matcher must not compare
            // engine-local hashes across engines; the snapshot is the
            // scope proof for foreign evidence.
            "validationScopeHash": String(repeating: "f", count: 64),
        ]
        if let battery {
            evidence["batteryResults"] = battery.map {
                [
                    "condition": $0.condition, "batteryHash": $0.batteryHash,
                    "total": $0.total, "correct": $0.correct,
                    "accuracy": $0.accuracy,
                ] as [String: Any]
            }
        }
        try JSONSerialization.data(withJSONObject: evidence, options: [.sortedKeys])
            .write(to: dir.appending(component: "validation-evidence.json"))
    }

    private func makeStudy(name: String) throws -> ExperimentManifest {
        var manifest = try ExperimentStore.create(
            name: name, description: "", modelID: "test/model",
            modelRevision: "abc123def456")
        manifest.concepts.append(
            .init(name: "french", stimulusSetHash: try realFrenchHash(), options: .init()))
        manifest.markersHash = ExperimentStore.liveMarkersHash(manifest)
        try ExperimentStore.save(manifest)
        return manifest
    }

    // MARK: The matcher keys on the run substrate

    @Test func importedServerEvidenceCountsForAServerBoundFreeze() throws {
        try withTempRoot {
            let manifest = try makeStudy(name: "mac-auth")
            try fabricateImportedServerValidateRun(for: manifest)
            // Historical rule unchanged: for a LOCAL run substrate the
            // foreign evidence never counts…
            #expect(ExperimentStore.validationEvidence(for: manifest) == nil)
            // …but for the SERVER run substrate it is substrate-MATCHED.
            #expect(
                ExperimentStore.validationEvidence(
                    for: manifest, runSubstrate: Self.serverEngine) != nil)
        }
    }

    @Test func localEvidenceDoesNotCountForAServerBoundFreeze() throws {
        try withTempRoot {
            let manifest = try makeStudy(name: "mac-auth-local-ev")
            try fabricateValidationEvidence(for: manifest)  // swift-mlx stamped
            #expect(ExperimentStore.validationEvidence(for: manifest) != nil)
            #expect(
                ExperimentStore.validationEvidence(
                    for: manifest, runSubstrate: Self.serverEngine) == nil)
        }
    }

    @Test func importedEvidenceForDifferentPinsNeverCounts() throws {
        try withTempRoot {
            let manifest = try makeStudy(name: "mac-auth-drift")
            // Snapshot with a DIFFERENT revision: the scope proof for
            // foreign evidence is the snapshot — a drifted snapshot must
            // refuse regardless of substrate.
            var drifted = manifest
            drifted.modelRevision = "other-revision"
            try fabricateImportedServerValidateRun(for: drifted)
            #expect(
                ExperimentStore.validationEvidence(
                    for: manifest, runSubstrate: Self.serverEngine) == nil)
        }
    }

    // MARK: No false advisory for server-evidence + server-run

    @Test func noCrossSubstrateAdvisoryWhenEvidenceMatchesTheRunEngine() throws {
        try withTempRoot {
            let manifest = try makeStudy(name: "mac-auth-adv")
            try fabricateImportedServerValidateRun(for: manifest)
            // Perspective = the RUN engine (server): matched, silent.
            #expect(
                ExperimentStore.crossSubstrateValidationAdvisory(
                    for: manifest, perspective: Self.serverEngine) == nil)
            let readiness = ExperimentStore.freezeReadiness(
                for: manifest, runSubstrate: Self.serverEngine)
            #expect(
                !readiness.advisories.contains(where: {
                    FreezeRouting.isCrossSubstrateAdvisory($0)
                }))
            #expect(
                !readiness.unmetGates.contains(where: {
                    $0.contains("no validate run matches")
                }))
        }
    }

    @Test func localEvidenceDrawsTheAdvisoryUnderAServerRunPerspective() throws {
        try withTempRoot {
            let manifest = try makeStudy(name: "mac-auth-adv2")
            try fabricateValidationEvidence(
                for: manifest, substrate: Self.localEngine)
            let advisory = ExperimentStore.crossSubstrateValidationAdvisory(
                for: manifest, perspective: Self.serverEngine)
            #expect(
                advisory
                    == "validation evidence was produced on \(Self.localEngine); "
                    + "runs on \(Self.serverEngine) should re-validate on-substrate")
        }
    }

    // MARK: The freeze itself

    @Test func localFreezeSucceedsOnImportedServerEvidence() throws {
        try withTempRoot {
            let manifest = try makeStudy(name: "mac-auth-freeze")
            try fabricateImportedServerValidateRun(for: manifest)
            // Local-substrate freeze still refuses — behavior unchanged.
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.freeze(name: "mac-auth-freeze")
            }
            // Mac-authority freeze: full local gate set, evidence matched
            // for the run substrate, stamped frozenBy this engine.
            let frozen = try ExperimentStore.freeze(
                name: "mac-auth-freeze",
                runSubstrate: Self.serverEngine)
            #expect(frozen.status == .frozen)
            #expect(frozen.frozenBy == "swift")
            #expect(frozen.freezeForced == nil)
        }
    }

    @Test func macAuthorityFreezeStillGatesOnMissingEvidence() throws {
        try withTempRoot {
            _ = try makeStudy(name: "mac-auth-gate")
            // No evidence at all: the gate refuses and names the run
            // substrate so the refusal is actionable.
            do {
                _ = try ExperimentStore.freeze(
                    name: "mac-auth-gate",
                    runSubstrate: Self.serverEngine)
                Issue.record("freeze should have refused without evidence")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("no validate run matches"))
                #expect(error.reason.contains(Self.serverEngine))
                // Evidence exists nowhere, so no substrate flag is offered.
                #expect(!error.reason.contains("--run-substrate"))
                #expect(error.freezeRefusal?.repairAction.contains("--run-substrate") == false)
            }
        }
    }

    // MARK: The refusal names the substrate that has the evidence (2026-10-20)

    /// The 0.9.12 smoke test validated on the cluster, imported the evidence,
    /// and met a freeze that read only this Mac's — whose repair named
    /// `experiment validate` and `--force`, never the one flag that fit.
    @Test func aLocalFreezeOfServerValidatedPinsNamesTheServerFlag() throws {
        try withTempRoot {
            _ = try makeStudy(name: "mac-auth-flag")
            let manifest = try ExperimentStore.load(name: "mac-auth-flag")
            try fabricateImportedServerValidateRun(for: manifest)
            do {
                _ = try ExperimentStore.freeze(name: "mac-auth-flag")
                Issue.record("a local freeze must not accept server evidence")
            } catch let error as ExperimentError {
                let repair = try #require(error.freezeRefusal?.repairAction)
                #expect(
                    repair.hasPrefix(
                        "A validate run on \(Self.serverEngine) matches these pins. "
                            + "If the measured runs execute there, freeze with "
                            + "'steerlab-cli experiment freeze mac-auth-flag "
                            + "--run-substrate server'. Otherwise run "))
                // The old repairs are still offered after it.
                #expect(repair.contains("experiment validate mac-auth-flag"))
                #expect(repair.contains("freeze --force"))
                #expect(error.reason.contains(repair))
                #expect(error.freezeRefusal?.gate == .validateEvidence)
            }
            // …and following it freezes.
            let frozen = try ExperimentStore.freeze(
                name: "mac-auth-flag", runSubstrate: Self.serverEngine)
            #expect(frozen.status == .frozen)
        }
    }

    @Test func aServerFreezeOfLocallyValidatedPinsNamesTheLocalFlag() throws {
        try withTempRoot {
            let manifest = try makeStudy(name: "mac-auth-local-flag")
            try fabricateValidationEvidence(for: manifest, substrate: Self.localEngine)
            do {
                _ = try ExperimentStore.freeze(
                    name: "mac-auth-local-flag", runSubstrate: Self.serverEngine)
                Issue.record("a server freeze must not accept this Mac's evidence")
            } catch let error as ExperimentError {
                let repair = try #require(error.freezeRefusal?.repairAction)
                #expect(repair.hasPrefix("A validate run on \(Self.localEngine) matches"))
                #expect(repair.contains("--run-substrate local'"))
                #expect(repair.contains("steerlab-server experiment validate"))
            }
        }
    }

    /// A study frozen for the server runs there: the CLI's next step is the
    /// bundle, not this Mac's `experiment run`.
    @Test func aServerFreezeNextActionIsTheBundle() async throws {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "mac-auth-next-\(UUID().uuidString)")
        ExperimentStore.rootOverride = temp
        defer {
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        do {
            let manifest = try makeStudy(name: "mac-auth-next")
            try fabricateImportedServerValidateRun(for: manifest)
            let result = await ExperimentCLIRunner(sink: .discarding).run(
                namespace: "experiment",
                ["freeze", "mac-auth-next", "--run-substrate", "server"])
            #expect(result.envelope.exitCode == 0)
            #expect(
                result.envelope.nextAction?.verb
                    == "remote package mac-auth-next --site <id>")
            #expect(result.envelope.nextAction?.detail?.contains("--verb run") == true)
        }
    }

    // MARK: Bundled-validate job rows offer evidence import

    @Test @MainActor func bundledValidateJobsOfferEvidenceImport() {
        #expect(
            ExperimentPanel.jobOffersEvidenceImport(
                verb: "validate (bundle)", state: "succeeded"))
        // Dry runs finish "prepared" and direct validates carry no bundle.
        #expect(
            !ExperimentPanel.jobOffersEvidenceImport(
                verb: "validate (bundle)", state: "prepared"))
        #expect(
            !ExperimentPanel.jobOffersEvidenceImport(
                verb: "validate", state: "succeeded"))
    }
}
