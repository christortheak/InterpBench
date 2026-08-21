import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// The cross-substrate lifecycle, over bytes the SERVER actually produced.
///
/// Every other cross-engine test in this suite pins a *value* computed by a
/// pure function. This one pins the **wire**: a real archive written by
/// `bundles.package_evidence`, around an agent minted by
/// `model_variant.save_variant`, beside a sweep run a pinned promotion can
/// name. It is consumed by the real `EvidenceBundleImporter`, then by real
/// discovery and real promotion.
///
/// That distinction is the whole point. The previous coverage planted a
/// hand-built directory into `runs/` — which pins the SWIFT author's belief
/// about what the server emits, and that belief drifting out of date is
/// exactly the bug this seam kept producing: twice the discovery layer was
/// wrong while every unit test stayed green.
///
/// Regenerate the fixture with
/// `scripts/regenerate-cross-engine-fixtures.py`; `test_fixture_staleness.py`
/// fails if it drifts from the code that made it.
struct CrossEngineLifecycleTests {

    private struct Fixture {
        var archive: URL
        var bundleSha256: String
        var agentName: String
        var agentRunSuffix: String
        var sweepRun: String
        var extractRun: String
        var vectorArtifactID: String
        var experimentHash: String
        var stimulusSetHash: String

        static func load() throws -> Fixture {
            let directory = CodeResources.compiledCheckoutPath.appending(
                components: "Tests", "Fixtures", "cross-engine")
            let meta = try #require(
                try JSONSerialization.jsonObject(
                    with: try Data(
                        contentsOf: directory.appending(
                            component: "server-evidence-bundle.json")))
                    as? [String: Any])
            return .init(
                archive: directory.appending(
                    component: try #require(meta["archive"] as? String)),
                bundleSha256: try #require(meta["bundleSha256"] as? String),
                agentName: try #require(meta["agentName"] as? String),
                agentRunSuffix: try #require(meta["agentRunSuffix"] as? String),
                sweepRun: try #require(meta["sweepRun"] as? String),
                extractRun: try #require(meta["extractRun"] as? String),
                vectorArtifactID: try #require(meta["vectorArtifactID"] as? String),
                experimentHash: try #require(meta["experimentHash"] as? String),
                stimulusSetHash: try #require(meta["stimulusSetHash"] as? String))
        }
    }

    private func withWorkspace<T>(_ body: (URL) throws -> T) rethrows -> T {
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "seam-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        ExperimentRootOverrideLock.acquire()
        WorkspaceRoot.programmaticOverride = temp
        defer {
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: temp)
        }
        return try body(temp)
    }

    /// Server packaging → Swift import → local discovery → resolvable vector.
    @Test func aServerPackagedBundleImportsAndItsAgentBecomesUsable() throws {
        let fixture = try Fixture.load()
        try withWorkspace { _ in
            // The importer verifies the archive hash BEFORE extracting, so a
            // wrong hash here would fail loudly rather than silently import.
            _ = try EvidenceBundleImporter.importEvidenceBundle(
                fixture.archive, expectedSHA256: fixture.bundleSha256)

            // Discovery: the agent the SERVER minted, found by the layout rule
            // rather than by a filename the Swift side guessed.
            let agents = ModelVariantStore.scan()
            let agent = try #require(
                agents.first { $0.artifact.name == fixture.agentName },
                "the server-minted agent did not appear in the local library")

            // Its birth certificate survived the round trip intact.
            let promotion = try #require(agent.artifact.promotion)
            #expect(promotion.promotedBy == "criterion")
            #expect(promotion.winningCell?.layer == 2)

            // Its vector reference is workspace-RELATIVE, as `promote` writes
            // it, and resolves against the local catalog after import.
            let injection = try #require(agent.artifact.injections.first)
            #expect(injection.vectorArtifactID.hasPrefix("runs/"))
            let catalogIDs = Set(VectorCatalog.scan().map(\.id))
            #expect(
                ArtifactIdentity.contains(catalogIDs, injection.vectorArtifactID),
                "the imported agent's vector did not resolve: this is the shape that made an imported agent visible but unusable")
        }
    }

    /// The sibling runs the bundle declared come home too, so the evidence a
    /// pinned promotion needs is present — not just the agent.
    @Test func theBundleCarriesTheEvidenceAPromotionWouldName() throws {
        let fixture = try Fixture.load()
        try withWorkspace { workspace in
            _ = try EvidenceBundleImporter.importEvidenceBundle(
                fixture.archive, expectedSHA256: fixture.bundleSha256)

            let runs = workspace.appending(component: "runs")
            for run in [fixture.sweepRun, fixture.extractRun] {
                var isDirectory: ObjCBool = false
                #expect(
                    FileManager.default.fileExists(
                        atPath: runs.appending(component: run).path,
                        isDirectory: &isDirectory),
                    "sibling run '\(run)' did not come home")
            }

            // The sweep run is READABLE by the catalog that a pinned promotion
            // consults, and carries the recommendation to pin against.
            let sweep = try SweepRunCatalog.load(
                directory: runs.appending(component: fixture.sweepRun))
            let recommendation = try #require(sweep.recommendations["fear"])
            guard case .selected(let provenance) = recommendation else {
                Issue.record("the imported sweep recorded no winning cell")
                return
            }
            #expect(provenance.winningCell.layer == 2)
        }
    }

    /// A promotion pinned to SERVER-produced sweep evidence is judged on
    /// whether the STUDY changed — not on which engine hashed it.
    ///
    /// This test used to assert the opposite, and the assertion was wrong in
    /// a way the fixture concealed. Epoch hashes are per-engine because the
    /// engines canonicalize differently, so comparing them across substrates
    /// is not a strict check — it is a check that cannot pass, and it refused
    /// legitimate cluster evidence in a workspace where every sweep is
    /// foreign by design (observed live 2026-07-26). The run carries its own
    /// manifest SNAPSHOT, which this engine can hash itself; that is the
    /// comparison, and it is the same one `ExperimentStore.validationEvidence`
    /// has used for validate evidence since 2026-07-21.
    ///
    /// The fixture concealed it because its hand-built sweep directory had no
    /// `experiment.json` — a file every real server run writes — so the test
    /// exercised the no-snapshot fallback while appearing to cover the real
    /// path. The generator now writes it.
    @Test func aPinnedPromotionFromForeignEvidenceIsJudgedOnTheStudyNotTheEngine() throws {
        let fixture = try Fixture.load()
        try withWorkspace { _ in
            _ = try EvidenceBundleImporter.importEvidenceBundle(
                fixture.archive, expectedSHA256: fixture.bundleSha256)

            // A study that DIFFERS from the one the sweep ran under: the
            // concept is attached here but the sweep's snapshot has its own
            // shape, so this must still refuse.
            var manifest = try ExperimentStore.create(
                name: "seam", description: "", modelID: "org/m",
                modelRevision: String(repeating: "a", count: 40))
            manifest.concepts.append(
                .init(
                    name: "fear", stimulusSetHash: fixture.stimulusSetHash,
                    options: .init()))
            try ExperimentStore.save(manifest)

            var message = ""
            #expect(throws: (any Error).self) {
                do {
                    _ = try AgentPromotion.promote(
                        experimentName: "seam", concept: "fear",
                        agentName: "locally-promoted",
                        pins: .init(sweepRun: fixture.sweepRun),
                        log: { _ in })
                } catch {
                    message = "\(error)"
                    throw error
                }
            }
            // It names WHAT changed, not merely that two hashes differ.
            #expect(message.contains("has changed since source run"))
            #expect(
                message.contains(":"),
                "the refusal carries no field diagnosis: \(message)")
            // The unactionable framings are both gone.
            #expect(!message.contains("do not cross substrates"))
            #expect(!message.contains("different manifest epoch"))
        }
    }

    /// And the case the live failure actually was: the study is UNCHANGED,
    /// the evidence came from the cluster, and promotion proceeds. Before the
    /// fix this was unreachable — no manifest, however faithful, could
    /// satisfy a cross-engine hash comparison.
    @Test func aPinnedPromotionFromForeignEvidenceSucceedsWhenTheStudyIsUnchanged() throws {
        let fixture = try Fixture.load()
        try withWorkspace { workspace in
            _ = try EvidenceBundleImporter.importEvidenceBundle(
                fixture.archive, expectedSHA256: fixture.bundleSha256)

            // Adopt the sweep's own snapshot as the live study — the state a
            // researcher is in when they sweep on the cluster and have not
            // edited since.
            let snapshot = try #require(
                RunEpoch.snapshot(
                    workspace.appending(components: "runs", fixture.sweepRun)))
            _ = try ExperimentStore.create(
                name: snapshot.name, description: "", modelID: snapshot.modelID)
            try ExperimentStore.save(snapshot)

            let record = try AgentPromotion.promote(
                experimentName: snapshot.name, concept: "fear",
                agentName: "cluster-promoted",
                pins: .init(sweepRun: fixture.sweepRun),
                log: { _ in })

            let promotion = try #require(record.artifact.promotion)
            #expect(promotion.promotedBy == "criterion")
            #expect(promotion.winningCell?.layer == 2)
            // The agent's settings apply to the substrate that produced them.
            #expect(promotion.substrate == "python-hf-transformers")
        }
    }

    /// A tampered archive must not import. The hash is checked before
    /// extraction, so a corrupted bundle never reaches the runs tree.
    @Test func aBundleWhoseHashDoesNotMatchIsRefused() throws {
        let fixture = try Fixture.load()
        try withWorkspace { workspace in
            #expect(throws: (any Error).self) {
                try EvidenceBundleImporter.importEvidenceBundle(
                    fixture.archive,
                    expectedSHA256: String(repeating: "0", count: 64))
            }
            let runs = workspace.appending(component: "runs")
            #expect(
                !FileManager.default.fileExists(
                    atPath: runs.appending(component: fixture.sweepRun).path))
        }
    }
}
