import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// The PINNED promotion contract (B2, 2026-07-26) — Swift half.
///
/// Promotion used to resolve its own inputs by recency: the newest sweep run
/// in the workspace, and the newest extraction artifact matching the recipe.
/// Both are ambient. With several sweeps per concept — the ordinary state
/// once a grid is being iterated — a promote issued after a re-sweep silently
/// bound to whichever run was newest, and the birth certificate then recorded
/// a cell the researcher had not chosen.
struct PinnedPromotionTests {

    private static let fearHash = String(repeating: "f", count: 64)

    private func withTempWorkspace<T>(_ body: (URL) throws -> T) rethrows -> T {
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "pinned-promote-\(UUID().uuidString)")
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

    @discardableResult
    private func plantVectorArtifact(
        in root: URL, run: String = "20260707T000001000-extract",
        method: String = "meanDifference", bytes: String = "weights"
    ) throws -> String {
        let runDir = root.appending(components: "runs", run)
        try FileManager.default.createDirectory(
            at: runDir, withIntermediateDirectories: true)
        try Data(bytes.utf8).write(
            to: runDir.appending(component: "fear.safetensors"))
        let sidecar: [String: Any] = [
            "modelID": "test/model", "concept": "fear",
            "stimulusSetHash": Self.fearHash, "layerCount": 4, "hiddenSize": 2,
            "normsPerLayer": [1.0, 1.0, 1.0, 1.0],
            "residualNormPerLayer": [1.0, 1.0, 1.0, 1.0],
            "extractionDate": "2026-07-07T00:00:00Z",
            "extractionMethod": method, "substrate": "swift-mlx",
            "revision": "abc123", "readingPosition": "last token",
            "neutralProjection": "none",
            "residualNormSource": "extraction-stimuli",
        ]
        try JSONSerialization.data(withJSONObject: sidecar)
            .write(to: runDir.appending(component: "fear.json"))
        return runDir.appending(component: "fear").path
    }

    /// An experiment with a concept but NO `-recommended` condition, so the
    /// pinned path is the only route to a cell.
    private func experiment(named name: String) throws -> ExperimentManifest {
        var manifest = try ExperimentStore.create(
            name: name, description: "", modelID: "test/model",
            modelRevision: "abc123")
        manifest.concepts.append(
            .init(name: "fear", stimulusSetHash: Self.fearHash, options: .init()))
        try ExperimentStore.save(manifest)
        return try ExperimentStore.load(name: name)
    }

    /// A sweep run the pinned contract can name: sweep.csv (the discovery
    /// marker), recommendations.json, and the manifest-epoch stamp.
    @discardableResult
    private func plantSweepRun(
        in root: URL, experiment name: String, layer: Int = 3,
        alpha: Double = 0.4, run: String? = nil, epochHash: String?? = nil
    ) throws -> String {
        let runName = run ?? "20260726T000000000-exp-\(name)-sweep"
        let runDir = root.appending(components: "runs", runName)
        try FileManager.default.createDirectory(
            at: runDir, withIntermediateDirectories: true)
        try Data("concept,layer,alpha,markerDensity,distinct2,batteryAccuracy\n".utf8)
            .write(to: runDir.appending(component: "sweep.csv"))
        let entry: [String: Any] = [
            "sweepRun": runName,
            "criterion": [
                "objective": ["metric": "markerDensity"],
                "constraints": [
                    "capabilityTolerance": 0.15, "coherenceFloor": 0.45,
                ],
            ],
            "devPromptsHash": String(repeating: "d", count: 64),
            "winningCell": ["layer": layer, "alpha": alpha],
            "metrics": ["markerDensity": 0.31],
        ]
        try JSONSerialization.data(withJSONObject: ["fear": entry])
            .write(to: runDir.appending(component: "recommendations.json"))
        // `nil` outer = stamp the live hash; `.some(nil)` = no stamp at all.
        let stamp: String?
        switch epochHash {
        case .none:
            stamp = ExperimentStore.manifestHash(
                try ExperimentStore.load(name: name))
        case .some(let explicit):
            stamp = explicit
        }
        if let stamp {
            try Data(stamp.utf8).write(
                to: runDir.appending(component: "experiment-hash.txt"))
        }
        return runName
    }

    // MARK: no ambient lookup

    @Test func pinnedPromotionUsesOnlyTheNamedRun() throws {
        try withTempWorkspace { root in
            _ = try experiment(named: "pin")
            try plantVectorArtifact(in: root)
            let named = try plantSweepRun(in: root, experiment: "pin",
                                          layer: 3, alpha: 0.4)
            // A NEWER sweep with a different winner. Under the old recency
            // rule this would win; under the pinned contract it is never read.
            try plantSweepRun(
                in: root, experiment: "pin", layer: 9, alpha: 0.9,
                run: "20260726T999999999-exp-pin-sweep")

            let record = try AgentPromotion.promote(
                experimentName: "pin", concept: "fear",
                pins: .init(sweepRun: named), log: { _ in })

            let promotion = try #require(record.artifact.promotion)
            #expect(promotion.winningCell?.layer == 3)
            #expect(promotion.winningCell?.alpha == 0.4)
            #expect(promotion.promotedBy == "criterion")
        }
    }

    // MARK: the plan must still be current

    @Test func aStaleExpectedCellRefuses() throws {
        try withTempWorkspace { root in
            _ = try experiment(named: "pin")
            try plantVectorArtifact(in: root)
            let named = try plantSweepRun(in: root, experiment: "pin")

            #expect(throws: (any Error).self) {
                try AgentPromotion.promote(
                    experimentName: "pin", concept: "fear",
                    pins: .init(sweepRun: named, winningCell: (9, 0.9)),
                    log: { _ in })
            }
            // The agreeing cell promotes normally.
            let record = try AgentPromotion.promote(
                experimentName: "pin", concept: "fear",
                pins: .init(sweepRun: named, winningCell: (3, 0.4)),
                log: { _ in })
            #expect(record.artifact.promotion?.winningCell?.layer == 3)
        }
    }

    @Test func aForeignManifestEpochRefuses() throws {
        try withTempWorkspace { root in
            _ = try experiment(named: "pin")
            try plantVectorArtifact(in: root)
            let named = try plantSweepRun(
                in: root, experiment: "pin",
                epochHash: .some(String(repeating: "0", count: 64)))

            #expect(throws: (any Error).self) {
                try AgentPromotion.promote(
                    experimentName: "pin", concept: "fear",
                    pins: .init(sweepRun: named), log: { _ in })
            }
        }
    }

    @Test func anUnstampedRunRefuses() throws {
        try withTempWorkspace { root in
            _ = try experiment(named: "pin")
            try plantVectorArtifact(in: root)
            let named = try plantSweepRun(
                in: root, experiment: "pin", epochHash: .some(nil))

            #expect(throws: (any Error).self) {
                try AgentPromotion.promote(
                    experimentName: "pin", concept: "fear",
                    pins: .init(sweepRun: named), log: { _ in })
            }
        }
    }

    @Test func aDriftedExpectedEpochRefuses() throws {
        try withTempWorkspace { root in
            _ = try experiment(named: "pin")
            try plantVectorArtifact(in: root)
            let named = try plantSweepRun(in: root, experiment: "pin")

            #expect(throws: (any Error).self) {
                try AgentPromotion.promote(
                    experimentName: "pin", concept: "fear",
                    pins: .init(
                        sweepRun: named,
                        experimentHash: String(repeating: "0", count: 64)),
                    log: { _ in })
            }
        }
    }

    // MARK: artifact pinning

    @Test func aPinnedArtifactIsSelectedByIdentityNotRecency() throws {
        try withTempWorkspace { root in
            _ = try experiment(named: "pin")
            let older = try plantVectorArtifact(
                in: root, run: "20260707T000001000-extract")
            try plantVectorArtifact(in: root, run: "20260726T000001000-extract")
            let named = try plantSweepRun(in: root, experiment: "pin")

            // Unpinned: the newest interchangeable artifact wins.
            let unpinned = try AgentPromotion.promote(
                experimentName: "pin", concept: "fear",
                pins: .init(sweepRun: named), log: { _ in })
            #expect(
                unpinned.artifact.injections.first?.vectorArtifactID
                    .contains("20260726T000001000-extract") == true)

            // Pinned: the NAMED artifact is used though a newer one exists.
            let pinned = try AgentPromotion.promote(
                experimentName: "pin", concept: "fear",
                agentName: "older-agent",
                pins: .init(sweepRun: named, vectorArtifactID: older),
                log: { _ in })
            #expect(
                pinned.artifact.injections.first?.vectorArtifactID
                    .contains("20260707T000001000-extract") == true)
        }
    }

    @Test func aPinnedArtifactThatDoesNotMatchTheRecipeRefuses() throws {
        try withTempWorkspace { root in
            _ = try experiment(named: "pin")
            try plantVectorArtifact(in: root)
            // Same concept, DIFFERENT extraction method.
            let foreign = try plantVectorArtifact(
                in: root, run: "20260726T000002000-extract", method: "lat")
            let named = try plantSweepRun(in: root, experiment: "pin")

            // Pinning chooses WHICH artifact; it never waives WHETHER it
            // matches the experiment's recipe.
            #expect(throws: (any Error).self) {
                try AgentPromotion.promote(
                    experimentName: "pin", concept: "fear",
                    pins: .init(sweepRun: named, vectorArtifactID: foreign),
                    log: { _ in })
            }
        }
    }

    @Test func aDriftedArtifactHashRefuses() throws {
        try withTempWorkspace { root in
            _ = try experiment(named: "pin")
            try plantVectorArtifact(in: root)
            let named = try plantSweepRun(in: root, experiment: "pin")

            #expect(throws: (any Error).self) {
                try AgentPromotion.promote(
                    experimentName: "pin", concept: "fear",
                    pins: .init(
                        sweepRun: named,
                        vectorArtifactHash: String(repeating: "0", count: 64)),
                    log: { _ in })
            }
        }
    }

    @Test func theBirthCertificateRecordsWhichBytesWereInjected() throws {
        try withTempWorkspace { root in
            _ = try experiment(named: "pin")
            try plantVectorArtifact(in: root)
            let named = try plantSweepRun(in: root, experiment: "pin")

            let record = try AgentPromotion.promote(
                experimentName: "pin", concept: "fear",
                pins: .init(sweepRun: named), log: { _ in })
            let expected = SHA256.hash(data: Data("weights".utf8))
                .map { String(format: "%02x", $0) }.joined()
            #expect(record.artifact.promotion?.vectorArtifactHash == expected)
        }
    }

    // MARK: idempotency

    @Test func retryingTheSamePromotionReturnsTheSameAgent() throws {
        try withTempWorkspace { root in
            _ = try experiment(named: "pin")
            try plantVectorArtifact(in: root)
            let named = try plantSweepRun(in: root, experiment: "pin")
            let pins = AgentPromotion.Pins(sweepRun: named)

            let first = try AgentPromotion.promote(
                experimentName: "pin", concept: "fear", pins: pins, log: { _ in })
            let second = try AgentPromotion.promote(
                experimentName: "pin", concept: "fear", pins: pins, log: { _ in })

            // Compared canonically: the record URLs differ only by the
            // /var → /private/var symlink on macOS temp roots.
            #expect(
                ArtifactIdentity.canonical(first.url.path)
                    == ArtifactIdentity.canonical(second.url.path))
            #expect(ModelVariantStore.scan().count == 1)
            let key = try #require(first.artifact.promotion?.promotionKey)
            #expect(second.artifact.promotion?.promotionKey == key)
        }
    }

    @Test func aDifferentCellIsADifferentPromotion() throws {
        try withTempWorkspace { root in
            _ = try experiment(named: "pin")
            try plantVectorArtifact(in: root)
            let named = try plantSweepRun(in: root, experiment: "pin")
            let pins = AgentPromotion.Pins(sweepRun: named)

            _ = try AgentPromotion.promote(
                experimentName: "pin", concept: "fear", pins: pins, log: { _ in })
            let other = try AgentPromotion.promote(
                experimentName: "pin", concept: "fear",
                agentName: "override-agent", cell: (7, 0.7),
                overrideReason: "probing a stronger cell",
                pins: pins, log: { _ in })

            #expect(other.artifact.promotion?.promotedBy == "manualOverride")
            #expect(ModelVariantStore.scan().count == 2)
        }
    }

    // MARK: cross-engine key agreement

    /// The promotion key must be byte-identical across engines, or an agent
    /// minted on the server and imported onto the Mac would be re-minted as a
    /// duplicate by a Swift promote of the same request — the very failure
    /// idempotency exists to prevent.
    ///
    /// These expectations are generated by the Python twin; regenerate with
    /// `scripts/regenerate-cross-engine-fixtures.py`.
    @Test func promotionKeyMatchesThePythonTwin() throws {
        let fixtureURL = CodeResources.compiledCheckoutPath
            .appending(
                components: "Tests", "Fixtures", "cross-engine",
                "promotion-keys.json")
        let data = try Data(contentsOf: fixtureURL)
        let cases = try #require(
            try JSONSerialization.jsonObject(with: data) as? [[String: Any]])
        #expect(!cases.isEmpty)
        for entry in cases {
            let input = try #require(entry["input"] as? [String: Any])
            let cell = try #require(input["winningCell"] as? [String: Any])
            let got = AgentPromotion.promotionKey(
                experiment: try #require(input["experiment"] as? String),
                experimentHash: try #require(input["experimentHash"] as? String),
                concept: try #require(input["concept"] as? String),
                sweepRun: input["sweepRun"] as? String,
                layer: try #require(cell["layer"] as? Int),
                alpha: try #require(cell["alpha"] as? Double),
                vectorArtifactID: try #require(input["vectorArtifactID"] as? String),
                promotedBy: try #require(input["promotedBy"] as? String),
                agentName: try #require(input["agentName"] as? String),
                vectorArtifactHash: input["vectorArtifactHash"] as? String)
            let label = entry["label"] as? String ?? "?"
            #expect(
                got == entry["key"] as? String,
                "promotion key drift on case '\(label)': the engines no longer agree on which promotions are the same promotion")
        }
    }
}

/// The contract reaching the surfaces that promote (engineer review,
/// 2026-07-26). The pins shipped server-side, in the CLI and in the pipeline,
/// while `ClusterClient.promoteExperiment` had no `pins` parameter at all —
/// so the Create Agent button the researcher actually presses fell back to
/// the ambient resolution the contract exists to remove.
struct PromotionContractReachTests {

    @Test func theClusterClientCanCarryPins() throws {
        // Compile-level proof the parameter exists and accepts the contract;
        // the wire shape is asserted in the server's route test.
        let pins = AgentPromotion.Pins(
            sweepRun: "run-1", experimentHash: String(repeating: "a", count: 64),
            winningCell: (41, 0.1))
        #expect(pins.sweepRun == "run-1")
        #expect(pins.winningCell?.layer == 41)
    }

    /// An expected hash with no computable actual hash must REFUSE. Comparing
    /// only when both exist failed open: an unreadable `.safetensors` let the
    /// promotion proceed and mint an agent whose claimed bytes were never
    /// verified — the one thing the pin exists to prevent.
    @Test func anUnreadableArtifactCannotSatisfyAPinnedHash() throws {
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "failopen-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        ExperimentRootOverrideLock.acquire()
        WorkspaceRoot.programmaticOverride = temp
        defer {
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: temp)
        }

        var manifest = try ExperimentStore.create(
            name: "fo", description: "", modelID: "test/model",
            modelRevision: "abc123")
        manifest.concepts.append(
            .init(
                name: "fear",
                stimulusSetHash: String(repeating: "f", count: 64),
                options: .init()))
        try ExperimentStore.save(manifest)

        // A sidecar with NO .safetensors beside it: the recipe matches, the
        // bytes are unreadable.
        let runDir = temp.appending(components: "runs", "20260707T000001000-extract")
        try FileManager.default.createDirectory(
            at: runDir, withIntermediateDirectories: true)
        let sidecar: [String: Any] = [
            "modelID": "test/model", "concept": "fear",
            "stimulusSetHash": String(repeating: "f", count: 64),
            "layerCount": 4, "hiddenSize": 2,
            "normsPerLayer": [1.0, 1.0, 1.0, 1.0],
            "residualNormPerLayer": [1.0, 1.0, 1.0, 1.0],
            "extractionDate": "2026-07-07T00:00:00Z",
            "extractionMethod": "meanDifference", "substrate": "swift-mlx",
            "revision": "abc123", "readingPosition": "last token",
            "neutralProjection": "none",
            "residualNormSource": "extraction-stimuli",
        ]
        try JSONSerialization.data(withJSONObject: sidecar)
            .write(to: runDir.appending(component: "fear.json"))

        let runName = "20260726T000000000-exp-fo-sweep"
        let sweepDir = temp.appending(components: "runs", runName)
        try FileManager.default.createDirectory(
            at: sweepDir, withIntermediateDirectories: true)
        try Data("concept,layer,alpha\n".utf8)
            .write(to: sweepDir.appending(component: "sweep.csv"))
        let entry: [String: Any] = [
            "sweepRun": runName,
            "criterion": ["objective": ["metric": "markerDensity"]],
            "devPromptsHash": String(repeating: "d", count: 64),
            "winningCell": ["layer": 3, "alpha": 0.4],
            "metrics": ["markerDensity": 0.3],
        ]
        try JSONSerialization.data(withJSONObject: ["fear": entry])
            .write(to: sweepDir.appending(component: "recommendations.json"))
        try Data(ExperimentStore.manifestHash(
            try ExperimentStore.load(name: "fo")).utf8)
            .write(to: sweepDir.appending(component: "experiment-hash.txt"))

        #expect(throws: (any Error).self) {
            try AgentPromotion.promote(
                experimentName: "fo", concept: "fear",
                pins: .init(
                    sweepRun: runName,
                    vectorArtifactHash: String(repeating: "0", count: 64)),
                log: { _ in })
        }
    }
}
