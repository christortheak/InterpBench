import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit

/// A multi-agent run directory must carry its scenario, not a pointer to it.
///
/// `experiment.json` pins `multiAgentScenarioPath` + `multiAgentScenarioHash`,
/// but the seat→variant attribution — which agent variant sat in which seat —
/// lives in the scenario's own `agents[]`
/// (`variantArtifactPath`/`variantArtifactHash`). A reader that sees only the
/// run directory (the Results Explorer's bridge serves `runs/` and nothing
/// else) therefore could not say what ran. The snapshot is the fix, and it is
/// only evidence if the bytes are the pinned bytes.
///
/// Server twin: `Server/tests/test_scenario_snapshot.py`.
@Suite(.serialized) struct ScenarioSnapshotTests {

    /// A workspace holding one panel whose single seat carries a variant, and
    /// a multi-agent manifest pinning it. Returns the raw scenario bytes so a
    /// test can compare against exactly what is on disk.
    private func makePanelWorkspace(
        in root: URL, pinnedHash: String?
    ) throws -> (manifest: ExperimentManifest, scenarioURL: URL, bytes: Data) {
        let panelsDirectory = root.appending(components: "prompts", "panels")
        try FileManager.default.createDirectory(
            at: panelsDirectory, withIntermediateDirectories: true)
        let scenarioURL = panelsDirectory.appending(component: "panel.json")
        // Deliberately NOT this engine's canonical encoding (indented, author
        // key order): a verbatim snapshot must preserve bytes a re-encode
        // would lose.
        let bytes = Data(
            """
            {
              "name": "panel",
              "baseModelID": "test/model",
              "sharedMaterials": "rules",
              "agents": [
                {
                  "id": "a",
                  "name": "Judge A",
                  "baseModelID": "test/model",
                  "variantArtifactPath": "runs/model-variants/seat-a.json",
                  "variantArtifactHash": "\(String(repeating: "b", count: 64))",
                  "systemPrompt": ""
                }
              ],
              "turns": [
                {
                  "id": "t1",
                  "title": "T",
                  "speakerAgentID": "a",
                  "promptTemplate": "go",
                  "outputLabel": "o",
                  "routing": "all",
                  "routedAgentIDs": [],
                  "includeScenarioMaterials": true,
                  "includeSpeakerContext": true
                }
              ]
            }
            """.utf8)
        try bytes.write(to: scenarioURL)

        var manifest = ExperimentManifest(
            name: "panel-study", description: "d", modelID: "test/model")
        manifest.studyKind = .multiAgent
        manifest.multiAgentScenarioPath = "prompts/panels/panel.json"
        manifest.multiAgentScenarioHash = pinnedHash
        return (manifest, scenarioURL, bytes)
    }

    /// A workspace whose single turn declares an endpoint — valid or typo'd.
    private func makeEndpointWorkspace(
        in root: URL, kind: String
    ) throws -> (manifest: ExperimentManifest, bytes: Data) {
        let panelsDirectory = root.appending(components: "prompts", "panels")
        try FileManager.default.createDirectory(
            at: panelsDirectory, withIntermediateDirectories: true)
        let bytes = Data(
            """
            {
              "name": "panel",
              "baseModelID": "test/model",
              "agents": [
                {"id": "a", "name": "Judge A", "baseModelID": "test/model",
                 "systemPrompt": ""}
              ],
              "turns": [
                {
                  "id": "t1",
                  "title": "Tentative vote",
                  "speakerAgentID": "a",
                  "promptTemplate": "State your vote as 'Vote: affirm'.",
                  "outputLabel": "vote",
                  "routing": "all",
                  "routedAgentIDs": [],
                  "includeScenarioMaterials": true,
                  "includeSpeakerContext": true,
                  "endpoint": {
                    "name": "vote",
                    "kind": "\(kind)",
                    "marker": "Vote:",
                    "vocabulary": ["affirm", "reverse"]
                  }
                }
              ]
            }
            """.utf8)
        try bytes.write(to: panelsDirectory.appending(component: "panel.json"))
        var manifest = ExperimentManifest(
            name: "panel-study", description: "d", modelID: "test/model")
        manifest.studyKind = .multiAgent
        manifest.multiAgentScenarioPath = "prompts/panels/panel.json"
        return (manifest, bytes)
    }

    private func withWorkspace(_ body: (URL) throws -> Void) throws {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "scenario-snapshot-\(UUID().uuidString)")
        ExperimentStore.rootOverride = temp
        // The scenario path resolves through VectorCatalog.projectRoot, so
        // the workspace root has to move too or a relative pin would escape
        // the temp tree.
        WorkspaceRoot.programmaticOverride = temp
        defer {
            WorkspaceRoot.programmaticOverride = nil
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        try body(temp)
    }

    private func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// (a) The run directory holds the scenario byte-for-byte.
    @Test func runDirectoryHoldsTheScenarioByteForByte() throws {
        try withWorkspace { root in
            let (manifest, _, bytes) = try makePanelWorkspace(
                in: root, pinnedHash: nil)
            let prepared = try ExperimentTasks.prepareMultiAgentRun(
                manifest: manifest)

            let snapshot = prepared.runDirectory.appending(
                component: "scenario.json")
            let copied = try Data(contentsOf: snapshot)
            #expect(copied == bytes)
            // Still hashes to what a manifest would pin — a re-encode would
            // not.
            #expect(sha256(copied) == prepared.scenarioHash)
            // Beside the manifest snapshot, not instead of it.
            #expect(
                FileManager.default.fileExists(
                    atPath: prepared.runDirectory
                        .appending(component: "experiment.json").path))
            // The point of the snapshot: seat→variant attribution answerable
            // from the run directory alone.
            let seats = try JSONDecoder()
                .decode(MultiAgentScenario.self, from: copied).agents
            #expect(seats.first?.variantArtifactPath
                == "runs/model-variants/seat-a.json")
            #expect(seats.first?.variantArtifactHash != nil)
        }
    }

    /// (b) A pinned-hash mismatch refuses before any generation — and before
    /// any run directory exists — naming both hashes.
    @Test func aDriftedScenarioRefusesBeforeAnyGeneration() throws {
        try withWorkspace { root in
            let stale = String(repeating: "a", count: 64)
            let (manifest, _, bytes) = try makePanelWorkspace(
                in: root, pinnedHash: stale)
            let live = sha256(bytes)

            do {
                _ = try ExperimentTasks.prepareMultiAgentRun(manifest: manifest)
                Issue.record("expected a refusal")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("changed since pinning"))
                #expect(error.reason.contains(live.prefix(12)))
                #expect(error.reason.contains(stale.prefix(12)))
            }

            // No run directory was created: no half-run to explain away, and
            // nothing generated.
            let runs = (try? FileManager.default.contentsOfDirectory(
                at: ExperimentStore.runsDirectory,
                includingPropertiesForKeys: nil)) ?? []
            #expect(runs.filter { $0.lastPathComponent.contains("panel-study") }
                .isEmpty)
        }
    }

    /// (c) A declared turn endpoint (Wave-2) survives the run prologue: the
    /// parse reaches the runner, and the verbatim snapshot carries the
    /// declaration into the run directory, so a reader with only the run can
    /// say what the stamps on its turns mean.
    @Test func aDeclaredEndpointSurvivesThePrologue() throws {
        try withWorkspace { root in
            let (manifest, bytes) = try makeEndpointWorkspace(
                in: root, kind: "choice")
            let prepared = try ExperimentTasks.prepareMultiAgentRun(
                manifest: manifest)

            let endpoint = try #require(prepared.scenario.turns.first?.endpoint)
            #expect(endpoint.name == "vote")
            #expect(endpoint.kind == .choice)
            #expect(endpoint.marker == "Vote:")
            let snapshot = try Data(
                contentsOf: prepared.runDirectory.appending(component: "scenario.json"))
            #expect(snapshot == bytes)
            #expect(
                try JSONDecoder().decode(MultiAgentScenario.self, from: snapshot)
                    .turns.first?.endpoint == endpoint)
        }
    }

    /// (d) A typo'd declaration refuses at load, before any run directory
    /// exists. The alternative is worse than a crash: a pinned, reviewed
    /// scenario that silently parses nothing for every turn of every
    /// replicate, indistinguishable from a panel that never answered.
    @Test func aTypodEndpointDeclarationRefusesTheRun() throws {
        try withWorkspace { root in
            let (manifest, _) = try makeEndpointWorkspace(in: root, kind: "chioce")
            do {
                _ = try ExperimentTasks.prepareMultiAgentRun(manifest: manifest)
                Issue.record("expected a refusal")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("unknown kind"))
                #expect(error.reason.contains("chioce"))
            }
        }
    }
}
