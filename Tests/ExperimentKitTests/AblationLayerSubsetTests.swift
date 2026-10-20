import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// An agent's ablation covers exactly the layers it declares.
///
/// Release review 2026-10-19: the agent cell builder ablated from the first
/// declared layer to the last, so `[3, 7, 10]` also removed the direction at
/// 4–6 and 8–9, an unsorted list such as `[10, 3]` formed an invalid closed
/// range and crashed, and a list naming only layers the vector does not have
/// ablated every layer, where the Python engine ablated none. Both engines
/// now ablate the listed layers and nothing else, ascending and once each,
/// which is what the preregistration's `L3+L7+L10` describes, and both refuse
/// a listed layer the vector does not have with the same sentence.
///
/// The layers and refusals in `Tests/Fixtures/cross-engine/ablation-layers.json`
/// were produced by the PYTHON engine's agent cell builder; this engine must
/// cover the same layers or refuse with the same sentence. Server twin:
/// `Server/tests/test_ablation_layer_subsets.py`.
@Suite struct AblationLayerSubsetTests {

    /// A saved 12-layer vector whose every layer is nonzero, so no cell is
    /// skipped for a zero norm.
    private func writeVector(layers: Int = 12) throws -> (dir: URL, id: String) {
        let vectors = ConceptVectors(
            perLayer: (0..<layers).map { [Float($0 + 1), 1, 0] })
        let sidecar = SteeringVectorSidecar(
            modelID: "test/model", concept: "fear",
            stimulusSetHash: "abc123", vectors: vectors)
        let dir = FileManager.default.temporaryDirectory
            .appending(component: "ablation-layers-\(UUID().uuidString)")
        try SteeringVectorStore.save(
            vectors: vectors, sidecar: sidecar, to: dir, name: "fear")
        return (dir, dir.appending(component: "fear").path)
    }

    private func agent(
        _ injections: [ModelVariantArtifact.InjectionRef], bandWidth: Int = 1
    ) -> ModelVariantArtifact {
        ModelVariantArtifact(
            name: "fear-ablation", baseModelID: "test/model",
            injections: injections,
            bandWidth: bandWidth, alphaInNormUnits: false,
            promptMode: "chatAssistant", qwenThinkingEnabled: false,
            temperature: 0, systemPrompt: "")
    }

    private func ablation(
        _ id: String, layers: [Int]?
    ) -> ModelVariantArtifact.InjectionRef {
        .init(
            concept: "fear", vectorArtifactID: id, layer: 3, alpha: 1,
            mode: .ablate, layers: layers)
    }

    // MARK: - The Python engine's layers

    @Test func everyDeclaredListCoversThePythonEnginesLayers() throws {
        let url = CodeResources.compiledCheckoutPath.appending(
            components: "Tests", "Fixtures", "cross-engine", "ablation-layers.json")
        let fixture = try #require(
            try JSONSerialization.jsonObject(with: Data(contentsOf: url))
                as? [String: Any])
        let layerCount = try #require(fixture["layerCount"] as? Int)
        let bandWidth = try #require(fixture["bandWidth"] as? Int)
        let cases = try #require(fixture["cases"] as? [[String: Any]])
        #expect(cases.count >= 12, "the declared-list coverage shrank")
        let (dir, id) = try writeVector(layers: layerCount)
        defer { try? FileManager.default.removeItem(at: dir) }
        for entry in cases {
            let label = try #require(entry["label"] as? String)
            var injection = try #require(entry["injection"] as? [String: Any])
            injection["vectorArtifactID"] = id
            let ref = try JSONDecoder().decode(
                ModelVariantArtifact.InjectionRef.self,
                from: JSONSerialization.data(withJSONObject: injection))
            let variant = agent([ref], bandWidth: bandWidth)
            if let refusal = entry["refusal"] as? String {
                do {
                    let cells = try ExperimentTasks.injections(for: variant)
                    Issue.record("\(label): ran \(cells.map(\.layer)), expected a refusal")
                } catch let error as ExperimentError {
                    #expect(error.reason == refusal, "\(label)")
                }
                continue
            }
            let expected = try #require(entry["layers"] as? [Int], "\(label)")
            let cells = try ExperimentTasks.injections(for: variant)
            #expect(cells.map(\.layer) == expected, "\(label)")
            #expect(
                cells.allSatisfy { $0.mode == ref.effectiveMode }, "\(label)")
        }
    }

    // MARK: - The three shapes the review found

    @Test func aListWithGapsAblatesExactlyThoseLayers() throws {
        let (dir, id) = try writeVector()
        defer { try? FileManager.default.removeItem(at: dir) }
        let cells = try ExperimentTasks.injections(
            for: agent([ablation(id, layers: [3, 7, 10])]))
        #expect(cells.map(\.layer) == [3, 7, 10])
        #expect(cells.allSatisfy { $0.mode == .ablate && $0.alpha == 1 })
    }

    /// `[10, 3]` used to form `10 ... 3`, which traps.
    @Test func anUnsortedOrRepeatedListIsTheSetItNames() throws {
        let (dir, id) = try writeVector()
        defer { try? FileManager.default.removeItem(at: dir) }
        let unsorted = try ExperimentTasks.injections(
            for: agent([ablation(id, layers: [10, 3])]))
        #expect(unsorted.map(\.layer) == [3, 10])
        let repeated = try ExperimentTasks.injections(
            for: agent([ablation(id, layers: [7, 3, 7])]))
        #expect(repeated.map(\.layer) == [3, 7])
    }

    /// nil and empty both mean every layer, as before.
    @Test func noListStillMeansEveryLayer() throws {
        let (dir, id) = try writeVector()
        defer { try? FileManager.default.removeItem(at: dir) }
        for layers in [nil, [Int]()] {
            let cells = try ExperimentTasks.injections(
                for: agent([ablation(id, layers: layers)]))
            #expect(cells.map(\.layer) == Array(0..<12))
        }
    }

    /// A list naming only layers the vector does not have used to ablate
    /// every layer here and none on the server. Now any listed layer the
    /// vector does not have refuses, on both engines, with the repair.
    @Test func layersTheVectorDoesNotHaveRefuseWithTheRepair() throws {
        let (dir, id) = try writeVector()
        defer { try? FileManager.default.removeItem(at: dir) }
        for (layers, named, pronoun) in [
            ([12, 40], "layers 12, 40", "them"), ([3, 12], "layer 12", "it"),
            ([-1, 2], "layer -1", "it"),
        ] {
            #expect {
                try ExperimentTasks.injections(for: agent([ablation(id, layers: layers)]))
            } throws: { error in
                (error as? ExperimentError)?.reason
                    == "variant 'fear-ablation': the ablation of 'fear' lists \(named), "
                    + "which its vector does not have (its layers are 0 to 11); "
                    + "remove \(pronoun) from the injection's layers, or delete the "
                    + "list to ablate every layer; in a frozen study, save the "
                    + "corrected agent as a new agent, duplicate the study, attach "
                    + "the new agent to the copy, then validate and freeze it"
            }
        }
        // Steering ignores the list, so it never refuses for it.
        let steering = ModelVariantArtifact.InjectionRef(
            concept: "fear", vectorArtifactID: id, layer: 5, alpha: 1, layers: [40])
        #expect(steering.ablationLayerProblem(layerCount: 12) == nil)
        #expect(try ExperimentTasks.injections(for: agent([steering])).map(\.layer) == [5])
    }

    @Test func declaredLayersReadAsASet() {
        #expect(ablation("v", layers: [10, 3, 7, 3]).declaredAblationLayers == [3, 7, 10])
        #expect(ablation("v", layers: []).declaredAblationLayers == nil)
        #expect(ablation("v", layers: nil).declaredAblationLayers == nil)
        #expect(ablation("v", layers: [10, 3]).resolvedLayers(layerCount: 12) == [3, 10])
    }

    // MARK: - The other readers of the list

    /// The mean-alignment preflight names the layer it measured, not the
    /// offset into a contiguous span.
    @Test func theMeanAlignmentWarningNamesADeclaredLayer() throws {
        let warning = ExperimentTasks.reportAblationMeanAlignment(
            concept: "fear",
            vectors: [[0, 1, 0], [1, 0, 0]],
            layers: [3, 10],
            neutralMean: [[1, 0, 0], [1, 0, 0]],
            where: "variant 'v'", remedy: "center it")
        #expect(try #require(warning).contains("at layer 10;"))
    }

    /// The Playground ablates every layer and says when an agent does not;
    /// a full list written in any order is every layer. A list naming a layer
    /// the vector does not have is said apart, because a study run refuses
    /// it rather than ablating "only some layers".
    @Test func thePlaygroundReadsTheListAsTheRunDoes() {
        let everyLayer = ChatService.ablationSettings(
            agent: "a",
            restored: [(ablation("v", layers: [3, 2, 1, 0, 2]), 4)],
            currentlyCentered: false)
        #expect(everyLayer.notes.isEmpty)
        let beyondTheVector = ChatService.ablationSettings(
            agent: "a",
            restored: [(ablation("v", layers: [0, 1, 2, 3, 7]), 4)],
            currentlyCentered: false)
        #expect(beyondTheVector.notes == [
            "agent 'a' lists ablation layers its vector does not have for fear; "
                + "a study run refuses it until the list is fixed",
        ])
        let subset = ChatService.ablationSettings(
            agent: "a",
            restored: [(ablation("v", layers: [3, 1]), 4)],
            currentlyCentered: false)
        #expect(subset.notes.count == 1 && subset.notes[0].contains("only some layers"))
    }

    /// Reordering or repeating a layer in the list is not an edit, so it
    /// keeps an agent's promotion record; adding a layer is.
    @Test func reorderingTheListIsNotAnEdit() {
        let stored = agent([ablation("runs/x/fear", layers: [3, 10])])
        #expect(AgentEditProvenance.doesTheSame(
            agent([ablation("runs/x/fear", layers: [10, 3, 10])]), as: stored))
        #expect(!AgentEditProvenance.doesTheSame(
            agent([ablation("runs/x/fear", layers: [3, 7, 10])]), as: stored))
    }
}
