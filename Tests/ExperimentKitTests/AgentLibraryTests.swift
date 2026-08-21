import Foundation
import Testing
@testable import ExperimentKit

/// Pure rules behind the Agents library UI: kind classification, readiness
/// chips, virtual baseline rows, and filters. All CPU-only — availability
/// arrives as data, never from a live workspace.
struct AgentLibraryTests {

    // MARK: fixtures

    private func artifact(
        name: String = "agent-1",
        baseModelID: String = "Qwen/Qwen3-4B-MLX-4bit",
        adapters: [ModelVariantArtifact.AdapterRef] = [],
        injections: [ModelVariantArtifact.InjectionRef] = [],
        promotion: ModelVariantArtifact.Promotion? = nil
    ) -> ModelVariantArtifact {
        ModelVariantArtifact(
            name: name,
            baseModelID: baseModelID,
            adapters: adapters,
            injections: injections,
            promptMode: "chatAssistant",
            qwenThinkingEnabled: false,
            temperature: 0,
            systemPrompt: "",
            promotion: promotion)
    }

    private func promotion(by promotedBy: String) -> ModelVariantArtifact.Promotion {
        ModelVariantArtifact.Promotion(
            experiment: "screen-1",
            experimentHash: "abc",
            promotedAt: "2026-07-07T00:00:00Z",
            promotedBy: promotedBy,
            substrate: "swift-mlx",
            appVersion: "test")
    }

    private var injection: ModelVariantArtifact.InjectionRef {
        .init(concept: "french", vectorArtifactID: "vec-1", layer: 12, alpha: 0.4)
    }

    private var adapter: ModelVariantArtifact.AdapterRef {
        .init(
            name: "lora-1", artifactPath: "runs/ft/artifact.json",
            adapterDirectory: "runs/ft/adapter", adapterHash: "hash-recorded")
    }

    // MARK: kind

    @Test func kindClassificationPriorities() {
        #expect(AgentLibrary.kind(of: artifact()) == .exploratory)
        #expect(
            AgentLibrary.kind(of: artifact(injections: [injection])) == .vectorOnly)
        #expect(AgentLibrary.kind(of: artifact(adapters: [adapter])) == .adapter)
        // Promotion provenance beats composition.
        #expect(
            AgentLibrary.kind(
                of: artifact(
                    injections: [injection], promotion: promotion(by: "criterion")))
                == .sweepPromoted)
        #expect(
            AgentLibrary.kind(
                of: artifact(
                    adapters: [adapter], injections: [injection],
                    promotion: promotion(by: "manualOverride")))
                == .overridePromoted)
    }

    // MARK: runnability + chips

    @Test func runnableLocallyRequiresEveryComponent() {
        let agent = artifact(adapters: [adapter], injections: [injection])
        let full = AgentLibrary.Availability(
            localModelIDs: ["Qwen/Qwen3-4B-MLX-4bit"],
            localVectorIDs: ["vec-1"],
            localAdapterKeys: ["runs/ft/adapter"])
        #expect(AgentLibrary.isRunnableLocally(agent, availability: full))

        let noModel = AgentLibrary.Availability(
            localVectorIDs: ["vec-1"], localAdapterKeys: ["runs/ft/adapter"])
        #expect(!AgentLibrary.isRunnableLocally(agent, availability: noModel))

        let noVector = AgentLibrary.Availability(
            localModelIDs: ["Qwen/Qwen3-4B-MLX-4bit"],
            localAdapterKeys: ["runs/ft/adapter"])
        #expect(!AgentLibrary.isRunnableLocally(agent, availability: noVector))

        let noAdapter = AgentLibrary.Availability(
            localModelIDs: ["Qwen/Qwen3-4B-MLX-4bit"], localVectorIDs: ["vec-1"])
        #expect(!AgentLibrary.isRunnableLocally(agent, availability: noAdapter))
    }

    @Test func chipsReportProvenanceCompositionAndRunnability() {
        let agent = artifact(
            injections: [injection], promotion: promotion(by: "criterion"))
        let chips = AgentLibrary.chips(
            for: agent,
            availability: .init(
                localModelIDs: ["Qwen/Qwen3-4B-MLX-4bit"],
                localVectorIDs: ["vec-1"]))
        let labels = chips.map(\.label)
        #expect(labels.contains("Sweep-promoted"))
        #expect(labels.contains("Uses steering"))
        #expect(labels.contains("Runnable locally"))
        #expect(!labels.contains("Uses adapter"))
        #expect(!labels.contains("Missing artifact"))
        #expect(!labels.contains("Runnable on server"))
    }

    @Test func overrideAndExploratoryChips() {
        let overridden = AgentLibrary.chips(
            for: artifact(
                injections: [injection], promotion: promotion(by: "manualOverride")),
            availability: .init())
        #expect(overridden.map(\.label).contains("Override"))

        let exploratory = AgentLibrary.chips(for: artifact(), availability: .init())
        #expect(exploratory.map(\.label).contains("Exploratory"))
    }

    @Test func missingArtifactChecksEveryProvidedCatalog() {
        let agent = artifact(injections: [injection])
        // Not in the local catalog, but present server-side: not missing.
        let serverHasIt = AgentLibrary.missingVectorRefs(
            agent, availability: .init(serverVectorIDs: ["vec-1"]))
        #expect(serverHasIt.isEmpty)
        // In no catalog at all: missing, and the chip says which ref.
        let nowhere = AgentLibrary.missingVectorRefs(agent, availability: .init())
        #expect(nowhere == ["vec-1"])
        let chips = AgentLibrary.chips(for: agent, availability: .init())
        #expect(chips.contains { $0.label == "Missing artifact" })
    }

    @Test func serverRunnabilityChipFollowsProvidedVerdict() {
        let agent = artifact(injections: [injection])
        let yes = AgentLibrary.chips(
            for: agent,
            availability: .init(serverApplicability: true, serverVectorIDs: ["vec-1"]))
        #expect(yes.contains { $0.label == "Runnable on server" })
        // nil = no server connected: no claim either way.
        let unknown = AgentLibrary.chips(
            for: agent, availability: .init(serverVectorIDs: ["vec-1"]))
        #expect(!unknown.contains { $0.label == "Runnable on server" })
    }

    @Test func hashDriftOnlyWhenBothSidesRecordAndDisagree() {
        let agent = artifact(adapters: [adapter])
        #expect(
            AgentLibrary.driftedAdapterRefs(
                agent, currentAdapterHashes: ["runs/ft/adapter": "hash-recorded"]
            ).isEmpty)
        #expect(
            AgentLibrary.driftedAdapterRefs(
                agent, currentAdapterHashes: ["runs/ft/adapter": "hash-changed"])
                == ["lora-1"])
        // Unknown current hash: silent, never guessed.
        #expect(AgentLibrary.driftedAdapterRefs(agent, currentAdapterHashes: [:]).isEmpty)
        let chips = AgentLibrary.chips(
            for: agent, availability: .init(),
            currentAdapterHashes: ["runs/ft/adapter": "hash-changed"])
        #expect(chips.contains { $0.label == "Hash drift" })
    }

    // MARK: virtual baselines

    @Test func baselineRowsAreDistinctPerBaseModelAndNamed() {
        let rows = AgentLibrary.baselineRows(for: [
            artifact(name: "a", baseModelID: "model-b"),
            artifact(name: "b", baseModelID: "model-a"),
            artifact(name: "c", baseModelID: "model-b"),
        ])
        #expect(rows.map(\.baseModelID) == ["model-a", "model-b"])
        #expect(rows[0].name == "model-a (baseline)")
    }

    // MARK: filters

    @Test func filtersMatchArtifactsAndExcludeBaselinesWhereInapplicable() {
        let promoted = artifact(
            injections: [injection], promotion: promotion(by: "criterion"))
        let plain = artifact(name: "plain")

        var filter = AgentLibrary.Filter(sweepPromotedOnly: true)
        #expect(AgentLibrary.matches(promoted, filter: filter, runnableHere: false))
        #expect(!AgentLibrary.matches(plain, filter: filter, runnableHere: false))
        #expect(filter.excludesBaselines)
        #expect(
            !AgentLibrary.matches(
                baseline: .init(baseModelID: "Qwen/Qwen3-4B-MLX-4bit"),
                filter: filter, runnableHere: true))

        filter = AgentLibrary.Filter(baseModelID: "other-model")
        #expect(!AgentLibrary.matches(promoted, filter: filter, runnableHere: true))

        filter = AgentLibrary.Filter(runnableHere: true)
        #expect(!AgentLibrary.matches(promoted, filter: filter, runnableHere: false))
        #expect(AgentLibrary.matches(promoted, filter: filter, runnableHere: true))
        // Base-model and runnable filters DO apply to baseline rows.
        #expect(
            AgentLibrary.matches(
                baseline: .init(baseModelID: "m"), filter: filter, runnableHere: true))
        #expect(
            !AgentLibrary.matches(
                baseline: .init(baseModelID: "m"), filter: filter, runnableHere: false))
    }
}
