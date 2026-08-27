import Foundation
import Testing

@testable import ExperimentKit

/// Pure-CPU tests for the pre-derivation reuse/cost planner: synthetic record
/// lists + counts in, a `DerivationAdvice` verdict out. No networking, no
/// filesystem, no model.
@MainActor
struct DerivationPlannerTests {

    private let serverWorkspace = ClusterConnectionStore.Workspace.server(UUID())

    private func record(
        id: String = "runs/2026-06-10T000000Z-x/french",
        concept: String = "french",
        modelID: String = "Qwen/Qwen3-4B-MLX-4bit",
        stimulusSetHash: String? = "hash-1",
        method: String? = "caaMeanDifference",
        date: String? = "2026-06-10T00:00:00Z"
    ) -> SubstrateVectorRecord {
        SubstrateVectorRecord(
            id: id, concept: concept, modelID: modelID,
            stimulusSetHash: stimulusSetHash, extractionMethod: method,
            extractionDate: date)
    }

    // MARK: Fresh / stale / absent

    @Test func freshLocalCAAMatchIsReusableWithDate() {
        let advice = DerivationPlanner.advise(
            records: [record()],
            concept: "french", method: .caaMeanDifference,
            workspace: .local, modelID: "Qwen/Qwen3-4B-MLX-4bit",
            stimulusHash: "hash-1", cost: .paired(stimulusCount: 24))
        #expect(advice == .reusable(record(), date: "2026-06-10T00:00:00Z"))
    }

    @Test func freshServerCAAMatchComparesTheServerMethodSpelling() {
        // The server catalog exposes the contrastive `extractionMethod`
        // ("meanDifference"), not the local recipe spelling.
        let serverRecord = record(method: "meanDifference")
        let advice = DerivationPlanner.advise(
            records: [serverRecord],
            concept: "french", method: .caaMeanDifference,
            workspace: serverWorkspace, modelID: "Qwen/Qwen3-4B-MLX-4bit",
            stimulusHash: "hash-1", cost: .paired(stimulusCount: 24))
        #expect(advice == .reusable(serverRecord, date: "2026-06-10T00:00:00Z"))
    }

    @Test func freshServerLATMatch() {
        let serverRecord = record(method: "lat")
        let advice = DerivationPlanner.advise(
            records: [serverRecord],
            concept: "french", method: .pairedDifferencePCA,
            workspace: serverWorkspace, modelID: "Qwen/Qwen3-4B-MLX-4bit",
            stimulusHash: "hash-1", cost: .paired(stimulusCount: 24))
        #expect(advice == .reusable(serverRecord, date: "2026-06-10T00:00:00Z"))
    }

    @Test func staleStimuliReportTheClassifierReasonVerbatim() {
        let advice = DerivationPlanner.advise(
            records: [record()],
            concept: "french", method: .caaMeanDifference,
            workspace: .local, modelID: "Qwen/Qwen3-4B-MLX-4bit",
            stimulusHash: "hash-2", cost: .paired(stimulusCount: 24))
        #expect(
            advice
                == .staleExists(
                    record(),
                    reason: "stimuli changed since extraction (hash mismatch)"))
    }

    @Test func methodDivergenceIsStaleNotReusable() {
        // A LAT artifact exists but the builder now asks for CAA.
        let latRecord = record(method: "repeLAT")
        let advice = DerivationPlanner.advise(
            records: [latRecord],
            concept: "french", method: .caaMeanDifference,
            workspace: .local, modelID: "Qwen/Qwen3-4B-MLX-4bit",
            stimulusHash: "hash-1", cost: .paired(stimulusCount: 24))
        #expect(
            advice == .staleExists(latRecord, reason: "different extraction method/options"))
    }

    @Test func absentConceptModelPairIsNewWithPairedCostLine() {
        let advice = DerivationPlanner.advise(
            records: [record()],
            concept: "sycophancy", method: .caaMeanDifference,
            workspace: .local, modelID: "Qwen/Qwen3-4B-MLX-4bit",
            stimulusHash: "hash-9", cost: .paired(stimulusCount: 24))
        #expect(advice == .new(costLine: "24 stimuli × Qwen/Qwen3-4B-MLX-4bit"))
    }

    @Test func absentServerGrandMeanIsNewWithServerJobNote() {
        let advice = DerivationPlanner.advise(
            records: [],
            concept: "fear", method: .emotionGrandMean,
            workspace: serverWorkspace, modelID: "Qwen/Qwen3-4B",
            stimulusHash: "hash-9",
            cost: .grandMean(storyCount: 120, conceptCount: 6))
        #expect(
            advice
                == .new(
                    costLine:
                        "120 stories across 6 concepts × Qwen/Qwen3-4B — runs as a server job"))
    }

    @Test func grandMeanFreshMatchUsesTheGrandMeanMethodOnBothWorkspaces() {
        let local = record(concept: "fear", method: "emotionGrandMean")
        #expect(
            DerivationPlanner.advise(
                records: [local],
                concept: "fear", method: .emotionGrandMean,
                workspace: .local, modelID: "Qwen/Qwen3-4B-MLX-4bit",
                stimulusHash: "hash-1",
                cost: .grandMean(storyCount: 120, conceptCount: 6))
                == .reusable(local, date: "2026-06-10T00:00:00Z"))
        #expect(
            DerivationPlanner.advise(
                records: [local],
                concept: "fear", method: .emotionGrandMean,
                workspace: serverWorkspace, modelID: "Qwen/Qwen3-4B-MLX-4bit",
                stimulusHash: "hash-1",
                cost: .grandMean(storyCount: 120, conceptCount: 6))
                == .reusable(local, date: "2026-06-10T00:00:00Z"))
    }

    // MARK: Never silent reuse

    @Test func unreadableLiveRecipeNeverOffersReuse() {
        // stimulusHash nil = the live recipe could not be hashed; even a
        // matching record must not be offered for reuse.
        let advice = DerivationPlanner.advise(
            records: [record()],
            concept: "french", method: .caaMeanDifference,
            workspace: .local, modelID: "Qwen/Qwen3-4B-MLX-4bit",
            stimulusHash: nil, cost: .paired(stimulusCount: 24))
        #expect(advice == .new(costLine: "24 stimuli × Qwen/Qwen3-4B-MLX-4bit"))
    }

    @Test func recordWithoutHashIsStaleNotReusable() {
        let hashless = record(stimulusSetHash: nil)
        let advice = DerivationPlanner.advise(
            records: [hashless],
            concept: "french", method: .caaMeanDifference,
            workspace: .local, modelID: "Qwen/Qwen3-4B-MLX-4bit",
            stimulusHash: "hash-1", cost: .paired(stimulusCount: 24))
        #expect(
            advice
                == .staleExists(
                    hashless,
                    reason: "stimuli changed since extraction (hash mismatch)"))
    }

    // MARK: Cost lines

    @Test func costLinesAreHonestArithmeticWithoutTimeEstimates() {
        #expect(
            DerivationPlanner.costLine(
                .paired(stimulusCount: 24), modelID: "m", workspace: .local)
                == "24 stimuli × m")
        #expect(
            DerivationPlanner.costLine(
                .grandMean(storyCount: 120, conceptCount: 6), modelID: "m",
                workspace: .local)
                == "120 stories across 6 concepts × m")
        #expect(
            DerivationPlanner.costLine(
                .probe(exampleCount: 240), modelID: "m", workspace: .local)
                == "240 probe examples × m")
        #expect(
            DerivationPlanner.costLine(
                .probe(exampleCount: 240), modelID: "m", workspace: serverWorkspace)
                == "240 probe examples × m — runs as a server job")
    }

    // MARK: Method pins

    @Test func methodPinMatchesWhatEachSubstrateStamps() {
        #expect(
            DerivationPlanner.methodPin(for: .caaMeanDifference, workspace: .local)
                == "caaMeanDifference")
        #expect(
            DerivationPlanner.methodPin(for: .pairedDifferencePCA, workspace: .local) == "repeLAT")
        #expect(
            DerivationPlanner.methodPin(for: .emotionGrandMean, workspace: .local)
                == "emotionGrandMean")
        #expect(
            DerivationPlanner.methodPin(for: .caaMeanDifference, workspace: serverWorkspace)
                == "meanDifference")
        #expect(
            DerivationPlanner.methodPin(for: .pairedDifferencePCA, workspace: serverWorkspace) == "lat")
        #expect(
            DerivationPlanner.methodPin(for: .emotionGrandMean, workspace: serverWorkspace)
                == "emotionGrandMean")
    }
}
