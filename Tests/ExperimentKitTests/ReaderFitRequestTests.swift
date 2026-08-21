import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// Pure-CPU tests for the server reader-fit request assembly
/// (`ConceptBuilder.readerFitRequest` and its shared encoders). The contract
/// under test: the inline `pairsJSONL` payload is byte-identical to the pinned
/// local pairs file lines (same row ids, same tail held-out split assignment,
/// same sorted-keys encoding — so both engines hash the same dataset), and the
/// template selection resolves to exactly one of templateID / templateJSON
/// (the server's `POST /api/reader/fit` contract).
@Suite struct ReaderFitRequestTests {

    private func expectChatServiceError(
        containing substring: String,
        _ body: () throws -> Void
    ) {
        do {
            try body()
            Issue.record("expected ChatServiceError containing '\(substring)'")
        } catch let error as ChatServiceError {
            #expect(error.reason.contains(substring), "got: \(error.reason)")
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }

    // MARK: - Pair rows (the shared local-file / inline-payload encoder)

    @Test func pairRowsAreSortedKeysWithTailHeldOutSplit() throws {
        let lines = try ConceptBuilder.readerPairRows(
            concept: "fear",
            positives: ["p0", "p1", "p2"],
            negatives: ["n0", "n1", "n2"],
            heldOutPairCount: 1,
            templateID: "amount-in-scenario-v1")

        // Byte-exact rows: sorted keys, nil topic omitted, ids "<concept>-pair-<i>",
        // the LAST k pairs split "test" — pinned so any encoding drift (which
        // would silently change the cross-engine dataset hash) fails loudly.
        #expect(
            lines == [
                #"{"concept":"fear","id":"fear-pair-0","negativeStimulus":"n0","positiveStimulus":"p0","split":"train","templateID":"amount-in-scenario-v1"}"#,
                #"{"concept":"fear","id":"fear-pair-1","negativeStimulus":"n1","positiveStimulus":"p1","split":"train","templateID":"amount-in-scenario-v1"}"#,
                #"{"concept":"fear","id":"fear-pair-2","negativeStimulus":"n2","positiveStimulus":"p2","split":"test","templateID":"amount-in-scenario-v1"}"#,
            ])

        // The rows round-trip through the shared SteeringKit loader with the
        // intended splits — the same parse the fit itself performs.
        let dataset = try RepEReader.parsePairs(
            Data((lines.joined(separator: "\n") + "\n").utf8), source: "test")
        #expect(dataset.concept == "fear")
        #expect(dataset.pairs.map(\.split) == ["train", "train", "test"])
        #expect(dataset.train.count == 2)
        #expect(dataset.heldOut.count == 1)
    }

    @Test func heldOutClampNeverStarvesTheTrainSplit() throws {
        // Same clamp as the local build path: at least 2 pairs stay "train".
        let lines = try ConceptBuilder.readerPairRows(
            concept: "fear",
            positives: ["p0", "p1", "p2"],
            negatives: ["n0", "n1", "n2"],
            heldOutPairCount: 99,
            templateID: "t-v1")
        let dataset = try RepEReader.parsePairs(
            Data((lines.joined(separator: "\n") + "\n").utf8), source: "test")
        #expect(dataset.pairs.map(\.split) == ["train", "train", "test"])
    }

    // MARK: - Template selection (templateID xor templateJSON)

    @Test func registryRequestPinsTemplateIDOnly() throws {
        let request = try ConceptBuilder.readerFitRequest(
            concept: "fear",
            positives: ["p0", "p1"],
            negatives: ["n0", "n1"],
            heldOutPairCount: 0,
            registryTemplateID: "amount-in-scenario-v1",
            customTemplateText: nil)
        #expect(request.concept == "fear")
        #expect(request.templateID == "amount-in-scenario-v1")
        #expect(request.templateJSON == nil)
        // No model pinned unless the caller passes one (the server then uses
        // whatever model is loaded — the pre-existing behavior).
        #expect(request.modelID == nil)
        // Payload is the pinned rows joined with a trailing newline — the
        // exact bytes the local path writes to prompts/readers/<name>/pairs.jsonl.
        let lines = try ConceptBuilder.readerPairRows(
            concept: "fear", positives: ["p0", "p1"], negatives: ["n0", "n1"],
            heldOutPairCount: 0, templateID: "amount-in-scenario-v1")
        #expect(request.pairsJSONL == lines.joined(separator: "\n") + "\n")
        #expect(request.pairsJSONL.hasSuffix("\n"))
    }

    @Test func selectedServerModelThreadsThroughVerbatim() throws {
        // Item A3: the builder's selected server model rides the request so
        // the server acquires exactly that model for the fit.
        let request = try ConceptBuilder.readerFitRequest(
            concept: "fear",
            positives: ["p0", "p1"],
            negatives: ["n0", "n1"],
            heldOutPairCount: 0,
            registryTemplateID: "amount-in-scenario-v1",
            customTemplateText: nil,
            modelID: "Qwen/Qwen3-4B")
        #expect(request.modelID == "Qwen/Qwen3-4B")
        // Pinning a model must not perturb the dataset payload bytes.
        let unpinned = try ConceptBuilder.readerFitRequest(
            concept: "fear",
            positives: ["p0", "p1"],
            negatives: ["n0", "n1"],
            heldOutPairCount: 0,
            registryTemplateID: "amount-in-scenario-v1",
            customTemplateText: nil)
        #expect(request.pairsJSONL == unpinned.pairsJSONL)
    }

    @Test func customRequestPinsTemplateJSONOnly() throws {
        let request = try ConceptBuilder.readerFitRequest(
            concept: "fear",
            positives: ["p0", "p1"],
            negatives: ["n0", "n1"],
            heldOutPairCount: 0,
            registryTemplateID: nil,
            customTemplateText: "S: {{stimulus}} q")
        #expect(request.templateID == nil)
        // Byte-exact custom template object: the same JSON the local path
        // persists into its run directory (sorted keys, id "custom-<name>-v1",
        // conceptSlot derived from a {{concept}} slot, stamped divergence).
        #expect(
            request.templateJSON
                == #"{"conceptSlot":false,"divergence":"custom-unregistered","id":"custom-fear-v1","latToken":"final","text":"S: {{stimulus}} q"}"#)
        // The rows pin the custom template's id, not a registry id.
        #expect(request.pairsJSONL.contains(#""templateID":"custom-fear-v1""#))
    }

    @Test func customTemplateWithConceptSlotDerivesConceptSlotTrue() throws {
        let request = try ConceptBuilder.readerFitRequest(
            concept: "fear",
            positives: ["p0", "p1"],
            negatives: ["n0", "n1"],
            heldOutPairCount: 0,
            registryTemplateID: nil,
            customTemplateText: "How much {{concept}}? S: {{stimulus}} —")
        #expect(request.templateJSON?.contains(#""conceptSlot":true"#) == true)
    }

    @Test func customTemplateRequiresStimulusSlot() {
        expectChatServiceError(containing: "{{stimulus}}") {
            _ = try ConceptBuilder.readerFitRequest(
                concept: "fear",
                positives: ["p0", "p1"],
                negatives: ["n0", "n1"],
                heldOutPairCount: 0,
                registryTemplateID: nil,
                customTemplateText: "no slot here")
        }
    }

    @Test func missingTemplateSelectionThrows() {
        expectChatServiceError(containing: "select a task template") {
            _ = try ConceptBuilder.readerFitRequest(
                concept: "fear",
                positives: ["p0", "p1"],
                negatives: ["n0", "n1"],
                heldOutPairCount: 0,
                registryTemplateID: nil,
                customTemplateText: nil)
        }
    }
}
