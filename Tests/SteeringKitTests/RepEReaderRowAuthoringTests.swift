import Foundation
import Testing

@testable import SteeringKit

/// The reader dataset's WRITE side, and the shape-only contrast derivation the
/// authoring UI needs.
///
/// Two gaps closed here, both of them "the type can parse a shape it cannot
/// produce" problems:
///
/// 1. `Pair`'s synthesized encoding emitted `positiveStimulus` /
///    `negativeStimulus` unconditionally — as empty strings on a template-pair
///    row — and `parsePairs` refuses exactly that combination. A row written by
///    the engine could not be read back by the engine.
/// 2. `resolveContrastMode` took a whole `Dataset`, so nothing could ask what a
///    (shape, template) choice MEANS before the rows existed. The editor has to
///    ask precisely then, and it must get the engine's refusal, not its own.
@Suite struct RepEReaderRowAuthoringTests {

    private static let stancePair = RepEReader.TaskTemplate(
        id: "instructed-stance-pair-v1", conceptSlot: false,
        text: "{{instruction}}\nScenario: {{stimulus}}\nThe described state is",
        latToken: "final", hash: "th",
        instructionPair: RepEReader.TaskTemplate.InstructionPair(
            experimental: "T plus", reference: "T minus"))

    private static let plain = RepEReader.TaskTemplate(
        id: "unnamed-scenario-v1", conceptSlot: false,
        text: "Scenario: {{stimulus}}\nThe intensity is", latToken: "final",
        hash: "th")

    private func expectReaderError(
        containing substring: String, _ body: () throws -> Void
    ) {
        do {
            try body()
            Issue.record("expected ReaderError containing '\(substring)'")
        } catch let error as RepEReader.ReaderError {
            #expect(error.reason.contains(substring), "got: \(error.reason)")
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }

    // MARK: - Encoding

    @Test func templatePairRowEncodesToTheShapeTheLoaderAccepts() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let row = RepEReader.Pair.templatePair(
            id: "candour-row-0", concept: "candour", stimulus: "s0",
            split: "train", templateID: "instructed-stance-pair-v1")
        let line = String(decoding: try encoder.encode(row), as: UTF8.self)

        // The two pair fields are ABSENT, not empty: writing them would make
        // the loader refuse the engine's own row.
        #expect(
            line
                == #"{"concept":"candour","id":"candour-row-0","split":"train","stimulus":"s0","templateID":"instructed-stance-pair-v1"}"#)

        let dataset = try RepEReader.parsePairs(
            Data((line + "\n").utf8), source: "authored")
        #expect(dataset.shape == .singleStimulus)
        #expect(dataset.pairs.first?.stimulus == "s0")
        #expect(dataset.pairs.first?.isTemplatePairRow == true)
    }

    /// The content-pair encoding is untouched — its bytes are the cross-engine
    /// dataset hash, and a shape that never existed before must not move it.
    @Test func contentPairRowEncodingIsUnchanged() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let row = RepEReader.Pair(
            id: "candour-pair-0", concept: "candour",
            positiveStimulus: "p0", negativeStimulus: "n0",
            split: "train", templateID: "unnamed-scenario-v1")
        #expect(
            String(decoding: try encoder.encode(row), as: UTF8.self)
                == #"{"concept":"candour","id":"candour-pair-0","negativeStimulus":"n0","positiveStimulus":"p0","split":"train","templateID":"unnamed-scenario-v1"}"#)
    }

    // MARK: - Shape-only contrast derivation

    @Test func shapeOverloadDerivesTheSameContrastAsTheDatasetOverload() throws {
        #expect(
            try RepEReader.resolveContrastMode(
                shape: .contentPair, template: Self.plain) == .supervisedContent)
        #expect(
            try RepEReader.resolveContrastMode(
                shape: .singleStimulus, template: Self.stancePair)
                == .unsupervisedTemplatePair)

        // And the dataset overload now routes through it, so there is one
        // switch and one set of refusal literals.
        let dataset = RepEReader.Dataset(
            concept: "candour",
            pairs: [
                RepEReader.Pair.templatePair(
                    concept: "candour", stimulus: "s",
                    templateID: "instructed-stance-pair-v1")
            ],
            hash: "dh")
        #expect(
            try RepEReader.resolveContrastMode(
                dataset: dataset, template: Self.stancePair)
                == .unsupervisedTemplatePair)
    }

    @Test func shapeOverloadRaisesThePinnedTwinRefusals() throws {
        expectReaderError(containing: "a second stimulus would be a confound") {
            _ = try RepEReader.resolveContrastMode(
                shape: .contentPair, template: Self.stancePair)
        }
        expectReaderError(containing: "nothing to contrast the stimulus against") {
            _ = try RepEReader.resolveContrastMode(
                shape: .singleStimulus, template: Self.plain)
        }
    }
}
