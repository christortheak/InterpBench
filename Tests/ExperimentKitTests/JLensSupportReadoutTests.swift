import Foundation
import Testing

@testable import ExperimentKit

/// Decoding the J-lens support readout, against JSON captured from the running
/// server (`POST /api/jlens/support` → job result, 2026-07-29, gemma-3-4b-it).
///
/// A wire-contract suite, not a numerics one: the Python side owns the solver and
/// tests it against constructed cases. What can silently break here is a key
/// rename or an optional that stops arriving, and that failure mode looks like an
/// empty panel rather than an error.
struct JLensSupportReadoutTests {

    /// Verbatim from the job result — including the fields the app does not read,
    /// so an added server key cannot break decoding.
    static let captured = """
        {
          "artifactType": "jlens-support",
          "budget": 8,
          "device": "mps",
          "directionConvention": "J_l^T (g . u_t)",
          "layers": [
            {
              "coneExhaustedAt": null,
              "energyFraction": 0.05192464217543602,
              "energyOverNull": 0.013099651783704758,
              "layer": 29,
              "nullConeExhaustedAt": null,
              "nullEnergyFraction": 0.03882499039173126,
              "support": [
                {"coefficient": 243.52786254882812, "piece": "\\u2581\\"",
                 "share": 0.2557300329208374, "tokenID": 623},
                {"coefficient": 141.5634002685547, "piece": "\\u2581exclaimed",
                 "share": 0.1486559957265854, "tokenID": 81569},
                {"coefficient": 112.17279052734375, "piece": "\\u2581hilarious",
                 "share": 0.11779300123453140, "tokenID": 51323}
              ]
            }
          ],
          "lensFitCorpus": "Salesforce/wikitext:wikitext-103-raw-v1",
          "lensFitPrompts": 546,
          "lensID": "google--gemma-3-4b-it--jlens-wikitext",
          "modelID": "google/gemma-3-4b-it",
          "nullSeed": 20260729,
          "revision": "093f9f388b31de276ce2de164bdc2081324b9767",
          "runDirectory": "/ws/runs/20260730T043042131-jlens-support-extraversion",
          "schemaVersion": 1,
          "solver": "non-negative matching pursuit + projected-gradient refinement",
          "sourceLensCommit": "a4114d7752d11eb546e6cf372213d7e75526d3a1",
          "supportIdentityHash": "b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90a1",
          "vector": {
            "concept": "extraversion",
            "extractionMethod": "meanDifference",
            "name": "extraversion",
            "recipeMethod": null,
            "runDirectory": "/ws/runs/20260728T212338386-concept-extraversion",
            "sha256": "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c4b5a69788796a5b4c3d2e1f0"
          }
        }
        """

    private func decoded() throws -> JLensSupportReadout {
        try JSONDecoder().decode(JLensSupportReadout.self,
                                 from: Data(Self.captured.utf8))
    }

    @Test func decodesTheServerPayload() throws {
        let readout = try decoded()
        #expect(readout.lensID == "google--gemma-3-4b-it--jlens-wikitext")
        #expect(readout.modelID == "google/gemma-3-4b-it")
        #expect(readout.budget == 8)
        #expect(readout.lensFitPrompts == 546)
        #expect(readout.vector.name == "extraversion")
        #expect(readout.vector.extractionMethod == "meanDifference")
        #expect(readout.layers.count == 1)
        #expect(readout.layers[0].support.count == 3)
    }

    @Test func keepsTheSentencePieceMarkerVerbatim() throws {
        // "▁exclaimed" is not "exclaimed": the marker says the token carries a
        // leading space, and two vocabulary entries can differ by exactly that.
        // Stripping it for display would make distinct atoms look identical.
        let pieces = try decoded().layers[0].support.map(\.piece)
        #expect(pieces[1] == "\u{2581}exclaimed")
        #expect(pieces[0] == "\u{2581}\"")
    }

    @Test func everyLayerCarriesItsOwnNull() throws {
        // The whole reason the readout is trustworthy. A matched-norm random
        // direction scores comparably in this dictionary, so a layer that
        // decoded an energy figure without its control would invite a
        // conclusion the number cannot support.
        for layer in try decoded().layers {
            #expect(layer.nullEnergyFraction > 0)
            #expect(abs(layer.energyOverNull
                        - (layer.energyFraction - layer.nullEnergyFraction)) < 1e-9)
        }
    }

    @Test func marginJustAboveNullIsNotDressedUpAsAResult() throws {
        // +1.3 points. `beatsNull` must report the sign honestly and nothing more —
        // the tokens are the finding, not this margin.
        let layer = try decoded().layers[0]
        #expect(layer.beatsNull)
        #expect(layer.energyOverNull < 0.02)
    }

    @Test func aNegativeMarginIsRepresentableAndNotAnError() throws {
        // The common case in practice: real concept vectors reconstructed BELOW
        // their nulls at four of five layers on gemma-3-4b-it. A view that
        // treated that as a failure would hide the readout exactly when it is
        // most worth reading.
        let json = Self.captured
            .replacingOccurrences(of: "\"energyOverNull\": 0.013099651783704758",
                                  with: "\"energyOverNull\": -0.03")
            .replacingOccurrences(of: "\"nullEnergyFraction\": 0.03882499039173126",
                                  with: "\"nullEnergyFraction\": 0.08192464217543602")
        let readout = try JSONDecoder().decode(JLensSupportReadout.self,
                                               from: Data(json.utf8))
        #expect(!readout.layers[0].beatsNull)
        #expect(readout.layers[0].support.count == 3)
    }

    @Test func sharesArriveDescendingAndInRange() throws {
        // What the view actually depends on: it draws a bar per share and lists
        // atoms in the order received. Shares sum to 1 over the FULL support,
        // but this fixture keeps the first three of the run's eight rows, so
        // asserting the sum here would test the fixture rather than the decoder —
        // and `share` is computed server-side, where its definition is tested.
        let support = try decoded().layers[0].support
        #expect(support == support.sorted { $0.share > $1.share })
        #expect(support.allSatisfy { $0.share > 0 && $0.share <= 1 })
        #expect(support.allSatisfy { $0.coefficient >= 0 })
    }

    @Test func strongestLayerPicksTheLargestMargin() throws {
        let base = try decoded()
        let weaker = JLensSupportLayer(
            layer: 5, support: [], energyFraction: 0.02,
            nullEnergyFraction: 0.09, energyOverNull: -0.07,
            coneExhaustedAt: 300, nullConeExhaustedAt: 290)
        var readout = base
        readout.layers = [weaker, base.layers[0]]
        #expect(readout.strongestLayer?.layer == 29)
    }

    @Test func optionalProvenanceMayBeAbsentOnOlderServers() throws {
        // An older server omits the fit-provenance and run-directory fields. The
        // readout still renders; it just says less about where it came from.
        var object = try JSONSerialization.jsonObject(
            with: Data(Self.captured.utf8)) as! [String: Any]
        for key in ["lensFitPrompts", "lensFitCorpus", "runDirectory",
                    "supportIdentityHash", "nullSeed", "revision"] {
            object.removeValue(forKey: key)
        }
        let data = try JSONSerialization.data(withJSONObject: object)
        let readout = try JSONDecoder().decode(JLensSupportReadout.self, from: data)
        #expect(readout.lensFitPrompts == nil)
        #expect(readout.runDirectory == nil)
        #expect(readout.layers.count == 1)
    }

    @Test func coneExhaustionDecodesWhenPresent() throws {
        let json = Self.captured.replacingOccurrences(
            of: "\"coneExhaustedAt\": null", with: "\"coneExhaustedAt\": 327")
        let readout = try JSONDecoder().decode(JLensSupportReadout.self,
                                               from: Data(json.utf8))
        #expect(readout.layers[0].coneExhaustedAt == 327)
    }
}
