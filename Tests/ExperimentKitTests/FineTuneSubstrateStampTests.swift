import Foundation
import Testing

@testable import ExperimentKit

/// Pure-CPU tests for the adapter substrate/format stamp — the pinned
/// cross-engine contract on adapter sidecars. This engine writes
/// `"substrate": "swift-mlx"` + `"adapterFormat": "mlx-lora"`; the Python
/// server writes "python-hf-transformers" + "hf-peft-lora". Application
/// paths refuse EXPLICITLY foreign stamps; unstamped legacy sidecars keep
/// today's behavior.
struct FineTuneSubstrateStampTests {

    private func makeArtifact(
        substrate: String? = nil, adapterFormat: String? = nil
    ) -> FineTuneArtifact {
        FineTuneArtifact(
            name: "judicial-lora",
            baseModelID: "Qwen/Qwen3-4B-MLX-4bit",
            adapterDirectory: "runs/fine-tunes/x/adapter",
            substrate: substrate,
            adapterFormat: adapterFormat)
    }

    // MARK: Codable round-trip (the exact sidecar keys)

    @Test func stampedSidecarRoundTripsWithTheExactKeys() throws {
        let artifact = makeArtifact(
            substrate: AdapterSubstrateGate.localSubstrate,
            adapterFormat: AdapterSubstrateGate.localAdapterFormat)

        let data = try JSONEncoder().encode(artifact)
        let object = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        // EXACT keys/values of the pinned contract — the server twin reads
        // and writes the same fields.
        #expect(object["substrate"] as? String == "swift-mlx")
        #expect(object["adapterFormat"] as? String == "mlx-lora")

        let decoded = try JSONDecoder().decode(FineTuneArtifact.self, from: data)
        #expect(decoded == artifact)
        #expect(decoded.substrate == "swift-mlx")
        #expect(decoded.adapterFormat == "mlx-lora")
    }

    @Test func serverStampedSidecarRoundTrips() throws {
        let artifact = makeArtifact(
            substrate: "python-hf-transformers", adapterFormat: "hf-peft-lora")
        let decoded = try JSONDecoder().decode(
            FineTuneArtifact.self, from: JSONEncoder().encode(artifact))
        #expect(decoded.substrate == "python-hf-transformers")
        #expect(decoded.adapterFormat == "hf-peft-lora")
    }

    @Test func legacySidecarWithoutStampsDecodesAsUnstamped() throws {
        // Pre-stamp sidecars must keep decoding (and behave as today).
        let json = """
            {"schemaVersion": 1, "name": "old", "baseModelID": "Qwen/Qwen3-4B-MLX-4bit",
             "adapterDirectory": "runs/fine-tunes/old/adapter", "fineTuneType": "lora",
             "rank": 8, "scale": 10, "adaptedLayers": 16, "batchSize": 4,
             "iterations": 1000, "learningRate": 0.00001,
             "createdAt": "2026-01-01T00:00:00Z", "notes": ""}
            """
        let decoded = try JSONDecoder().decode(
            FineTuneArtifact.self, from: Data(json.utf8))
        #expect(decoded.substrate == nil)
        #expect(decoded.adapterFormat == nil)
        #expect(
            !AdapterSubstrateGate.isExplicitlyForeign(
                substrate: decoded.substrate, adapterFormat: decoded.adapterFormat))
    }

    // MARK: Refusal at application

    @Test func explicitlyForeignAdapterIsRefusedWithTheContractMessage() {
        let message = AdapterSubstrateGate.refusalMessage(
            name: "peft-lora",
            substrate: "python-hf-transformers",
            adapterFormat: "hf-peft-lora")
        #expect(
            message
                == "adapter 'peft-lora' was trained as 'hf-peft-lora' on "
                + "'python-hf-transformers'; this engine loads mlx-lora adapters "
                + "— retrain on this substrate")
    }

    @Test func halfStampedForeignRecordsStillRefuse() {
        // Either field alone naming another engine refuses — the missing
        // half must not launder a foreign adapter.
        #expect(
            AdapterSubstrateGate.refusalMessage(
                name: "a", substrate: "python-hf-transformers", adapterFormat: nil)
                != nil)
        #expect(
            AdapterSubstrateGate.refusalMessage(
                name: "a", substrate: nil, adapterFormat: "hf-peft-lora") != nil)
    }

    @Test func localAndUnstampedAdaptersLoad() {
        #expect(
            AdapterSubstrateGate.refusalMessage(
                name: "mine",
                substrate: AdapterSubstrateGate.localSubstrate,
                adapterFormat: AdapterSubstrateGate.localAdapterFormat) == nil)
        // Absent stamps behave as today: loadable.
        #expect(
            AdapterSubstrateGate.refusalMessage(
                name: "legacy", substrate: nil, adapterFormat: nil) == nil)
    }

    // MARK: Picker filtering

    @Test func pickerFilterExcludesForeignKeepsLocalAndUnstamped() {
        let records = [
            makeArtifact(substrate: "swift-mlx", adapterFormat: "mlx-lora"),
            makeArtifact(substrate: nil, adapterFormat: nil),  // legacy stays
            makeArtifact(
                substrate: "python-hf-transformers", adapterFormat: "hf-peft-lora"),
        ]
        let offered = records.filter {
            !AdapterSubstrateGate.isExplicitlyForeign(
                substrate: $0.substrate, adapterFormat: $0.adapterFormat)
        }
        #expect(offered.count == 2)
        #expect(offered.allSatisfy { $0.substrate != "python-hf-transformers" })
    }

    @Test func localConstantsReuseThePinnedContract() {
        // "swift-mlx" must be the SAME constant vector sidecars use — a
        // drifted copy would stamp adapters with a string nothing filters on.
        #expect(AdapterSubstrateGate.localSubstrate == "swift-mlx")
        #expect(AdapterSubstrateGate.localAdapterFormat == "mlx-lora")
    }
}
