import Foundation
import Testing

@testable import ExperimentKit

/// Open-issues #11 — the Mac must be able to READ a server-minted adapter
/// agent.
///
/// The live failure (2026-08-16): `steerlab-cli experiment verify` threw
/// `DecodingError.keyNotFound: 'name'` at
/// `variantConditions[0].artifact.adapters[0]` on every study whose adapter
/// agent came from the server/driver path, while `steerlab-server experiment
/// verify` read the same manifest clean. Not a corrupt artifact and not the
/// config backfill: the server carries adapter entries VERBATIM
/// (`ModelVariant.adapters` is a list of opaque dicts through
/// `from_dict`/`to_dict`), so the shape a driver posts to
/// `POST /api/model-variant/save` — `{adapterDirectory, adapterHash,
/// configHash}` — is the shape on disk. Every server-side reader tolerates a
/// missing name (`adapter.get('name') or adapter.get('adapterDirectory')`);
/// only Swift's decoder demanded the key, and demanded it hard enough to
/// refuse the whole manifest.
///
/// The fixture is PRODUCED BY THE PYTHON ENGINE
/// (`scripts/regenerate-cross-engine-fixtures.py` →
/// `Tests/Fixtures/cross-engine/server-minted-adapter-agent.json`) for the
/// same reason as the other cross-engine fixtures: a Swift test that
/// hand-writes what it believes the server emits pins the Swift author's
/// belief, and that belief drifting out of date is exactly this bug.
struct ServerMintedAdapterDecodeTests {

    private static func fixture() throws -> [String: Any] {
        let url = CodeResources.compiledCheckoutPath.appending(
            components: "Tests", "Fixtures", "cross-engine",
            "server-minted-adapter-agent.json")
        let object = try JSONSerialization.jsonObject(
            with: try Data(contentsOf: url)) as? [String: Any]
        return try #require(object)
    }

    private static func data(for value: Any) throws -> Data {
        try JSONSerialization.data(
            withJSONObject: value, options: [.sortedKeys])
    }

    @Test("a server-minted adapter agent decodes, name derived from the directory")
    func serverMintedAdapterAgentDecodes() throws {
        let artifact = try #require(try Self.fixture()["artifact"] as? [String: Any])
        // The precondition this test exists for: the server really does write
        // an adapter entry with no `name`. If a future server change starts
        // emitting one, this assertion fails and the fixture diff says so —
        // rather than the test quietly passing for the wrong reason.
        let adapters = try #require(artifact["adapters"] as? [[String: Any]])
        #expect(adapters.count == 1)
        #expect(adapters[0]["name"] == nil)
        #expect(adapters[0]["artifactPath"] == nil)

        let decoded = try JSONDecoder().decode(
            ModelVariantArtifact.self, from: try Self.data(for: artifact))
        let adapter = try #require(decoded.adapters.first)
        #expect(adapter.adapterDirectory == "adapters/sympathy-lora")
        // Derived from the directory's basename — the same label the server's
        // own `/api/adapters` listing mints for an unnamed adapter.
        #expect(adapter.name == "sympathy-lora")
        // An absent `artifactPath` falls back to the directory rather than
        // becoming an empty string that resolves to the workspace root.
        #expect(adapter.artifactPath == "adapters/sympathy-lora")
        #expect(adapter.adapterHash == String(repeating: "d", count: 64))
        #expect(adapter.configHash == String(repeating: "e", count: 64))
    }

    @Test("the whole variantConditions envelope decodes — the path that failed")
    func variantConditionEnvelopeDecodes() throws {
        let conditions = try #require(
            try Self.fixture()["variantConditions"] as? [[String: Any]])
        // A REAL manifest, round-tripped through its own encoder, with the
        // server's condition list dropped in — so this test exercises the
        // adapter decode, not a hand-built manifest's missing keys.
        let empty = ExperimentManifest(
            name: "adapter-study",
            description: "an adapter arm minted on the server",
            modelID: "google/gemma-3-27b-it")
        var manifest = try #require(
            try JSONSerialization.jsonObject(
                with: try JSONEncoder().encode(empty)) as? [String: Any])
        manifest["variantConditions"] = conditions
        // This is the exact decode that threw keyNotFound 'name' at
        // `variantConditions[0].artifact.adapters[0]`.
        let decoded = try JSONDecoder().decode(
            ExperimentManifest.self, from: try Self.data(for: manifest))
        let condition = try #require(decoded.variantConditions.first)
        let adapter = try #require(condition.artifact.adapters.first)
        #expect(adapter.name == "sympathy-lora")
        #expect(condition.artifactHash.count == 64)
    }

    @Test("a declared adapter name still wins over the derived one")
    func declaredNameIsUnchanged() throws {
        let json = """
            {"name": "trained-on-sympathy",
             "artifactPath": "adapters/x/adapter_model.safetensors",
             "adapterDirectory": "adapters/x"}
            """
        let adapter = try JSONDecoder().decode(
            ModelVariantArtifact.AdapterRef.self, from: Data(json.utf8))
        #expect(adapter.name == "trained-on-sympathy")
        #expect(adapter.artifactPath == "adapters/x/adapter_model.safetensors")
    }

    @Test("the derived name never renders as an empty label")
    func derivedNameFallsBackToALabel() {
        typealias Ref = ModelVariantArtifact.AdapterRef
        #expect(Ref.derivedName(adapterDirectory: "runs/x/lora-gemma")
            == "lora-gemma")
        // A trailing slash is a path-writing habit, not a different adapter.
        #expect(Ref.derivedName(adapterDirectory: "runs/x/lora-gemma/")
            == "lora-gemma")
        #expect(Ref.derivedName(adapterDirectory: "") == "adapter")
        #expect(Ref.derivedName(adapterDirectory: "/") == "adapter")
    }
}
