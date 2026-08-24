import CryptoKit
import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// Phase-0 of the portability program: the cross-engine contracts a future
/// cross-platform Python client will depend on, pinned as goldens — Swift half.
///
/// The client the later phases build has to do three things the two engines
/// already do to each other: **author a workspace this engine reads**,
/// **submit a hash-pinned run bundle either engine produced**, and **import an
/// evidence bundle with verification**. Nothing here changes behaviour; it
/// records what today's behaviour IS, so a later phase that breaks one of
/// these seams fails a test instead of failing in a workspace.
///
/// The idioms are the suite's existing ones, not new ones:
///
/// - **Producer-generated fixtures.** Bytes under `Tests/Fixtures/cross-engine/`
///   come from the engine that really produces them
///   (`scripts/regenerate-cross-engine-fixtures.py`, and now
///   `Server/tests/test_portability_contracts.py`), so a consumer's test
///   cannot pin its author's *belief* about the other engine — the failure
///   mode `CrossEngineLifecycleTests` was written to end.
/// - **Write-if-missing**, for the one fixture this engine produces
///   (`swift-authored-manifest.json`, consumed by the server suite): the
///   structural assertions always run on the fresh document, and only the byte
///   comparison waits for the file to be committed
///   (`ExperimentCLIEnvelopeTests.check`). To regenerate, delete the file and
///   re-run this suite.
/// - **Twin literals** (`CLIEnvelopeParityTests`): a constant belonging to the
///   other engine is written out here with a naming cross-reference, so
///   neither engine can quietly follow the other.
///
/// Server twin: `Server/tests/test_portability_contracts.py`.
/// Inventory: `docs/PORTABILITY-CONTRACTS.md`.
///
/// Serialized, and holding `ExperimentRootOverrideLock`: `rootOverride` is a
/// process-global seam shared with every other lifecycle suite.
@Suite(.serialized) struct PortabilityContractTests {

    // MARK: - Harness

    static var fixtureDirectory: URL {
        CodeResources.compiledCheckoutPath.appending(
            components: "Tests", "Fixtures", "cross-engine")
    }

    /// Volatile freeze stamps: keys describing WHEN and BY WHAT a manifest was
    /// stamped, never what the study measures. This engine's own list lives in
    /// `ExperimentStore.volatileFreezeKeys` (private); the server spells the
    /// same set inline in `Manifest.content_hash` and
    /// `experiment_store._write_freeze_canonical`. Server twin test:
    /// `test_portability_contracts.py::test_volatile_freeze_stamps_are_outside_the_content_hash`.
    static let volatileFreezeKeys = [
        "appVersion", "createdAt", "forcedGatesSkipped", "freezeForced",
        "freezeHash", "frozenAt", "frozenBy", "gitCommit", "status",
    ]

    /// The `steerlab-bundle.json` header keys for a run bundle. Copied from
    /// `bundles.package_experiment` / `bundles.BUNDLE_SCHEMA`
    /// (`Server/steerlab_server/experiment/bundles.py`) — and identical to the
    /// literal this engine's `RunBundlePackager.packageExperiment` writes,
    /// which is the whole point: either engine's bundle is submittable to
    /// either engine. Server twin test:
    /// `test_a_run_bundle_metadata_fixture_is_current`.
    static let runBundleHeaderKeys = [
        "createdAt", "entries", "experiment", "experimentContentHash", "kind",
        "rootRelative", "schemaVersion", "validationScopeHash",
        "verificationViolations",
    ]

    /// Copied from the server's `BundleEntry.to_dict`.
    static let bundleEntryKeys = ["bytes", "path", "sha256"]

    private func loadFixture(_ name: String) throws -> [String: Any] {
        let url = Self.fixtureDirectory.appending(component: name)
        let data = try Data(contentsOf: url)
        return try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any],
            "\(name) is not a JSON object")
    }

    /// The experiment-root seam ONLY, so `attachConcept` still reads the
    /// checkout's committed `prompts/concepts/` — the same harness
    /// `ExperimentCLIEnvelopeTests.withTempRoot` uses for its attach/verify
    /// goldens.
    private func withTempRoot<T>(_ body: (URL) throws -> T) rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "portability-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        ExperimentStore.rootOverride = temp
        defer {
            ExperimentStore.rootOverride = nil
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: temp)
        }
        return try body(temp)
    }

    /// BOTH process-global root seams, as `RunBundleClosureTests` sets them:
    /// the packer resolves pins (and writes its staging tree) against
    /// `VectorCatalog.projectRoot`/`WorkspaceRoot`, the manifest store against
    /// `ExperimentStore.rootOverride`. Without the workspace seam the packer
    /// would stage inside the CHECKOUT.
    private func withTempWorkspace<T>(_ body: (URL) throws -> T) rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "portability-ws-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        ExperimentStore.rootOverride = temp
        WorkspaceRoot.programmaticOverride = temp
        defer {
            ExperimentStore.rootOverride = nil
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: temp)
        }
        return try body(temp)
    }

    @discardableResult
    private func write(_ root: URL, _ relative: String, _ text: String) throws -> String {
        let url = root.appending(path: relative)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let data = Data(text.utf8)
        try data.write(to: url)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - 1. Manifest interop, server → this engine

    /// CONTRACT: manifest-interop — a manifest the PYTHON engine authored, in
    /// the create → attach → pin state a client leaves a workspace in, decodes
    /// here with every declared field intact.
    ///
    /// The bytes are the server's own `experiment_store` output, so this pins
    /// the server's behaviour rather than a Swift author's model of it.
    @Test func aPythonAuthoredManifestReadsWithEveryFieldIntact() throws {
        let fixture = try loadFixture("manifest-interop.json")
        let draft = try #require(fixture["draft"] as? [String: Any])
        let data = try JSONSerialization.data(withJSONObject: draft)
        let manifest = try JSONDecoder().decode(ExperimentManifest.self, from: data)

        // Identity.
        #expect(manifest.name == draft["name"] as? String)
        #expect(manifest.modelID == draft["modelID"] as? String)
        #expect(manifest.modelRevision == draft["modelRevision"] as? String)
        #expect(manifest.status == .draft)
        #expect(
            manifest.experimentDescription
                == draft["experimentDescription"] as? String)
        #expect(manifest.taskDescription == draft["taskDescription"] as? String)

        // The pin surface — the fields a run is reproducible from.
        #expect(manifest.taskPromptsFile == draft["taskPromptsFile"] as? String)
        #expect(manifest.taskPromptsHash == draft["taskPromptsHash"] as? String)
        #expect(manifest.judgeRubricFile == draft["judgeRubricFile"] as? String)
        #expect(manifest.judgeRubricHash == draft["judgeRubricHash"] as? String)
        #expect(
            manifest.capabilityBatteryFile
                == draft["capabilityBatteryFile"] as? String)
        #expect(
            manifest.capabilityBatteryHash
                == draft["capabilityBatteryHash"] as? String)
        #expect(manifest.neutralCorpusHash == draft["neutralCorpusHash"] as? String)

        // Arms.
        let declaredConcepts = try #require(draft["concepts"] as? [[String: Any]])
        #expect(
            manifest.concepts.map(\.name)
                == declaredConcepts.compactMap { $0["name"] as? String })
        #expect(
            manifest.concepts.map(\.stimulusSetHash)
                == declaredConcepts.compactMap { $0["stimulusSetHash"] as? String })
        #expect(
            manifest.concepts.first?.stimulusSetHash
                == fixture["stimulusSetHash"] as? String)
        let declaredConditions = try #require(draft["conditions"] as? [[String: Any]])
        #expect(
            manifest.conditions.map(\.name)
                == declaredConditions.compactMap { $0["name"] as? String })

        // Sampling policy: a reader that gets these wrong runs a different
        // study while reporting the same manifest.
        #expect(manifest.seeds.map(Int.init) == draft["seeds"] as? [Int])
        #expect(manifest.temperature == draft["temperature"] as? Double)
        #expect(manifest.maxTokens == draft["maxTokens"] as? Int)
        #expect(manifest.samplesPerItem == draft["samplesPerItem"] as? Int)
        #expect(manifest.seedPolicy == draft["seedPolicy"] as? String)
        #expect(manifest.promptMode?.rawValue == draft["promptMode"] as? String)
        #expect(manifest.systemPrompt == draft["systemPrompt"] as? String)
    }

    /// CONTRACT: manifest-interop-lossless — decoding a server-authored
    /// manifest and re-encoding it LOSES NOTHING.
    ///
    /// `Codable` silently discards keys the model does not declare, so a
    /// server key this engine has not learned about would vanish the first
    /// time the Mac saved the study — a silent unpinning, not an error. The
    /// pinned literal below is the set of dropped keys, which must stay empty.
    @Test func aPythonAuthoredManifestSurvivesTheSwiftModelWithoutLosingKeys() throws {
        let fixture = try loadFixture("manifest-interop.json")
        let draft = try #require(fixture["draft"] as? [String: Any])
        let data = try JSONSerialization.data(withJSONObject: draft)
        let manifest = try JSONDecoder().decode(ExperimentManifest.self, from: data)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let reencoded = try encoder.encode(manifest)
        let object = try #require(
            try JSONSerialization.jsonObject(with: reencoded) as? [String: Any])

        let dropped = Set(draft.keys).subtracting(object.keys).sorted()
        #expect(
            dropped == [],
            "this engine drops server manifest key(s) \(dropped): a study saved on the Mac would silently lose them")

        // Byte stability: the same model encodes to the same bytes every
        // time, which is what makes any hash over it comparable at all.
        #expect(try encoder.encode(manifest) == reencoded)
        let firstHash = ExperimentStore.canonicalManifestBodyHash(reencoded)
        #expect(firstHash != nil)
        #expect(ExperimentStore.canonicalManifestBodyHash(reencoded) == firstHash)
    }

    /// CONTRACT: condition-globals-are-required — a condition authored WITHOUT
    /// `bandWidth`/`alphaInNormUnits` does not decode here at all.
    ///
    /// Found by this phase. The server never writes such a condition — its
    /// `experiment_store._condition_entry` stamps both keys with defaults — so
    /// today's traffic is safe. But nothing pinned that, and a cross-platform
    /// client that builds a condition dictionary by hand (the obvious thing to
    /// do: `{"name": …, "slots": […]}`) authors a manifest this engine refuses
    /// to open, with a `keyNotFound` deep inside `conditions[0]` rather than a
    /// typed refusal naming the problem. That is the exact failure shape of
    /// open-issues #11 (the adapter entry with no `name`), one level over.
    ///
    /// Pinned as it IS, deliberately: making the decoder tolerant would be a
    /// behaviour change, and Phase 0 changes nothing. Recorded as a Phase-1
    /// gap in docs/PORTABILITY-CONTRACTS.md.
    @Test func aConditionWithoutItsGlobalsIsUnreadableHere() throws {
        let fixture = try loadFixture("manifest-interop.json")
        var draft = try #require(fixture["draft"] as? [String: Any])
        let conditions = try #require(draft["conditions"] as? [[String: Any]])
        // The server's own conditions carry both globals…
        for condition in conditions {
            #expect(condition["bandWidth"] != nil)
            #expect(condition["alphaInNormUnits"] != nil)
        }
        // …and stripping them makes the manifest unreadable here.
        draft["conditions"] = conditions.map { condition -> [String: Any] in
            var stripped = condition
            stripped.removeValue(forKey: "bandWidth")
            stripped.removeValue(forKey: "alphaInNormUnits")
            return stripped
        }
        let data = try JSONSerialization.data(withJSONObject: draft)
        #expect(throws: DecodingError.self) {
            _ = try JSONDecoder().decode(ExperimentManifest.self, from: data)
        }
    }

    /// GAP (found by this phase, recorded not repaired): declaring a condition
    /// without naming `alphaInNormUnits` yields TRUE here and FALSE on the
    /// server.
    ///
    /// `Condition.init` defaults it to true;
    /// `experiment_store._condition_entry` defaults it to False. Neither
    /// engine reads the other's default — both always WRITE the key, so a
    /// condition that has crossed the wire is unambiguous, which is why
    /// nothing is broken today. But the same client call produces a different
    /// study depending on which engine served it, and α units are not a
    /// cosmetic setting.
    ///
    /// Pinned as it IS, on both sides, over the two committed fixtures. Server
    /// twin: `test_the_condition_alpha_unit_default_diverges_from_swift`.
    @Test func theConditionAlphaUnitDefaultDivergesFromTheServer() throws {
        let local = ExperimentManifest.Condition(name: "arm", slots: [])
        #expect(
            local.alphaInNormUnits,
            "this engine's condition default changed — the divergence with the server's `False` may be closed; re-check docs/PORTABILITY-CONTRACTS.md G6")
        #expect(local.bandWidth == 1)   // this one DOES agree

        // Visible in the committed bytes: the same two conditions, declared
        // the same way through each engine's own store API.
        let python = try loadFixture("manifest-interop.json")
        for condition in try #require(
            (python["draft"] as? [String: Any])?["conditions"] as? [[String: Any]])
        {
            #expect(condition["alphaInNormUnits"] as? Bool == false)
        }
        let swift = try loadFixture("swift-authored-manifest.json")
        for condition in try #require(
            (swift["manifest"] as? [String: Any])?["conditions"] as? [[String: Any]])
        {
            #expect(condition["alphaInNormUnits"] as? Bool == true)
        }
    }

    /// CONTRACT: manifest-canonicalization — the identity of a study is what
    /// it MEASURES, so every key describing when/by-what it was stamped sits
    /// outside the hash on BOTH engines.
    ///
    /// The hash VALUES are per-engine by design (`Manifest.content_hash` says
    /// so; `CrossEngineLifecycleTests` documents why comparing them across
    /// substrates is a check that cannot pass). What must agree is the RULE.
    /// Server twin:
    /// `test_volatile_freeze_stamps_are_outside_the_content_hash`.
    @Test func volatileFreezeStampsAreOutsideTheContentHash() throws {
        let fixture = try loadFixture("manifest-interop.json")
        // The server publishes the list it applies; this engine's own list is
        // private, so the agreement is pinned through behaviour below.
        #expect(
            (fixture["volatileFreezeKeys"] as? [String])?.sorted()
                == Self.volatileFreezeKeys)

        let draft = try #require(fixture["draft"] as? [String: Any])
        let manifest = try JSONDecoder().decode(
            ExperimentManifest.self,
            from: try JSONSerialization.data(withJSONObject: draft))
        let baseline = ExperimentStore.manifestHash(manifest)

        // Every volatile stamp: the hash holds.
        var stamped = manifest
        stamped.createdAt = "2099-01-01T00:00:00Z"
        stamped.status = .frozen
        stamped.frozenAt = "2099-01-01T00:00:00Z"
        stamped.freezeHash = String(repeating: "a", count: 64)
        stamped.frozenBy = "server"
        stamped.gitCommit = String(repeating: "b", count: 40)
        stamped.appVersion = "a different build"
        #expect(
            ExperimentStore.manifestHash(stamped) == baseline,
            "a freeze stamp moved the content hash: the two engines then disagree about whether a study changed")

        // Measured surface: the hash moves.
        for mutate in [
            { (m: inout ExperimentManifest) in m.modelID = "org/other" },
            { (m: inout ExperimentManifest) in m.temperature = 0.1 },
            { (m: inout ExperimentManifest) in m.seeds = [9] },
            { (m: inout ExperimentManifest) in
                m.taskPromptsHash = String(repeating: "0", count: 64) },
        ] {
            var mutated = manifest
            mutate(&mutated)
            #expect(ExperimentStore.manifestHash(mutated) != baseline)
        }
    }

    /// Keys the Swift encoder ALWAYS writes and the server OMITS when they
    /// hold their (identical) default. Both engines default
    /// `multiAgentIncludeBaseline` to true and `recordTokenIDs` to false, so
    /// no study is described differently — but the server-freeze check
    /// compares parsed DOCUMENTS, and a key present on one side only is a
    /// difference to a key-set comparison. Pinned so the day either engine
    /// changes, this is revisited on purpose.
    static let keysSwiftAlwaysWrites = ["multiAgentIncludeBaseline", "recordTokenIDs"]

    /// Plants a server-frozen study (manifest + the server's exact
    /// `freeze-canonical.json` bytes) and returns the loaded manifest.
    private func plantServerFrozen(
        _ root: URL, document: [String: Any], canonical: String
    ) throws -> ExperimentManifest {
        let name = try #require(document["name"] as? String)
        let directory = root.appending(components: "experiments", name)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try JSONSerialization
            .data(withJSONObject: document, options: [.sortedKeys])
            .write(to: directory.appending(component: "experiment.json"))
        // The server's exact bytes, written verbatim — re-serializing them
        // would defeat the point of a byte pin.
        try Data(canonical.utf8)
            .write(to: directory.appending(component: "freeze-canonical.json"))
        return try ExperimentStore.load(name: name)
    }

    private func freezeViolations(_ manifest: ExperimentManifest) -> [String] {
        ExperimentStore.verify(manifest).filter {
            $0.contains("changed after freeze")
                || $0.contains("freeze-canonical.json")
        }
    }

    /// CONTRACT: server-freeze-canonical — a manifest the SERVER froze
    /// verifies on this engine against the exact canonical bytes the server
    /// hashed, PROVIDED the server's document carries the two boolean keys
    /// this engine's encoder always writes.
    ///
    /// This is the one place the two canonicalizations meet for real:
    /// `sha256(freeze-canonical.json)` must equal `freezeHash`, and the file's
    /// parsed content must equal this engine's re-encoding of the manifest
    /// minus the volatile stamps — a DEEP equality, so a single field this
    /// engine drops, renames, or re-types fails it.
    ///
    /// The existing coverage on this path (`FreezeFirewallClosureTests`,
    /// `ExperimentStoreTests`) FABRICATES the server's bytes in Swift, which
    /// pins the Swift author's belief about the server. These bytes came out
    /// of `experiment_store.freeze`, and the difference is not academic — see
    /// `aServerAuthoredFrozenManifestReadsAsDriftedHere`, which is what the
    /// fabricated bytes were hiding.
    @Test func aServerFrozenManifestVerifiesAgainstItsOwnCanonicalBytes() throws {
        let fixture = try loadFixture("manifest-interop.json")
        let frozen = try #require(fixture["frozenWithSwiftDefaults"] as? [String: Any])
        let canonical = try #require(
            fixture["freezeCanonicalWithSwiftDefaults"] as? String)
        let freezeHash = try #require(
            fixture["freezeHashWithSwiftDefaults"] as? String)
        #expect(frozen["frozenBy"] as? String == "server")
        #expect(frozen["freezeHash"] as? String == freezeHash)
        for key in Self.keysSwiftAlwaysWrites {
            #expect(frozen[key] != nil, "fixture gap: this half must carry \(key)")
        }

        try withTempRoot { root in
            let manifest = try plantServerFrozen(
                root, document: frozen, canonical: canonical)
            #expect(manifest.status == .frozen)

            // Step 1, stated separately so a failure says WHICH half broke:
            // the bytes are the hash.
            let canonicalData = Data(canonical.utf8)
            #expect(
                ExperimentStore.sha256Hex(canonicalData) == freezeHash,
                "sha256(freeze-canonical.json) is not the freezeHash")

            // Step 2: the same document seen through this engine's model.
            // Reimplemented here only to NAME any difference — the contract
            // itself is the production `verify` call below.
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            let mine = try #require(
                try JSONSerialization.jsonObject(with: try encoder.encode(manifest))
                    as? [String: Any])
            let theirs = try #require(
                try JSONSerialization.jsonObject(with: canonicalData)
                    as? [String: Any])
            let volatile = Set(Self.volatileFreezeKeys)
            let mineKeys = Set(mine.keys).subtracting(volatile)
            let theirsKeys = Set(theirs.keys).subtracting(volatile)
            #expect(
                mineKeys.subtracting(theirsKeys).sorted() == [],
                "this engine writes key(s) the server's canonical bytes lack: \(mineKeys.subtracting(theirsKeys).sorted())")
            #expect(
                theirsKeys.subtracting(mineKeys).sorted() == [],
                "the server's canonical bytes carry key(s) this engine drops: \(theirsKeys.subtracting(mineKeys).sorted())")
            for key in mineKeys.intersection(theirsKeys).sorted() {
                #expect(
                    (mine[key] as? NSObject)?.isEqual(theirs[key] as? NSObject) ?? false,
                    "'\(key)' differs between the engines: \(mine[key] ?? "nil") vs \(theirs[key] ?? "nil")")
            }

            let violations = freezeViolations(manifest)
            #expect(
                violations == [],
                "a server-frozen manifest failed this engine's canonical verification: \(violations)")
        }
    }

    /// GAP (found by this phase, recorded not repaired): a study authored
    /// ENTIRELY on the server — which is exactly what a cross-platform client
    /// would do — reads as "changed after freeze" here, though nothing
    /// changed.
    ///
    /// The cause is presence, not meaning. The server omits
    /// `multiAgentIncludeBaseline` and `recordTokenIDs` when they hold their
    /// defaults; this engine's encoder always writes them; and
    /// `serverFreezeCanonicalViolations` compares the two parsed documents,
    /// where a key on one side only is a difference. The default VALUES agree
    /// on both engines, so the two documents describe the same study.
    ///
    /// It has stayed invisible because today's server-frozen studies were
    /// authored on the Mac first, so their documents already carry both keys
    /// (`aServerFrozenManifestVerifiesAgainstItsOwnCanonicalBytes`). Phase 0
    /// changes nothing, so this pins the behaviour AS IT IS — including that
    /// the refusal is the unhelpful generic one, with no named field. Closing
    /// it is Phase-1 work; see docs/PORTABILITY-CONTRACTS.md.
    @Test func aServerAuthoredFrozenManifestReadsAsDriftedHere() throws {
        let fixture = try loadFixture("manifest-interop.json")
        let frozen = try #require(fixture["frozen"] as? [String: Any])
        let canonical = try #require(fixture["freezeCanonical"] as? String)
        #expect(
            (fixture["keysSwiftAlwaysWritesAndThisEngineOmitsAtDefault"] as? [String])
                == Self.keysSwiftAlwaysWrites)
        // The server's own document really does omit them.
        for key in Self.keysSwiftAlwaysWrites {
            #expect(
                frozen[key] == nil,
                "the server now writes \(key) — this gap may be closed; re-check")
        }

        try withTempRoot { root in
            let manifest = try plantServerFrozen(
                root, document: frozen, canonical: canonical)

            // The bytes are intact: this is NOT tampering being caught.
            #expect(
                ExperimentStore.sha256Hex(Data(canonical.utf8))
                    == fixture["freezeHash"] as? String)

            // The difference is exactly the two keys, and nothing else.
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            let mine = try #require(
                try JSONSerialization.jsonObject(with: try encoder.encode(manifest))
                    as? [String: Any])
            let theirs = try #require(
                try JSONSerialization.jsonObject(with: Data(canonical.utf8))
                    as? [String: Any])
            let volatile = Set(Self.volatileFreezeKeys)
            let mineKeys = Set(mine.keys).subtracting(volatile)
            let theirsKeys = Set(theirs.keys).subtracting(volatile)
            #expect(
                mineKeys.subtracting(theirsKeys).sorted() == Self.keysSwiftAlwaysWrites)
            #expect(theirsKeys.subtracting(mineKeys).sorted() == [])
            for key in mineKeys.intersection(theirsKeys).sorted() {
                #expect(
                    (mine[key] as? NSObject)?.isEqual(theirs[key] as? NSObject) ?? false,
                    "'\(key)' differs beyond the known gap")
            }
            // The two defaults this engine supplies are the server's defaults
            // too, so the documents describe the SAME study.
            #expect(manifest.multiAgentIncludeBaseline == true)
            #expect(manifest.recordTokenIDs == false)

            // And the current, unhelpful consequence.
            #expect(
                freezeViolations(manifest)
                    == ["manifest content changed after freeze (hash mismatch)"],
                "the server-authored freeze gap changed shape — re-read docs/PORTABILITY-CONTRACTS.md before adjusting this")
        }
    }

    // MARK: - 2. Manifest interop, this engine → server

    /// CONTRACT: manifest-interop-reverse (producer half) — publishes a
    /// manifest THIS engine authored for the server suite to read.
    ///
    /// Write-if-missing: the structural assertions below always run; only the
    /// byte comparison waits for the file to be committed. Consumer:
    /// `test_portability_contracts.py::test_a_swift_authored_manifest_loads_with_every_field_intact`.
    @Test func theSwiftAuthoredManifestFixtureIsCurrent() throws {
        try withTempRoot { root in
            var manifest = try ExperimentStore.create(
                name: "portability-interop-swift", description: "Phase-0 interop fixture",
                modelID: "org/m",
                modelRevision: "0123456789abcdef0123456789abcdef01234567")
            manifest = try ExperimentStore.attachConcept(
                "french", method: .meanDifference,
                experimentName: manifest.name)

            let promptsHash = try write(
                root, "prompts/tasks/cases.jsonl",
                "{\"id\":\"c1\",\"text\":\"decide\"}\n")
            let batteryHash = try write(
                root, "prompts/batteries/b1.jsonl",
                "{\"prompt\":\"2+2\",\"answer\":\"4\"}\n")
            manifest.taskDescription = "answer the item"
            manifest.taskPromptsFile = "prompts/tasks/cases.jsonl"
            manifest.taskPromptsHash = promptsHash
            manifest.capabilityBatteryFile = "prompts/batteries/b1.jsonl"
            manifest.capabilityBatteryHash = batteryHash
            manifest.conditions = [
                .init(name: "baseline", slots: []),
                .init(
                    name: "arm-a",
                    slots: [.init(concept: "french", layer: 1, alpha: 0.1)]),
            ]
            manifest.seeds = [0, 1, 2]
            manifest.temperature = 0.7
            manifest.maxTokens = 512
            manifest.samplesPerItem = 3
            manifest.seedPolicy = "derivedSHA256"
            manifest.systemPrompt = "Respond in JSON."
            try ExperimentStore.save(manifest)

            let stored = try #require(ExperimentStore.manifestData(name: manifest.name))
            var document = try #require(
                try JSONSerialization.jsonObject(with: stored) as? [String: Any])

            // Structural assertions on the FRESH document, so a regression
            // fails even before any byte comparison.
            #expect(document["name"] as? String == manifest.name)
            #expect(document["status"] as? String == "draft")
            #expect((document["concepts"] as? [[String: Any]])?.count == 1)
            #expect((document["conditions"] as? [[String: Any]])?.count == 2)
            #expect(document["taskPromptsHash"] as? String == manifest.taskPromptsHash)
            #expect(ExperimentStore.verify(manifest) == [])

            // `createdAt` is a volatile stamp on both engines (it is outside
            // every canonicalization here), so normalizing it is safe rather
            // than a fudge — and without it the fixture would churn on every
            // run, which teaches reviewers to ignore its diffs.
            document["createdAt"] = "1970-01-01T00:00:00Z"

            let payload: [String: Any] = [
                "note": "produced by the Swift engine's ExperimentStore "
                    + "(Tests/ExperimentKitTests/PortabilityContractTests.swift) "
                    + "— do not hand-edit; delete this file and re-run that "
                    + "suite to regenerate",
                "manifest": document,
                "manifestHash": ExperimentStore.manifestHash(manifest),
                "stimulusSetHash": manifest.concepts.first?.stimulusSetHash ?? "",
            ]
            let text = String(
                decoding: try JSONSerialization.data(
                    withJSONObject: payload,
                    options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]),
                as: UTF8.self) + "\n"

            let url = Self.fixtureDirectory.appending(
                component: "swift-authored-manifest.json")
            guard let existing = try? String(contentsOf: url, encoding: .utf8) else {
                try FileManager.default.createDirectory(
                    at: Self.fixtureDirectory, withIntermediateDirectories: true)
                try text.write(to: url, atomically: true, encoding: .utf8)
                return
            }
            #expect(
                text == existing,
                "swift-authored-manifest.json drifted: delete it and re-run this suite, then run the server suite")
        }
    }

    // MARK: - 3. Run-bundle interop

    /// CONTRACT: run-bundle-metadata — the `steerlab-bundle.json` header this
    /// engine writes is key-for-key the one the server writes and reads.
    ///
    /// A bundle is the submission unit: the Mac packages, the server executes.
    /// A header key added on one side and not the other is how a bundle
    /// becomes engine-specific without anyone deciding that it should.
    ///
    /// `RunBundleClosureTests.runBundlePacksTheEntirePinSurface` pins WHAT is
    /// packed; this pins the SHAPE of the document describing it.
    @Test func theRunBundleMetadataShapeMatchesTheServerLiteral() throws {
        let fixture = try loadFixture("run-bundle-metadata.json")
        #expect(fixture["kind"] as? String == "runBundle")
        #expect(fixture["schemaVersion"] as? Int == 1)
        #expect(fixture["headerKeys"] as? [String] == Self.runBundleHeaderKeys)
        #expect(fixture["entryKeys"] as? [String] == Self.bundleEntryKeys)

        try withTempWorkspace { root in
            var manifest = try ExperimentStore.create(
                name: "bundle-shape", description: "", modelID: "org/m",
                modelRevision: "0123456789abcdef0123456789abcdef01234567")
            manifest.taskPromptsFile = "prompts/tasks/cases.jsonl"
            manifest.taskPromptsHash = try write(
                root, "prompts/tasks/cases.jsonl",
                "{\"id\":\"c1\",\"text\":\"decide\"}\n")
            try ExperimentStore.save(manifest)

            let bundle = try RunBundlePackager.packageExperiment(manifest)
            defer {
                try? FileManager.default.removeItem(
                    at: bundle.deletingLastPathComponent())
            }
            let extracted = root.appending(component: "extracted")
            try FileManager.default.createDirectory(
                at: extracted, withIntermediateDirectories: true)
            let process = Process()
            process.executableURL = URL(filePath: "/usr/bin/tar")
            process.arguments = ["-xzf", bundle.path, "-C", extracted.path]
            try process.run()
            process.waitUntilExit()
            try #require(process.terminationStatus == 0)

            let meta = try #require(
                try JSONSerialization.jsonObject(
                    with: try Data(
                        contentsOf: extracted.appending(
                            component: "steerlab-bundle.json")))
                    as? [String: Any])
            #expect(meta.keys.sorted() == Self.runBundleHeaderKeys)
            #expect(meta["kind"] as? String == "runBundle")
            #expect(meta["schemaVersion"] as? Int == 1)
            for entry in try #require(meta["entries"] as? [[String: Any]]) {
                #expect(entry.keys.sorted() == Self.bundleEntryKeys)
                #expect((entry["sha256"] as? String)?.count == 64)
            }
        }
    }

    // MARK: - 4. Envelope fields a cross-platform client dispatches on

    /// CONTRACT: envelope-dispatch-fields — a cross-platform client branches
    /// on exactly four things, so all four must be present and USABLE in every
    /// committed golden, not merely allowed by the closed key set.
    ///
    /// `ExperimentCLIEnvelopeTests.everyCommittedFixtureIsOneValidEnvelope`
    /// already pins the closed key set, the schema version, the engine stamp,
    /// and error-present-iff-not-success. What it does not pin is that
    /// `error.code` names something and that `error.repairAction` is an
    /// actionable instruction — the two fields an agent cannot recover
    /// without, and the ones AGENTS.md promises every refusal carries. Server
    /// twin: `test_every_committed_golden_carries_the_fields_a_client_dispatches_on`.
    @Test func everyGoldenCarriesTheFieldsAClientDispatchesOn() throws {
        let directory = CodeResources.compiledCheckoutPath.appending(
            components: "Tests", "Fixtures", "cli-envelopes")
        let files =
            (try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil)) ?? []
        let goldens = files.filter { $0.pathExtension == "json" }.sorted {
            $0.lastPathComponent < $1.lastPathComponent
        }
        #expect(!goldens.isEmpty, "no committed cli-envelope goldens")

        var sawRefusal = false
        for url in goldens {
            let name = url.lastPathComponent
            let text = try String(contentsOf: url, encoding: .utf8)
            let envelope = try SteerLabCLIEnvelope.decode(fromJSON: text)
            let object = try #require(
                try JSONSerialization.jsonObject(with: Data(text.utf8))
                    as? [String: Any])

            // 1. `state` — always in the closed vocabulary, always resolvable
            //    to an exit code without consulting a second table.
            let state = try #require(object["state"] as? String)
            #expect(
                SteerLabCLIState(rawValue: state) != nil,
                "\(name): state '\(state)' is outside the vocabulary")
            #expect(envelope.state.rawValue == state)

            // 2. `workspace` — a client addressing a remote engine has to know
            //    WHICH workspace answered. Optional by the contract; when
            //    present it must never be an empty placeholder a client would
            //    read as "the default".
            if let workspace = object["workspace"] {
                let path = try #require(
                    workspace as? String, "\(name): workspace is not a string")
                #expect(
                    !path.trimmingCharacters(in: .whitespaces).isEmpty,
                    "\(name): empty workspace")
            }

            guard !envelope.state.isSuccess else {
                #expect(object["error"] == nil, "\(name): success carries an error")
                continue
            }
            sawRefusal = true
            let error = try #require(
                object["error"] as? [String: Any], "\(name): refusal has no error")

            // 3. `error.code` — the branch key.
            let code = try #require(error["code"] as? String, "\(name): no error.code")
            #expect(!code.isEmpty, "\(name): empty error.code")

            // 4. `error.repairAction` — what the client DOES next. A refusal
            //    without one sends an agent in a circle, which is the failure
            //    the typed-refusal contract exists to prevent.
            let repair = try #require(
                error["repairAction"] as? String,
                "\(name): refusal has no repairAction")
            #expect(
                !repair.trimmingCharacters(in: .whitespaces).isEmpty,
                "\(name): empty repairAction")
            #expect(repair.count > 8, "\(name): repairAction is not an instruction")
        }
        #expect(
            sawRefusal,
            "no committed golden is a refusal: the dispatch fields that matter most to a client are then untested")
    }
}
