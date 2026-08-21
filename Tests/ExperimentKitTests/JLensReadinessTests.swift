import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// `data check` coverage for the J-lens readout block.
///
/// Server-only by hard requirement, so this engine never resolves or loads a
/// lens. What it checks is what the study DECLARED — and every one of these is
/// a freeze refusal the researcher would otherwise meet only at the end of
/// authoring, or (for retention) only after the run, when it is too late.
@Suite(.serialized) struct JLensReadinessTests {

    private func manifest(
        readout: JSONValue? = nil, revision: String? = "005ad3404e59",
        dtype: String? = "bfloat16", recordTokenIDs: Bool = false
    ) -> ExperimentManifest {
        var m = ExperimentManifest(
            name: "jspace-pilot", description: "d",
            modelID: "google/gemma-3-27b-it")
        m.jlensReadout = readout
        m.modelRevision = revision
        m.dtype = dtype
        m.recordTokenIDs = recordTokenIDs
        return m
    }

    private func block(
        _ overrides: [String: JSONValue] = [:], drop: [String] = []
    ) -> JSONValue {
        var pins: [String: JSONValue] = [
            "lensID": .string("google--gemma-3-27b-it--jlens-wikitext"),
            "lensSHA256": .string(String(repeating: "a", count: 64)),
            "configHash": .string(String(repeating: "c", count: 64)),
            "tokenizerHash": .string(String(repeating: "t", count: 64)),
            "layers": .array([.number(31), .number(40), .number(48)]),
            "watchlist": .array([.number(44598)]),
            // A frozen study must name the exact acceptance it rests on, or a
            // later re-qualification moves the ground under it.
            "qualificationID": .string("q-0123456789abcdef"),
        ]
        for key in drop { pins.removeValue(forKey: key) }
        for (key, value) in overrides { pins[key] = value }
        return .object(pins)
    }

    @Test func noReadoutIsOptionalNotABlocker() {
        let row = StudyDataReadiness.jlensReadoutRequirement(
            manifest: manifest())
        #expect(row.status == .optional)
        // The engine boundary is stated where a researcher will read it.
        #expect(row.detail.contains("server-only"))
    }

    @Test func everyMissingPinIsNamed() {
        for key in ["lensID", "lensSHA256", "configHash", "tokenizerHash",
                    "layers", "qualificationID"] {
            let row = StudyDataReadiness.jlensReadoutRequirement(
                manifest: manifest(readout: block(drop: [key])))
            #expect(row.status == .missing, "dropping \(key) was not a blocker")
            #expect(row.detail.contains(key),
                    "the refusal did not name \(key)")
        }
    }

    @Test func aReadoutThatWouldRecordNothingIsInvalid() {
        // Neither a watchlist nor a top-k width: the run completes, reports
        // normally, and contains no readout — indistinguishable afterwards
        // from a study that never asked for one.
        let row = StudyDataReadiness.jlensReadoutRequirement(
            manifest: manifest(readout: block(drop: ["watchlist"])))
        #expect(row.status == .invalid)
        #expect(row.detail.contains("record nothing"))
    }

    @Test func aTopKOnlyReadoutRecordsSomething() {
        let row = StudyDataReadiness.jlensReadoutRequirement(
            manifest: manifest(
                readout: block(["topK": .number(10)], drop: ["watchlist"]),
                recordTokenIDs: true))
        #expect(row.status == .present)
    }

    @Test func anUnpinnedRuntimeBlocksBecauseQualificationIsKeyedByIt() {
        for (revision, dtype, expected) in [
            (nil as String?, "bfloat16" as String?, "modelRevision"),
            ("005ad3404e59" as String?, nil as String?, "dtype"),
        ] {
            let row = StudyDataReadiness.jlensReadoutRequirement(
                manifest: manifest(
                    readout: block(), revision: revision, dtype: dtype))
            #expect(row.status == .missing)
            #expect(row.detail.contains(expected))
        }
    }

    @Test func retentionOffIsPartialNotMissing() {
        // It RUNS. What it costs is the ability to go back — and retention is
        // not retroactive, which is why this is said before the run.
        let row = StudyDataReadiness.jlensReadoutRequirement(
            manifest: manifest(readout: block(), recordTokenIDs: false))
        #expect(row.status == .partial)
        #expect(row.detail.contains("recordTokenIDs"))
        // Accurate about what retention buys: the SEQUENCE is kept, not a
        // bit-identical re-run (external review, 2026-08-16).
        #expect(!row.detail.contains("exactly replayable"))
        #expect(!StudyDataReadiness.summary([row]).blockers.contains(row),
                "a partial row must not read as a blocker")
    }

    @Test func aFullyDeclaredReadoutIsPresentAndStillNamesTheQualificationGate() {
        let row = StudyDataReadiness.jlensReadoutRequirement(
            manifest: manifest(readout: block(), recordTokenIDs: true))
        #expect(row.status == .present)
        // Present here never means "freezable": only the server can produce a
        // passing qualification, and saying so avoids a false all-clear.
        #expect(row.detail.contains("jlens qualify"))
    }
}

/// `qualificationID` is a pin, not a nicety (external review round 3).
///
/// Without it a frozen study resolves whichever qualification is newest, so
/// appending one later moves the ground under an already-frozen study. Both
/// engines enforce the same list; the server asserts its copy in Python.
@Suite(.serialized) struct JLensQualificationPinTests {

    @Test func aReadoutWithoutAQualificationIDIsABlocker() {
        var m = ExperimentManifest(
            name: "jspace-pilot", description: "d",
            modelID: "google/gemma-3-27b-it")
        m.modelRevision = "005ad3404e59"
        m.dtype = "bfloat16"
        m.recordTokenIDs = true
        m.jlensReadout = .object([
            "lensID": .string("google--gemma-3-27b-it--jlens-wikitext"),
            "lensSHA256": .string(String(repeating: "a", count: 64)),
            "configHash": .string(String(repeating: "c", count: 64)),
            "tokenizerHash": .string(String(repeating: "t", count: 64)),
            "layers": .array([.number(31)]),
            "watchlist": .array([.number(44598)]),
        ])
        let row = StudyDataReadiness.jlensReadoutRequirement(manifest: m)
        #expect(row.status == .missing)
        #expect(row.detail.contains("qualificationID"))
    }
}

/// The J-lens contract in the FREEZE firewall, not only the UI checklist.
///
/// `data check` grew coverage first, but freeze runs through `verify()` — so a
/// Swift/CLI freeze could accept a readout the server would refuse, and the
/// supposedly cross-engine firewall had already diverged (external review
/// round 4).
@Suite(.serialized) struct JLensVerifyMirrorTests {

    private func manifest(_ block: [String: JSONValue]?,
                          instruments: [String]? = ["sampledText"])
        -> ExperimentManifest
    {
        var m = ExperimentManifest(
            name: "s", description: "d", modelID: "google/gemma-3-27b-it")
        m.outcomeInstruments = instruments
        m.jlensReadout = block.map { .object($0) }
        return m
    }

    private var complete: [String: JSONValue] {
        [
            "lensID": .string("google--gemma-3-27b-it--jlens-wikitext"),
            "lensSHA256": .string(String(repeating: "a", count: 64)),
            "configHash": .string(String(repeating: "c", count: 64)),
            "tokenizerHash": .string(String(repeating: "t", count: 64)),
            "qualificationID": .string("q-0123456789abcdef"),
            "layers": .array([.number(31)]),
            "watchlist": .array([.number(44598)]),
        ]
    }

    @Test func noReadoutIsNoViolation() {
        #expect(ExperimentStore.jlensReadoutViolations(manifest(nil)).isEmpty)
    }

    @Test func aCompleteReadoutVerifies() {
        #expect(ExperimentStore.jlensReadoutViolations(
            manifest(complete)).isEmpty)
    }

    @Test func everyMissingPinIsAViolationIncludingQualificationID() {
        for key in ["lensID", "lensSHA256", "configHash", "tokenizerHash",
                    "qualificationID", "layers"] {
            var block = complete
            block.removeValue(forKey: key)
            let found = ExperimentStore.jlensReadoutViolations(manifest(block))
            #expect(found.contains { $0.contains(key) },
                    "dropping \(key) produced no violation naming it")
        }
    }

    @Test func aReadoutThatRecordsNothingIsAViolation() {
        var block = complete
        block.removeValue(forKey: "watchlist")
        #expect(ExperimentStore.jlensReadoutViolations(manifest(block))
            .contains { $0.contains("record nothing") })
    }

    @Test func aChoiceOnlyPlanIsAViolation() {
        let found = ExperimentStore.jlensReadoutViolations(
            manifest(complete, instruments: ["answerTokenLogprob"]))
        #expect(found.contains { $0.contains("no sampled text") })
    }

    @Test func verifyItselfCarriesTheseViolations() {
        // The point of the finding: the checks must be reachable from the
        // path freeze actually runs, not only from the checklist.
        var block = complete
        block.removeValue(forKey: "qualificationID")
        var m = manifest(block)
        m.concepts = []
        #expect(ExperimentStore.verify(m).contains {
            $0.contains("qualificationID")
        })
    }
}

/// Agent adapter identity must pin the CONFIGURATION too (round 5).
@Suite(.serialized) struct AdapterConfigPinTests {

    @Test func adapterRefCarriesAndRoundTripsConfigHash() throws {
        let ref = ModelVariantArtifact.AdapterRef(
            name: "a", artifactPath: "runs/ft/a.json",
            adapterDirectory: "runs/ft/a",
            adapterHash: String(repeating: "1", count: 64),
            configHash: String(repeating: "2", count: 64))
        let round = try JSONDecoder().decode(
            ModelVariantArtifact.AdapterRef.self,
            from: try JSONEncoder().encode(ref))
        // Without this an agent verified its weights and silently skipped its
        // configuration: editing adapter_config.json changes what the forward
        // pass does while the declared identity stays identical.
        #expect(round.configHash == ref.configHash)
        #expect(round.adapterHash == ref.adapterHash)
    }

    @Test func aLegacyRefDecodesWithConfigurationUnpinned() throws {
        let json = """
        {"name":"a","artifactPath":"runs/ft/a.json",
         "adapterDirectory":"runs/ft/a","adapterHash":"\(String(repeating: "1", count: 64))"}
        """
        let ref = try JSONDecoder().decode(
            ModelVariantArtifact.AdapterRef.self, from: Data(json.utf8))
        // nil is "unpinned", never "verified" — consumers stamp it as such.
        #expect(ref.configHash == nil)
    }
}

extension JLensVerifyMirrorTests {

    @Test func aPresentNonObjectBlockIsAViolation() {
        var m = ExperimentManifest(
            name: "s", description: "d", modelID: "google/gemma-3-27b-it")
        m.jlensReadout = .string("not-an-object")
        let found = ExperimentStore.jlensReadoutViolations(m)
        #expect(found.contains { $0.contains("not an object") })
    }
}

// MARK: - External review round 6: the agent's configuration is half its identity

@Suite("Adapter configuration pinning")
struct AdapterConfigurationPinTests {

    private func manifest(configHash: String?) -> ExperimentManifest {
        var m = ExperimentManifest(
            name: "jspace-pilot", description: "d",
            modelID: "google/gemma-3-27b-it")
        m.variantConditions = [
            ExperimentManifest.VariantCondition(
                name: "sympathy-agent", artifactPath: "p", artifactHash: "h",
                artifact: ModelVariantArtifact(
                    name: "sympathy-agent", baseModelID: "google/gemma-3-27b-it",
                    adapters: [
                        ModelVariantArtifact.AdapterRef(
                            name: "sympathy-lora",
                            artifactPath: "adapters/sympathy",
                            adapterDirectory: "adapters/sympathy",
                            adapterHash: String(repeating: "ab", count: 32),
                            configHash: configHash)],
                    promptMode: "chatAssistant", qwenThinkingEnabled: false,
                    temperature: 0, systemPrompt: ""))
        ]
        return m
    }

    /// Every agent minted before 2026-08-16 lacks `configHash` — all 20 in
    /// the live study workspace. Gating freeze on it would push whole
    /// existing studies onto `--force`, which stamps them non-citable: a
    /// worse outcome than a loud advisory for a gap that is recomputable
    /// from bytes already on disk.
    @Test("an unpinned adapter config advises, and does not refuse the freeze")
    func unpinnedConfigIsAnAdvisory() throws {
        let advisories = ExperimentStore.freezeAdvisories(for: manifest(configHash: nil))
        let hit = try #require(advisories.first { $0.contains("configHash") })
        // It must say the gap is REPAIRABLE, not merely that it exists.
        #expect(hit.contains("computable"))
        #expect(hit.contains("sympathy-agent"))
    }

    @Test("a pinned adapter config raises no advisory")
    func pinnedConfigIsQuiet() {
        #expect(!ExperimentStore
            .freezeAdvisories(for: manifest(configHash: String(repeating: "cd", count: 32)))
            .contains { $0.contains("configHash") })
    }

    /// The server's `/api/adapters` catalog reports the sidecar's own key
    /// names (`adapterBytesHash`/`adapterConfigHash`). Before round 6 the
    /// remote path decoded neither, so a server-workspace agent could not be
    /// pinned at all — and J-lens work runs server-side.
    @Test("a remote adapter record carries the server's two hashes through")
    func remoteAdapterRecordCarriesHashes() throws {
        let bytes = String(repeating: "ab", count: 32)
        let config = String(repeating: "cd", count: 32)
        let json = Data("""
        {"name": "sympathy-lora", "adapterDirectory": "runs/lora/sympathy",
         "adapterBytesHash": "\(bytes)", "adapterConfigHash": "\(config)"}
        """.utf8)
        let record = try JSONDecoder().decode(RemoteAdapterRecord.self, from: json)
        #expect(record.adapterBytesHash == bytes)
        #expect(record.adapterConfigHash == config)
    }

    /// Absent stays absent: a legacy server adapter is "unpinned", never
    /// silently "verified".
    @Test("a remote adapter record without hashes decodes as unpinned")
    func remoteAdapterRecordToleratesLegacy() throws {
        let json = Data(#"{"name": "old", "adapterDirectory": "runs/lora/old"}"#.utf8)
        let record = try JSONDecoder().decode(RemoteAdapterRecord.self, from: json)
        #expect(record.adapterBytesHash == nil)
        #expect(record.adapterConfigHash == nil)
    }
}

/// External review round 7 (P3): the decode test proved `RemoteAdapterRecord`
/// parses both hashes; it did not prove the composition path USES them. This
/// drives a real `/api/adapters` response through `ChatService`'s own
/// refresh + compose, which is where the mapping actually has to hold.
@Suite("Adapter catalog to composed agent")
struct AdapterCatalogCompositionTests {

    private final class StubProtocol: URLProtocol, @unchecked Sendable {
        nonisolated(unsafe) static var body = Data()
        override class func canInit(with request: URLRequest) -> Bool { true }
        override class func canonicalRequest(for r: URLRequest) -> URLRequest { r }
        override func startLoading() {
            let response = HTTPURLResponse(
                url: request.url!, statusCode: 200,
                httpVersion: nil, headerFields: nil)!
            client?.urlProtocol(self, didReceive: response,
                                cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: Self.body)
            client?.urlProtocolDidFinishLoading(self)
        }
        override func stopLoading() {}
    }

    @MainActor
    private func service(catalogJSON: String) -> (ChatService, UserDefaults, String) {
        let suite = "AdapterCatalogComposition.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        let store = clusterStore(defaults: defaults)
        StubProtocol.body = Data(catalogJSON.utf8)
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubProtocol.self]
        store.clientSessionOverride = URLSession(configuration: configuration)
        let entry = store.addServer(name: "test", urlString: "http://server.test")
        store.activeWorkspace = .server(entry.id)
        return (ChatService(cluster: store), defaults, suite)
    }

    @MainActor
    @Test("a server adapter's hashes survive catalog → composed agent")
    func hashesSurviveComposition() async throws {
        let bytes = String(repeating: "ab", count: 32)
        let config = String(repeating: "cd", count: 32)
        let (chat, defaults, suite) = service(catalogJSON: """
        {"adapters": [{"name": "sympathy-lora",
                       "adapterDirectory": "runs/lora/sympathy",
                       "adapterBytesHash": "\(bytes)",
                       "adapterConfigHash": "\(config)"}]}
        """)
        defer { defaults.removePersistentDomain(forName: suite) }

        await chat.refreshServerSteeringArtifacts()
        chat.adaptersEnabled = true
        chat.selectedServerAdapterID = "runs/lora/sympathy"

        let adapter = try #require(chat.serverControlState().adapters.first)
        #expect(adapter.adapterHash == bytes)
        #expect(adapter.configHash == config)
    }

    /// Absent stays absent through composition too: a legacy server adapter
    /// must compose as UNPINNED, never silently verified.
    @MainActor
    @Test("a legacy server adapter composes as unpinned")
    func legacyComposesUnpinned() async throws {
        let (chat, defaults, suite) = service(catalogJSON: """
        {"adapters": [{"name": "old-lora", "adapterDirectory": "runs/lora/old"}]}
        """)
        defer { defaults.removePersistentDomain(forName: suite) }

        await chat.refreshServerSteeringArtifacts()
        chat.adaptersEnabled = true
        chat.selectedServerAdapterID = "runs/lora/old"

        let adapter = try #require(chat.serverControlState().adapters.first)
        #expect(adapter.adapterHash == nil)
        #expect(adapter.configHash == nil)
    }
}

/// External review round 11 (P2): condition names are a CROSS-ENGINE
/// record-identity contract — both engines' `record_key` leads with them — so
/// the uniqueness invariant must exist on both, and must be checked against
/// what a run EXECUTES rather than what it declares.
@Suite("Executable condition-name uniqueness")
struct ConditionNameUniquenessTests {

    private func manifest(
        conditions: [String] = [], variants: [String] = [],
        studyType: String? = nil
    ) -> ExperimentManifest {
        var m = ExperimentManifest(
            name: "names", description: "d", modelID: "google/gemma-3-27b-it")
        m.conditions = conditions.map {
            ExperimentManifest.Condition(name: $0, slots: [])
        }
        m.variantConditions = variants.map {
            ExperimentManifest.VariantCondition(
                name: $0, artifactPath: "p", artifactHash: "h",
                artifact: ModelVariantArtifact(
                    name: $0, baseModelID: "google/gemma-3-27b-it",
                    promptMode: "chatAssistant", qwenThinkingEnabled: false,
                    temperature: 0, systemPrompt: ""))
        }
        m.studyType = studyType
        return m
    }

    /// The collision a declared-collections scan cannot see: every run
    /// executes a baseline, and it is in no declared collection.
    @Test("a variant named baseline collides with the implicit baseline")
    func variantNamedBaselineCollides() {
        let violations = ExperimentStore.verify(manifest(variants: ["baseline"]))
        #expect(violations.contains { $0.contains("duplicate condition name") })
    }

    @Test("two variants sharing a name collide")
    func twoVariantsCollide() {
        let violations = ExperimentStore.verify(manifest(variants: ["arm", "arm"]))
        #expect(violations.contains { $0.contains("duplicate condition name") })
    }

    /// The other direction: a variant comparison runs baseline + variants
    /// only, so duplicates among carried injection conditions cannot alias
    /// any record key and must not block the study.
    @Test("inert carried conditions do not cause a false refusal")
    func inertCarriedConditionsDoNotRefuse() {
        let m = manifest(conditions: ["carried", "carried"],
                         variants: ["sympathy-arm"], studyType: "agentComparison")
        #expect(!ExperimentStore.verify(m)
            .contains { $0.contains("duplicate condition name") })
    }

    @Test("distinct executable names verify clean")
    func distinctNamesVerifyClean() {
        #expect(!ExperimentStore.verify(manifest(variants: ["sympathy-arm"]))
            .contains { $0.contains("duplicate condition name") })
    }

    /// The name list must come from the run's own resolver, so the gate and
    /// execution cannot disagree about what a study runs.
    @Test("the effective name list matches what a variant comparison executes")
    func effectiveNamesMatchExecution() {
        let m = manifest(conditions: ["carried"], variants: ["sympathy-arm"])
        #expect(ExperimentTasks.effectiveConditionNames(for: m)
            == ["baseline", "sympathy-arm"])
    }
}

/// External review round 12: a multi-agent study executes a SCENARIO, not the
/// model-output collections — which it may legally carry under the app's
/// never-delete rule.
@Suite("Multi-agent condition identity")
struct MultiAgentConditionNameTests {

    private func scenarioManifest(
        conditions: [String] = [], variants: [String] = [],
        includeBaseline: Bool = true
    ) -> ExperimentManifest {
        var m = ExperimentManifest(
            name: "panel", description: "d", modelID: "google/gemma-3-27b-it")
        m.studyKind = .multiAgent
        m.multiAgentScenarioPath = "scenarios/s.json"
        m.multiAgentScenarioHash = "h"
        m.multiAgentIncludeBaseline = includeBaseline
        m.conditions = conditions.map {
            ExperimentManifest.Condition(name: $0, slots: [])
        }
        m.variantConditions = variants.map {
            ExperimentManifest.VariantCondition(
                name: $0, artifactPath: "p", artifactHash: "h",
                artifact: ModelVariantArtifact(
                    name: $0, baseModelID: "google/gemma-3-27b-it",
                    promptMode: "chatAssistant", qwenThinkingEnabled: false,
                    temperature: 0, systemPrompt: ""))
        }
        return m
    }

    @Test("a carried variant named baseline does not refuse a panel study")
    func carriedBaselineVariantIsLegal() {
        #expect(!ExperimentStore.verify(scenarioManifest(variants: ["baseline"]))
            .contains { $0.contains("duplicate condition name") })
    }

    @Test("carried duplicate conditions do not refuse a panel study")
    func carriedDuplicatesAreLegal() {
        #expect(!ExperimentStore
            .verify(scenarioManifest(conditions: ["dup", "dup"]))
            .contains { $0.contains("duplicate condition name") })
    }

    /// Not "no names" — the RIGHT names. A panel keys records on "configured"
    /// plus "baseline" when a baseline play-through is included, so the
    /// resolver's contract stays true for this study kind too.
    @Test("panel names are the scenario's own vocabulary")
    func panelNamesAreTheScenarios() {
        #expect(ExperimentTasks.effectiveConditionNames(for: scenarioManifest())
            == ["baseline", "configured"])
        #expect(ExperimentTasks.effectiveConditionNames(
            for: scenarioManifest(includeBaseline: false)) == ["configured"])
    }

    @Test("carried model-output config contributes no panel names")
    func carriedConfigContributesNoNames() {
        let names = ExperimentTasks.effectiveConditionNames(
            for: scenarioManifest(conditions: ["carried"],
                                  variants: ["carried-agent"]))
        #expect(!names.contains("carried") && !names.contains("carried-agent"))
    }
}

/// The round-12 sweep, Swift side: every helper this branch added that takes a
/// manifest, checked against `studyKind`. J-lens is a model-output instrument;
/// a panel study may carry a block across a kind switch and never arms one.
@Suite("Carried readout blocks on a panel study")
struct CarriedReadoutOnPanelTests {

    private func panel(withReadout: Bool) -> ExperimentManifest {
        var m = ExperimentManifest(
            name: "panel", description: "d", modelID: "google/gemma-3-27b-it")
        m.studyKind = .multiAgent
        m.multiAgentScenarioPath = "scenarios/s.json"
        m.multiAgentScenarioHash = "h"
        if withReadout {
            // Deliberately INVALID for a model-output study — half-pinned —
            // so a kind-blind gate would certainly fire.
            m.jlensReadout = .object(["lensID": .string("lens-1")])
        }
        return m
    }

    @Test("a carried readout block does not fail a panel study's verify")
    func carriedBlockDoesNotFailVerify() {
        #expect(!ExperimentStore.verify(panel(withReadout: true))
            .contains { $0.lowercased().contains("jlensreadout") })
    }

    /// `data check` exits 2 on `.missing`/`.invalid`, so grading a carried
    /// block here would fail a legal panel study's readiness too.
    @Test("the readiness row is not-applicable, never a blocker, for a panel")
    func readinessRowIsNotABlocker() {
        for carried in [true, false] {
            let row = StudyDataReadiness.jlensReadoutRequirement(
                manifest: panel(withReadout: carried))
            #expect(row.status == .optional)
            #expect(row.detail.contains("not applicable"))
        }
    }

    /// The guard must be exactly the study kind, not a blanket exemption: a
    /// model-output study with the same half-pinned block must still fail.
    @Test("a model-output study with the same block still fails")
    func modelOutputStillValidated() {
        var m = ExperimentManifest(
            name: "study", description: "d", modelID: "google/gemma-3-27b-it")
        m.jlensReadout = .object(["lensID": .string("lens-1")])
        #expect(!ExperimentStore.jlensReadoutViolations(m).isEmpty)
        #expect(ExperimentStore.verify(m)
            .contains { $0.lowercased().contains("jlensreadout") })
    }
}

/// External review round 13: freeze must answer what `verify()` answers. The
/// round-12 fix made a panel carrying model-output configuration verify
/// clean; freeze still refused it, and `freeze_advisories` was simultaneously
/// telling the researcher that carried configuration is not verified for this
/// study kind.
@Suite("Freeze surfaces by study kind")
struct FreezeSurfacesByStudyKindTests {

    @Test("the shared decision is exactly the study kind")
    func decisionIsTheStudyKind() {
        var panel = ExperimentManifest(
            name: "p", description: "d", modelID: "google/gemma-3-27b-it")
        panel.studyKind = .multiAgent
        #expect(!ExperimentStore.modelOutputSurfacesOperative(panel))

        var study = panel
        study.studyKind = .modelOutput
        #expect(ExperimentStore.modelOutputSurfacesOperative(study))
    }

    /// Readiness must not auto-pin a capability battery a panel never uses —
    /// carried state may advise, and may not be auto-pinned.
    @Test("readiness does not pin a battery for a panel's carried variants")
    func readinessDoesNotPinBatteryForPanel() {
        var panel = ExperimentManifest(
            name: "p", description: "d", modelID: "google/gemma-3-27b-it")
        panel.studyKind = .multiAgent
        panel.multiAgentScenarioPath = "scenarios/s.json"
        panel.multiAgentScenarioHash = "h"
        panel.variantConditions = [
            ExperimentManifest.VariantCondition(
                name: "carried", artifactPath: "p", artifactHash: "h",
                artifact: ModelVariantArtifact(
                    name: "carried", baseModelID: "google/gemma-3-27b-it",
                    promptMode: "chatAssistant", qwenThinkingEnabled: false,
                    temperature: 0, systemPrompt: ""))
        ]
        let readiness = ExperimentStore.freezeReadiness(for: panel)
        // Battery/validate gates belong to the model-output surfaces.
        #expect(!readiness.unmetGates.contains { $0.contains("battery") })
        #expect(!readiness.unmetGates.contains { $0.contains("validate run") })
    }

    /// The exemption must be exactly the study kind, not a hole in the
    /// firewall: the same carried variant on a model-output study still gates.
    @Test("a model-output study still meets the variant gates")
    func modelOutputStillGated() {
        var study = ExperimentManifest(
            name: "s", description: "d", modelID: "google/gemma-3-27b-it")
        study.variantConditions = [
            ExperimentManifest.VariantCondition(
                name: "arm", artifactPath: "p", artifactHash: "h",
                artifact: ModelVariantArtifact(
                    name: "arm", baseModelID: "google/gemma-3-27b-it",
                    promptMode: "chatAssistant", qwenThinkingEnabled: false,
                    temperature: 0, systemPrompt: ""))
        ]
        #expect(!ExperimentStore.freezeReadiness(for: study).unmetGates.isEmpty)
    }
}

/// External review round 14: drive the REAL freeze. Round 13's Swift test
/// called `freezeReadiness`, which is exactly why the unconditional
/// pin-and-copy operations escaped — the gate evaluator was scoped and the
/// transaction around it was not.
@Suite("Panel freeze transaction", .serialized)
struct PanelFreezeTransactionTests {

    private func writePanel(
        in root: URL, carriedExists: Bool, kind: ExperimentManifest.StudyKind
    ) throws -> String {
        let fm = FileManager.default
        let runs = root.appending(path: "runs/model-variants")
        try fm.createDirectory(at: runs, withIntermediateDirectories: true)
        let scenarios = root.appending(path: "scenarios")
        try fm.createDirectory(at: scenarios, withIntermediateDirectories: true)
        let scenarioBody = Data("{}".utf8)
        try scenarioBody.write(to: scenarios.appending(component: "s.json"))

        let artifactPath = "runs/model-variants/carried.json"
        var artifactHash = "h"
        if carriedExists {
            let body = Data(#"{"name":"carried","baseModelID":"google/gemma-3-27b-it"}"#.utf8)
            try body.write(to: root.appending(path: artifactPath))
            artifactHash = ExperimentStore.sha256Hex(body)
        }
        var m = ExperimentManifest(
            name: "panel", description: "d", modelID: "google/gemma-3-27b-it")
        m.studyKind = kind
        m.modelRevision = String(repeating: "a", count: 40)
        m.multiAgentScenarioPath = "scenarios/s.json"
        m.multiAgentScenarioHash = ExperimentStore.sha256Hex(scenarioBody)
        m.variantConditions = [
            ExperimentManifest.VariantCondition(
                name: "carried", artifactPath: artifactPath,
                artifactHash: artifactHash,
                artifact: ModelVariantArtifact(
                    name: "carried", baseModelID: "google/gemma-3-27b-it",
                    promptMode: "chatAssistant", qwenThinkingEnabled: false,
                    temperature: 0, systemPrompt: ""))
        ]
        try ExperimentStore.save(m, allowCreate: true)
        return artifactPath
    }

    /// The stale case: freeze promised carried state is "not bundled", then
    /// tried to bundle it.
    @Test("a panel with a stale carried agent still freezes")
    func staleCarriedAgentStillFreezes() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "panel-stale") { root in
            _ = try writePanel(in: root, carriedExists: false, kind: .multiAgent)
            let frozen = try ExperimentStore.freeze(name: "panel", force: true)
            #expect(frozen.status == .frozen)
        }
    }

    /// The present case is the more dangerous one: freeze SUCCEEDED while
    /// silently rewriting the carried configuration.
    @Test("a panel's carried agent is not copied, repointed, or auto-pinned")
    func carriedAgentIsLeftAlone() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "panel-present") { root in
            let path = try writePanel(in: root, carriedExists: true, kind: .multiAgent)
            let frozen = try ExperimentStore.freeze(name: "panel", force: true)

            #expect(frozen.variantConditions[0].artifactPath == path)
            #expect(!FileManager.default.fileExists(
                atPath: root.appending(
                    path: "experiments/panel/pinned/variant-carried.json").path))
            #expect(frozen.capabilityBatteryFile == nil)
            #expect(frozen.markersHash == nil)
        }
    }

    /// The exemption is exactly the study kind, not a hole in the firewall.
    @Test("the same agent on a model-output study is still relocated")
    func modelOutputAgentIsStillRelocated() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "panel-mo") { root in
            let path = try writePanel(in: root, carriedExists: true, kind: .modelOutput)
            let frozen = try ExperimentStore.freeze(name: "panel", force: true)

            #expect(frozen.variantConditions[0].artifactPath != path)
            #expect(frozen.variantConditions[0].artifactPath
                .hasPrefix("experiments/panel/pinned/"))
        }
    }
}

/// External review round 15: the mirror of the carried-agent case. A scenario
/// is the PANEL's input, not a kind-neutral one — the round-14 comment saying
/// otherwise was wrong.
@Suite("Carried scenario on a model-output study", .serialized)
struct CarriedScenarioTests {

    private func writeStudy(
        in root: URL, scenarioExists: Bool, kind: ExperimentManifest.StudyKind
    ) throws -> String {
        let fm = FileManager.default
        try fm.createDirectory(at: root.appending(path: "runs/scenarios"),
                               withIntermediateDirectories: true)
        let scenarioPath = "runs/scenarios/carried.json"
        let body = Data("{}".utf8)
        if scenarioExists { try body.write(to: root.appending(path: scenarioPath)) }

        var m = ExperimentManifest(
            name: "study", description: "d", modelID: "google/gemma-3-27b-it")
        m.studyKind = kind
        m.modelRevision = String(repeating: "a", count: 40)
        m.multiAgentScenarioPath = scenarioPath
        m.multiAgentScenarioHash = ExperimentStore.sha256Hex(body)
        if kind == .modelOutput {
            // A model-output study must have something attached to verify at
            // all; the carried scenario is what this test is about.
            let variantBody = Data(#"{"name":"arm","baseModelID":"google/gemma-3-27b-it"}"#.utf8)
            try fm.createDirectory(at: root.appending(path: "runs"),
                                   withIntermediateDirectories: true)
            try variantBody.write(to: root.appending(path: "runs/v.json"))
            m.variantConditions = [
                ExperimentManifest.VariantCondition(
                    name: "arm", artifactPath: "runs/v.json",
                    artifactHash: ExperimentStore.sha256Hex(variantBody),
                    artifact: ModelVariantArtifact(
                        name: "arm", baseModelID: "google/gemma-3-27b-it",
                        promptMode: "chatAssistant", qwenThinkingEnabled: false,
                        temperature: 0, systemPrompt: ""))
            ]
        }
        try ExperimentStore.save(m, allowCreate: true)
        return scenarioPath
    }

    @Test("a study carrying a stale panel scenario still freezes")
    func staleCarriedScenarioStillFreezes() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "sc-stale") { root in
            let path = try writeStudy(in: root, scenarioExists: false,
                                      kind: .modelOutput)
            let frozen = try ExperimentStore.freeze(name: "study", force: true)
            #expect(frozen.status == .frozen)
            #expect(frozen.multiAgentScenarioPath == path)
        }
    }

    @Test("a carried panel scenario is not relocated or committed")
    func presentCarriedScenarioIsLeftAlone() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "sc-present") { root in
            let path = try writeStudy(in: root, scenarioExists: true,
                                      kind: .modelOutput)
            let frozen = try ExperimentStore.freeze(name: "study", force: true)
            #expect(frozen.multiAgentScenarioPath == path)
            #expect(!FileManager.default.fileExists(
                atPath: root.appending(
                    path: "experiments/study/pinned/scenario.json").path))
        }
    }

    /// The exemption is exactly the study kind: a panel's scenario is its live
    /// input and must still be pinned into the experiment directory.
    @Test("a panel still relocates its own scenario")
    func panelStillRelocatesItsOwnScenario() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "sc-panel") { root in
            _ = try writeStudy(in: root, scenarioExists: true, kind: .multiAgent)
            let frozen = try ExperimentStore.freeze(name: "study", force: true)
            #expect(frozen.multiAgentScenarioPath?
                .hasPrefix("experiments/study/pinned/") == true)
        }
    }
}
