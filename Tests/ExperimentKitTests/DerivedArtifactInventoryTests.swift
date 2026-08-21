import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// `DerivedArtifactInventory` (Data section phase 3, part A) and
/// `DatasetDerivation` (part B).
///
/// The claims under test: every derived kind is enumerated by the STORE that
/// already owns it (so a fixture written to that store's real schema shows
/// up, and nothing else does), provenance comes from the artifact's own
/// sidecar, the scan follows the resolved workspace, and the derive actions
/// a source dataset offers are gated by ROLE with a stated reason when they
/// are empty.
@Suite(.serialized) @MainActor
struct DerivedArtifactInventoryTests {

    // MARK: Harness

    private func withTempWorkspace<T>(_ body: (URL) throws -> T) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "derived-inventory") { root in
            let previous = WorkspaceRoot.programmaticOverride
            WorkspaceRoot.programmaticOverride = root
            defer { WorkspaceRoot.programmaticOverride = previous }
            try? FileManager.default.createDirectory(
                at: root, withIntermediateDirectories: true)
            return try body(root)
        }
    }

    private func write(_ contents: String, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try contents.write(to: url, atomically: true, encoding: .utf8)
    }

    // MARK: Fixtures — each written to its OWN store's real schema

    @discardableResult
    private func makeVector(
        _ name: String, in root: URL, run: String = "2026-01-01-extract",
        concept: String = "courage", model: String = "test/model",
        method: String = "meanDifference", layers: Int = 3,
        date: String = "2026-01-01T00:00:00Z"
    ) throws -> URL {
        let directory = root.appending(components: "runs", run)
        // VectorCatalog.scan pairs `<name>.safetensors` with `<name>.json`;
        // the tensor bytes are never parsed, so an empty file is enough.
        try write("", to: directory.appending(component: "\(name).safetensors"))
        try write(
            """
            {"modelID": "\(model)", "concept": "\(concept)", \
            "stimulusSetHash": "abc123", "layerCount": \(layers), \
            "hiddenSize": 8, "normsPerLayer": [1.0, 1.0, 1.0], \
            "extractionDate": "\(date)", "extractionMethod": "\(method)", \
            "revision": "deadbeef", "readingPosition": "last token"}
            """,
            to: directory.appending(component: "\(name).json"))
        return directory
    }

    private func makeProbe(
        _ name: String, in root: URL, run: String = "2026-01-02-probe",
        concept: String = "courage", layer: Int = 14,
        date: String = "2026-01-02T00:00:00Z"
    ) throws {
        try write(
            """
            {"modelID": "test/model", "concept": "\(concept)", "layer": \(layer), \
            "recipeName": "linear-probe", "createdAt": "\(date)", \
            "validationHash": "v0ffee", \
            "probe": {"direction": [1.0], "projectionCenter": 0, \
            "projectionScale": 1, "orientation": 1, "positiveMean": 1, \
            "negativeMean": -1}}
            """,
            to: root.appending(components: "runs", run, "\(name).probe.json"))
    }

    private func makeAdapter(
        _ name: String, in root: URL, date: String = "2026-01-03T00:00:00Z"
    ) throws {
        try write(
            """
            {"schemaVersion": 1, "name": "\(name)", "baseModelID": "test/model", \
            "adapterDirectory": "adapters/\(name)", "fineTuneType": "lora", \
            "rank": 8, "scale": 10, "adaptedLayers": 16, "batchSize": 4, \
            "iterations": 1000, "learningRate": 0.00001, \
            "createdAt": "\(date)", "notes": "", "adapterFormat": "mlx-lora"}
            """,
            to: root.appending(
                components: "runs", "fine-tunes", "run-\(name)", "fine-tune.json"))
    }

    @discardableResult
    private func makeNeutralPCBasis(
        _ run: String, in root: URL, date: String = "2026-01-04T00:00:00Z"
    ) throws -> URL {
        let url = root.appending(
            components: "runs", "neutral-pcs", run, "neutral-pcs.json")
        try write(
            """
            {"schemaVersion": 1, "modelID": "test/model", "corpusHash": "c0ffee", \
            "corpusPath": "prompts/neutral/corpus.jsonl", \
            "readingPosition": "last token", "selectionDescription": "top-3", \
            "layers": [10, 11], "componentsByLayer": [[[1.0]], [[1.0]]], \
            "residualNormPerLayer": [1.0, 1.0], "tokenRowCount": 128, \
            "createdAt": "\(date)", \
            "screening": {"readingPosition": "last token", "minimumTokenCount": 1, \
            "sourceCount": 1, "includedCount": 1, "excludedShortCount": 0}}
            """,
            to: url)
        return url
    }

    private func makeAgent(
        _ name: String, in root: URL, promoted: Bool,
        date: String = "2026-01-05T00:00:00Z"
    ) throws {
        let promotion =
            promoted
            ? """
            , "promotion": {"experiment": "screen-1", "experimentHash": "e0f1", \
            "promotedAt": "\(date)", "promotedBy": "criterion", \
            "substrate": "swift-mlx", "appVersion": "test", \
            "winningCell": {"layer": 14, "alpha": 0.08}}
            """
            : ""
        try write(
            """
            {"name": "\(name)", "baseModelID": "test/model", \
            "createdAt": "\(date)"\(promotion)}
            """,
            to: root.appending(
                components: "runs", "model-variants", "run-\(name)",
                "model-variant.json"))
    }

    // MARK: Empty

    @Test func anEmptyWorkspaceHasNothingDerived() throws {
        try withTempWorkspace { root in
            #expect(DerivedArtifactInventory.scan(root: root).isEmpty)
        }
    }

    /// Source datasets are NOT derived artifacts. A workspace full of stimuli
    /// and corpora with no builds yet has an empty derived scope — the two
    /// tables never double-count.
    @Test func sourceDatasetsNeverAppearInTheDerivedScope() throws {
        try withTempWorkspace { root in
            try write(
                #"{"text": "p"}"# + "\n",
                to: root.appending(
                    components: "prompts", "concepts", "courage", "positive.jsonl"))
            try write(
                #"{"text": "n"}"# + "\n",
                to: root.appending(
                    components: "prompts", "concepts", "courage", "negative.jsonl"))
            #expect(DerivedArtifactInventory.scan(root: root).isEmpty)
            #expect(DatasetInventory.scan(root: root).count == 1)
        }
    }

    // MARK: Each kind, through its own store

    @Test func aSavedVectorIsEnumeratedWithItsSidecarProvenance() throws {
        try withTempWorkspace { root in
            try makeVector("courage-v1", in: root)

            let entries = DerivedArtifactInventory.scan(root: root)
            #expect(entries.count == 1)
            let vector = try #require(entries.first)
            #expect(vector.kind == .steeringVector)
            #expect(vector.name == "courage-v1")
            #expect(vector.modelID == "test/model")
            #expect(vector.concept == "courage")
            #expect(vector.detail.contains("meanDifference"))
            #expect(vector.detail.contains("3 layers"))
            #expect(vector.route == .analysis)
            // The row points at the artifact; provenance at its sidecar.
            #expect(vector.primaryURL.lastPathComponent == "courage-v1.safetensors")
            #expect(vector.sidecarURL.lastPathComponent == "courage-v1.json")
            #expect(vector.displayPath.hasPrefix("runs/"))

            let facts = Dictionary(
                uniqueKeysWithValues: vector.facts.map { ($0.label, $0.value) })
            #expect(facts["Concept"] == "courage")
            #expect(facts["Model"] == "test/model")
            #expect(facts["Method"] == "meanDifference")
            #expect(facts["Layers"] == "3")
            #expect(facts["Revision"] == "deadbeef")
            #expect(facts["Stimulus sha256"] == "abc123")
            // The sidecar carries no recipe name, so no row claims one.
            #expect(facts["Recipe"] == nil)
        }
    }

    @Test func aTrainedProbeIsEnumeratedFromItsProbeArtifact() throws {
        try withTempWorkspace { root in
            try makeProbe("courage-probe", in: root)

            let probe = try #require(DerivedArtifactInventory.scan(root: root).first)
            #expect(probe.kind == .readingProbe)
            // `.probe.json` is a double extension — the name is neither.
            #expect(probe.name == "courage-probe")
            #expect(probe.concept == "courage")
            #expect(probe.detail == "linear-probe · layer 14")
            #expect(probe.route == .conceptsAndVectors)
            let facts = Dictionary(
                uniqueKeysWithValues: probe.facts.map { ($0.label, $0.value) })
            #expect(facts["Layer"] == "14")
            #expect(facts["Validation sha256"] == "v0ffee")
        }
    }

    @Test func aTrainedAdapterIsEnumeratedFromTheFineTuneStore() throws {
        try withTempWorkspace { root in
            try makeAdapter("formality-lora", in: root)

            let adapter = try #require(DerivedArtifactInventory.scan(root: root).first)
            #expect(adapter.kind == .adapter)
            #expect(adapter.name == "formality-lora")
            #expect(adapter.modelID == "test/model")
            #expect(adapter.detail.contains("lora"))
            #expect(adapter.detail.contains("rank 8"))
            #expect(adapter.route == .adapterTraining)
            let facts = Dictionary(
                uniqueKeysWithValues: adapter.facts.map { ($0.label, $0.value) })
            #expect(facts["Adapter format"] == "mlx-lora")
            #expect(facts["Hyperparameters"]?.contains("rank 8") == true)
            // An empty `notes` is absent, not a blank row.
            #expect(facts["Notes"] == nil)
        }
    }

    /// The neutral-PC row carries BOTH digests, and the pinned one is the
    /// SHA-256 of the FILE BYTES — the digest `ExperimentStore.verify()`
    /// re-checks — not the corpus hash the artifact also records.
    @Test func aNeutralPCBasisCarriesThePinnedFileHashNotJustTheCorpusHash() throws {
        try withTempWorkspace { root in
            let url = try makeNeutralPCBasis("2026-neutral-pcs-test", in: root)

            let basis = try #require(DerivedArtifactInventory.scan(root: root).first)
            #expect(basis.kind == .neutralPCBasis)
            #expect(basis.route == .conceptsAndVectors)
            let facts = Dictionary(
                uniqueKeysWithValues: basis.facts.map { ($0.label, $0.value) })
            #expect(facts["Corpus sha256"] == "c0ffee")
            let pinned = try #require(facts["Basis sha256 (pinned)"])
            let bytes = try Data(contentsOf: url)
            #expect(pinned == ExperimentStore.sha256Hex(bytes))
            #expect(pinned != "c0ffee")
            #expect(facts["Components"] == "top-3")
            #expect(facts["Token rows"] == "128")
        }
    }

    /// The agent row is a CROSS-REFERENCE: it routes to the Agents section
    /// and it reads the promotion birth certificate rather than restating it.
    @Test func agentsAreCrossReferencedWithTheirBirthCertificate() throws {
        try withTempWorkspace { root in
            try makeAgent("promoted-agent", in: root, promoted: true)
            try makeAgent("hand-made", in: root, promoted: false)

            let agents = DerivedArtifactInventory.scan(root: root)
                .filter { $0.kind == .agent }
            #expect(agents.count == 2)
            #expect(agents.allSatisfy { $0.route == .agents })
            #expect(DerivedArtifactKind.agent.detail.contains("Agents section"))

            let promoted = try #require(agents.first { $0.name == "promoted-agent" })
            let facts = Dictionary(
                uniqueKeysWithValues: promoted.facts.map { ($0.label, $0.value) })
            #expect(facts["Promoted by"] == "criterion")
            #expect(facts["From study"] == "screen-1")
            #expect(facts["Winning cell"] == "layer 14 · α 0.08")

            let hand = try #require(agents.first { $0.name == "hand-made" })
            let handFacts = Dictionary(
                uniqueKeysWithValues: hand.facts.map { ($0.label, $0.value) })
            #expect(handFacts["Promotion"]?.contains("hand-created") == true)
            #expect(handFacts["Promoted by"] == nil)
        }
    }

    // MARK: Ordering, ids, workspace resolution

    @Test func rowsGroupByKindThenNewestFirst() throws {
        try withTempWorkspace { root in
            try makeVector(
                "older", in: root, run: "run-a", date: "2026-01-01T00:00:00Z")
            try makeVector(
                "newer", in: root, run: "run-b", date: "2026-02-01T00:00:00Z")
            try makeProbe("a-probe", in: root)
            try makeAdapter("an-adapter", in: root)

            let entries = DerivedArtifactInventory.scan(root: root)
            #expect(entries.map(\.kind) == [
                .steeringVector, .steeringVector, .readingProbe, .adapter,
            ])
            #expect(entries[0].name == "newer")
            #expect(entries[1].name == "older")
            #expect(Set(entries.map(\.id)).count == entries.count)
        }
    }

    /// An unparseable stamp is shown VERBATIM and sorts last — never
    /// re-formatted into a date the artifact does not claim.
    @Test func aFreeFormCreationStampIsShownVerbatimAndSortsLast() throws {
        try withTempWorkspace { root in
            try makeVector("legacy", in: root, run: "run-a", date: "sometime in 2024")
            try makeVector(
                "modern", in: root, run: "run-b", date: "2026-02-01T00:00:00Z")

            let entries = DerivedArtifactInventory.scan(root: root)
            #expect(entries.map(\.name) == ["modern", "legacy"])
            let legacy = try #require(entries.last)
            #expect(legacy.created == nil)
            #expect(legacy.createdText == "sometime in 2024")
            #expect(legacy.sortableCreated == .distantPast)
        }
    }

    /// Fractional-second stamps (what `makeUniqueRunDirectory` writes) parse
    /// too — both ISO-8601 spellings the producing stores emit.
    @Test func bothISO8601SpellingsParse() {
        #expect(DerivedArtifactEntry.parse("2026-01-01T00:00:00Z") != nil)
        #expect(DerivedArtifactEntry.parse("2026-01-01T00:00:00.123Z") != nil)
        #expect(DerivedArtifactEntry.parse("") == nil)
        #expect(DerivedArtifactEntry.parse("not a date") == nil)
    }

    @Test func scanFollowsTheResolvedWorkspace() throws {
        let first = FileManager.default.temporaryDirectory
            .appending(component: "derived-a-\(UUID().uuidString)")
        let second = FileManager.default.temporaryDirectory
            .appending(component: "derived-b-\(UUID().uuidString)")
        defer {
            try? FileManager.default.removeItem(at: first)
            try? FileManager.default.removeItem(at: second)
        }
        try makeVector("alpha", in: first)
        try makeVector("beta", in: second)

        ExperimentRootOverrideLock.acquire()
        let previous = WorkspaceRoot.programmaticOverride
        defer {
            WorkspaceRoot.programmaticOverride = previous
            ExperimentRootOverrideLock.release()
        }

        WorkspaceRoot.programmaticOverride = first
        #expect(DerivedArtifactInventory.scan().map(\.name) == ["alpha"])
        WorkspaceRoot.programmaticOverride = second
        #expect(DerivedArtifactInventory.scan().map(\.name) == ["beta"])
    }

    /// One malformed sidecar must not stop the scan — the owning stores all
    /// skip what they cannot decode, and this asserts the projection inherits
    /// that rather than throwing.
    @Test func aMalformedSidecarIsSkippedAndTheRestStillScan() throws {
        try withTempWorkspace { root in
            try makeVector("intact", in: root, run: "run-a")
            let broken = root.appending(components: "runs", "run-b")
            try write("", to: broken.appending(component: "broken.safetensors"))
            try write("{ this is not json", to: broken.appending(component: "broken.json"))

            let entries = DerivedArtifactInventory.scan(root: root)
            #expect(entries.map(\.name) == ["intact"])
        }
    }

    // MARK: Part B — role-gated derivations

    @Test func eachSourceRoleOffersTheDerivationItsRecipeOwns() throws {
        try withTempWorkspace { root in
            try write(
                #"{"text": "p"}"# + "\n",
                to: root.appending(
                    components: "prompts", "concepts", "courage", "positive.jsonl"))
            try write(
                #"{"text": "n"}"# + "\n",
                to: root.appending(
                    components: "prompts", "concepts", "courage", "negative.jsonl"))
            try write(
                #"{"positive": "p", "negative": "n"}"# + "\n",
                to: root.appending(
                    components: "prompts", "repe", "courage", "pairs.jsonl"))
            try write(
                #"{"concept": "sympathy", "text": "a story"}"# + "\n",
                to: root.appending(
                    components: "prompts", "emotions", "sympathy", "stories.jsonl"))
            try write(
                #"{"text": "item", "expresses": true}"# + "\n",
                to: root.appending(
                    components: "prompts", "probes", "courage", "items.jsonl"))
            try write(
                #"{"text": "neutral"}"# + "\n",
                to: root.appending(components: "prompts", "neutral", "corpus.jsonl"))

            let byKind = Dictionary(
                grouping: DatasetInventory.scan(root: root), by: \.kind)

            func actions(_ kind: DatasetKind) throws -> [DatasetDerivationAction] {
                DatasetDerivation.actions(for: try #require(byKind[kind]?.first))
            }

            #expect(try actions(.conceptStimuli) == [.buildVector(concept: "courage")])
            #expect(try actions(.pairedStimuli) == [.buildVector(concept: "courage")])
            #expect(
                try actions(.grandMeanCorpus)
                    == [.buildGrandMeanVector(concept: "sympathy")])
            #expect(try actions(.probeItems) == [.trainProbe(concept: "courage")])
            #expect(try actions(.neutralCorpus) == [.buildNeutralPCs])
        }
    }

    /// Only the grand-mean route claims to set the recipe, because that is
    /// the only one with a clean seam that would not contradict the files a
    /// paired concept already has.
    @Test func onlyTheGrandMeanRouteClaimsToSetTheRecipe() {
        #expect(DatasetDerivationAction.buildGrandMeanVector(concept: "s")
            .setsGrandMeanRecipe)
        #expect(!DatasetDerivationAction.buildVector(concept: "c").setsGrandMeanRecipe)
        #expect(!DatasetDerivationAction.trainProbe(concept: "c").setsGrandMeanRecipe)
        #expect(!DatasetDerivationAction.buildNeutralPCs.setsGrandMeanRecipe)
        // The neutral-PC action carries no concept — nothing to preselect.
        #expect(DatasetDerivationAction.buildNeutralPCs.concept == nil)
        #expect(DatasetDerivationAction.buildVector(concept: "c").concept == "c")
    }

    /// Every action names its destination honestly — the sentence rendered as
    /// the button's help must say where it goes.
    @Test func everyActionNamesWhereItGoes() {
        let actions: [DatasetDerivationAction] = [
            .buildVector(concept: "courage"),
            .buildGrandMeanVector(concept: "sympathy"),
            .trainProbe(concept: "courage"),
            .buildNeutralPCs,
        ]
        for action in actions {
            #expect(action.destination.contains("Concepts & Vectors"))
            #expect(!action.title.isEmpty)
        }
        #expect(
            DatasetDerivationAction.buildGrandMeanVector(concept: "s")
                .destination.contains("grand-mean"))
    }

    /// The measurement-side families derive NOTHING, and each says why —
    /// an empty action row must never read as a missing feature.
    @Test func measurementSideDatasetsDeriveNothingAndSayWhy() throws {
        try withTempWorkspace { root in
            try write(
                #"{"text": "held out", "expresses": true}"# + "\n",
                to: root.appending(
                    components: "prompts", "concepts", "courage", "validation.jsonl"))
            try write(
                #"{"batteryFormat": 2, "scoring": "choiceProbability"}"# + "\n"
                    + #"{"prompt": "2+2?", "answer": "4", "options": ["3", "4"]}"# + "\n",
                to: root.appending(
                    components: "prompts", "batteries", "basic.jsonl"))
            try write(
                #"{"id": "t1", "prompt": "decide"}"# + "\n",
                to: root.appending(
                    components: "prompts", "tasks", "case-prompts.jsonl"))

            for entry in DatasetInventory.scan(root: root) where entry.kind != .conceptStimuli {
                guard [.validationSet, .capabilityBattery, .promptSet].contains(entry.kind)
                else { continue }
                #expect(DatasetDerivation.actions(for: entry).isEmpty)
                let reason = try #require(DatasetDerivation.noDerivationReason(for: entry))
                #expect(!reason.isEmpty)
            }

            let validation = try #require(
                DatasetInventory.scan(root: root).first { $0.kind == .validationSet })
            let reason = try #require(
                DatasetDerivation.noDerivationReason(for: validation))
            #expect(reason.contains("EVIDENCE"))
            #expect(reason.contains("validationHash"))
        }
    }

    /// A row that DOES offer actions offers no reason — the two are
    /// mutually exclusive by construction, so the view can render either.
    @Test func aDerivableRowHasNoNoDerivationReason() throws {
        try withTempWorkspace { root in
            try write(
                #"{"text": "neutral"}"# + "\n",
                to: root.appending(components: "prompts", "neutral", "corpus.jsonl"))
            let corpus = try #require(DatasetInventory.scan(root: root).first)
            #expect(!DatasetDerivation.actions(for: corpus).isEmpty)
            #expect(DatasetDerivation.noDerivationReason(for: corpus) == nil)
        }
    }
}
