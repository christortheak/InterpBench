import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// `DatasetInventory` (Data section phase 1): the inventory must REPORT the
/// resolved workspace and nothing else — no seeding, no invented counts, no
/// fatal stop on one malformed file.
@Suite(.serialized) @MainActor
struct DatasetInventoryTests {

    // MARK: Harness

    /// Same override discipline as the rest of this target: ONE process-global
    /// lock around both root seams (`ExperimentRootOverrideLock`), with
    /// `WorkspaceRoot.programmaticOverride` set so `VectorCatalog.projectRoot`
    /// — the inventory's default root — resolves to the temp tree.
    private func withTempWorkspace<T>(_ body: (URL) throws -> T) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "dataset-inventory") { root in
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

    private func textRows(_ texts: [String]) -> String {
        texts.map { #"{"text": "\#($0)"}"# }.joined(separator: "\n") + "\n"
    }

    private func makeConcept(
        _ name: String, in root: URL, positives: Int = 3, negatives: Int = 3
    ) throws -> URL {
        let directory = root.appending(components: "prompts", "concepts", name)
        try write(
            textRows((0 ..< positives).map { "\(name) positive \($0)" }),
            to: directory.appending(component: "positive.jsonl"))
        try write(
            textRows((0 ..< negatives).map { "\(name) negative \($0)" }),
            to: directory.appending(component: "negative.jsonl"))
        return directory
    }

    private func makeNeutralCorpus(in root: URL, rows: Int = 4) throws {
        try write(
            textRows((0 ..< rows).map { "neutral line \($0)" }),
            to: root.appending(components: "prompts", "neutral", "corpus.jsonl"))
    }

    // MARK: Empty workspace

    @Test func anEmptyWorkspaceProducesAnEmptyInventory() throws {
        try withTempWorkspace { root in
            #expect(DatasetInventory.scan(root: root).isEmpty)
        }
    }

    /// The scan reports; it never seeds. A workspace whose prompts/ tree
    /// exists but is EMPTY (the shape `WorkspaceStore.create` leaves before
    /// any concept is authored) yields nothing — in particular no
    /// norm-calibration row standing in for a corpus file that is not there.
    @Test func aConceptEmptyWorkspaceGetsNoPhantomRows() throws {
        try withTempWorkspace { root in
            for sub in WorkspaceStore.promptSubdirectories {
                try FileManager.default.createDirectory(
                    at: root.appending(components: "prompts", sub),
                    withIntermediateDirectories: true)
            }
            let entries = DatasetInventory.scan(root: root)
            #expect(entries.isEmpty)
            #expect(FileManager.default.fileExists(
                atPath: root.appending(
                    components: "prompts", "neutral", "corpus.jsonl").path) == false)
        }
    }

    /// …but the generic instruments a seeded workspace GENUINELY carries do
    /// appear. `prompts/neutral/corpus.jsonl` is in `seedManifest`, so a
    /// fresh workspace's inventory is exactly one row.
    @Test func aSeededWorkspaceShowsItsGenericInstrumentsOnly() throws {
        try withTempWorkspace { root in
            try makeNeutralCorpus(in: root, rows: 5)
            let entries = DatasetInventory.scan(root: root)
            #expect(entries.count == 1)
            let corpus = try #require(entries.first)
            #expect(corpus.kind == .neutralCorpus)
            #expect(corpus.itemCount == 5)
            #expect(corpus.conceptName == nil)
            // The corpus hash is the one NeutralCorpusStore already computes.
            #expect(corpus.contentHash?.count == 64)
        }
    }

    // MARK: Kinds and counts

    @Test func conceptDirectoryAndCorpusProduceTheRightKindsAndCounts() throws {
        try withTempWorkspace { root in
            _ = try makeConcept("courage", in: root, positives: 4, negatives: 6)
            try makeNeutralCorpus(in: root, rows: 3)

            let entries = DatasetInventory.scan(root: root)
            #expect(entries.count == 2)

            let stimuli = try #require(entries.first { $0.kind == .conceptStimuli })
            #expect(stimuli.name == "courage")
            #expect(stimuli.itemCount == 10)
            #expect(stimuli.conceptName == "courage")
            #expect(stimuli.files.count == 2)
            #expect(stimuli.issue == nil)
            #expect(stimuli.byteSize > 0)
            #expect(stimuli.modified != nil)
            // The hash is StimulusSet's — the one extraction pins.
            let expected = try StimulusSet(
                directory: root.appending(components: "prompts", "concepts", "courage")
            ).hash
            #expect(stimuli.contentHash == expected)

            #expect(entries.contains { $0.kind == .neutralCorpus })
        }
    }

    @Test func everyPerConceptFamilyIsEnumerated() throws {
        try withTempWorkspace { root in
            _ = try makeConcept("fairness", in: root, positives: 2, negatives: 2)
            try write(
                #"{"text": "held out scenario", "expresses": true}"# + "\n"
                    + #"{"text": "another one", "expresses": false}"# + "\n",
                to: root.appending(
                    components: "prompts", "concepts", "fairness", "validation.jsonl"))
            try write(
                #"{"positive": "p one", "negative": "n one"}"# + "\n"
                    + #"{"positive": "p two", "negative": "n two"}"# + "\n"
                    + #"{"positive": "p three", "negative": "n three"}"# + "\n",
                to: root.appending(
                    components: "prompts", "repe", "fairness", "pairs.jsonl"))
            // The READER root's real row shape — a different one from the
            // RepE mirror above, which is why each family is counted by its
            // own loader (phase 4).
            try write(
                #"{"concept": "fairness", "positiveStimulus": "r one", "#
                    + #""negativeStimulus": "r two", "split": "train", "#
                    + #""templateID": "t1"}"# + "\n",
                to: root.appending(
                    components: "prompts", "readers", "fairness", "pairs.jsonl"))
            try write(
                #"{"concept": "fairness", "text": "a story", "topic": "t1"}"# + "\n"
                    + #"{"concept": "fairness", "text": "another story", "topic": "t2"}"#
                    + "\n",
                to: root.appending(
                    components: "prompts", "emotions", "fairness", "stories.jsonl"))
            try write(
                #"{"text": "probe item one", "expresses": true}"# + "\n"
                    + #"{"text": "probe item two", "expresses": false}"# + "\n"
                    + #"{"text": "probe item three", "expresses": true}"# + "\n",
                to: root.appending(
                    components: "prompts", "probes", "fairness", "items.jsonl"))

            let entries = DatasetInventory.scan(root: root)
            let byKind = Dictionary(grouping: entries, by: \.kind)

            #expect(byKind[.conceptStimuli]?.first?.itemCount == 4)
            #expect(byKind[.validationSet]?.first?.itemCount == 2)
            #expect(byKind[.grandMeanCorpus]?.first?.itemCount == 2)
            #expect(byKind[.probeItems]?.first?.itemCount == 3)
            // Two paired roots, both named `fairness`, both listed and
            // distinguishable by their sub-family label.
            let paired = byKind[.pairedStimuli] ?? []
            #expect(paired.count == 2)
            #expect(Set(paired.compactMap(\.familyLabel)) == ["repe", "readers"])
            #expect(paired.first { $0.familyLabel == "repe" }?.itemCount == 3)
            #expect(paired.first { $0.familyLabel == "readers" }?.itemCount == 1)
            // Each family parsed cleanly with ITS loader — before phase 4
            // both were read as RepE pairs, so a real reader dataset was
            // reported as malformed.
            #expect(paired.allSatisfy { $0.issue == nil })
            #expect(paired.allSatisfy { $0.contentHash?.count == 64 })

            #expect(byKind[.validationSet]?.first?.familyLabel == "paired")

            // Distinct ids even where two kinds share one directory.
            #expect(Set(entries.map(\.id)).count == entries.count)
            // The validation set carries the hash the manifest pins.
            let validation = try #require(byKind[.validationSet]?.first)
            let pinned = ExperimentStore.conceptValidationHash(
                fileURL: validation.files[0])
            #expect(validation.contentHash == pinned)
            #expect(validation.contentHash?.count == 64)

            // No store defines a pinned hash for probe items — absent, not faked.
            #expect(byKind[.probeItems]?.first?.contentHash == nil)
        }
    }

    /// `ExperimentStore.conceptValidationRelativePath` puts the held-out set
    /// under prompts/concepts/ for paired recipes and prompts/emotions/ for
    /// grand-mean — the study's PRIMARY recipe. Both homes are scanned, both
    /// are labelled, and both hash to what the manifest would pin.
    @Test func bothValidationHomesAreScannedAndLabelled() throws {
        try withTempWorkspace { root in
            let held = #"{"text": "held out", "expresses": true}"# + "\n"
            try write(
                held,
                to: root.appending(
                    components: "prompts", "concepts", "sympathy", "validation.jsonl"))
            try write(
                held + #"{"text": "second", "expresses": false}"# + "\n",
                to: root.appending(
                    components: "prompts", "emotions", "sympathy", "validation.jsonl"))

            let validations = DatasetInventory.scan(root: root)
                .filter { $0.kind == .validationSet }
            #expect(validations.count == 2)
            #expect(Set(validations.compactMap(\.familyLabel)) == ["paired", "grand-mean"])
            #expect(validations.first { $0.familyLabel == "paired" }?.itemCount == 1)
            #expect(validations.first { $0.familyLabel == "grand-mean" }?.itemCount == 2)

            let grandMean = try #require(
                validations.first { $0.familyLabel == "grand-mean" })
            #expect(
                grandMean.contentHash
                    == ExperimentStore.conceptValidationHash(
                        name: "sympathy", isPaired: false))

            // Two rows, one concept name, one directory each — still distinct.
            #expect(Set(validations.map(\.id)).count == 2)
        }
    }

    // MARK: Malformed input is reported, not fatal

    @Test func aMalformedDatasetIsReportedAndTheRestStillScan() throws {
        try withTempWorkspace { root in
            _ = try makeConcept("intact", in: root)
            let broken = root.appending(components: "prompts", "concepts", "broken")
            try write("this is not JSONL at all\n", to: broken.appending(
                component: "positive.jsonl"))
            try write(
                textRows(["fine"]), to: broken.appending(component: "negative.jsonl"))

            let entries = DatasetInventory.scan(root: root)
            #expect(entries.count == 2)

            let bad = try #require(entries.first { $0.name == "broken" })
            #expect(bad.issue != nil)
            #expect(bad.itemCount == nil)  // unknown, never 0
            #expect(bad.contentHash == nil)
            #expect(bad.byteSize > 0)  // stat-level facts survive

            let good = try #require(entries.first { $0.name == "intact" })
            #expect(good.issue == nil)
            #expect(good.itemCount == 6)
        }
    }

    @Test func aHalfWrittenConceptNamesTheMissingFile() throws {
        try withTempWorkspace { root in
            try write(
                textRows(["only positives here"]),
                to: root.appending(
                    components: "prompts", "concepts", "halfway", "positive.jsonl"))

            let entry = try #require(DatasetInventory.scan(root: root).first)
            #expect(entry.kind == .conceptStimuli)
            #expect(entry.itemCount == nil)
            #expect(entry.issue?.contains("negative.jsonl") == true)
            #expect(entry.files.count == 1)
        }
    }

    /// A directory that exists but holds none of the family's files is not a
    /// dataset — it must not become a phantom zero-item row.
    @Test func anEmptyConceptDirectoryIsNotARow() throws {
        try withTempWorkspace { root in
            try FileManager.default.createDirectory(
                at: root.appending(components: "prompts", "concepts", "placeholder"),
                withIntermediateDirectories: true)
            #expect(DatasetInventory.scan(root: root).isEmpty)
        }
    }

    // MARK: Workspace resolution

    /// The inventory's DEFAULT root is the resolved workspace, so switching
    /// workspaces changes the entries with no caller involvement.
    @Test func scanFollowsTheResolvedWorkspaceAndChangesOnSwitch() throws {
        let first = FileManager.default.temporaryDirectory
            .appending(component: "inventory-a-\(UUID().uuidString)")
        let second = FileManager.default.temporaryDirectory
            .appending(component: "inventory-b-\(UUID().uuidString)")
        defer {
            try? FileManager.default.removeItem(at: first)
            try? FileManager.default.removeItem(at: second)
        }
        _ = try makeConcept("alpha", in: first)
        _ = try makeConcept("beta", in: second)
        _ = try makeConcept("gamma", in: second)

        ExperimentRootOverrideLock.acquire()
        let previous = WorkspaceRoot.programmaticOverride
        defer {
            WorkspaceRoot.programmaticOverride = previous
            ExperimentRootOverrideLock.release()
        }

        WorkspaceRoot.programmaticOverride = first
        // No explicit root: this is the resolution chain under test.
        let fromFirst = DatasetInventory.scan()
        #expect(fromFirst.map(\.name) == ["alpha"])

        WorkspaceRoot.programmaticOverride = second
        let fromSecond = DatasetInventory.scan()
        #expect(fromSecond.map(\.name) == ["beta", "gamma"])
    }

    // MARK: Ordering, ids, and display

    @Test func entriesAreGroupedByKindThenNamed() throws {
        try withTempWorkspace { root in
            _ = try makeConcept("zeta", in: root)
            _ = try makeConcept("alpha", in: root)
            try makeNeutralCorpus(in: root)

            let entries = DatasetInventory.scan(root: root)
            #expect(entries.map(\.name) == ["alpha", "zeta", "norm-calibration"])
        }
    }

    @Test func displayPathIsWorkspaceRelative() throws {
        try withTempWorkspace { root in
            _ = try makeConcept("delta", in: root)
            let entry = try #require(DatasetInventory.scan(root: root).first)
            #expect(entry.displayPath(root: root) == "prompts/concepts/delta")
            #expect(entry.shortHash?.count == 12)
        }
    }

    /// Row counts and hashes come from loaders that read whole files; past the
    /// budget the scan degrades to stat-level rather than pulling the file
    /// into memory on a UI refresh.
    @Test func anOversizeDatasetDegradesToStatLevel() throws {
        try withTempWorkspace { root in
            let directory = root.appending(
                components: "prompts", "concepts", "enormous")
            let filler = String(
                repeating: "x", count: Int(DatasetInventory.maximumParsedBytes) + 16)
            try write(textRows([filler]), to: directory.appending(
                component: "positive.jsonl"))
            try write(textRows(["small"]), to: directory.appending(
                component: "negative.jsonl"))

            let entry = try #require(DatasetInventory.scan(root: root).first)
            #expect(entry.itemCount == nil)
            #expect(entry.contentHash == nil)
            #expect(entry.issue?.contains("parse budget") == true)
            #expect(entry.byteSize > DatasetInventory.maximumParsedBytes)
        }
    }

    // MARK: Capability batteries (phase 3)

    private func battery(_ items: [String], header: String? = nil) -> String {
        ((header.map { [$0] } ?? []) + items).joined(separator: "\n") + "\n"
    }

    private static let choiceItem =
        #"{"id": "b1", "prompt": "2+2?", "answer": "4", "options": ["3", "4"]}"#
    private static let legacyItem =
        #"{"prompt": "Capital of France?", "answer": "Paris"}"#

    /// Both formats enumerate, and each row says WHICH — the format is the
    /// sub-family label, because a v1 and a v2 battery are read differently
    /// even when their bytes hash the same way.
    @Test func batteriesOfBothFormatsAreEnumeratedWithCountAndFormat() throws {
        try withTempWorkspace { root in
            try write(
                battery(
                    [Self.choiceItem, Self.choiceItem],
                    header: #"{"batteryFormat": 2, "scoring": "choiceProbability", "#
                        + #""maxTokens": 24, "description": "Two arithmetic probes."}"#),
                to: root.appending(components: "prompts", "batteries", "modern.jsonl"))
            try write(
                battery([Self.legacyItem, Self.legacyItem, Self.legacyItem]),
                to: root.appending(components: "prompts", "batteries", "legacy.jsonl"))

            let batteries = DatasetInventory.scan(root: root)
                .filter { $0.kind == .capabilityBattery }
            #expect(batteries.count == 2)

            let modern = try #require(batteries.first { $0.name == "modern" })
            #expect(modern.itemCount == 2)
            #expect(modern.familyLabel == "format 2")
            #expect(modern.issue == nil)
            // The v2 header's declared description, surfaced through the same
            // header parser the loader uses to identify the format.
            #expect(modern.note?.contains("Two arithmetic probes.") == true)
            #expect(modern.note?.contains("legacy headerless") != true)

            let legacy = try #require(batteries.first { $0.name == "legacy" })
            #expect(legacy.itemCount == 3)
            #expect(legacy.familyLabel == "format 1")
            #expect(legacy.issue == nil)
            // An honest caption, not a defect: it still pins and runs.
            #expect(legacy.note?.contains("legacy headerless") == true)
            #expect(legacy.note?.contains("battery lint") == true)

            // One directory, many datasets — the rows are distinct.
            #expect(Set(batteries.map(\.id)).count == 2)
        }
    }

    /// The hash a battery row shows is the hash the PIN uses: raw file bytes
    /// through `ExperimentStore.sha256Hex`, byte-identical to what
    /// `pinSweepInputs` writes as `sweep.batteryHash`. Asserted against the
    /// pin path itself rather than against a re-implementation.
    @Test func aBatteryRowsHashIsThePinsHash() throws {
        try withTempWorkspace { root in
            let contents = battery(
                [Self.choiceItem],
                header: #"{"batteryFormat": 2, "scoring": "choiceProbability"}"#)
            let url = root.appending(
                components: "prompts", "batteries", "pinned.jsonl")
            try write(contents, to: url)

            let row = try #require(
                DatasetInventory.scan(root: root).first { $0.kind == .capabilityBattery })
            var manifest = ExperimentManifest(name: "s", description: "", modelID: "test/model")
            manifest.sweep = ExperimentManifest.SweepSpec(
                batteryFile: "prompts/batteries/pinned.jsonl")
            ExperimentStore.pinSweepInputs(into: &manifest)

            #expect(row.contentHash == manifest.sweep?.batteryHash)
            #expect(row.contentHash == ExperimentStore.sha256Hex(try Data(contentsOf: url)))
            #expect(row.contentHash?.count == 64)
        }
    }

    /// A battery the loader rejects is REPORTED, not dropped — and the hash
    /// still shows, because that is exactly the case where the researcher is
    /// chasing which bytes a pin refers to.
    @Test func aMalformedBatteryIsReportedWithItsHashIntact() throws {
        try withTempWorkspace { root in
            try write(
                battery(
                    [#"{"prompt": "2+2?", "answer": "4"}"#],
                    header: #"{"batteryFormat": 2, "scoring": "choiceProbability"}"#),
                to: root.appending(components: "prompts", "batteries", "broken.jsonl"))

            let row = try #require(
                DatasetInventory.scan(root: root).first { $0.kind == .capabilityBattery })
            // A choiceProbability item with no options cannot be scored.
            #expect(row.issue?.contains("options") == true)
            #expect(row.itemCount == nil)
            #expect(row.familyLabel == nil)
            #expect(row.contentHash?.count == 64)
        }
    }

    /// Only `*.jsonl` directly inside the root, and only regular files. The
    /// roots are flat by convention; descending would be inventing a layout.
    @Test func batteryEnumerationIsFlatAndJSONLOnly() throws {
        try withTempWorkspace { root in
            let batteries = root.appending(components: "prompts", "batteries")
            try write(battery([Self.legacyItem]), to: batteries.appending(
                component: "real.jsonl"))
            try write("notes", to: batteries.appending(component: "README.md"))
            try write(battery([Self.legacyItem]), to: batteries.appending(
                components: "archive", "buried.jsonl"))

            let rows = DatasetInventory.scan(root: root)
                .filter { $0.kind == .capabilityBattery }
            #expect(rows.map(\.name) == ["real"])
        }
    }

    // MARK: Task and dev prompt sets (phase 3)

    /// The two roots that have a real layout authority, each counted by the
    /// loader that OWNS it and hashed with the digest its pin uses.
    @Test func taskAndDevPromptSetsAreEnumeratedByTheirOwnLoaders() throws {
        try withTempWorkspace { root in
            try write(
                #"{"id": "case-1", "prompt": "Decide.", "options": ["a", "b"], "target": "a"}"#
                    + "\n"
                    + #"{"id": "case-2", "prompt": "Decide again."}"# + "\n",
                to: root.appending(
                    components: "prompts", "tasks", "tidal-power-prompts.jsonl"))
            try write(
                textRows(["dev one", "dev two", "dev three"]),
                to: root.appending(
                    components: "prompts", "dev", "dev-prompts.jsonl"))

            let sets = DatasetInventory.scan(root: root).filter { $0.kind == .promptSet }
            #expect(sets.count == 2)

            let task = try #require(sets.first { $0.familyLabel == "task" })
            #expect(task.name == "tidal-power-prompts")
            #expect(task.itemCount == 2)
            #expect(task.issue == nil)
            #expect(task.note?.contains("taskPromptsHash") == true)

            let dev = try #require(sets.first { $0.familyLabel == "dev" })
            #expect(dev.name == "dev-prompts")
            #expect(dev.itemCount == 3)
            #expect(dev.note?.contains("devPromptsHash") == true)
            // The residual, stated on the row rather than papered over.
            #expect(dev.note?.contains("DEFAULT task-prompt file") == true)
        }
    }

    /// Both prompt-set hashes are the digests their pins write: the task
    /// set's is `taskPromptsHash`, the dev set's is `sweep.devPromptsHash`.
    @Test func promptSetHashesAreThePinsHashes() throws {
        try withTempWorkspace { root in
            try write(
                #"{"id": "case-1", "prompt": "Decide."}"# + "\n",
                to: root.appending(components: "prompts", "tasks", "study-prompts.jsonl"))
            try write(
                textRows(["dev one"]),
                to: root.appending(components: "prompts", "dev", "dev-prompts.jsonl"))

            let sets = DatasetInventory.scan(root: root).filter { $0.kind == .promptSet }
            let task = try #require(sets.first { $0.familyLabel == "task" })
            let dev = try #require(sets.first { $0.familyLabel == "dev" })

            var manifest = ExperimentManifest(name: "s", description: "", modelID: "test/model")
            let pinnedTask = try ExperimentStore.pinTaskPrompts(
                "prompts/tasks/study-prompts.jsonl", into: &manifest)
            #expect(task.contentHash == pinnedTask)
            #expect(task.contentHash == manifest.taskPromptsHash)

            manifest.sweep = ExperimentManifest.SweepSpec(
                devPromptsFile: "prompts/dev/dev-prompts.jsonl")
            ExperimentStore.pinSweepInputs(into: &manifest)
            #expect(dev.contentHash == manifest.sweep?.devPromptsHash)
        }
    }

    /// The shape divergence is real and reported, not hidden: a dev file must
    /// be `{"text": …}` rows, so a task-shaped file filed under prompts/dev/
    /// is an ISSUE — while its hash still shows.
    @Test func aPromptSetTheFamilysLoaderRejectsIsReported() throws {
        try withTempWorkspace { root in
            try write(
                #"{"id": "x", "prompt": "task-shaped row"}"# + "\n",
                to: root.appending(components: "prompts", "dev", "misfiled.jsonl"))
            try write(
                "this is not JSONL at all\n",
                to: root.appending(components: "prompts", "tasks", "garbage.jsonl"))

            let sets = DatasetInventory.scan(root: root).filter { $0.kind == .promptSet }
            let dev = try #require(sets.first { $0.familyLabel == "dev" })
            #expect(dev.itemCount == nil)
            #expect(dev.issue != nil)
            #expect(dev.contentHash?.count == 64)

            let task = try #require(sets.first { $0.familyLabel == "task" })
            #expect(task.itemCount == nil)
            #expect(task.issue != nil)
        }
    }

    /// A workspace with only the seeded instruments (no concepts) shows them
    /// as instruments — batteries and prompt sets after the concept-scoped
    /// families, never mixed into them.
    @Test func instrumentFamiliesSortAfterTheConceptScopedOnes() throws {
        try withTempWorkspace { root in
            _ = try makeConcept("courage", in: root)
            try makeNeutralCorpus(in: root)
            try write(
                battery([Self.legacyItem]),
                to: root.appending(components: "prompts", "batteries", "basic.jsonl"))
            try write(
                textRows(["dev one"]),
                to: root.appending(components: "prompts", "dev", "dev-prompts.jsonl"))

            #expect(
                DatasetInventory.scan(root: root).map(\.kind) == [
                    .conceptStimuli, .neutralCorpus, .promptSet, .capabilityBattery,
                ])
        }
    }

    // MARK: The observable model

    @Test func theModelScansTheResolvedWorkspaceOffTheMainActor() async throws {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "inventory-model-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        _ = try makeConcept("observable", in: root)

        ExperimentRootOverrideLock.acquire()
        let previous = WorkspaceRoot.programmaticOverride
        WorkspaceRoot.programmaticOverride = root
        defer {
            WorkspaceRoot.programmaticOverride = previous
            ExperimentRootOverrideLock.release()
        }

        let model = DatasetInventoryModel()
        #expect(model.hasScanned == false)
        #expect(model.entries.isEmpty)

        model.refresh()
        // The scan runs detached; poll the main-actor state it publishes.
        for _ in 0 ..< 200 where !model.hasScanned {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(model.hasScanned)
        #expect(model.isScanning == false)
        #expect(model.scannedRoot?.standardizedFileURL == root.standardizedFileURL)
        #expect(model.entries.map(\.name) == ["observable"])
        #expect(model.presentKinds == [.conceptStimuli])
        #expect(model.issueCount == 0)
        // ONE refresh covers BOTH scopes (phase 3): the scope switch is a
        // view choice, never a reason to re-scan — so the single existing
        // refresh seam keeps the derived table current too.
        #expect(model.derived.isEmpty)
        #expect(model.presentDerivedKinds.isEmpty)
    }

    /// The derived half of that same refresh: one scan populates both lists.
    @Test func oneRefreshPopulatesBothScopes() async throws {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "inventory-both-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        _ = try makeConcept("courage", in: root)
        let run = root.appending(components: "runs", "2026-01-01-extract")
        try write("", to: run.appending(component: "courage-v1.safetensors"))
        try write(
            """
            {"modelID": "test/model", "concept": "courage", \
            "stimulusSetHash": "abc", "layerCount": 1, "hiddenSize": 4, \
            "normsPerLayer": [1.0], "extractionDate": "2026-01-01T00:00:00Z"}
            """,
            to: run.appending(component: "courage-v1.json"))

        ExperimentRootOverrideLock.acquire()
        let previous = WorkspaceRoot.programmaticOverride
        WorkspaceRoot.programmaticOverride = root
        defer {
            WorkspaceRoot.programmaticOverride = previous
            ExperimentRootOverrideLock.release()
        }

        let model = DatasetInventoryModel()
        model.refresh()
        for _ in 0 ..< 200 where !model.hasScanned {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(model.entries.map(\.name) == ["courage"])
        #expect(model.derived.map(\.name) == ["courage-v1"])
        #expect(model.presentDerivedKinds == [.steeringVector])
    }
}
