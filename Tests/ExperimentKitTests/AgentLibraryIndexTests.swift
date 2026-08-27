import Foundation
import Testing

@testable import ExperimentKit

/// The Agent Library's list layer: what a ROW costs, and what it deliberately
/// does not.
///
/// The regression these guard (2026-08-27): switching to the Agents section
/// scanned the library, walked `runs/` for robustness reports, and SHA-256'd
/// every saved agent's file — all synchronously, before the tab could draw.
/// The split is now index (rows) vs evidence (the expensive overlay), and the
/// `LoadStats` counting seam is the only way to assert from the outside that
/// the index did not quietly do the evidence pass's work.
///
/// Everything here is hermetic: synthetic agents in a temp directory, roots
/// passed explicitly, no workspace override and no model.
@Suite struct AgentLibraryIndexTests {

    // MARK: - Fixtures

    private func withDirectory<T>(_ body: (URL) throws -> T) rethrows -> T {
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "agent-index-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temp) }
        return try body(temp)
    }

    private func makeArtifact(
        name: String,
        createdAt: String = "2026-08-01T09:15:03Z",
        adapters: [ModelVariantArtifact.AdapterRef] = [],
        injections: [ModelVariantArtifact.InjectionRef] = [],
        promotion: ModelVariantArtifact.Promotion? = nil
    ) -> ModelVariantArtifact {
        var artifact = ModelVariantArtifact(
            name: name,
            baseModelID: "test/base-model",
            adapters: adapters,
            injections: injections,
            promptMode: "chatAssistant",
            qwenThinkingEnabled: false,
            temperature: 0,
            systemPrompt: "",
            promotion: promotion)
        artifact.createdAt = createdAt
        return artifact
    }

    private func makePromotion(
        promotedBy: String = "criterion",
        overrideReason: String? = nil
    ) -> ModelVariantArtifact.Promotion {
        ModelVariantArtifact.Promotion(
            experiment: "batch-1",
            experimentHash: "abc123",
            promotedAt: "2026-08-01T09:00:00Z",
            promotedBy: promotedBy,
            overrideReason: overrideReason,
            sweepRun: "2026-08-01-batch-1-sweep",
            criterion: ExperimentManifest.SweepSelection(
                objective: .init(metric: "judgeScore")),
            winningCell: .init(layer: 14, alpha: 1.5),
            substrate: "test-substrate",
            appVersion: "test")
    }

    /// Plant one agent in the native library layout.
    @discardableResult
    private func plant(
        _ artifact: ModelVariantArtifact, in library: URL
    ) throws -> URL {
        let directory = library.appending(
            component: "model-variant-\(artifact.name)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let url = directory.appending(component: "model-variant.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(artifact).write(to: url)
        return url
    }

    private func makeReport(
        variantName: String,
        artifactHash: String?,
        generatedAt: String = "2026-08-02T10:00:00Z"
    ) -> VariantRobustnessReport {
        VariantRobustnessReport(
            variantName: variantName,
            variantArtifactPath: nil,
            variantArtifactHash: artifactHash,
            baseModelID: "test/base-model",
            substrate: "test-substrate",
            presetID: nil,
            batteryFile: "battery.jsonl",
            coherencePromptsFile: "prompts.jsonl",
            judgeModel: nil,
            generatedAt: generatedAt,
            baselineBatteryAccuracy: 0.95,
            variantBatteryAccuracy: 0.92,
            meanBaselineDistinct2: 0.8,
            meanVariantDistinct2: 0.78,
            meanBaselineWords: 100,
            meanVariantWords: 98,
            batteryItems: [],
            coherenceItems: [],
            warnings: [])
    }

    private func plantReport(
        _ report: VariantRobustnessReport, in runs: URL, runID: String
    ) throws {
        let directory = runs.appending(component: runID)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try JSONEncoder().encode(report).write(
            to: directory.appending(component: VariantRobustness.reportFileName))
    }

    // MARK: - The index does the list's work and no more

    @Test func loadingManyAgentsTouchesOnlyTheIndex() throws {
        try withDirectory { library in
            for index in 0 ..< 25 {
                try plant(
                    makeArtifact(
                        name: "agent-\(index)",
                        promotion: index.isMultiple(of: 2)
                            ? makePromotion() : nil),
                    in: library)
            }

            let snapshot = AgentLibraryIndex.load(
                directory: library, importedRoot: nil, root: library)

            #expect(snapshot.entries.count == 25)
            // One decode per agent — the list cannot know a name or a kind
            // without it…
            #expect(snapshot.stats.artifactsDecoded == 25)
            // …and nothing else. These two are the evidence pass's cost, and
            // the tab-switch path must never pay them.
            #expect(snapshot.stats.artifactsHashed == 0)
            #expect(snapshot.stats.robustnessReportsRead == 0)
        }
    }

    /// A `runs/` tree full of robustness reports must not be read by the
    /// index — the list does not render them, the overlay does.
    @Test func theIndexIgnoresTheRunsTreeEntirely() throws {
        try withDirectory { root in
            let library = root.appending(component: "model-variants")
            let runs = root.appending(component: "runs")
            try FileManager.default.createDirectory(
                at: library, withIntermediateDirectories: true)
            try plant(makeArtifact(name: "agent-a"), in: library)
            for index in 0 ..< 10 {
                try plantReport(
                    makeReport(variantName: "agent-a", artifactHash: nil),
                    in: runs, runID: "run-\(index)")
            }

            let snapshot = AgentLibraryIndex.load(
                directory: library, importedRoot: nil, root: root)

            #expect(snapshot.entries.count == 1)
            #expect(snapshot.stats.robustnessReportsRead == 0)
            #expect(snapshot.stats.artifactsHashed == 0)
        }
    }

    // MARK: - A row renders from its summary alone

    @Test func theSummaryCarriesEverythingARowDisplays() throws {
        try withDirectory { library in
            let adapter = ModelVariantArtifact.AdapterRef(
                name: "adapter-a",
                artifactPath: "adapters/a/adapter.safetensors",
                adapterDirectory: "adapters/a",
                adapterHash: "hash-a")
            let injection = ModelVariantArtifact.InjectionRef(
                concept: "concept-a",
                vectorArtifactID: "runs/extract-1/concept-a",
                layer: 14,
                alpha: 1.5)
            let artifact = makeArtifact(
                name: "agent-a", adapters: [adapter], injections: [injection],
                promotion: makePromotion())
            try plant(artifact, in: library)

            let entry = try #require(
                AgentLibraryIndex.load(
                    directory: library, importedRoot: nil, root: library
                ).entries.first)

            #expect(entry.name == "agent-a")
            #expect(entry.baseModelID == "test/base-model")
            #expect(entry.dateLabel == "2026-08-01 09:15")
            #expect(entry.kind == .sweepPromoted)
            #expect(entry.componentsSummary == "1 adapter · 1 vector")
            #expect(entry.promotionLine?.contains("from 'batch-1'") == true)
            #expect(entry.promotionLine?.contains("criterion judgeScore") == true)
            #expect(entry.promotionLine?.contains("L14 α1.5") == true)
            #expect(entry.promotedExperiment == "batch-1")
            #expect(entry.overrideReason == nil)
        }
    }

    /// Anti-drift: the summary's derived fields must agree with the rules
    /// applied to the full artifact. If someone teaches `AgentLibrary` a new
    /// distinction and forgets the summary, this fails rather than the list
    /// quietly disagreeing with the editor.
    @Test func theSummaryAgreesWithTheArtifactRules() throws {
        let cases: [ModelVariantArtifact] = [
            makeArtifact(name: "plain"),
            makeArtifact(
                name: "vector-only",
                injections: [
                    .init(
                        concept: "c", vectorArtifactID: "runs/x/c", layer: 3,
                        alpha: 1)
                ]),
            makeArtifact(
                name: "adapter-only",
                adapters: [
                    .init(
                        name: "a", artifactPath: "adapters/a/w.safetensors",
                        adapterDirectory: "adapters/a")
                ]),
            makeArtifact(name: "promoted", promotion: makePromotion()),
            makeArtifact(
                name: "override",
                promotion: makePromotion(
                    promotedBy: "manualOverride", overrideReason: "documented")),
        ]
        let availability = AgentLibrary.Availability(
            localModelIDs: ["test/base-model"],
            localVectorIDs: ["runs/x/c"],
            localAdapterKeys: ["adapters/a"])

        try withDirectory { library in
            for artifact in cases { try plant(artifact, in: library) }
            let entries = AgentLibraryIndex.load(
                directory: library, importedRoot: nil, root: library
            ).entries
            #expect(entries.count == cases.count)

            for artifact in cases {
                let entry = try #require(
                    entries.first { $0.name == artifact.name })
                #expect(entry.kind == AgentLibrary.kind(of: artifact))
                #expect(
                    entry.componentsSummary
                        == AgentLibrary.componentsSummary(artifact))
                #expect(
                    AgentLibrary.chips(
                        for: entry.components, availability: availability)
                        == AgentLibrary.chips(
                            for: artifact, availability: availability))
                #expect(
                    AgentLibrary.isRunnableLocally(
                        entry.components, availability: availability)
                        == AgentLibrary.isRunnableLocally(
                            artifact, availability: availability))
            }
        }
    }

    // MARK: - The evidence pass is where the expensive work lives

    @Test func theEvidencePassHashesAndReadsReports() throws {
        try withDirectory { root in
            let library = root.appending(component: "model-variants")
            let runs = root.appending(component: "runs")
            try FileManager.default.createDirectory(
                at: library, withIntermediateDirectories: true)
            let url = try plant(makeArtifact(name: "agent-a"), in: library)
            let hash = try ModelVariantStore.hash(url)
            try plantReport(
                makeReport(variantName: "agent-a", artifactHash: hash),
                in: runs, runID: "run-1")
            try plantReport(
                makeReport(
                    variantName: "someone-else", artifactHash: "other",
                    generatedAt: "2026-08-03T10:00:00Z"),
                in: runs, runID: "run-2")

            let entries = AgentLibraryIndex.load(
                directory: library, importedRoot: nil, root: root
            ).entries
            let evidence = AgentLibraryIndex.evidence(
                for: entries, runsDirectory: runs)

            #expect(evidence.stats.artifactsHashed == 1)
            #expect(evidence.stats.robustnessReportsRead == 2)
            let matched = try #require(evidence.byEntryID[entries[0].id])
            #expect(matched.report.variantName == "agent-a")
            #expect(matched.runDirectory.lastPathComponent == "run-1")
        }
    }

    @Test func anAgentWithNoReportSimplyHasNoEvidence() throws {
        try withDirectory { root in
            let library = root.appending(component: "model-variants")
            let runs = root.appending(component: "runs")
            try FileManager.default.createDirectory(
                at: library, withIntermediateDirectories: true)
            try plant(makeArtifact(name: "agent-a"), in: library)

            let entries = AgentLibraryIndex.load(
                directory: library, importedRoot: nil, root: root
            ).entries
            let evidence = AgentLibraryIndex.evidence(
                for: entries, runsDirectory: runs)

            #expect(evidence.byEntryID.isEmpty)
            // Absent evidence is still one hash per agent — the cost the tab
            // switch no longer pays up front.
            #expect(evidence.stats.artifactsHashed == 1)
            #expect(evidence.stats.robustnessReportsRead == 0)
        }
    }

    // MARK: - Filtering asks the expensive question only when it must

    /// `runnableHere` is a full availability judgement (and, on a server, a
    /// catalog round of the applicability rule). The list evaluated it for
    /// every row on every body evaluation even with the filter off.
    @Test func runnabilityIsOnlyAskedWhenTheFilterIsOn() {
        let components = AgentLibrary.Components(baseModelID: "test/base-model")
        var asked = 0
        let ask: () -> Bool = {
            asked += 1
            return true
        }

        _ = AgentLibrary.matches(
            components, filter: AgentLibrary.Filter(), runnableHere: ask)
        #expect(asked == 0)

        _ = AgentLibrary.matches(
            components, filter: AgentLibrary.Filter(sweepPromotedOnly: true),
            runnableHere: ask)
        #expect(asked == 0)

        _ = AgentLibrary.matches(
            components, filter: AgentLibrary.Filter(runnableHere: true),
            runnableHere: ask)
        #expect(asked == 1)
    }

    @Test func baselineRowsAlsoDeferTheQuestion() {
        let row = AgentLibrary.BaselineRow(baseModelID: "test/base-model")
        var asked = 0
        let ask: () -> Bool = {
            asked += 1
            return true
        }

        _ = AgentLibrary.matches(
            baseline: row, filter: AgentLibrary.Filter(), runnableHere: ask)
        #expect(asked == 0)

        _ = AgentLibrary.matches(
            baseline: row, filter: AgentLibrary.Filter(runnableHere: true),
            runnableHere: ask)
        #expect(asked == 1)
    }

    @Test func baselineRowsDeriveFromBaseModelIDsAlone() {
        let rows = AgentLibrary.baselineRows(
            forBaseModelIDs: ["b/two", "a/one", "b/two"])
        #expect(rows.map(\.baseModelID) == ["a/one", "b/two"])
    }

    // MARK: - The evidence overlay's latest-wins rule (review round 6, #6)

    /// The Library starts the evidence pass TWICE by design: once in `.task`,
    /// for the rows carried over from a previous visit, and again from
    /// `onChange(of: agentIndex)` when the rescan reports new ones. Both used
    /// to capture the library scan's token, which cannot tell them apart, so
    /// whichever `runs/` walk finished last won — including the older one,
    /// over evidence computed for a NEWER snapshot.
    @Test func aSupersededEvidencePassMayNotLandOverANewerOne() {
        let root = URL(filePath: "/w")
        let ids: Set<String> = ["a", "b"]
        // The second pass, landing against its own state: applied.
        #expect(
            FineTuningPanel.evidenceMayLand(
                scanToken: 3, liveScanToken: 3,
                evidenceToken: 2, liveEvidenceToken: 2,
                root: root, liveRoot: root,
                entryIDs: ids, liveEntryIDs: ids))
        // The FIRST pass, finishing after the second started: dropped. Under
        // the shared token both of these read as "token 3 == token 3".
        #expect(
            !FineTuningPanel.evidenceMayLand(
                scanToken: 3, liveScanToken: 3,
                evidenceToken: 1, liveEvidenceToken: 2,
                root: root, liveRoot: root,
                entryIDs: ids, liveEntryIDs: ids))
    }

    @Test func evidenceForAnotherWorkspaceOrAnotherRowSetIsDropped() {
        let root = URL(filePath: "/w")
        let ids: Set<String> = ["a", "b"]
        // A library rescan superseded the walk.
        #expect(
            !FineTuningPanel.evidenceMayLand(
                scanToken: 3, liveScanToken: 4,
                evidenceToken: 1, liveEvidenceToken: 1,
                root: root, liveRoot: root,
                entryIDs: ids, liveEntryIDs: ids))
        // The workspace moved under it.
        #expect(
            !FineTuningPanel.evidenceMayLand(
                scanToken: 3, liveScanToken: 3,
                evidenceToken: 1, liveEvidenceToken: 1,
                root: root, liveRoot: URL(filePath: "/other"),
                entryIDs: ids, liveEntryIDs: ids))
        // The rows changed without either counter moving — evidence keyed on
        // ids that are no longer on screen renders under the wrong agent.
        #expect(
            !FineTuningPanel.evidenceMayLand(
                scanToken: 3, liveScanToken: 3,
                evidenceToken: 1, liveEvidenceToken: 1,
                root: root, liveRoot: root,
                entryIDs: ids, liveEntryIDs: ["a", "c"]))
    }
}
