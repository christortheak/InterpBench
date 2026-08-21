import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// The Data workbench's retirement pass (phase 4 of 4).
///
/// Three claims, each of which was a real hole before this pass:
///
/// 1. **One path authority for the paired roots.** `prompts/repe/<name>/` and
///    `prompts/readers/<name>/pairs.jsonl` were assembled inline in five
///    places inside `ConceptBuilder` plus the inventory's own family loop.
///    `VectorCatalog.pairedStimuli*` is now the single spelling, and the
///    tests below pin the literal paths so a drift in either direction fails
///    here rather than silently orphaning a dataset.
/// 2. **One creation entry.** The Concepts & Vectors builder's own
///    from-scratch concept form is retired; the role-first sheet drives
///    `ConceptBuilder.saveNewConcept()` instead, and everything the builder
///    could EDIT afterwards still works.
/// 3. **Selection seams.** The Analysis and Agents routes moved their
///    selections onto the panel models, so a derived row can preselect the
///    artifact it names instead of switching sections and asking the
///    researcher to find it again.
@Suite(.serialized) @MainActor
struct DataWorkbenchPhase4Tests {

    // MARK: Harness

    private func withTempWorkspace<T>(_ body: (URL) throws -> T) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "data-workbench-4") { root in
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

    private func makeConcept(_ name: String, in root: URL, rows: Int = 5) throws {
        let directory = root.appending(components: "prompts", "concepts", name)
        try write(
            textRows((0 ..< rows).map { "\(name) positive \($0)" }),
            to: directory.appending(component: "positive.jsonl"))
        try write(
            textRows((0 ..< rows).map { "\(name) negative \($0)" }),
            to: directory.appending(component: "negative.jsonl"))
    }

    // MARK: 1 — the paired path authority

    /// The literal spellings, pinned. Every retired call site built one of
    /// these by hand; if the authority ever disagrees with them, the datasets
    /// already on disk stop being found.
    @Test func theAuthorityReproducesEveryRetiredCallSitesPath() throws {
        try withTempWorkspace { root in
            let name = "tidiness"

            // The concept-delete sweep's root (was: projectRoot + prompts,
            // repe, name).
            #expect(
                VectorCatalog.pairedStimuliDirectory(family: .repe, name: name, root: root)
                    == root.appending(components: "prompts", "repe", name))
            // The on-disk recipe probe's two files (was: projectRoot +
            // prompts, {repe,readers}, name, pairs.jsonl).
            #expect(
                VectorCatalog.pairedStimuliFile(family: .repe, name: name, root: root)
                    == root.appending(
                        components: "prompts", "repe", name, "pairs.jsonl"))
            #expect(
                VectorCatalog.pairedStimuliFile(family: .readers, name: name, root: root)
                    == root.appending(
                        components: "prompts", "readers", name, "pairs.jsonl"))
            // The two reader-fit writers' directory (was: projectRoot +
            // prompts, readers, name).
            #expect(
                VectorCatalog.pairedStimuliDirectory(
                    family: .readers, name: name, root: root)
                    == root.appending(components: "prompts", "readers", name))
            // The RepE build's provenance stamp (was the string literal
            // "prompts/repe/\(name)/pairs.jsonl").
            #expect(
                VectorCatalog.pairedStimuliRelativePath(family: .repe, name: name)
                    == "prompts/repe/\(name)/pairs.jsonl")
            #expect(
                VectorCatalog.pairedStimuliRelativePath(family: .readers, name: name)
                    == "prompts/readers/\(name)/pairs.jsonl")
            // And the family roots the inventory's loop used to name as bare
            // strings.
            #expect(
                VectorCatalog.PairedStimulusFamily.allCases.map(\.label)
                    == ["repe", "readers"])
            #expect(
                VectorCatalog.PairedStimulusFamily.allCases.map(\.relativeDirectory)
                    == ["prompts/repe", "prompts/readers"])
            #expect(VectorCatalog.pairedStimuliFileName == "pairs.jsonl")

            // The relative path and the URL are two views of ONE rule.
            for family in VectorCatalog.PairedStimulusFamily.allCases {
                #expect(
                    VectorCatalog.pairedStimuliFile(
                        family: family, name: name, root: root)
                        == root.appending(
                            path: VectorCatalog.pairedStimuliRelativePath(
                                family: family, name: name)))
            }
        }
    }

    /// The READER of those paths agrees with the authority: a mirror written
    /// through it is the mirror `ConceptBuilder.pairedRecipeFamilyOnDisk`
    /// finds, including the most-recently-written tie-break.
    @Test func theOnDiskRecipeProbeReadsTheAuthoritysPaths() throws {
        try withTempWorkspace { root in
            let builder = ConceptBuilder()
            try makeConcept("tidiness", in: root)
            #expect(
                ConceptBuilder.pairedRecipeFamilyOnDisk(for: "tidiness")
                    == .caaMeanDifference)

            let repePairs = VectorCatalog.pairedStimuliFile(
                family: .repe, name: "tidiness", root: root)
            try write(#"{"positive": "p", "negative": "n"}"# + "\n", to: repePairs)
            #expect(
                ConceptBuilder.pairedRecipeFamilyOnDisk(for: "tidiness") == .repeLAT)

            // Both mirrors present: the probe breaks the tie by modification
            // time ("most recently written wins"), which is exactly what the
            // new paired-set advisory warns about.
            try write(
                #"{"concept": "tidiness", "positiveStimulus": "p", "#
                    + #""negativeStimulus": "n", "templateID": "t1"}"# + "\n",
                to: VectorCatalog.pairedStimuliFile(
                    family: .readers, name: "tidiness", root: root))
            try FileManager.default.setAttributes(
                [.modificationDate: Date(timeIntervalSince1970: 4_000_000)],
                ofItemAtPath: repePairs.path)
            #expect(
                ConceptBuilder.pairedRecipeFamilyOnDisk(for: "tidiness")
                    == .repeReaderLAT)
            _ = builder
        }
    }

    /// The delete sweep's scope, unchanged by the move to the authority: the
    /// RepE mirror goes with the concept's editable datasets, the READERS
    /// mirror does not (a fitted reader's pinned dataset outlives them).
    @Test func conceptDeleteSweepsTheAuthoritysRepeMirrorOnly() throws {
        try withTempWorkspace { root in
            try makeConcept("tidiness", in: root)
            let repe = VectorCatalog.pairedStimuliFile(
                family: .repe, name: "tidiness", root: root)
            let readers = VectorCatalog.pairedStimuliFile(
                family: .readers, name: "tidiness", root: root)
            try write(#"{"positive": "p", "negative": "n"}"# + "\n", to: repe)
            try write(
                #"{"concept": "tidiness", "positiveStimulus": "p", "#
                    + #""negativeStimulus": "n", "templateID": "t1"}"# + "\n",
                to: readers)

            let builder = ConceptBuilder()
            builder.selectConcept("tidiness")
            builder.deleteSelectedConcept()

            let fm = FileManager.default
            #expect(!fm.fileExists(atPath: repe.path))
            #expect(fm.fileExists(atPath: readers.path))
            #expect(
                !fm.fileExists(
                    atPath: root.appending(
                        components: "prompts", "concepts", "tidiness").path))
        }
    }

    /// The inventory's family loop resolves through the same authority — and
    /// counts each family with ITS loader, so a dataset the creation flow
    /// just validated is not listed as malformed.
    @Test func theInventoryReadsBothPairedRootsThroughTheAuthority() throws {
        try withTempWorkspace { root in
            try DatasetCreationPlanner.plan(
                DatasetCreationRequest(
                    role: .pairedStimuli, rawName: "tidiness", pairedFamily: .repe),
                root: root
            ).createDirectory()
            try write(
                #"{"positive": "p1", "negative": "n1"}"# + "\n"
                    + #"{"positive": "p2", "negative": "n2"}"# + "\n",
                to: VectorCatalog.pairedStimuliFile(
                    family: .repe, name: "tidiness", root: root))
            try write(
                #"{"concept": "brevity", "positiveStimulus": "p", "#
                    + #""negativeStimulus": "n", "templateID": "t1"}"# + "\n",
                to: VectorCatalog.pairedStimuliFile(
                    family: .readers, name: "brevity", root: root))

            let paired = DatasetInventory.scan(root: root)
                .filter { $0.kind == .pairedStimuli }
            #expect(paired.count == 2)
            #expect(paired.allSatisfy { $0.issue == nil })
            #expect(paired.first { $0.familyLabel == "repe" }?.itemCount == 2)
            #expect(paired.first { $0.familyLabel == "readers" }?.itemCount == 1)
            #expect(
                paired.first { $0.familyLabel == "readers" }?.name == "brevity")
        }
    }

    // MARK: 2 — the retired creation entry

    /// The sheet's "author in the builder" path is exactly what the retired
    /// in-panel form did — `saveNewConcept()` over the plan's name — and the
    /// concept it produces is editable in the builder exactly as before.
    @Test func creationThroughTheSheetLeavesTheConceptEditableInTheBuilder() async throws {
        // Manual root override rather than `withTempWorkspace`: the editing
        // half is async (the builder's row importer), and the harness closure
        // is synchronous.
        let root = FileManager.default.temporaryDirectory
            .appending(component: "data-workbench-4-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        ExperimentRootOverrideLock.acquire()
        let previous = WorkspaceRoot.programmaticOverride
        WorkspaceRoot.programmaticOverride = root
        ExperimentStore.rootOverride = root
        defer {
            ExperimentStore.rootOverride = nil
            WorkspaceRoot.programmaticOverride = previous
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: root)
        }

        do {
            let plan = DatasetCreationPlanner.plan(
                DatasetCreationRequest(role: .conceptStimuli, rawName: "Tidiness Set"),
                root: root)
            #expect(plan.name == "tidiness-set")

            // Exactly the sheet's `authorInBuilder` sequence.
            let builder = ConceptBuilder()
            try plan.createDirectory()
            builder.conceptName = plan.name
            builder.saveNewConcept()

            // Selectable, and BOTH of the panel's selections landed on it.
            #expect(builder.existingConcepts.contains("tidiness-set"))
            #expect(builder.selectedExisting == "tidiness-set")
            #expect(builder.vectorBuilderSelectedExisting == "tidiness-set")
            #expect(builder.currentConceptName == "tidiness-set")

            // Nothing seeded: the concept primitive and nothing else.
            let contents = try FileManager.default.contentsOfDirectory(
                atPath: plan.directory.path)
            #expect(contents == ["concept.json"])

            // And still EDITABLE: the builder's own row editor writes into
            // the concept the sheet created.
            builder.probeDraft =
                #"{"text": "probe row", "expresses": true}"# + "\n"
            await builder.addProbeDrafts()
            #expect(builder.probeExamples.count == 1)
            #expect(
                FileManager.default.fileExists(
                    atPath: root.appending(
                        components: "prompts", "probes", "tidiness-set",
                        "items.jsonl").path))
        }
    }

    /// The retired form's own seam is INTACT — the sheet is what drives it
    /// now, and the on-disk result is identical either way.
    @Test func theSheetPathAndTheRetiredFormWriteTheSameConceptPrimitive() throws {
        try withTempWorkspace { root in
            let builder = ConceptBuilder()
            builder.conceptName = "brevity"
            builder.saveNewConcept()
            let direct = try Data(
                contentsOf: root.appending(
                    components: "prompts", "concepts", "brevity", "concept.json"))

            let plan = DatasetCreationPlanner.plan(
                DatasetCreationRequest(role: .conceptStimuli, rawName: "gadgetry"),
                root: root)
            try plan.createDirectory()
            builder.conceptName = plan.name
            builder.saveNewConcept()
            let viaSheet = try Data(
                contentsOf: plan.directory.appending(component: "concept.json"))

            func shape(_ data: Data) throws -> Set<String> {
                let object =
                    try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
                return Set(object.keys)
            }
            #expect(try shape(direct) == shape(viaSheet))
        }
    }

    /// Every kind the inventory lists is either CREATABLE through the one
    /// flow or carries a reason — the phase-4 coherence promise. `promptSet`
    /// is the one deliberate absence, and phase 4 is also what made its rows
    /// name the study that pins them.
    @Test func everyInventoryKindIsCreatableOrHasAStatedReason() {
        let creatable = Set(DatasetRole.allCases.map(\.kind))
        let uncovered = Set(DatasetKind.allCases).subtracting(creatable)
        #expect(uncovered == [.promptSet])
        // The reason is stated where a researcher meets it.
        #expect(
            DatasetDerivation.noDerivationReason(
                for: DatasetInventoryEntry(
                    kind: .promptSet, name: "x", directory: URL(filePath: "/tmp"),
                    files: [], itemCount: nil, byteSize: 0, modified: nil,
                    contentHash: nil, issue: nil, conceptName: nil)
            )?.contains("Pin it on the study") == true)
        // Paired stimulus sets are creatable now — phase 2's stated gap.
        #expect(creatable.contains(.pairedStimuli))
        // Every role the builder can author rows for offers the hand-back.
        #expect(
            DatasetRole.allCases.filter(\.authorsInConceptBuilder)
                == [.conceptStimuli, .pairedStimuli, .storyCorpus, .probeItems])
    }

    /// The one seam every route into a concept selection uses.
    @Test func selectConceptRefreshesTheIndexThenMovesBothSelections() throws {
        try withTempWorkspace { root in
            let builder = ConceptBuilder()
            #expect(builder.selectConcept("tidiness") == false)
            #expect(builder.selectedExisting == nil)

            // Authored AFTER the builder existed — the seam re-scans, which
            // is why a concept the sheet just created is selectable.
            try makeConcept("tidiness", in: root)
            #expect(builder.selectConcept("tidiness"))
            #expect(builder.selectedExisting == "tidiness")
            #expect(builder.vectorBuilderSelectedExisting == "tidiness")

            // Re-routing to the already-selected concept still lands the
            // build target on it.
            builder.vectorBuilderSelectedExisting = nil
            #expect(builder.selectConcept("tidiness"))
            #expect(builder.vectorBuilderSelectedExisting == "tidiness")
        }
    }

    // MARK: 3 — the selection seams

    @discardableResult
    private func makeVector(
        _ name: String, in root: URL, model: String = "test/model"
    ) throws -> URL {
        let directory = root.appending(components: "runs", "2026-01-01-extract")
        try write("", to: directory.appending(component: "\(name).safetensors"))
        try write(
            """
            {"modelID": "\(model)", "concept": "tidiness", \
            "stimulusSetHash": "abc123", "layerCount": 3, "hiddenSize": 8, \
            "normsPerLayer": [1.0, 1.0, 1.0], \
            "extractionDate": "2026-01-01T00:00:00Z", \
            "extractionMethod": "meanDifference", "revision": "deadbeef", \
            "readingPosition": "last token"}
            """,
            to: directory.appending(component: "\(name).json"))
        return directory
    }

    private func makeAgent(_ name: String, in root: URL) throws {
        try write(
            """
            {"name": "\(name)", "baseModelID": "test/model", \
            "createdAt": "2026-01-05T00:00:00Z"}
            """,
            to: root.appending(
                components: "runs", "model-variants", "run-\(name)",
                "model-variant.json"))
    }

    /// A derived row's `selectionKey` is the DESTINATION tool's own identity
    /// formula, not a new one — which is the whole reason the route can
    /// preselect. A vector's key drops the `.safetensors`; an agent's key is
    /// its artifact path.
    @Test func selectionKeysMatchTheDestinationToolsOwnIdentities() throws {
        try withTempWorkspace { root in
            try makeVector("tidiness-v1", in: root)
            try makeAgent("agent-1", in: root)

            let entries = DerivedArtifactInventory.scan(root: root)
            let vectorEntry = try #require(entries.first { $0.kind == .steeringVector })
            let agentEntry = try #require(entries.first { $0.kind == .agent })

            let artifact = try #require(
                VectorCatalog.scan(runsDirectory: VectorCatalog.runsDirectory(root: root))
                    .first)
            #expect(vectorEntry.selectionKey == artifact.id)

            let record = try #require(ModelVariantStore.scan().first)
            #expect(agentEntry.selectionKey == record.id)

            // Only the routes that preselect carry a key.
            #expect(DerivedArtifactRoute.analysis.preselects)
            #expect(DerivedArtifactRoute.agents.preselects)
            #expect(!DerivedArtifactRoute.adapterTraining.preselects)
            #expect(!DerivedArtifactRoute.conceptsAndVectors.preselects)
            for entry in entries {
                #expect((entry.selectionKey != nil) == (entry.route?.preselects == true))
            }
        }
    }

    /// Analysis: the selection moved onto `GeometryPanel`, so the route can
    /// set it. Changing the selection retires any table computed from the
    /// previous one — the phase-0 orphan rule, now enforced at the model
    /// layer rather than by the view's state dying on a section switch.
    @Test func theAnalysisSeamPreselectsAndRetiresAnOrphanedTable() throws {
        try withTempWorkspace { root in
            try makeVector("tidiness-v1", in: root)
            let entry = try #require(
                DerivedArtifactInventory.scan(root: root)
                    .first { $0.kind == .steeringVector })
            let key = try #require(entry.selectionKey)

            let geometry = GeometryPanel()
            #expect(geometry.selectedIDs.isEmpty)

            // What `DataSectionView.route(.derived(.analysis, …))` performs.
            geometry.select(vectorIDs: [key])
            #expect(geometry.selectedIDs == [key])

            // A stale layer/table from an earlier selection is dropped when
            // the selection changes; an identical re-selection is a no-op.
            geometry.selectedLayer = 7
            geometry.select(vectorIDs: [key])
            #expect(geometry.selectedLayer == 7)
            geometry.select(vectorIDs: [key, "other"])
            #expect(geometry.selectedLayer == 0)

            // An empty selection retires the table; a non-empty one keeps it.
            geometry.selectedLayer = 3
            geometry.retireOrphanedLocalTable()
            #expect(geometry.selectedLayer == 3)
            geometry.clearSelection()
            geometry.selectedLayer = 3
            geometry.retireOrphanedLocalTable()
            #expect(geometry.selectedLayer == 0)
        }
    }

    /// The one condition the Analysis help text still names: the pane's local
    /// list is filtered to the LOADED model, so a preselected vector from
    /// another model is selected-but-unlisted. Reported, never silently
    /// dropped.
    @Test func theAnalysisSeamReportsASelectionItsListCannotShow() throws {
        try withTempWorkspace { root in
            try makeVector("tidiness-v1", in: root)
            let artifacts = VectorCatalog.scan(
                runsDirectory: VectorCatalog.runsDirectory(root: root))
            let artifact = try #require(artifacts.first)

            let geometry = GeometryPanel()
            geometry.select(vectorIDs: [artifact.id])
            #expect(geometry.unlistableSelection(among: artifacts).isEmpty)
            // The other model is loaded, so nothing compatible is listed.
            #expect(geometry.unlistableSelection(among: []) == [artifact.id])
            #expect(
                DerivedArtifactRoute.analysis.help.contains("loaded model"))
        }
    }

    /// Agents: the selection was ALREADY model-side; what was missing was the
    /// refresh-then-set seam a route could call. It refuses an id the library
    /// does not hold rather than selecting into nothing.
    @Test func theAgentsSeamPreselectsTheLibraryRow() throws {
        try withTempWorkspace { root in
            let panel = FineTuningPanel()
            #expect(panel.selectAgent(id: "/nowhere/model-variant.json") == false)
            #expect(panel.selectedVariantID == nil)

            // Minted AFTER the panel existed — the seam re-scans.
            try makeAgent("agent-1", in: root)
            let entry = try #require(
                DerivedArtifactInventory.scan(root: root).first { $0.kind == .agent })
            let key = try #require(entry.selectionKey)

            #expect(panel.selectAgent(id: key))
            #expect(panel.selectedVariantID == key)
            #expect(panel.selectedVariant?.artifact.name == "agent-1")
            #expect(DerivedArtifactRoute.agents.help.contains("Library"))
        }
    }
}
