import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// `DatasetCreationPlanner` / `DatasetCreationPlan` (Data section phase 2):
/// the role-first creation flow's model layer.
///
/// The properties under test are the ones that make misfiling structurally
/// impossible: the destination comes from the engine's own path authorities,
/// an existing file is never silently overwritten, a rejected import leaves
/// nothing behind, and what lands is a row the phase-1 inventory finds.
@Suite(.serialized) @MainActor
struct DatasetCreationPlanTests {

    // MARK: Harness

    private func withTempWorkspace<T>(_ body: (URL) throws -> T) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "dataset-creation") { root in
            let previous = WorkspaceRoot.programmaticOverride
            WorkspaceRoot.programmaticOverride = root
            defer { WorkspaceRoot.programmaticOverride = previous }
            try? FileManager.default.createDirectory(
                at: root, withIntermediateDirectories: true)
            return try body(root)
        }
    }

    /// Source files live OUTSIDE the workspace — an import is bytes coming in
    /// from wherever the researcher had them.
    private func makeSource(_ contents: String, named name: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "dataset-creation-src-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let url = directory.appending(component: name)
        try contents.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    private func textRows(_ texts: [String]) -> String {
        texts.map { #"{"text": "\#($0)"}"# }.joined(separator: "\n") + "\n"
    }

    private func storyRows(concept: String, count: Int) -> String {
        (0 ..< count)
            .map { #"{"concept": "\#(concept)", "topic": "t\#($0)", "text": "story \#($0)"}"# }
            .joined(separator: "\n") + "\n"
    }

    private func labelledRows(_ count: Int) -> String {
        (0 ..< count)
            .map { #"{"text": "scenario \#($0)", "expresses": \#($0.isMultiple(of: 2))}"# }
            .joined(separator: "\n") + "\n"
    }

    /// A format-2 battery: header line, then choice items whose answer is
    /// among their options (what `PinShapeValidation` requires).
    private func batteryFile(items: Int) -> String {
        let header =
            #"{"batteryFormat": 2, "scoring": "choiceProbability", "maxTokens": 24}"#
        let rows = (0 ..< items).map {
            #"{"id": "cap-\#($0)", "prompt": "\#($0) + 1?", "answer": "\#($0 + 1)", "#
                + #""options": ["\#($0 + 1)", "\#($0 + 2)"]}"#
        }
        return ([header] + rows).joined(separator: "\n") + "\n"
    }

    private func request(
        _ role: DatasetRole,
        name: String = "",
        family: DatasetRecipeFamily? = nil,
        neutral: NeutralCorpusTarget? = nil,
        paired: VectorCatalog.PairedStimulusFamily? = nil
    ) -> DatasetCreationRequest {
        DatasetCreationRequest(
            role: role, rawName: name, recipeFamily: family, neutralTarget: neutral,
            pairedFamily: paired)
    }

    /// `StimulusSet.PairedStimulus` rows — the RepE/LAT mirror's shape.
    private func repePairRows(_ count: Int) -> String {
        (0 ..< count)
            .map { #"{"id": "p\#($0)", "positive": "yes \#($0)", "negative": "no \#($0)"}"# }
            .joined(separator: "\n") + "\n"
    }

    /// `RepEReader.Pair` rows — the reader dataset's shape. Same filename,
    /// different keys; neither loader reads the other's rows.
    private func readerPairRows(concept: String, count: Int) -> String {
        (0 ..< count)
            .map {
                #"{"concept": "\#(concept)", "positiveStimulus": "yes \#($0)", "#
                    + #""negativeStimulus": "no \#($0)", "split": "train", "#
                    + #""templateID": "scaffold-v1"}"#
            }
            .joined(separator: "\n") + "\n"
    }

    // MARK: Destinations come from the path authorities

    @Test func conceptStimuliLandUnderTheConceptsAuthority() throws {
        try withTempWorkspace { root in
            let plan = DatasetCreationPlanner.plan(
                request(.conceptStimuli, name: "sympathy"), root: root)
            #expect(plan.isResolved)
            #expect(
                plan.directory
                    == VectorCatalog.conceptsDirectory(root: root)
                        .appending(component: "sympathy"))
            #expect(
                plan.files.map(\.slot) == [.positiveStimuli, .negativeStimuli])
            #expect(
                plan.files.map(\.relativePath) == [
                    "prompts/concepts/sympathy/positive.jsonl",
                    "prompts/concepts/sympathy/negative.jsonl",
                ])
            #expect(plan.kind == .conceptStimuli)
        }
    }

    @Test func storyCorpusLandsUnderTheEmotionsAuthority() throws {
        try withTempWorkspace { root in
            let plan = DatasetCreationPlanner.plan(
                request(.storyCorpus, name: "sympathy"), root: root)
            #expect(
                plan.directory
                    == VectorCatalog.emotionsDirectory(root: root)
                        .appending(component: "sympathy"))
            #expect(
                plan.files.first?.relativePath
                    == "prompts/emotions/sympathy/stories.jsonl")
            #expect(plan.kind == .grandMeanCorpus)
        }
    }

    @Test func probeItemsLandUnderTheProbesAuthority() throws {
        try withTempWorkspace { root in
            let plan = DatasetCreationPlanner.plan(
                request(.probeItems, name: "sympathy"), root: root)
            #expect(
                plan.directory
                    == VectorCatalog.probesDirectory(root: root)
                        .appending(component: "sympathy"))
            #expect(
                plan.files.first?.relativePath == "prompts/probes/sympathy/items.jsonl")
        }
    }

    /// The recipe family is not a label — it picks the canonical ROOT, and
    /// the rule is `ExperimentStore.conceptValidationRelativePath`, the same
    /// one the pin/verify reader and the readiness checklist resolve through.
    @Test func validationSetsFollowTheRecipeFamilyToTheirRoot() throws {
        try withTempWorkspace { root in
            let paired = DatasetCreationPlanner.plan(
                request(.validationSet, name: "sympathy", family: .paired), root: root)
            let grandMean = DatasetCreationPlanner.plan(
                request(.validationSet, name: "sympathy", family: .grandMean),
                root: root)

            #expect(
                paired.files.first?.relativePath
                    == ExperimentStore.conceptValidationRelativePath(
                        name: "sympathy", isPaired: true))
            #expect(
                grandMean.files.first?.relativePath
                    == ExperimentStore.conceptValidationRelativePath(
                        name: "sympathy", isPaired: false))
            #expect(
                paired.files.first?.relativePath
                    == "prompts/concepts/sympathy/validation.jsonl")
            #expect(
                grandMean.files.first?.relativePath
                    == "prompts/emotions/sympathy/validation.jsonl")
            // Same kind, different directory: two DISTINCT inventory rows.
            #expect(paired.kind == .validationSet && grandMean.kind == .validationSet)
            #expect(paired.inventoryEntryID != grandMean.inventoryEntryID)
        }
    }

    @Test func neutralCorporaComeFromTheNeutralStore() throws {
        try withTempWorkspace { root in
            let calibration = DatasetCreationPlanner.plan(
                request(.neutralCorpus, neutral: .normCalibration), root: root)
            #expect(calibration.isResolved)
            // No name is asked for: the workspace has exactly one, and the
            // store names it.
            #expect(calibration.name == NeutralCorpusStore.normCorpusID)
            #expect(
                calibration.files.first?.url
                    == NeutralCorpusStore.normCorpusURL(root: root))

            let projection = DatasetCreationPlanner.plan(
                request(
                    .neutralCorpus, name: "Assistant Dialogue",
                    neutral: .projection),
                root: root)
            #expect(
                projection.files.first?.url
                    == NeutralCorpusStore.projectionRoot(root: root)
                        .appending(components: "assistant-dialogue", "corpus.jsonl"))
            // The neutral family's own slug rule, not the concept sanitizer.
            #expect(projection.name == NeutralCorpusStore.slugify("Assistant Dialogue"))
        }
    }

    // MARK: Requirements

    @Test func anUndeclaredNameOrSubChoiceHasNoDestinationToShow() throws {
        try withTempWorkspace { root in
            let unnamed = DatasetCreationPlanner.plan(
                request(.conceptStimuli), root: root)
            #expect(unnamed.requirements == [.nameRequired])
            #expect(unnamed.files.isEmpty)
            #expect(!unnamed.isResolved)

            let noFamily = DatasetCreationPlanner.plan(
                request(.validationSet, name: "sympathy"), root: root)
            #expect(noFamily.requirements == [.recipeFamilyRequired])
            #expect(noFamily.files.isEmpty)

            let noTarget = DatasetCreationPlanner.plan(
                request(.neutralCorpus), root: root)
            #expect(noTarget.requirements == [.neutralTargetRequired])

            // A projection basis still needs its own name.
            let unnamedProjection = DatasetCreationPlanner.plan(
                request(.neutralCorpus, neutral: .projection), root: root)
            #expect(unnamedProjection.requirements == [.nameRequired])
        }
    }

    @Test func namesAreSanitizedThroughTheConceptLabsOwnRule() throws {
        try withTempWorkspace { root in
            let plan = DatasetCreationPlanner.plan(
                request(.conceptStimuli, name: "  Judicial Sympathy!  "), root: root)
            #expect(plan.name == ConceptBuilder.sanitizedName("Judicial Sympathy!"))
            #expect(plan.name == "judicial-sympathy")
            #expect(plan.wasSanitized)
            #expect(
                plan.files.first?.relativePath
                    == "prompts/concepts/judicial-sympathy/positive.jsonl")

            let unusable = DatasetCreationPlanner.plan(
                request(.conceptStimuli, name: "***"), root: root)
            #expect(unusable.requirements == [.nameUnusable("***")])
            #expect(unusable.files.isEmpty)
        }
    }

    // MARK: Collisions and the verb

    @Test func anEmptyDestinationIsACreateAndAnOccupiedOneIsNot() throws {
        try withTempWorkspace { root in
            let fresh = DatasetCreationPlanner.plan(
                request(.storyCorpus, name: "sympathy"), root: root)
            #expect(fresh.verb == .create)
            #expect(fresh.collidingFiles.isEmpty)

            // Directory present, file absent: adding to an existing dataset.
            try FileManager.default.createDirectory(
                at: fresh.directory, withIntermediateDirectories: true)
            #expect(
                DatasetCreationPlanner.plan(
                    request(.storyCorpus, name: "sympathy"), root: root
                ).verb == .addTo)

            // File present: replacing.
            try storyRows(concept: "sympathy", count: 2)
                .write(
                    to: fresh.directory.appending(component: "stories.jsonl"),
                    atomically: true, encoding: .utf8)
            let occupied = DatasetCreationPlanner.plan(
                request(.storyCorpus, name: "sympathy"), root: root)
            #expect(occupied.verb == .replace)
            #expect(occupied.collidingFiles.map(\.slot) == [.stories])
            #expect(occupied.collidingFiles.first?.existingByteSize ?? 0 > 0)
        }
    }

    /// The held-out set filed under the OTHER recipe's root is exactly the
    /// misfiling the dual-root lookup exists to survive. Creating a second one
    /// is legal but ambiguous, and the plan says so before anything is written.
    @Test func aTwinValidationSetUnderTheOtherRootIsAnAdvisory() throws {
        try withTempWorkspace { root in
            let clean = DatasetCreationPlanner.plan(
                request(.validationSet, name: "sympathy", family: .grandMean),
                root: root)
            #expect(clean.advisories.isEmpty)

            let twin = root.appending(
                components: "prompts", "concepts", "sympathy", "validation.jsonl")
            try FileManager.default.createDirectory(
                at: twin.deletingLastPathComponent(), withIntermediateDirectories: true)
            try labelledRows(2).write(to: twin, atomically: true, encoding: .utf8)

            let advised = DatasetCreationPlanner.plan(
                request(.validationSet, name: "sympathy", family: .grandMean),
                root: root)
            #expect(advised.advisories.count == 1)
            #expect(
                advised.advisories[0].contains(
                    "prompts/concepts/sympathy/validation.jsonl"))
            // Still a create — the advisory does not block, it informs.
            #expect(advised.verb == .create)
        }
    }

    // MARK: Import validation

    @Test func aWellFormedImportIsAcceptedWithItsRowCountAndHash() throws {
        try withTempWorkspace { root in
            let plan = DatasetCreationPlanner.plan(
                request(.conceptStimuli, name: "sympathy"), root: root)
            let positive = try makeSource(
                textRows(["p1", "p2", "p3"]), named: "anything.jsonl")
            let negative = try makeSource(textRows(["n1", "n2"]), named: "other.txt")

            let outcome = try plan.apply(
                imports: [.positiveStimuli: positive, .negativeStimuli: negative])

            #expect(outcome.previews[.positiveStimuli]?.rowCount == 3)
            #expect(outcome.previews[.negativeStimuli]?.rowCount == 2)
            #expect(outcome.totalRows == 5)
            #expect(outcome.previews[.positiveStimuli]?.contentHash.count == 64)

            // Copied to the canonical NAME as well as the canonical place.
            let set = try StimulusSet(directory: plan.directory)
            #expect(set.positive.count == 3)
            #expect(set.negative.count == 2)
        }
    }

    @Test func eachRoleValidatesWithItsOwnFamilysParser() throws {
        try withTempWorkspace { root in
            // Story corpus: the multi-concept row shape.
            let stories = DatasetCreationPlanner.plan(
                request(.storyCorpus, name: "sympathy"), root: root)
            try stories.apply(
                imports: [
                    .stories: try makeSource(
                        storyRows(concept: "sympathy", count: 4), named: "s.jsonl")
                ])
            #expect(
                try StimulusSet.loadMultiConceptTexts(
                    url: stories.directory.appending(component: "stories.jsonl")
                ).rows.count == 4)

            // Validation: the {"text", "expresses"} shape.
            let validation = DatasetCreationPlanner.plan(
                request(.validationSet, name: "sympathy", family: .grandMean),
                root: root)
            let validationOutcome = try validation.apply(
                imports: [.validation: try makeSource(labelledRows(6), named: "v.jsonl")])
            #expect(validationOutcome.previews[.validation]?.rowCount == 6)

            // Probe items: the Concept Lab's own parser.
            let probes = DatasetCreationPlanner.plan(
                request(.probeItems, name: "sympathy"), root: root)
            let probeOutcome = try probes.apply(
                imports: [.probeItems: try makeSource(labelledRows(5), named: "i.jsonl")])
            #expect(probeOutcome.previews[.probeItems]?.rowCount == 5)

            // Neutral corpus: the plain text-row loader the store reads with.
            let neutral = DatasetCreationPlanner.plan(
                request(.neutralCorpus, neutral: .normCalibration), root: root)
            try neutral.apply(
                imports: [
                    .neutralCorpus: try makeSource(
                        textRows(["a", "b", "c"]), named: "c.jsonl")
                ])
            #expect(NeutralCorpusStore.scan(root: root).first?.count == 3)
        }
    }

    /// A rejection names the parser's own error — file and line where the
    /// loader gives one — and writes NOTHING.
    @Test func aMalformedImportIsRefusedWithItsParseError() throws {
        try withTempWorkspace { root in
            let plan = DatasetCreationPlanner.plan(
                request(.storyCorpus, name: "sympathy"), root: root)
            let bad = try makeSource(
                #"{"concept": "sympathy", "text": "ok"}"# + "\n"
                    + "this is not JSON at all\n",
                named: "s.jsonl")

            var rejected: DatasetCreationError?
            do {
                try plan.apply(imports: [.stories: bad])
            } catch let error as DatasetCreationError {
                rejected = error
            }
            let error = try #require(rejected)
            #expect("\(error)".contains("stories.jsonl"))
            #expect("\(error)".contains(":2"))
            #expect(!FileManager.default.fileExists(
                atPath: plan.directory.appending(component: "stories.jsonl").path))
        }
    }

    /// Zero usable rows is a refusal, not an empty success: the text loaders
    /// return `[]` for a file of blank lines, and an empty dataset landing in
    /// the canonical place is the failure this flow exists to prevent.
    @Test func anEmptyImportIsRefused() throws {
        try withTempWorkspace { root in
            let plan = DatasetCreationPlanner.plan(
                request(.neutralCorpus, neutral: .normCalibration), root: root)
            let blank = try makeSource("\n\n\n", named: "corpus.jsonl")
            #expect(throws: DatasetCreationError.self) {
                try plan.apply(imports: [.neutralCorpus: blank])
            }
            #expect(!FileManager.default.fileExists(
                atPath: NeutralCorpusStore.normCorpusURL(root: root).path))
        }
    }

    // MARK: No partial writes

    /// The transaction: EVERY file is staged and parsed before ANY file is
    /// published. A malformed second file must not leave a validated first
    /// one half-installed, and no staging debris may survive.
    @Test func aFailureMidwayLeavesNeitherFileNorStagingDebris() throws {
        try withTempWorkspace { root in
            let plan = DatasetCreationPlanner.plan(
                request(.conceptStimuli, name: "sympathy"), root: root)
            let good = try makeSource(textRows(["p1", "p2"]), named: "p.jsonl")
            let bad = try makeSource("not jsonl\n", named: "n.jsonl")

            #expect(throws: DatasetCreationError.self) {
                try plan.apply(
                    imports: [.positiveStimuli: good, .negativeStimuli: bad])
            }

            let fm = FileManager.default
            #expect(!fm.fileExists(
                atPath: plan.directory.appending(component: "positive.jsonl").path))
            #expect(!fm.fileExists(
                atPath: plan.directory.appending(component: "negative.jsonl").path))
            let leftovers = (try? fm.contentsOfDirectory(
                at: plan.directory, includingPropertiesForKeys: nil)) ?? []
            #expect(leftovers.isEmpty)
            // And the workspace still holds no dataset.
            #expect(DatasetInventory.scan(root: root).isEmpty)
        }
    }

    /// An existing valid dataset survives a refused replacement byte for byte.
    @Test func aRefusedReplacementLeavesTheOriginalUntouched() throws {
        try withTempWorkspace { root in
            let plan = DatasetCreationPlanner.plan(
                request(.storyCorpus, name: "sympathy"), root: root)
            try plan.apply(
                imports: [
                    .stories: try makeSource(
                        storyRows(concept: "sympathy", count: 3), named: "s.jsonl")
                ])
            let destination = plan.directory.appending(component: "stories.jsonl")
            let before = try Data(contentsOf: destination)

            let replacement = try makeSource(
                storyRows(concept: "sympathy", count: 9), named: "s2.jsonl")
            let second = DatasetCreationPlanner.plan(
                request(.storyCorpus, name: "sympathy"), root: root)
            #expect(second.verb == .replace)

            var refusal: DatasetCreationError?
            do {
                try second.apply(imports: [.stories: replacement])
            } catch let error as DatasetCreationError {
                refusal = error
            }
            #expect(
                refusal == .destinationExists(["prompts/emotions/sympathy/stories.jsonl"]))
            #expect(try Data(contentsOf: destination) == before)

            // Explicit consent replaces it, and only then.
            let outcome = try second.apply(
                imports: [.stories: replacement], replacingExisting: true)
            #expect(outcome.previews[.stories]?.rowCount == 9)
            #expect(try Data(contentsOf: destination) != before)
        }
    }

    /// The collision check re-stats at apply time: a destination that appeared
    /// after the preview is still a refusal, not an overwrite.
    @Test func aDestinationThatAppearsAfterThePreviewIsStillARefusal() throws {
        try withTempWorkspace { root in
            let plan = DatasetCreationPlanner.plan(
                request(.probeItems, name: "sympathy"), root: root)
            #expect(plan.verb == .create)

            try FileManager.default.createDirectory(
                at: plan.directory, withIntermediateDirectories: true)
            let destination = plan.directory.appending(component: "items.jsonl")
            try labelledRows(2).write(to: destination, atomically: true, encoding: .utf8)
            let before = try Data(contentsOf: destination)

            #expect(throws: DatasetCreationError.self) {
                try plan.apply(
                    imports: [
                        .probeItems: try makeSource(labelledRows(7), named: "i.jsonl")
                    ])
            }
            #expect(try Data(contentsOf: destination) == before)
        }
    }

    /// A concept stimulus set is both files or it is not a dataset.
    @Test func aHalfImportedPairIsRefused() throws {
        try withTempWorkspace { root in
            let plan = DatasetCreationPlanner.plan(
                request(.conceptStimuli, name: "sympathy"), root: root)
            var refusal: DatasetCreationError?
            do {
                try plan.apply(
                    imports: [
                        .positiveStimuli: try makeSource(
                            textRows(["p1"]), named: "p.jsonl")
                    ])
            } catch let error as DatasetCreationError {
                refusal = error
            }
            #expect(refusal == .incompleteSet([.negativeStimuli]))
            #expect(DatasetInventory.scan(root: root).isEmpty)
            // The refusal comes BEFORE any mkdir: an incomplete declaration
            // does not even leave an empty directory behind.
            #expect(!FileManager.default.fileExists(atPath: plan.directory.path))

            // …but completing an existing half IS legal: only the missing
            // slot is imported, and the set becomes whole.
            try FileManager.default.createDirectory(
                at: plan.directory, withIntermediateDirectories: true)
            try textRows(["p1", "p2"]).write(
                to: plan.directory.appending(component: "positive.jsonl"),
                atomically: true, encoding: .utf8)
            let reread = DatasetCreationPlanner.plan(
                request(.conceptStimuli, name: "sympathy"), root: root)
            try reread.apply(
                imports: [
                    .negativeStimuli: try makeSource(textRows(["n1"]), named: "n.jsonl")
                ])
            #expect(try StimulusSet(directory: plan.directory).negative.count == 1)
        }
    }

    @Test func aSlotFromAnotherRoleIsRefused() throws {
        try withTempWorkspace { root in
            let plan = DatasetCreationPlanner.plan(
                request(.probeItems, name: "sympathy"), root: root)
            #expect(
                throws: DatasetCreationError.unexpectedSlot(.stories)
            ) {
                try plan.apply(
                    imports: [
                        .stories: try makeSource(
                            storyRows(concept: "s", count: 1), named: "s.jsonl")
                    ])
            }
        }
    }

    @Test func anUnresolvedPlanCannotBeApplied() throws {
        try withTempWorkspace { root in
            let plan = DatasetCreationPlanner.plan(request(.conceptStimuli), root: root)
            #expect(throws: DatasetCreationError.self) {
                try plan.apply(
                    imports: [
                        .positiveStimuli: try makeSource(
                            textRows(["p"]), named: "p.jsonl")
                    ])
            }
        }
    }

    // MARK: Landing on the new row

    /// The point of the flow: after an import the phase-1 inventory finds the
    /// dataset, and the id the outcome hands back is the row to select.
    @Test func theInventoryFindsWhatTheFlowCreated() throws {
        try withTempWorkspace { root in
            #expect(DatasetInventory.scan(root: root).isEmpty)

            let plan = DatasetCreationPlanner.plan(
                request(.validationSet, name: "Judicial Sympathy", family: .grandMean),
                root: root)
            let outcome = try plan.apply(
                imports: [.validation: try makeSource(labelledRows(8), named: "v.jsonl")])

            let entries = DatasetInventory.scan(root: root)
            let entry = try #require(entries.first { $0.id == outcome.inventoryEntryID })
            #expect(entry.kind == .validationSet)
            #expect(entry.familyLabel == "grand-mean")
            #expect(entry.name == "judicial-sympathy")
            #expect(entry.itemCount == 8)
            #expect(entry.issue == nil)
            // The hash the inventory shows is the hash the import previewed,
            // which is the hash a manifest would pin.
            #expect(entry.contentHash == outcome.previews[.validation]?.contentHash)
            #expect(
                entry.contentHash
                    == ExperimentStore.conceptValidationHash(
                        name: "judicial-sympathy", isPaired: false))
        }
    }

    @Test func everyRolesInventoryEntryIDMatchesTheRowItProduces() throws {
        try withTempWorkspace { root in
            let plans: [(DatasetCreationPlan, [DatasetFileSlot: URL])] = [
                (
                    DatasetCreationPlanner.plan(
                        request(.conceptStimuli, name: "alpha"), root: root),
                    [
                        .positiveStimuli: try makeSource(
                            textRows(["p"]), named: "p.jsonl"),
                        .negativeStimuli: try makeSource(
                            textRows(["n"]), named: "n.jsonl"),
                    ]
                ),
                (
                    DatasetCreationPlanner.plan(
                        request(.storyCorpus, name: "alpha"), root: root),
                    [
                        .stories: try makeSource(
                            storyRows(concept: "alpha", count: 2), named: "s.jsonl")
                    ]
                ),
                (
                    DatasetCreationPlanner.plan(
                        request(.validationSet, name: "alpha", family: .paired),
                        root: root),
                    [.validation: try makeSource(labelledRows(2), named: "v.jsonl")]
                ),
                (
                    DatasetCreationPlanner.plan(
                        request(.probeItems, name: "alpha"), root: root),
                    [.probeItems: try makeSource(labelledRows(2), named: "i.jsonl")]
                ),
                (
                    DatasetCreationPlanner.plan(
                        request(.neutralCorpus, neutral: .normCalibration), root: root),
                    [.neutralCorpus: try makeSource(textRows(["x"]), named: "c.jsonl")]
                ),
                (
                    DatasetCreationPlanner.plan(
                        request(.neutralCorpus, name: "dialogue", neutral: .projection),
                        root: root),
                    [.neutralCorpus: try makeSource(textRows(["y"]), named: "c.jsonl")]
                ),
                (
                    DatasetCreationPlanner.plan(
                        request(.capabilityBattery, name: "alpha"), root: root),
                    [
                        .capabilityBattery: try makeSource(
                            batteryFile(items: 2), named: "b.jsonl")
                    ]
                ),
                // Both paired FAMILIES: same dataset name, two roots, two
                // rows — so the id formula has to distinguish them.
                (
                    DatasetCreationPlanner.plan(
                        request(.pairedStimuli, name: "alpha", paired: .repe),
                        root: root),
                    [.repePairs: try makeSource(repePairRows(3), named: "r.jsonl")]
                ),
                (
                    DatasetCreationPlanner.plan(
                        request(.pairedStimuli, name: "alpha", paired: .readers),
                        root: root),
                    [
                        .readerPairs: try makeSource(
                            readerPairRows(concept: "alpha", count: 3),
                            named: "rd.jsonl")
                    ]
                ),
            ]

            var expected: Set<String> = []
            for (plan, imports) in plans {
                let outcome = try plan.apply(imports: imports)
                expected.insert(outcome.inventoryEntryID)
            }

            let ids = Set(DatasetInventory.scan(root: root).map(\.id))
            #expect(expected.isSubset(of: ids))
            #expect(expected.count == plans.count)
        }
    }

    // MARK: The author-in-the-builder path

    /// `createDirectory` makes EMPTY structure. Nothing is seeded — a
    /// skeleton with an example row in it would be study content this
    /// instrument invented.
    @Test func createDirectoryMakesEmptyStructureAndNoRows() throws {
        try withTempWorkspace { root in
            let plan = DatasetCreationPlanner.plan(
                request(.conceptStimuli, name: "sympathy"), root: root)
            let directory = try plan.createDirectory()
            #expect(directory == plan.directory)
            let contents = try FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil)
            #expect(contents.isEmpty)
            // An empty directory is not a dataset, so the inventory stays
            // honest about what the workspace holds.
            #expect(DatasetInventory.scan(root: root).isEmpty)
        }
    }

    // MARK: Capability batteries (phase 3) — a FLAT-FILE role

    /// The battery role's destination is the batteries authority, and the
    /// NAME is the filename rather than a folder: `prompts/batteries/` is
    /// shared by every battery in the workspace.
    @Test func batteriesLandUnderTheBatteriesAuthorityNamedByTheDataset() throws {
        try withTempWorkspace { root in
            let plan = DatasetCreationPlanner.plan(
                request(.capabilityBattery, name: "study-guardrail"), root: root)
            #expect(plan.isResolved)
            #expect(plan.directory == VectorCatalog.batteriesDirectory(root: root))
            #expect(plan.files.count == 1)
            let file = try #require(plan.files.first)
            #expect(file.slot == .capabilityBattery)
            #expect(file.filename == "study-guardrail.jsonl")
            #expect(file.relativePath == "prompts/batteries/study-guardrail.jsonl")
            #expect(plan.verb == .create)
        }
    }

    /// Two batteries in one directory are two DATASETS. The second one is a
    /// `create` (the directory exists, the file does not) and their inventory
    /// rows are distinct — the flat-file identity rule.
    @Test func twoBatteriesInOneDirectoryAreTwoDistinctDatasets() throws {
        try withTempWorkspace { root in
            let first = DatasetCreationPlanner.plan(
                request(.capabilityBattery, name: "basic"), root: root)
            try first.apply(imports: [
                .capabilityBattery: try makeSource(batteryFile(items: 2), named: "a.jsonl")
            ])

            let second = DatasetCreationPlanner.plan(
                request(.capabilityBattery, name: "reasoning"), root: root)
            #expect(second.verb == .addTo)  // the directory exists; the file does not
            try second.apply(imports: [
                .capabilityBattery: try makeSource(batteryFile(items: 3), named: "b.jsonl")
            ])

            let rows = DatasetInventory.scan(root: root)
                .filter { $0.kind == .capabilityBattery }
            #expect(rows.map(\.name).sorted() == ["basic", "reasoning"])
            #expect(Set(rows.map(\.id)).count == 2)
            #expect(rows.first { $0.name == "basic" }?.itemCount == 2)
            #expect(rows.first { $0.name == "reasoning" }?.itemCount == 3)
            #expect(first.inventoryEntryID != second.inventoryEntryID)
        }
    }

    /// Import validation runs the ENGINE's own battery loader, so a file this
    /// accepts is a file a run reads — and a header that would fail the pin's
    /// shape check is refused here, before it lands.
    @Test func aBatteryImportIsValidatedByTheEnginesOwnLoader() throws {
        try withTempWorkspace { root in
            let plan = DatasetCreationPlanner.plan(
                request(.capabilityBattery, name: "guardrail"), root: root)

            let good = try makeSource(batteryFile(items: 4), named: "good.jsonl")
            let outcome = try plan.apply(imports: [.capabilityBattery: good])
            #expect(outcome.totalRows == 4)
            let preview = try #require(outcome.previews[.capabilityBattery])
            #expect(preview.rowCount == 4)
            #expect(preview.contentHash.count == 64)
            // The landed bytes hash to what the pin would pin.
            let landed = try Data(
                contentsOf: root.appending(
                    path: "prompts/batteries/guardrail.jsonl"))
            #expect(preview.contentHash == ExperimentStore.sha256Hex(landed))
        }
    }

    /// Legacy headerless batteries still import — their pins are unchanged,
    /// so refusing them would be this flow inventing a rule the engine does
    /// not have.
    @Test func aLegacyHeaderlessBatteryStillImports() throws {
        try withTempWorkspace { root in
            let plan = DatasetCreationPlanner.plan(
                request(.capabilityBattery, name: "legacy"), root: root)
            let legacy = try makeSource(
                #"{"prompt": "Capital of France?", "answer": "Paris"}"# + "\n"
                    + #"{"prompt": "2+2?", "answer": "4", "grading": "exact_number"}"#
                    + "\n",
                named: "legacy.jsonl")
            let outcome = try plan.apply(imports: [.capabilityBattery: legacy])
            #expect(outcome.totalRows == 2)

            let row = try #require(
                DatasetInventory.scan(root: root).first { $0.kind == .capabilityBattery })
            #expect(row.familyLabel == "format 1")
            #expect(row.note?.contains("legacy headerless") == true)
        }
    }

    /// A malformed battery is refused with the loader's own reason, and
    /// nothing lands — same atomicity as every other role.
    @Test func aMalformedBatteryIsRefusedAndNothingLands() throws {
        try withTempWorkspace { root in
            let plan = DatasetCreationPlanner.plan(
                request(.capabilityBattery, name: "broken"), root: root)
            // A choiceProbability item with no options can never be scored.
            let bad = try makeSource(
                #"{"batteryFormat": 2, "scoring": "choiceProbability"}"# + "\n"
                    + #"{"prompt": "2+2?", "answer": "4"}"# + "\n",
                named: "bad.jsonl")

            var rejected: DatasetCreationError?
            do {
                try plan.apply(imports: [.capabilityBattery: bad])
            } catch let error as DatasetCreationError {
                rejected = error
            }
            let error = try #require(rejected)
            #expect("\(error)".contains("options"))
            #expect(!FileManager.default.fileExists(
                atPath: root.appending(path: "prompts/batteries/broken.jsonl").path))
            // No staging debris either.
            let contents =
                (try? FileManager.default.contentsOfDirectory(
                    at: plan.directory, includingPropertiesForKeys: nil)) ?? []
            #expect(contents.isEmpty)
            #expect(DatasetInventory.scan(root: root).isEmpty)
        }
    }

    @Test func anUnnamedBatteryHasNoDestinationToShow() throws {
        try withTempWorkspace { root in
            let plan = DatasetCreationPlanner.plan(
                request(.capabilityBattery), root: root)
            #expect(!plan.isResolved)
            #expect(plan.requirements == [.nameRequired])
            #expect(plan.files.isEmpty)
            // The family root is still nameable before a name exists.
            #expect(plan.directory == VectorCatalog.batteriesDirectory(root: root))
        }
    }

    // MARK: Role vocabulary

    // MARK: Paired stimulus sets (phase 4)

    /// The two paired roots come from the ONE authority phase 4 named —
    /// the same `VectorCatalog.pairedStimuliDirectory` the Concept Builder's
    /// writers resolve through. Phase 2 had to leave this role out precisely
    /// because no such authority existed.
    @Test func pairedSetsLandUnderThePairedStimuliAuthority() throws {
        try withTempWorkspace { root in
            let repe = DatasetCreationPlanner.plan(
                request(.pairedStimuli, name: "tidiness", paired: .repe), root: root)
            let readers = DatasetCreationPlanner.plan(
                request(.pairedStimuli, name: "tidiness", paired: .readers), root: root)

            #expect(repe.isResolved)
            #expect(readers.isResolved)
            #expect(
                repe.directory
                    == VectorCatalog.pairedStimuliDirectory(
                        family: .repe, name: "tidiness", root: root))
            #expect(
                readers.directory
                    == VectorCatalog.pairedStimuliDirectory(
                        family: .readers, name: "tidiness", root: root))
            #expect(
                repe.files.map(\.relativePath)
                    == ["prompts/repe/tidiness/pairs.jsonl"])
            #expect(
                readers.files.map(\.relativePath)
                    == ["prompts/readers/tidiness/pairs.jsonl"])
            // One filename, two SLOTS — the slot is what dispatches the
            // import to the right loader.
            #expect(repe.files.map(\.slot) == [.repePairs])
            #expect(readers.files.map(\.slot) == [.readerPairs])
            #expect(repe.kind == .pairedStimuli)
            #expect(readers.kind == .pairedStimuli)
        }
    }

    /// The family is a DESTINATION-deciding sub-choice, exactly like the
    /// validation set's recipe family: with none chosen there is no path to
    /// preview.
    @Test func aPairedSetWithNoFamilyHasNoDestinationToShow() throws {
        try withTempWorkspace { root in
            let plan = DatasetCreationPlanner.plan(
                request(.pairedStimuli, name: "tidiness"), root: root)
            #expect(!plan.isResolved)
            #expect(plan.files.isEmpty)
            #expect(plan.requirements.contains(.pairedFamilyRequired))
            // The preview still names the family root it can honestly show.
            #expect(
                plan.directory
                    == VectorCatalog.pairedStimuliRoot(family: .repe, root: root))

            let named = DatasetCreationPlanner.plan(
                request(.pairedStimuli, paired: .readers), root: root)
            #expect(named.requirements.contains(.nameRequired))
        }
    }

    /// Each family is validated by the loader ITS recipe reads with — and
    /// the other family's rows are refused, because the two shapes are not
    /// interchangeable.
    @Test func eachPairedFamilyValidatesWithItsOwnLoader() throws {
        try withTempWorkspace { root in
            let repe = DatasetCreationPlanner.plan(
                request(.pairedStimuli, name: "tidiness", paired: .repe), root: root)
            let repeOutcome = try repe.apply(
                imports: [.repePairs: try makeSource(repePairRows(4), named: "r.jsonl")])
            #expect(repeOutcome.previews[.repePairs]?.rowCount == 4)
            #expect(
                try StimulusSet.loadPairs(
                    url: VectorCatalog.pairedStimuliFile(
                        family: .repe, name: "tidiness", root: root)
                ).pairs.count == 4)

            let readers = DatasetCreationPlanner.plan(
                request(.pairedStimuli, name: "tidiness", paired: .readers), root: root)
            let readerOutcome = try readers.apply(
                imports: [
                    .readerPairs: try makeSource(
                        readerPairRows(concept: "tidiness", count: 5), named: "rd.jsonl")
                ])
            #expect(readerOutcome.previews[.readerPairs]?.rowCount == 5)
            #expect(
                try RepEReader.loadPairs(
                    url: VectorCatalog.pairedStimuliFile(
                        family: .readers, name: "tidiness", root: root)
                ).pairs.count == 5)
        }
    }

    /// The cross-family refusal, and its atomicity: reader rows offered to
    /// the RepE mirror are rejected by the RepE loader, and nothing lands.
    @Test func pairedRowsOfTheWrongFamilyAreRefusedAndNothingLands() throws {
        try withTempWorkspace { root in
            let repe = DatasetCreationPlanner.plan(
                request(.pairedStimuli, name: "brevity", paired: .repe), root: root)
            #expect(
                throws: DatasetCreationError.self,
                performing: {
                    try repe.apply(
                        imports: [
                            .repePairs: try makeSource(
                                readerPairRows(concept: "brevity", count: 3),
                                named: "rd.jsonl")
                        ])
                })
            #expect(
                !FileManager.default.fileExists(
                    atPath: VectorCatalog.pairedStimuliFile(
                        family: .repe, name: "brevity", root: root).path))
            // No staging debris either.
            let leftovers =
                (try? FileManager.default.contentsOfDirectory(
                    atPath: repe.directory.path)) ?? []
            #expect(leftovers.isEmpty)

            let readers = DatasetCreationPlanner.plan(
                request(.pairedStimuli, name: "brevity", paired: .readers), root: root)
            #expect(
                throws: DatasetCreationError.self,
                performing: {
                    try readers.apply(
                        imports: [
                            .readerPairs: try makeSource(
                                repePairRows(3), named: "r.jsonl")
                        ])
                })
            #expect(
                !FileManager.default.fileExists(
                    atPath: VectorCatalog.pairedStimuliFile(
                        family: .readers, name: "brevity", root: root).path))
        }
    }

    /// A twin under the OTHER paired root is an advisory, not a refusal:
    /// both may coexist, but the Concept Builder restores a concept's recipe
    /// from whichever mirror was written last
    /// (`ConceptBuilder.pairedRecipeFamilyOnDisk`).
    @Test func aTwinPairedSetUnderTheOtherRootIsAnAdvisory() throws {
        try withTempWorkspace { root in
            try DatasetCreationPlanner.plan(
                request(.pairedStimuli, name: "tidiness", paired: .repe), root: root
            ).apply(imports: [.repePairs: try makeSource(repePairRows(3), named: "r.jsonl")])

            let readers = DatasetCreationPlanner.plan(
                request(.pairedStimuli, name: "tidiness", paired: .readers), root: root)
            #expect(readers.isResolved)
            #expect(readers.advisories.count == 1)
            #expect(
                readers.advisories[0].contains("prompts/repe/tidiness/pairs.jsonl"))

            // The reverse direction, and the no-twin case.
            let lonely = DatasetCreationPlanner.plan(
                request(.pairedStimuli, name: "gadgetry", paired: .readers), root: root)
            #expect(lonely.advisories.isEmpty)
        }
    }

    /// An existing paired file is never silently overwritten — same rule as
    /// every other role.
    @Test func anExistingPairedFileIsAReplacement() throws {
        try withTempWorkspace { root in
            let source = try makeSource(repePairRows(3), named: "r.jsonl")
            let first = DatasetCreationPlanner.plan(
                request(.pairedStimuli, name: "tidiness", paired: .repe), root: root)
            #expect(first.verb == .create)
            try first.apply(imports: [.repePairs: source])

            let again = DatasetCreationPlanner.plan(
                request(.pairedStimuli, name: "tidiness", paired: .repe), root: root)
            #expect(again.verb == .replace)
            #expect(
                throws: DatasetCreationError.self,
                performing: { try again.apply(imports: [.repePairs: source]) })
            try again.apply(imports: [.repePairs: source], replacingExisting: true)
        }
    }

    @Test func everyRoleMapsToAPhaseOneInventoryKind() {
        let kinds = Set(DatasetRole.allCases.map(\.kind))
        #expect(kinds.count == DatasetRole.allCases.count)
        #expect(kinds.isSubset(of: Set(DatasetKind.allCases)))
        // Every role explains what reads it and where it goes — the whole
        // point of choosing a role rather than a directory.
        for role in DatasetRole.allCases {
            #expect(!role.feeds.isEmpty)
            #expect(!role.canonicalLocationHint.isEmpty)
            #expect(!role.title.isEmpty)
        }
    }
}
