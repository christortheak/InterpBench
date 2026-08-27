import CryptoKit
import Foundation
import SteeringKit

/// The Data section's role-first CREATION model (WP-Data phase 2).
///
/// Phase 1 made the workspace's data visible (`DatasetInventory`). This file
/// makes new data land in the RIGHT PLACE. The researcher declares what a
/// dataset IS — its role — before any file exists, and the plan computes the
/// canonical destination from the engine's own path authorities. The class of
/// mistake this closes is misfiling: a `validation.jsonl` written under the
/// paired recipe's root for a grand-mean concept is invisible to that recipe
/// unless the dual-root fallback catches it
/// (`ExperimentStore.resolveConceptValidation`), and a set filed by hand into
/// a directory nobody reads is simply not measured.
///
/// Design rules this file keeps:
///
/// - **Every destination comes from an existing authority.** No path is
///   assembled from string literals here:
///   `VectorCatalog.conceptsDirectory/emotionsDirectory/probesDirectory`,
///   `ExperimentStore.conceptValidationRelativePath` (which recipe root a
///   held-out set belongs to), and `NeutralCorpusStore.normCorpusURL` /
///   `projectionRoot`. A layout change moves the flow with it.
/// - **Nothing is seeded.** A plan creates DIRECTORIES and copies the
///   researcher's own bytes. No row, no template, no example content is
///   invented — a fresh workspace stays concept-empty until someone authors
///   into it.
/// - **Never a silent overwrite.** A destination that already exists is
///   reported as a collision and the verb changes; `apply` refuses to replace
///   unless the caller explicitly says so.
/// - **Stage, validate, then publish.** Imported bytes are copied into a
///   staging directory beside the destination, parsed by the family's OWN
///   loader, and only then renamed into place (same-volume, atomic). A
///   malformed file leaves nothing behind — the staging directory is removed
///   on every exit path. Same convention as the transactional artifact fetch
///   in `RemoteVectorLocalization`.
public enum DatasetCreationPlanner {

    /// Build the plan for a request. Pure: it stats the filesystem to report
    /// collisions and never writes.
    ///
    /// `root` defaults to the RESOLVED workspace, like `DatasetInventory.scan`.
    public static func plan(
        _ request: DatasetCreationRequest, root: URL = VectorCatalog.projectRoot
    ) -> DatasetCreationPlan {
        let role = request.role
        var requirements: [DatasetCreationRequirement] = []

        // Sub-choices that DECIDE the destination, not decorations: a
        // validation set's recipe family picks which root owns it, and a
        // neutral corpus's target picks calibration vs projection.
        if role == .validationSet, request.recipeFamily == nil {
            requirements.append(.recipeFamilyRequired)
        }
        if role == .neutralCorpus, request.neutralTarget == nil {
            requirements.append(.neutralTargetRequired)
        }
        // Same shape as the validation set's recipe family: the paired family
        // is not a label — it picks the root AND the row shape the import is
        // parsed with.
        if role == .pairedStimuli, request.pairedFamily == nil {
            requirements.append(.pairedFamilyRequired)
        }

        let raw = request.rawName.trimmingCharacters(in: .whitespacesAndNewlines)
        let needsName = role.requiresName(neutralTarget: request.neutralTarget)
        var name = ""
        if needsName {
            if raw.isEmpty {
                requirements.append(.nameRequired)
            } else {
                // Two sanitizers, each the authority for its own family:
                // concept directories use the Concept Lab's rule, projection
                // corpora use the neutral store's slug.
                name = role == .neutralCorpus
                    ? NeutralCorpusStore.slugify(raw)
                    : ConceptBuilder.sanitizedName(raw)
                if name.isEmpty { requirements.append(.nameUnusable(raw)) }
            }
        } else if role == .neutralCorpus {
            // The calibration corpus is a singleton the store already names.
            name = NeutralCorpusStore.normCorpusID
        }

        guard requirements.isEmpty else {
            return DatasetCreationPlan(
                role: role,
                root: root,
                name: name,
                rawName: request.rawName,
                recipeFamily: request.recipeFamily,
                neutralTarget: request.neutralTarget,
                pairedFamily: request.pairedFamily,
                kind: role.kind,
                directory: role.familyRoot(
                    root: root, target: request.neutralTarget,
                    pairedFamily: request.pairedFamily),
                files: [],
                requirements: requirements,
                advisories: [])
        }

        let (directory, slots) = destination(
            role: role, name: name, root: root,
            recipeFamily: request.recipeFamily, neutralTarget: request.neutralTarget,
            pairedFamily: request.pairedFamily)

        let files = slots.map { slot -> DatasetPlannedFile in
            let filename = slot.filename(datasetName: name)
            let url = directory.appending(component: filename)
            let stat = DatasetInventory.FileStat.total(of: [url])
            return DatasetPlannedFile(
                slot: slot,
                filename: filename,
                url: url,
                relativePath: Self.relativePath(of: url, root: root),
                exists: !stat.present.isEmpty,
                existingByteSize: stat.present.isEmpty ? nil : stat.bytes)
        }

        return DatasetCreationPlan(
            role: role,
            root: root,
            name: name,
            rawName: request.rawName,
            recipeFamily: request.recipeFamily,
            neutralTarget: request.neutralTarget,
            pairedFamily: request.pairedFamily,
            kind: role.kind,
            directory: directory,
            files: files,
            requirements: [],
            advisories: advisories(
                role: role, name: name, root: root,
                recipeFamily: request.recipeFamily,
                pairedFamily: request.pairedFamily))
    }

    /// THE destination rule. Every branch resolves through the store that
    /// already owns that family's layout — see the file comment.
    private static func destination(
        role: DatasetRole,
        name: String,
        root: URL,
        recipeFamily: DatasetRecipeFamily?,
        neutralTarget: NeutralCorpusTarget?,
        pairedFamily: VectorCatalog.PairedStimulusFamily?
    ) -> (directory: URL, slots: [DatasetFileSlot]) {
        switch role {
        case .pairedStimuli:
            // Both paired roots come from the ONE authority phase 4 named
            // (`VectorCatalog.pairedStimuliDirectory`) — the same one the
            // Concept Builder's five write/read sites now resolve through.
            let family = pairedFamily ?? .repe
            return (
                VectorCatalog.pairedStimuliDirectory(
                    family: family, name: name, root: root),
                [family.slot]
            )
        case .conceptStimuli:
            return (
                VectorCatalog.conceptsDirectory(root: root).appending(component: name),
                [.positiveStimuli, .negativeStimuli]
            )
        case .storyCorpus:
            return (
                VectorCatalog.emotionsDirectory(root: root).appending(component: name),
                [.stories]
            )
        case .validationSet:
            // The one rule for which recipe root owns a held-out set.
            let relative = ExperimentStore.conceptValidationRelativePath(
                name: name, isPaired: (recipeFamily ?? .paired).isPaired)
            let url = root.appending(path: relative)
            return (url.deletingLastPathComponent(), [.validation])
        case .probeItems:
            return (
                VectorCatalog.probesDirectory(root: root).appending(component: name),
                [.probeItems]
            )
        case .neutralCorpus:
            switch neutralTarget ?? .normCalibration {
            case .normCalibration:
                let url = NeutralCorpusStore.normCorpusURL(root: root)
                return (url.deletingLastPathComponent(), [.neutralCorpus])
            case .projection:
                return (
                    NeutralCorpusStore.projectionRoot(root: root)
                        .appending(component: name),
                    [.neutralCorpus]
                )
            }
        case .capabilityBattery:
            // A FLAT-FILE family: the directory is shared and the NAME is the
            // file. The slot's filename is therefore per-plan rather than
            // fixed (see `DatasetFileSlot.filename(name:)`).
            return (VectorCatalog.batteriesDirectory(root: root), [.capabilityBattery])
        }
    }

    /// Non-blocking notes about the destination's neighbourhood.
    ///
    /// The one that matters: a held-out set already filed under the OTHER
    /// recipe's root. Reading tolerates it (the dual-root lookup in
    /// `ExperimentStore.resolveConceptValidation` finds it and says so), but
    /// filing a SECOND one makes the concept's held-out set ambiguous, and
    /// the canonical home silently wins. Better said before the file exists
    /// than discovered at freeze.
    private static func advisories(
        role: DatasetRole, name: String, root: URL, recipeFamily: DatasetRecipeFamily?,
        pairedFamily: VectorCatalog.PairedStimulusFamily?
    ) -> [String] {
        switch role {
        case .validationSet:
            guard let family = recipeFamily else { return [] }
            let otherRelative = ExperimentStore.conceptValidationRelativePath(
                name: name, isPaired: !family.isPaired)
            let other = root.appending(path: otherRelative)
            guard FileManager.default.fileExists(atPath: other.path) else { return [] }
            return [
                "a validation.jsonl for '\(name)' already exists under the OTHER "
                    + "recipe's root (\(otherRelative)). Filing a second one here "
                    + "makes the concept's held-out set ambiguous — a recipe reads "
                    + "and pins its own root's file and only falls back to the "
                    + "other. Move or merge rather than keeping both."
            ]
        case .pairedStimuli:
            // The twin under the other paired root is not an error — a
            // concept can legitimately have both — but it makes the Concept
            // Builder's recipe restore depend on MODIFICATION TIME
            // (`ConceptBuilder.pairedRecipeFamilyOnDisk` breaks the tie by
            // "most recently written wins"). Better said before the file
            // exists than discovered when the picker shows the other recipe.
            guard let family = pairedFamily else { return [] }
            let other: VectorCatalog.PairedStimulusFamily =
                family == .repe ? .readers : .repe
            let otherURL = VectorCatalog.pairedStimuliFile(
                family: other, name: name, root: root)
            guard FileManager.default.fileExists(atPath: otherURL.path) else { return [] }
            return [
                "'\(name)' already has pairs under the other paired root "
                    + "(\(VectorCatalog.pairedStimuliRelativePath(family: other, name: name)))"
                    + ". Both may coexist, but the Concept Builder restores a "
                    + "concept's recipe from whichever mirror was written most "
                    + "recently — so writing here also changes which recipe the "
                    + "builder shows for '\(name)'."
            ]
        case .conceptStimuli, .storyCorpus, .probeItems, .neutralCorpus,
            .capabilityBattery:
            return []
        }
    }

    static func relativePath(of url: URL, root: URL) -> String {
        let rootPath = root.standardizedFileURL.path
        let path = url.standardizedFileURL.path
        if path.hasPrefix(rootPath + "/") {
            return String(path.dropFirst(rootPath.count + 1))
        }
        return path
    }

    // MARK: Import validation

    /// Parse a candidate file with the loader the SLOT's family actually
    /// uses, so a file this accepts is a file the recipe can read — and a row
    /// this rejects is a row extraction would have choked on later.
    ///
    /// Zero usable rows is a refusal, not an empty success: the text loaders
    /// return `[]` for a file of blank lines, and an empty dataset landing
    /// silently in the canonical place is exactly the failure this flow
    /// exists to prevent.
    public static func validate(
        fileAt url: URL, as slot: DatasetFileSlot, concept: String
    ) throws -> DatasetImportPreview {
        let filename = url.lastPathComponent
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw DatasetCreationError.sourceUnreadable(slot: slot, path: url.path)
        }

        let rows: Int
        do {
            switch slot {
            case .positiveStimuli, .negativeStimuli, .neutralCorpus:
                rows = try StimulusSet.loadTexts(url: url).texts.count
            case .repePairs:
                // The loader the RepE-LAT build reads its own mirror with
                // (`ConceptBuilder.saveConceptAndBuildVector` → the hash it
                // stamps as the canonical dataset).
                rows = try StimulusSet.loadPairs(url: url).pairs.count
            case .readerPairs:
                // The reader's own loader — a DIFFERENT row shape from the
                // RepE mirror, and the one a fit would choke on later.
                rows = try RepEReader.loadPairs(url: url).pairs.count
            case .stories:
                rows = try StimulusSet.loadMultiConceptTexts(url: url).rows.count
            case .validation:
                // The loader is directory-addressed and looks for
                // `validation.jsonl` — the staged file is already named that.
                rows = try StimulusSet.loadValidation(
                    directory: url.deletingLastPathComponent())?.count ?? 0
            case .probeItems:
                rows = try ConceptBuilder.parseProbeExamples(
                    data, filename: filename, concept: concept).count
            case .capabilityBattery:
                // The engine's own battery loader — both formats — so a file
                // this accepts is a file `run`'s per-condition battery pass
                // reads, and a format-2 header that would fail the pin's
                // shape check is refused HERE, before it lands.
                rows = try CapabilityBattery(data: data, file: filename).items.count
            }
        } catch {
            throw DatasetCreationError.importRejected(
                slot: slot, reason: "\(error)")
        }

        guard rows > 0 else {
            throw DatasetCreationError.importRejected(
                slot: slot,
                reason: "\(filename) parsed to zero rows — \(slot.rowShapeHint)")
        }

        return DatasetImportPreview(
            slot: slot,
            rowCount: rows,
            byteSize: Int64(data.count),
            contentHash: SHA256.hash(data: data)
                .map { String(format: "%02x", $0) }.joined())
    }
}

// MARK: - Roles

/// What a dataset IS. Role-first: this is declared BEFORE a file exists, and
/// it decides the destination. Deliberately closed — there is no "misc" role,
/// because a dataset nobody's recipe reads is not a dataset.
///
/// The vocabulary is phase 1's (`DatasetKind`) minus `promptSet`. Phase 2
/// also deferred `pairedStimuli` — the repe/readers `pairs.jsonl` roots had
/// no named path authority to plan against, because `ConceptBuilder`
/// assembled them inline. Phase 4 named that authority
/// (`VectorCatalog.pairedStimuliDirectory`), so the role is creatable here
/// like every other. `promptSet` remains absent deliberately: a task/dev
/// prompt file is authored against a study's design and pinned BY the study
/// (`ExperimentStore.pinTaskPrompts`), which is where the Studies section
/// files it — a second creation entry here would be a second convention.
public enum DatasetRole: String, Sendable, CaseIterable, Identifiable, Codable {
    case conceptStimuli
    case pairedStimuli
    case storyCorpus
    case validationSet
    case probeItems
    case neutralCorpus
    case capabilityBattery

    public var id: String { rawValue }

    /// The inventory row this role produces once the files land — the same
    /// vocabulary phase 1 displays, so creating and listing agree.
    public var kind: DatasetKind {
        switch self {
        case .conceptStimuli: .conceptStimuli
        case .pairedStimuli: .pairedStimuli
        case .storyCorpus: .grandMeanCorpus
        case .validationSet: .validationSet
        case .probeItems: .probeItems
        case .neutralCorpus: .neutralCorpus
        case .capabilityBattery: .capabilityBattery
        }
    }

    public var title: String {
        switch self {
        case .conceptStimuli: "Concept stimulus set"
        case .pairedStimuli: "Paired stimulus set"
        case .storyCorpus: "Story corpus"
        case .validationSet: "Validation set"
        case .probeItems: "Probe items"
        case .neutralCorpus: "Neutral corpus"
        case .capabilityBattery: "Capability battery"
        }
    }

    /// What READS this dataset — the sentence that makes the role choosable
    /// without knowing the directory layout.
    public var feeds: String {
        switch self {
        case .conceptStimuli:
            "Contrastive positive/negative stimuli. Feeds the CAA "
                + "(mean-difference) and RepE/LAT extraction recipes."
        case .pairedStimuli:
            "One row per matched pair. Feeds the RepE/LAT vector build "
                + "(prompts/repe/) or a fitted RepE reading instrument "
                + "(prompts/readers/) — the two have different row shapes, so "
                + "the family is a choice, not a label."
        case .storyCorpus:
            "Multi-concept story rows. Feeds the grand-mean recipe, which "
                + "reads a concept's mean against the whole corpus's mean."
        case .validationSet:
            "Never-named held-out scenarios. Feeds the convergent validate "
                + "gate; its hash is a measurement-side pin checked at freeze."
        case .probeItems:
            "Labelled items a linear reading probe trains and scores on."
        case .neutralCorpus:
            "The pinned neutral denominator: residual-norm calibration (so α "
                + "is comparable across concepts) and neutral-PC projection bases."
        case .capabilityBattery:
            "Deterministic capability probes. Feeds the per-condition "
                + "capability control inside a run, the sweep's coherence "
                + "constraint, and agent robustness checks — the control that "
                + "tells a real effect from a broken model."
        }
    }

    /// The canonical location, shown before anything is typed. `<name>` is
    /// filled in live once the researcher names the dataset.
    public var canonicalLocationHint: String {
        switch self {
        case .conceptStimuli: "prompts/concepts/<name>/{positive,negative}.jsonl"
        case .pairedStimuli:
            "prompts/repe/<name>/pairs.jsonl (RepE/LAT vectors) or "
                + "prompts/readers/<name>/pairs.jsonl (reader instruments)"
        case .storyCorpus: "prompts/emotions/<name>/stories.jsonl"
        case .validationSet:
            "prompts/concepts/<name>/validation.jsonl (paired recipes) or "
                + "prompts/emotions/<name>/validation.jsonl (grand-mean)"
        case .probeItems: "prompts/probes/<name>/items.jsonl"
        case .neutralCorpus:
            "prompts/neutral/corpus.jsonl (calibration) or "
                + "prompts/neutral/projection/<name>/corpus.jsonl"
        case .capabilityBattery:
            "prompts/batteries/<name>.jsonl"
        }
    }

    /// True when the role is concept-bearing (or a named projection corpus,
    /// or a flat-file dataset whose NAME is its filename).
    /// The calibration corpus is the one singleton — the workspace has
    /// exactly one, and the store names it.
    public func requiresName(neutralTarget: NeutralCorpusTarget?) -> Bool {
        switch self {
        case .conceptStimuli, .pairedStimuli, .storyCorpus, .validationSet,
            .probeItems, .capabilityBattery:
            true
        case .neutralCorpus: (neutralTarget ?? .normCalibration) == .projection
        }
    }

    /// True where the Concept Builder can author this role's ROWS by hand
    /// (paste, generate, type) once the concept exists.
    ///
    /// This is what the New Dataset sheet offers "Create and open the
    /// builder" for, and it is load-bearing: phase 4 retired the builder's
    /// own new-concept field, so an empty concept skeleton is created HERE or
    /// nowhere. The roles left out are the ones the builder has no editor for
    /// — a held-out validation set, a neutral corpus (its own section owns
    /// the corpus, not a per-concept folder), and a capability battery — and
    /// each of those imports a file instead.
    public var authorsInConceptBuilder: Bool {
        switch self {
        case .conceptStimuli, .pairedStimuli, .storyCorpus, .probeItems: true
        case .validationSet, .neutralCorpus, .capabilityBattery: false
        }
    }

    /// The family root a plan points at before it has a name — what the
    /// destination preview can honestly show while the name field is empty.
    func familyRoot(
        root: URL, target: NeutralCorpusTarget?,
        pairedFamily: VectorCatalog.PairedStimulusFamily? = nil
    ) -> URL {
        switch self {
        case .conceptStimuli, .validationSet:
            VectorCatalog.conceptsDirectory(root: root)
        case .pairedStimuli:
            VectorCatalog.pairedStimuliRoot(
                family: pairedFamily ?? .repe, root: root)
        case .storyCorpus:
            VectorCatalog.emotionsDirectory(root: root)
        case .probeItems:
            VectorCatalog.probesDirectory(root: root)
        case .neutralCorpus:
            (target ?? .normCalibration) == .projection
                ? NeutralCorpusStore.projectionRoot(root: root)
                : NeutralCorpusStore.normCorpusURL(root: root).deletingLastPathComponent()
        case .capabilityBattery:
            VectorCatalog.batteriesDirectory(root: root)
        }
    }
}

/// Which extraction family a held-out validation set belongs to. This is not
/// a label — it decides the canonical root
/// (`ExperimentStore.conceptValidationRelativePath`).
public enum DatasetRecipeFamily: String, Sendable, CaseIterable, Identifiable, Codable {
    case paired
    case grandMean

    public var id: String { rawValue }
    public var isPaired: Bool { self == .paired }

    public var label: String {
        switch self {
        case .paired: "Paired (CAA / RepE-LAT)"
        case .grandMean: "Grand-mean (story corpus)"
        }
    }

    public var detail: String {
        switch self {
        case .paired:
            "the held-out set for a concept extracted from positive/negative "
                + "stimuli — filed under prompts/concepts/"
        case .grandMean:
            "the held-out set for a concept extracted from a story corpus — "
                + "filed under prompts/emotions/"
        }
    }
}

/// Which neutral corpus is being created. Both destinations come from
/// `NeutralCorpusStore`.
public enum NeutralCorpusTarget: String, Sendable, CaseIterable, Identifiable, Codable {
    case normCalibration
    case projection

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .normCalibration: "Norm calibration (the workspace's one denominator)"
        case .projection: "Named projection basis"
        }
    }
}

// MARK: - File slots

/// One file a role's dataset is made of. The filename is the family's, not a
/// choice — an imported file is copied to the canonical NAME as well as the
/// canonical directory, so a `held-out-v3.jsonl` the researcher picked lands
/// as `validation.jsonl`.
public enum DatasetFileSlot: String, Sendable, CaseIterable, Identifiable, Codable {
    case positiveStimuli
    case negativeStimuli
    /// `prompts/repe/<name>/pairs.jsonl` — `StimulusSet.PairedStimulus` rows.
    case repePairs
    /// `prompts/readers/<name>/pairs.jsonl` — `RepEReader.Pair` rows. Same
    /// filename, different shape, hence a separate slot: the slot is what
    /// dispatches `validate(fileAt:as:concept:)` to the right loader.
    case readerPairs
    case stories
    case validation
    case probeItems
    case neutralCorpus
    case capabilityBattery

    public var id: String { rawValue }

    /// The canonical filename this slot lands as.
    ///
    /// Most families own a DIRECTORY per dataset, so the filename is fixed
    /// and the name is the folder. The flat-file families share one
    /// directory (`prompts/batteries/`), so there the NAME is the file — the
    /// same rule the pin uses when a manifest names a battery.
    public func filename(datasetName: String) -> String {
        switch self {
        case .positiveStimuli: "positive.jsonl"
        case .negativeStimuli: "negative.jsonl"
        case .repePairs, .readerPairs: VectorCatalog.pairedStimuliFileName
        case .stories: "stories.jsonl"
        case .validation: "validation.jsonl"
        case .probeItems: "items.jsonl"
        case .neutralCorpus: "corpus.jsonl"
        case .capabilityBattery: "\(datasetName).jsonl"
        }
    }

    /// The filename to NAME in a message where only the slot is in scope.
    /// Identical to the real filename for the fixed families; the
    /// name-addressed ones show the SHAPE rather than inventing a name.
    public var displayFilename: String { filename(datasetName: "<name>") }

    public var label: String {
        switch self {
        case .positiveStimuli: "Positive stimuli"
        case .negativeStimuli: "Negative stimuli"
        case .repePairs: "Paired-difference PCA pairs"
        case .readerPairs: "Reader pairs"
        case .stories: "Story rows"
        case .validation: "Held-out scenarios"
        case .probeItems: "Probe items"
        case .neutralCorpus: "Corpus rows"
        case .capabilityBattery: "Battery items"
        }
    }

    /// The row shape the family's loader requires — named up front, because
    /// discovering it from a rejection one line at a time is the slow way.
    public var rowShapeHint: String {
        switch self {
        case .positiveStimuli, .negativeStimuli, .neutralCorpus:
            StimulusSetError.textRowShape
        case .repePairs:
            #"each line must be a JSON object with "positive" and "negative" "#
                + #"strings (the sign contract: "positive" is the "#
                + #"concept-PRESENT side), e.g. "#
                + #"{"id": "…", "positive": "…", "negative": "…"}"#
        case .readerPairs:
            #"each line must be a JSON object with "concept", "#
                + #""positiveStimulus", "negativeStimulus", and "templateID" "#
                + #"(plus an optional "split" of "train"/"test"); every row "#
                + #"must name the SAME concept — a reader measures exactly one"#
        case .stories:
            #"each line must be a JSON object with "concept" and "text" strings, "#
                + #"e.g. {"concept": "…", "topic": "…", "text": "…"}"#
        case .validation:
            #"each line must be a JSON object with "text" and a boolean "#
                + #""expresses", e.g. {"text": "…", "expresses": true}"#
        case .probeItems:
            #"each line must be a JSON object with "text" and "expresses" "#
                + #"(or a "label"), e.g. {"text": "…", "expresses": false}"#
        case .capabilityBattery:
            #"format 2: a header line {"batteryFormat": 2, "scoring": "#
                + #""choiceProbability", "maxTokens": 24, "promptMode": "#
                + #""chatAssistant"} then one item per line, "#
                + #"{"id", "prompt", "answer", "options"} with the answer "#
                + #"among the options. Legacy headerless files "#
                + #"({"prompt", "answer", "grading"}) are still accepted."#
        }
    }
}

extension VectorCatalog.PairedStimulusFamily {
    /// The creation slot this family's file lands in. Declared HERE rather
    /// than on the path authority so `VectorCatalog` stays free of the
    /// creation flow's vocabulary — but declared ONCE, so a family's root and
    /// the loader its import is parsed with cannot drift apart.
    var slot: DatasetFileSlot {
        switch self {
        case .repe: .repePairs
        case .readers: .readerPairs
        }
    }
}

// MARK: - Request / plan

public struct DatasetCreationRequest: Sendable, Equatable {
    public var role: DatasetRole
    public var rawName: String
    /// Validation sets only — the choice that picks the canonical root.
    public var recipeFamily: DatasetRecipeFamily?
    /// Neutral corpora only.
    public var neutralTarget: NeutralCorpusTarget?
    /// Paired stimulus sets only — the choice that picks the root AND the
    /// row shape the import is parsed with.
    public var pairedFamily: VectorCatalog.PairedStimulusFamily?

    public init(
        role: DatasetRole,
        rawName: String = "",
        recipeFamily: DatasetRecipeFamily? = nil,
        neutralTarget: NeutralCorpusTarget? = nil,
        pairedFamily: VectorCatalog.PairedStimulusFamily? = nil
    ) {
        self.role = role
        self.rawName = rawName
        self.recipeFamily = recipeFamily
        self.neutralTarget = neutralTarget
        self.pairedFamily = pairedFamily
    }
}

/// A declaration that is still missing something the DESTINATION depends on.
/// While any of these stand, the plan has no files to preview — it cannot
/// honestly show a path it does not yet know.
public enum DatasetCreationRequirement: Sendable, Equatable {
    case nameRequired
    case nameUnusable(String)
    case recipeFamilyRequired
    case neutralTargetRequired
    case pairedFamilyRequired

    public var message: String {
        switch self {
        case .pairedFamilyRequired:
            "choose which paired family this set belongs to — it decides the "
                + "root AND the row shape (RepE/LAT vector pairs vs reader "
                + "pairs are not interchangeable)"
        case .nameRequired:
            "name this dataset — the name is the directory the recipe reads"
        case .nameUnusable(let raw):
            "'\(raw)' has no usable characters for a directory name (letters, "
                + "numbers, and hyphens survive sanitization)"
        case .recipeFamilyRequired:
            "choose which recipe family this validation set belongs to — it "
                + "decides which root owns the file"
        case .neutralTargetRequired:
            "choose whether this is the calibration corpus or a named "
                + "projection basis"
        }
    }
}

public struct DatasetPlannedFile: Sendable, Equatable, Identifiable {
    public let slot: DatasetFileSlot
    /// The canonical filename resolved for THIS dataset — fixed for the
    /// directory-owning families, `<name>.jsonl` for the flat-file ones.
    public let filename: String
    public let url: URL
    public let relativePath: String
    public let exists: Bool
    public let existingByteSize: Int64?

    public var id: String { slot.rawValue }
}

/// What the flow will do, computed before it does it.
public struct DatasetCreationPlan: Sendable, Equatable {
    public let role: DatasetRole
    public let root: URL
    /// The sanitized directory name — what actually lands on disk.
    public let name: String
    /// What the researcher typed, kept so the sheet can say "filed as …".
    public let rawName: String
    public let recipeFamily: DatasetRecipeFamily?
    public let neutralTarget: NeutralCorpusTarget?
    public let pairedFamily: VectorCatalog.PairedStimulusFamily?
    public let kind: DatasetKind
    public let directory: URL
    public let files: [DatasetPlannedFile]
    public let requirements: [DatasetCreationRequirement]
    public let advisories: [String]

    public var isResolved: Bool { requirements.isEmpty && !files.isEmpty }

    public var wasSanitized: Bool {
        !name.isEmpty && name != rawName.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public var collidingFiles: [DatasetPlannedFile] { files.filter(\.exists) }

    public var directoryExists: Bool {
        FileManager.default.fileExists(atPath: directory.path)
    }

    /// Create / add-to / replace. The verb is the honest description of what
    /// pressing the button will do — it is never "create" over a file that is
    /// already there.
    public var verb: DatasetCreationVerb {
        if !collidingFiles.isEmpty { return .replace }
        return directoryExists ? .addTo : .create
    }

    public func relativePath(of url: URL) -> String {
        DatasetCreationPlanner.relativePath(of: url, root: root)
    }

    public var directoryRelativePath: String { relativePath(of: directory) }

    /// The `DatasetInventoryEntry.ID` this dataset will have once it exists —
    /// how the flow lands the researcher on the new row after a re-scan.
    /// Built through phase 1's own id function, not a second formula.
    public var inventoryEntryID: String {
        // Flat-file families share a directory, so the row they produce is
        // identified by its FILE — the same branch the scan takes.
        if kind.identifiesByFile, let file = files.first {
            return DatasetInventoryEntry.id(kind: kind, fileURL: file.url)
        }
        return DatasetInventoryEntry.id(kind: kind, directory: directory)
    }

    public func file(for slot: DatasetFileSlot) -> DatasetPlannedFile? {
        files.first { $0.slot == slot }
    }

    // MARK: Apply

    /// Copy the chosen files into the canonical destination.
    ///
    /// Transactional in the way that matters: EVERY file is staged and parsed
    /// before ANY file is published, so a malformed second file cannot leave
    /// a validated first one half-installed. Publication itself is one atomic
    /// same-volume rename per file; the staging directory is removed on every
    /// exit path.
    ///
    /// - Parameters:
    ///   - imports: source file per slot. A slot may be omitted only when the
    ///     canonical destination already holds that file — a dataset must not
    ///     come into existence half-formed.
    ///   - replacingExisting: required to overwrite. Without it, an existing
    ///     destination is a refusal, never a silent overwrite.
    @discardableResult
    public func apply(
        imports: [DatasetFileSlot: URL], replacingExisting: Bool = false
    ) throws -> DatasetCreationOutcome {
        guard requirements.isEmpty else {
            throw DatasetCreationError.unresolvedRequirements(requirements)
        }
        guard !files.isEmpty else { throw DatasetCreationError.nothingToImport }
        guard !imports.isEmpty else { throw DatasetCreationError.nothingToImport }

        let plannedSlots = Set(files.map(\.slot))
        for slot in imports.keys where !plannedSlots.contains(slot) {
            throw DatasetCreationError.unexpectedSlot(slot)
        }

        // Collisions are re-stated HERE, not read off the plan's snapshot: a
        // plan can be minutes old, and "it did not exist when I previewed it"
        // is not a licence to overwrite.
        let fm = FileManager.default
        func present(_ file: DatasetPlannedFile) -> Bool {
            fm.fileExists(atPath: file.url.path)
        }

        // A dataset is complete or it is not created: every slot must be
        // imported now or already present on disk.
        let missing = files.filter { imports[$0.slot] == nil && !present($0) }
        guard missing.isEmpty else {
            throw DatasetCreationError.incompleteSet(missing.map(\.slot))
        }

        let colliding = files.filter { imports[$0.slot] != nil && present($0) }
        if !colliding.isEmpty, !replacingExisting {
            throw DatasetCreationError.destinationExists(colliding.map(\.relativePath))
        }

        try fm.createDirectory(at: directory, withIntermediateDirectories: true)

        // Sibling of the destination so the publish is a same-volume rename.
        let staging = directory.appending(
            component: ".steerlab-dataset-import-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: staging) }
        try fm.createDirectory(at: staging, withIntermediateDirectories: true)

        // 1. Stage under the CANONICAL filename, and validate there.
        var staged: [DatasetFileSlot: URL] = [:]
        var previews: [DatasetFileSlot: DatasetImportPreview] = [:]
        for planned in files {
            guard let source = imports[planned.slot] else { continue }
            let slotDirectory = staging.appending(component: planned.slot.rawValue)
            try fm.createDirectory(at: slotDirectory, withIntermediateDirectories: true)
            let target = slotDirectory.appending(component: planned.filename)
            do {
                try fm.copyItem(at: source, to: target)
            } catch {
                throw DatasetCreationError.sourceUnreadable(
                    slot: planned.slot, path: source.path)
            }
            previews[planned.slot] = try DatasetCreationPlanner.validate(
                fileAt: target, as: planned.slot, concept: name)
            staged[planned.slot] = target
        }

        // 2. Publish. Nothing above here wrote into the destination.
        var written: [DatasetFileSlot: URL] = [:]
        for planned in files {
            guard let source = staged[planned.slot] else { continue }
            if fm.fileExists(atPath: planned.url.path) {
                _ = try fm.replaceItemAt(planned.url, withItemAt: source)
            } else {
                try fm.moveItem(at: source, to: planned.url)
            }
            written[planned.slot] = planned.url
        }

        return DatasetCreationOutcome(
            plan: self,
            written: written,
            previews: previews,
            inventoryEntryID: inventoryEntryID)
    }

    /// Create the destination DIRECTORY and nothing else — the "author it in
    /// the builder" path, where the files arrive later from the Concept Lab.
    /// Deliberately empty structure: no seed rows, no template file.
    @discardableResult
    public func createDirectory() throws -> URL {
        guard requirements.isEmpty else {
            throw DatasetCreationError.unresolvedRequirements(requirements)
        }
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        return directory
    }
}

public enum DatasetCreationVerb: String, Sendable, Equatable {
    case create
    case addTo
    case replace

    public var label: String {
        switch self {
        case .create: "Create"
        case .addTo: "Add to"
        case .replace: "Replace"
        }
    }
}

public struct DatasetImportPreview: Sendable, Equatable {
    public let slot: DatasetFileSlot
    public let rowCount: Int
    public let byteSize: Int64
    /// SHA-256 of the bytes as they will land — the same digest the family's
    /// loader and the inventory report for the published file.
    public let contentHash: String
}

public struct DatasetCreationOutcome: Sendable {
    public let plan: DatasetCreationPlan
    public let written: [DatasetFileSlot: URL]
    public let previews: [DatasetFileSlot: DatasetImportPreview]
    /// The inventory row to select after the re-scan.
    public let inventoryEntryID: String

    public var totalRows: Int { previews.values.reduce(0) { $0 + $1.rowCount } }
}

// MARK: - Errors

public enum DatasetCreationError: Error, CustomStringConvertible, Equatable {
    case unresolvedRequirements([DatasetCreationRequirement])
    case nothingToImport
    case unexpectedSlot(DatasetFileSlot)
    case incompleteSet([DatasetFileSlot])
    case destinationExists([String])
    case sourceUnreadable(slot: DatasetFileSlot, path: String)
    case importRejected(slot: DatasetFileSlot, reason: String)

    public var description: String {
        switch self {
        case .unresolvedRequirements(let requirements):
            "the dataset is not fully declared: "
                + requirements.map(\.message).joined(separator: "; ")
        case .nothingToImport:
            "no file was chosen to import"
        case .unexpectedSlot(let slot):
            "\(slot.displayFilename) is not part of this dataset role"
        case .incompleteSet(let slots):
            "this dataset needs "
                + slots.map(\.displayFilename).joined(separator: " and ")
                + " — a half-written set is not created"
        case .destinationExists(let paths):
            paths.joined(separator: ", ")
                + " already exists — confirm the replacement, or choose a "
                + "different name"
        case .sourceUnreadable(let slot, let path):
            "could not read the file chosen for \(slot.displayFilename): \(path)"
        case .importRejected(let slot, let reason):
            "\(slot.displayFilename) was not imported — \(reason)"
        }
    }
}
